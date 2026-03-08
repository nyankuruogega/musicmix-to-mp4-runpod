#!/bin/bash
# ============================================================
#  Music → MP4  |  RunPod Start Script (GPU-Accelerated)
#  - Uses h264_nvenc (NVIDIA GPU) for fast video encoding
#  - Flux Dev / Flux Schnell for AI background generation
#  - Falls back to CPU if no GPU available
#
#  Required:
#    - Expose port 7860
#    - Set env var GROQ_API_KEY (optional but recommended)
#    - GPU pod: A100 40GB for Flux Dev, RTX 4090/3090 for Flux Schnell
#    - HuggingFace token env var: HF_TOKEN (for Flux model download)
# ============================================================

set -e

echo "===== Installing system dependencies ====="
apt-get update -qq && apt-get install -y -qq ffmpeg

echo "===== Installing Python packages ====="
pip install -q gradio groq torch torchvision diffusers transformers accelerate sentencepiece protobuf huggingface_hub

echo "===== Detecting GPU ====="
if nvidia-smi &>/dev/null; then
    echo "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
    echo "No GPU detected - CPU fallback active"
fi

echo "===== Writing app ====="
cat > /app.py << 'PYEOF'
import os
import sys
import types

for _mod in ('audioop', 'pyaudioop'):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

import gradio as gr
import zipfile
import shutil
import tempfile
import subprocess
import json
from pathlib import Path

SUPPORTED_AUDIO = {'.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a', '.opus'}

# ── Style presets ──────────────────────────────────────────────────────────────
STYLE_PRESETS = {
    "None":                   "",
    "Cinematic":              "cinematic film still, dramatic lighting, anamorphic lens, golden hour, ultra detailed, 8k",
    "African Art":            "vibrant African traditional art, bold geometric patterns, warm earth tones, tribal motifs, richly colored",
    "Watercolor":             "soft watercolor painting, flowing colors, artistic brushstrokes, delicate washes, painterly",
    "Dark Mystical":          "dark mystical atmosphere, ethereal fog, moonlit, ancient and mysterious, dramatic shadows, deep colors",
    "Kenyan Landscape":       "sweeping Kenyan savanna, acacia trees, golden sunset, Mount Kenya in distance, vast plains, photorealistic",
    "Abstract Music":         "abstract colorful music visualization, sound waves, flowing neon colors, dynamic energy, digital art",
    "Vintage Poster":         "vintage retro music poster, aged texture, classic typography style, warm sepia tones, nostalgic",
    "Neon City":              "neon-lit city at night, rain-soaked streets, reflections, cyberpunk atmosphere, vivid colors",
    "Cultural Celebration":   "joyful African cultural celebration, colorful traditional clothing, festive atmosphere, warm community",
}

# ── Encoder detection ──────────────────────────────────────────────────────────
def detect_gpu_encoder():
    try:
        test = subprocess.run(
            ['ffmpeg', '-hide_banner', '-f', 'lavfi', '-i', 'color=black:s=128x128:d=1',
             '-c:v', 'h264_nvenc', '-f', 'null', '-'],
            capture_output=True, text=True, timeout=10
        )
        if test.returncode == 0:
            return 'h264_nvenc', True
    except Exception:
        pass
    return 'libx264', False


# ── Audio helpers ──────────────────────────────────────────────────────────────
def extract_zip(zip_path, extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)
    audio_files = []
    for root, dirs, files in os.walk(extract_dir):
        dirs.sort()
        for f in sorted(files):
            if Path(f).suffix.lower() in SUPPORTED_AUDIO:
                audio_files.append(os.path.join(root, f))
    return sorted(audio_files)


def get_audio_duration(audio_path):
    result = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', audio_path],
        capture_output=True, text=True
    )
    info = json.loads(result.stdout)
    for stream in info.get('streams', []):
        if stream.get('codec_type') == 'audio':
            return float(stream.get('duration', 0))
    return 0


def build_visualizer_filter(visualizer_type, width, height):
    if visualizer_type == "None":
        return None
    vis_map = {
        "Waveform":       f"showwaves=s={width}x{height//4}:mode=line:colors=0x00FF88:scale=sqrt,format=yuva420p",
        "Spectrum":       f"showspectrum=s={width}x{height//4}:mode=combined:color=rainbow:scale=sqrt,format=yuva420p",
        "Frequency Bars": f"avectorscope=s={width}x{height//4}:zoom=1.5:rc=0:gc=200:bc=255:rf=0:gf=40:bf=80,format=yuva420p",
        "Circular":       f"showwaves=s={width}x{height//4}:mode=p2p:colors=0xFF6B35:scale=lin,format=yuva420p",
    }
    return vis_map.get(visualizer_type)


def format_timestamp(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def build_tracklist(audio_files):
    tracks = []
    cursor = 0.0
    for af in audio_files:
        dur = get_audio_duration(af)
        tracks.append((Path(af).stem, cursor, dur))
        cursor += dur
    return tracks


def write_tracklist_file(tracks, output_path):
    lines = ["TRACKLIST", "=" * 44, ""]
    for i, (name, start, _) in enumerate(tracks, 1):
        lines.append(f"{i:>2}. [{format_timestamp(start)}]  {name}")
    lines += ["", "=" * 44,
        f"Total tracks : {len(tracks)}",
        f"Total length : {format_timestamp(sum(d for _, _, d in tracks))}"]
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


# ── Flux image generation ──────────────────────────────────────────────────────
_flux_pipe = None
_flux_loaded_model = None

def load_flux(model_choice):
    global _flux_pipe, _flux_loaded_model
    import torch
    from diffusers import FluxPipeline

    model_id = "black-forest-labs/FLUX.1-dev" if model_choice == "Flux Dev" else "black-forest-labs/FLUX.1-schnell"

    if _flux_pipe is not None and _flux_loaded_model == model_id:
        return _flux_pipe  # already loaded

    # Unload previous model if switching
    if _flux_pipe is not None:
        del _flux_pipe
        torch.cuda.empty_cache()

    hf_token = os.environ.get("HF_TOKEN", None)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    _flux_pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        token=hf_token,
    )

    if torch.cuda.is_available():
        _flux_pipe = _flux_pipe.to("cuda")
    else:
        _flux_pipe.enable_sequential_cpu_offload()

    _flux_loaded_model = model_id
    return _flux_pipe


def generate_background(prompt, style_preset, model_choice, width, height, progress=gr.Progress()):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter an image prompt.")

    progress(0.1, desc=f"Loading {model_choice}...")

    # Build full prompt with style
    style_suffix = STYLE_PRESETS.get(style_preset, "")
    full_prompt = f"{prompt.strip()}, {style_suffix}".strip(", ") if style_suffix else prompt.strip()

    try:
        import torch
        pipe = load_flux(model_choice)

        progress(0.4, desc="Generating image...")

        # Flux Schnell uses fewer steps
        num_steps = 4 if model_choice == "Flux Schnell" else 28
        guidance = 0.0 if model_choice == "Flux Schnell" else 3.5

        result = pipe(
            prompt=full_prompt,
            width=width,
            height=height,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
        )

        image = result.images[0]

        # Save to temp file
        img_path = os.path.join(tempfile.gettempdir(), "flux_background.png")
        image.save(img_path)

        progress(1.0, desc="Done!")
        return img_path, img_path, f"Generated with {model_choice}\nPrompt: {full_prompt}"

    except Exception as e:
        raise gr.Error(f"Image generation failed: {str(e)}")


# ── AI metadata ────────────────────────────────────────────────────────────────
def generate_metadata(groq_api_key, mix_type, channel_name, tracks):
    if not groq_api_key or not groq_api_key.strip():
        raise gr.Error("Groq API key not found. Set GROQ_API_KEY env var or enter it manually.")
    if not mix_type or not mix_type.strip():
        raise gr.Error("Please enter the mix type.")
    if not tracks:
        raise gr.Error("No tracklist found. Please generate the MP4 first.")
    from groq import Groq
    tracklist_str = "\n".join([f"{i}. [{format_timestamp(s)}] {n}" for i, (n, s, _) in enumerate(tracks, 1)])
    total_duration = format_timestamp(sum(d for _, _, d in tracks))
    channel_line = f"Channel: {channel_name.strip()}" if channel_name and channel_name.strip() else ""
    prompt = f"""You are a YouTube content strategist specializing in African music compilations.

I have created a YouTube music compilation video with the following details:
- Mix type: {mix_type}
- Total duration: {total_duration}
- Number of tracks: {len(tracks)}
{channel_line}

Tracklist:
{tracklist_str}

Please generate the following:

1. TITLES: Give me exactly 5 compelling YouTube video title options for this mix. Make them engaging, include relevant keywords, and vary the style (some with emojis, some without, some with year/era references). Number them 1-5.

2. DESCRIPTION: Write one complete YouTube video description that includes:
   - An engaging opening paragraph about this mix (2-3 sentences)
   - The full tracklist with timestamps (use the exact timestamps provided above)
   - A line encouraging viewers to like, subscribe and turn on notifications{"for " + channel_name.strip() if channel_name and channel_name.strip() else ""}
   - Relevant hashtags at the end (15-20 hashtags relevant to the music genre, African music, and the specific mix type)

Format your response exactly like this:
---TITLES---
1. [title]
2. [title]
3. [title]
4. [title]
5. [title]
---DESCRIPTION---
[full description here]
"""
    client = Groq(api_key=groq_api_key.strip())
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8, max_tokens=2000,
    )
    raw = response.choices[0].message.content
    if "---TITLES---" in raw and "---DESCRIPTION---" in raw:
        parts = raw.split("---DESCRIPTION---")
        return parts[0].replace("---TITLES---", "").strip(), parts[1].strip()
    return raw, raw


# ── Video process ──────────────────────────────────────────────────────────────
VIDEO_ENCODER, USING_GPU = detect_gpu_encoder()
print(f"[Encoder] Using: {VIDEO_ENCODER} ({'GPU' if USING_GPU else 'CPU'})")

_last_tracks = []

def process(zip_file, bg_image, visualizer_type, video_quality, resolution, progress=gr.Progress(track_tqdm=True)):
    global _last_tracks
    if zip_file is None:
        raise gr.Error("Please upload a ZIP file containing music.")
    def get_path(f):
        return f if isinstance(f, str) else f.name

    progress(0, desc="Setting up workspace...")
    work_dir = tempfile.mkdtemp()
    extract_dir = os.path.join(work_dir, "audio")
    os.makedirs(extract_dir)

    try:
        progress(0.05, desc="Extracting ZIP...")
        audio_files = extract_zip(get_path(zip_file), extract_dir)
        if not audio_files:
            raise gr.Error("No supported audio files found.")

        progress(0.1, desc=f"Found {len(audio_files)} files. Reading durations...")
        tracks = build_tracklist(audio_files)
        _last_tracks = tracks
        tracklist_path = os.path.join(tempfile.gettempdir(), "tracklist.txt")
        write_tracklist_file(tracks, tracklist_path)

        progress(0.15, desc="Concatenating audio...")
        concat_list = os.path.join(work_dir, "concat.txt")
        with open(concat_list, 'w') as f:
            for af in audio_files:
                f.write(f"file '{af}'\n")
        combined_audio = os.path.join(work_dir, "combined.wav")
        subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', concat_list, '-ar', '44100', '-ac', '2', combined_audio],
            check=True, capture_output=True)

        progress(0.3, desc="Preparing video...")
        res_map = {"1920x1080 (Full HD)": (1920, 1080), "1280x720 (HD)": (1280, 720), "854x480 (SD)": (854, 480)}
        width, height = res_map.get(resolution, (1280, 720))
        output_path = os.path.join(work_dir, "output.mp4")

        if USING_GPU:
            qp = {"High (slow)": "18", "Medium": "23", "Low (fast)": "28"}.get(video_quality, "23")
            encode_params = ['-c:v', 'h264_nvenc', '-qp', qp, '-preset', 'fast', '-gpu', '0']
        else:
            crf = {"High (slow)": "18", "Medium": "23", "Low (fast)": "28"}.get(video_quality, "23")
            encode_params = ['-c:v', 'libx264', '-crf', crf, '-preset', 'fast', '-threads', '0']

        if bg_image is not None:
            bg_input = ['-loop', '1', '-i', get_path(bg_image)]
            bg_filter = f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},setsar=1[bg]"
        else:
            bg_input = []
            bg_filter = f"color=c=0x0D0D1A:s={width}x{height}:r=24[bg]"

        vis_filter = build_visualizer_filter(visualizer_type, width, height)
        audio_input_idx = 1 if bg_image else 0

        if vis_filter:
            vis_offset_y = int(height * 0.75)
            filter_complex = (f"{bg_filter};"
                f"[{audio_input_idx}:a]{vis_filter}[vis];"
                f"[bg][vis]overlay=0:{vis_offset_y}:format=auto[outv]")
        else:
            filter_complex = f"{bg_filter};[bg]setpts=PTS-STARTPTS[outv]" if bg_image else f"{bg_filter}[outv]"

        encoder_label = "GPU (NVENC)" if USING_GPU else "CPU"
        progress(0.4, desc=f"Rendering MP4 with {encoder_label}...")

        cmd = ['ffmpeg', '-y', *bg_input, '-i', combined_audio,
            '-filter_complex', filter_complex,
            '-map', '[outv]', '-map', f'{audio_input_idx}:a',
            *encode_params,
            '-c:a', 'aac', '-b:a', '192k', '-shortest', output_path]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0 and USING_GPU:
            progress(0.4, desc="GPU encode failed, retrying with CPU...")
            cpu_cmd = ['ffmpeg', '-y', *bg_input, '-i', combined_audio,
                '-filter_complex', filter_complex,
                '-map', '[outv]', '-map', f'{audio_input_idx}:a',
                '-c:v', 'libx264', '-crf', '23', '-preset', 'fast', '-threads', '0',
                '-c:a', 'aac', '-b:a', '192k', '-shortest', output_path]
            result = subprocess.run(cpu_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise gr.Error(f"FFmpeg error:\n{result.stderr[-1000:]}")

        progress(0.95, desc="Finalizing...")
        final_output = os.path.join(tempfile.gettempdir(), "music_video_output.mp4")
        shutil.copy2(output_path, final_output)
        song_list = "\n".join([f"  [{format_timestamp(s)}]  {n}" for n, s, _ in tracks])
        progress(1.0, desc="Complete!")
        return final_output, tracklist_path, f"Done! {len(audio_files)} songs | Encoder: {encoder_label}\n\nTracklist:\n{song_list}"

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def run_ai_metadata(groq_api_key, mix_type, channel_name):
    global _last_tracks
    key = os.environ.get("GROQ_API_KEY", "").strip() or groq_api_key
    return generate_metadata(key, mix_type, channel_name, _last_tracks)


# ── UI ─────────────────────────────────────────────────────────────────────────
css = """
body, .gradio-container { background: #080810 !important; font-family: 'Segoe UI', Ubuntu, sans-serif !important; color: #e8e8f0 !important; }
.gradio-container { max-width: 980px !important; margin: 0 auto !important; }
h1.app-title { font-size: 2.6rem; font-weight: 800; background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin: 1.5rem 0 0.3rem; letter-spacing: -1px; }
p.subtitle { text-align: center; color: #6b7280; font-size: 0.9rem; margin-bottom: 0.5rem; font-family: monospace; }
.panel { background: #0f0f1f !important; border: 1px solid #1e1e3f !important; border-radius: 16px !important; padding: 1.2rem !important; }
label { color: #a78bfa !important; font-size: 0.78rem !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.08em !important; font-family: monospace !important; }
.generate-btn { background: linear-gradient(135deg, #7c3aed, #2563eb) !important; border: none !important; border-radius: 12px !important; font-size: 1.05rem !important; font-weight: 700 !important; color: white !important; padding: 0.85rem 2rem !important; transition: all 0.2s !important; }
.generate-btn:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 30px rgba(124,58,237,0.4) !important; }
.ai-btn { background: linear-gradient(135deg, #059669, #0284c7) !important; border: none !important; border-radius: 12px !important; font-size: 1.05rem !important; font-weight: 700 !important; color: white !important; padding: 0.85rem 2rem !important; transition: all 0.2s !important; }
.ai-btn:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 30px rgba(5,150,105,0.4) !important; }
.flux-btn { background: linear-gradient(135deg, #b45309, #92400e) !important; border: none !important; border-radius: 12px !important; font-size: 1.05rem !important; font-weight: 700 !important; color: white !important; padding: 0.85rem 2rem !important; transition: all 0.2s !important; }
.flux-btn:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 30px rgba(180,83,9,0.4) !important; }
.use-btn { background: linear-gradient(135deg, #7c3aed, #059669) !important; border: none !important; border-radius: 10px !important; font-size: 0.95rem !important; font-weight: 700 !important; color: white !important; padding: 0.6rem 1.5rem !important; }
footer { display: none !important; }
"""

gpu_status = "⚡ GPU Encoding Active (h264_nvenc)" if USING_GPU else "🖥️ CPU Encoding (no GPU detected)"
gpu_color = "#34d399" if USING_GPU else "#f59e0b"

# Shared state: path to the generated background image
generated_bg_path = gr.State(None)

with gr.Blocks(title="Music to MP4") as demo:
    gr.HTML(f"""
        <h1 class='app-title'>Music to MP4</h1>
        <p class='subtitle'>ZIP of songs → MP4 video + AI background + YouTube metadata</p>
        <p style='text-align:center;font-size:0.82rem;color:{gpu_color};font-family:monospace;margin-bottom:0.8rem'>{gpu_status}</p>
    """)

    # Shared state for generated image path
    flux_image_path = gr.State(value=None)

    with gr.Tabs():

        # ── TAB 1: AI Background Generator ──────────────────────────────────
        with gr.Tab("🎨 AI Background"):
            gr.HTML("<p style='color:#6b7280;font-size:0.85rem;margin-bottom:1rem'>Generate a background image with Flux. Then click <b>Use as Background</b> to send it to the video generator.</p>")
            with gr.Row():
                with gr.Column(scale=1, elem_classes="panel"):
                    gr.Markdown("### 🖼️ Image Settings")
                    flux_model = gr.Dropdown(
                        label="Flux Model",
                        choices=["Flux Schnell", "Flux Dev"],
                        value="Flux Schnell",
                        info="Schnell = fast (4 steps). Dev = higher quality (28 steps)."
                    )
                    flux_prompt = gr.Textbox(
                        label="Image Prompt",
                        placeholder="e.g. Kenyan warriors dancing at sunset, savanna landscape, dramatic sky",
                        lines=3
                    )
                    flux_style = gr.Dropdown(
                        label="Style Preset",
                        choices=list(STYLE_PRESETS.keys()),
                        value="Cinematic"
                    )
                    flux_res = gr.Dropdown(
                        label="Image Resolution",
                        choices=["1280x720 (HD)", "1920x1080 (Full HD)", "854x480 (SD)"],
                        value="1280x720 (HD)"
                    )
                    gr.HTML("<p style='color:#6b7280;font-size:0.75rem;margin-top:0.5rem'>⚠️ Flux Dev requires HF_TOKEN env var and 24GB+ VRAM. Flux Schnell needs ~12GB VRAM.</p>")

                with gr.Column(scale=1, elem_classes="panel"):
                    gr.Markdown("### 👁️ Preview")
                    flux_preview = gr.Image(label="Generated Background", interactive=False)
                    flux_status = gr.Textbox(label="Status", lines=2, interactive=False)
                    use_as_bg_btn = gr.Button("✅ Use as Background in Video", elem_classes="use-btn")

            flux_btn = gr.Button("🎨 Generate Background Image", elem_classes="flux-btn")

            def flux_res_to_dims(res_str):
                m = {"1280x720 (HD)": (1280, 720), "1920x1080 (Full HD)": (1920, 1080), "854x480 (SD)": (854, 480)}
                return m.get(res_str, (1280, 720))

            def run_flux(prompt, style, model, res_str, progress=gr.Progress()):
                w, h = flux_res_to_dims(res_str)
                img_path, _, status = generate_background(prompt, style, model, w, h, progress)
                return img_path, img_path, status

            flux_btn.click(
                fn=run_flux,
                inputs=[flux_prompt, flux_style, flux_model, flux_res],
                outputs=[flux_preview, flux_image_path, flux_status]
            )

        # ── TAB 2: Video Generator ───────────────────────────────────────────
        with gr.Tab("🎬 Generate Video"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes="panel"):
                    gr.Markdown("### 📁 Input")
                    zip_input = gr.File(label="ZIP File (audio files)", file_types=[".zip"])
                    gr.Markdown("**Background Image**")
                    bg_image = gr.File(label="Upload Image (or use AI Generated below)", file_types=[".jpg", ".jpeg", ".png", ".webp"])
                    ai_bg_display = gr.Image(label="AI Generated Background (active)", interactive=False)
                    clear_ai_bg_btn = gr.Button("✖ Clear AI Background", size="sm")

                with gr.Column(scale=1, elem_classes="panel"):
                    gr.Markdown("### ⚙️ Settings")
                    visualizer = gr.Dropdown(label="Visualizer", choices=["None", "Waveform", "Spectrum", "Frequency Bars", "Circular"], value="Waveform")
                    resolution = gr.Dropdown(label="Resolution", choices=["1920x1080 (Full HD)", "1280x720 (HD)", "854x480 (SD)"], value="1280x720 (HD)")
                    quality = gr.Dropdown(label="Video Quality", choices=["High (slow)", "Medium", "Low (fast)"], value="Medium")

            generate_btn = gr.Button("🎬 Generate MP4", elem_classes="generate-btn", variant="primary")

            with gr.Column(elem_classes="panel"):
                output_video = gr.Video(label="Output MP4", interactive=False)
                output_tracklist = gr.File(label="Download Tracklist", interactive=False)
                output_log = gr.Textbox(label="Summary", lines=8, interactive=False)

            # Wire "Use as Background" button from AI tab
            def apply_ai_bg(img_path):
                return img_path, img_path

            use_as_bg_btn.click(
                fn=apply_ai_bg,
                inputs=[flux_image_path],
                outputs=[ai_bg_display, flux_image_path]
            )

            clear_ai_bg_btn.click(
                fn=lambda: (None, None),
                outputs=[ai_bg_display, flux_image_path]
            )

            # Process uses uploaded bg_image OR ai-generated path (ai takes priority if set)
            def process_with_ai_bg(zip_file, uploaded_bg, ai_bg_path, visualizer_type, video_quality, resolution, progress=gr.Progress(track_tqdm=True)):
                # AI generated background takes priority over manual upload
                effective_bg = ai_bg_path if ai_bg_path else uploaded_bg

                class FakePath:
                    def __init__(self, p): self.name = p

                if effective_bg and isinstance(effective_bg, str):
                    effective_bg = FakePath(effective_bg)

                return process(zip_file, effective_bg, visualizer_type, video_quality, resolution, progress)

            generate_btn.click(
                fn=process_with_ai_bg,
                inputs=[zip_input, bg_image, flux_image_path, visualizer, quality, resolution],
                outputs=[output_video, output_tracklist, output_log]
            )
            gr.HTML("<p style='text-align:center;color:#374151;font-size:0.78rem;margin-top:0.8rem;font-family:monospace'>Supported: MP3 · WAV · OGG · FLAC · AAC · M4A · OPUS</p>")

        # ── TAB 3: AI YouTube Metadata ───────────────────────────────────────
        with gr.Tab("✨ AI YouTube Metadata"):
            gr.HTML("<p style='color:#6b7280;font-size:0.85rem;margin-bottom:1rem'>Generate titles and a full description using your tracklist. Run the video generator first.</p>")
            with gr.Row():
                with gr.Column(elem_classes="panel"):
                    gr.Markdown("### 🔑 Setup")
                    groq_key = gr.Textbox(label="Groq API Key (optional if GROQ_API_KEY env var set)", placeholder="gsk_...", type="password")
                    channel_name = gr.Textbox(label="YouTube Channel Name", placeholder="e.g. Nyankuru Stories")
                    mix_type = gr.Textbox(label="Mix Type / Description", placeholder="e.g. Kenyan Benga Classics, 90s East African Rhumba...", lines=2)
            ai_btn = gr.Button("✨ Generate Titles and Description", elem_classes="ai-btn")
            with gr.Row():
                with gr.Column(elem_classes="panel"):
                    output_titles = gr.Textbox(label="5 Title Options", lines=8, interactive=True)
                with gr.Column(elem_classes="panel"):
                    output_description = gr.Textbox(label="YouTube Description", lines=20, interactive=True)
            ai_btn.click(fn=run_ai_metadata, inputs=[groq_key, mix_type, channel_name], outputs=[output_titles, output_description])

demo.launch(server_name="0.0.0.0", server_port=7860, css=css)
PYEOF

echo "===== Launching app ====="
python /app.py
