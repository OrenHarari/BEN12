from __future__ import annotations

import io
import subprocess
import sys
import tempfile
import threading
import time
from functools import lru_cache
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (
    compute_transition_plan,
    get_config,
    perceptual_duration_options,
)
from app.state import get_state, reset_state
from utils.device import DeviceManager
from utils.ffmpeg import resolve_ffmpeg_binary

st.set_page_config(
    page_title="AI Growing Up Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CSS = """
<style>
.main-title { font-size: 2.4rem; font-weight: 700; margin-bottom: 0.2rem; }
.subtitle   { color: #888; margin-bottom: 1.5rem; }
div[data-testid="stProgress"] > div { height: 12px; border-radius: 6px; }
.caption-input input { font-size: 0.85rem; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# ── Resolution presets ─────────────────────────────────────────────────
RESOLUTION_MAP = {
    "720p":  (1280, 720),
    "1080p": (1920, 1080),
    "4K":    (3840, 2160),
}
MAX_UPLOAD_EDGE = 2048
TRANSITION_STYLE_OPTIONS = ["Emotional", "Balanced", "Fast"]
TRANSITION_DURATION_OPTIONS = perceptual_duration_options()

PERFORMANCE_PRESETS = {
    "fast": {
        "sdxl_steps": 18,
        "refiner_steps": 0,
        "rife_multiplier": 1,
        "ffmpeg_preset": "fast",
        "ffmpeg_gpu_preset": "p4",
        "enable_gfpgan": False,
        "enable_refiner": False,
    },
    "balanced": {
        "sdxl_steps": 25,
        "refiner_steps": 15,
        "rife_multiplier": 2,
        "ffmpeg_preset": "medium",
        "ffmpeg_gpu_preset": "p5",
        "enable_gfpgan": True,
        "enable_refiner": True,
    },
    "quality": {
        "sdxl_steps": 35,
        "refiner_steps": 25,
        "rife_multiplier": 4,
        "ffmpeg_preset": "slow",
        "ffmpeg_gpu_preset": "p6",
        "enable_gfpgan": True,
        "enable_refiner": True,
    },
    "extreme_speed_3090": {
        "sdxl_steps": 8,
        "refiner_steps": 0,
        "rife_multiplier": 1,
        "ffmpeg_preset": "fast",
        "ffmpeg_gpu_preset": "p1",
        "enable_gfpgan": False,
        "enable_refiner": False,
    },
    "extreme_quality_3090": {
        "sdxl_steps": 24,
        "refiner_steps": 18,
        "rife_multiplier": 2,
        "ffmpeg_preset": "slow",
        "ffmpeg_gpu_preset": "p5",
        "enable_gfpgan": True,
        "enable_refiner": True,
    },
}
PERFORMANCE_PRESET_ORDER = [
    "fast",
    "balanced",
    "quality",
    "extreme_speed_3090",
    "extreme_quality_3090",
]


@lru_cache(maxsize=8)
def _ffmpeg_supports_encoder(encoder_name: str) -> bool:
    try:
        res = subprocess.run(
            [resolve_ffmpeg_binary(), "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return False
    if res.returncode != 0:
        return False
    return encoder_name in (res.stdout + "\n" + res.stderr)


def _safe_rerun() -> None:
    """
    Support both old and new Streamlit rerun APIs.
    """
    if hasattr(st, "rerun"):
        st.rerun()
        return
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
        return
    raise RuntimeError("This Streamlit build does not support rerun APIs.")


def _nearest_duration_option(value: float) -> float:
    return min(TRANSITION_DURATION_OPTIONS, key=lambda x: abs(x - float(value)))


def _sidebar(state):
    st.sidebar.header("⚙️ Settings")

    # Resolution selector
    state.resolution = st.sidebar.selectbox(
        "Resolution",
        options=list(RESOLUTION_MAP.keys()),
        index=list(RESOLUTION_MAP.keys()).index(state.resolution),
        help="Higher resolution = larger file & longer render time",
        disabled=state.is_processing,
    )

    # Transition timing system (style + exact duration)
    state.transition_style = st.sidebar.selectbox(
        "Transition Style",
        options=TRANSITION_STYLE_OPTIONS,
        index=TRANSITION_STYLE_OPTIONS.index(state.transition_style)
        if state.transition_style in TRANSITION_STYLE_OPTIONS
        else 1,
        help="Emotional=cinematic slow, Balanced=natural, Fast=snappy.",
        disabled=state.is_processing,
    )
    state.transition_duration_seconds = st.sidebar.select_slider(
        "Transition Duration (seconds)",
        options=TRANSITION_DURATION_OPTIONS,
        value=_nearest_duration_option(state.transition_duration_seconds),
        format_func=lambda x: f"{x:.2f}s",
        help="Per-transition exact duration (0.2s to 12s).",
        disabled=state.is_processing,
    )

    # Fade in/out toggle
    state.fade_enabled = st.sidebar.checkbox(
        "Fade In / Fade Out",
        value=state.fade_enabled,
        help="Add a cinematic black fade at the beginning and end of the video",
        disabled=state.is_processing,
    )

    # Performance preset
    st.sidebar.divider()
    st.sidebar.subheader("⚡ Performance")
    state.performance_mode = st.sidebar.selectbox(
        "Render Preset",
        options=PERFORMANCE_PRESET_ORDER,
        index=PERFORMANCE_PRESET_ORDER.index(state.performance_mode)
        if state.performance_mode in PERFORMANCE_PRESET_ORDER
        else 1,
        help="Fast = quick preview. Quality = best detail, slower render.",
        disabled=state.is_processing,
    )
    state.turbo_mode = st.sidebar.checkbox(
        "Super Turbo (large albums)",
        value=state.turbo_mode,
        help="Aggressively reduces morph workload for much faster renders.",
        disabled=state.is_processing,
    )
    state.transition_process_workers = st.sidebar.slider(
        "Transition Workers (Processes)",
        min_value=1,
        max_value=3,
        value=int(state.transition_process_workers),
        help="Parallel CPU transition generation. 3 is best for large albums.",
        disabled=state.is_processing,
    )
    state.chunked_parallel = st.sidebar.checkbox(
        "Chunked Parallel Transitions",
        value=state.chunked_parallel,
        help="Generate transition chunks in parallel processes, then stream-encode in order.",
        disabled=state.is_processing,
    )
    state.transition_chunk_size = st.sidebar.slider(
        "Chunk Size (transitions)",
        min_value=1,
        max_value=6,
        value=int(state.transition_chunk_size),
        help="How many transitions each process computes per chunk.",
        disabled=state.is_processing or not state.chunked_parallel,
    )

    # Background music upload
    st.sidebar.divider()
    st.sidebar.subheader("🎵 Background Music")
    music_file = st.sidebar.file_uploader(
        "Upload MP3 (optional)",
        type=["mp3", "wav", "m4a", "ogg"],
        help="Audio will be added as background music, trimmed to video length",
        disabled=state.is_processing,
    )
    if music_file is not None:
        music_bytes = music_file.read()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.write(music_bytes)
        tmp.close()
        state.music_path = tmp.name
        st.sidebar.success(f"✓ {music_file.name}")
    elif not music_file:
        state.music_path = None

    st.sidebar.divider()
    show_system = st.sidebar.checkbox(
        "Show system info",
        value=False,
        disabled=state.is_processing,
    )
    if show_system:
        st.sidebar.header("System")
        device = DeviceManager.get_device()
        st.sidebar.metric("Device", device.type.upper())
        if device.type == "cuda":
            import torch
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / 1e9
            st.sidebar.metric("GPU", props.name)
            st.sidebar.metric("VRAM", f"{vram_gb:.1f} GB")
        elif device.type == "mps":
            st.sidebar.info("Apple Silicon (MPS)")
        else:
            st.sidebar.warning("No GPU — CPU mode (slower)")

        st.sidebar.divider()
        st.sidebar.caption(
            "All processing happens locally.  \n"
            "No data is sent to any server."
        )


def _run_pipeline(images, captions, config, state):
    try:
        from pipeline.orchestrator import GrowingUpPipeline

        # Apply user settings to config
        w, h = RESOLUTION_MAP[state.resolution]
        config.video.output_width = w
        config.video.output_height = h
        preset = PERFORMANCE_PRESETS[state.performance_mode]
        config.pipeline.sdxl_steps = preset["sdxl_steps"]
        config.pipeline.sdxl_refiner_steps = preset["refiner_steps"]
        config.pipeline.enable_refiner = preset["enable_refiner"]
        config.pipeline.enable_gfpgan = preset["enable_gfpgan"]
        config.pipeline.enable_turbo_mode = state.turbo_mode
        config.pipeline.transition_process_workers = int(state.transition_process_workers)
        config.pipeline.enable_chunked_parallel = bool(state.chunked_parallel)
        config.pipeline.transition_chunk_size = int(state.transition_chunk_size)
        config.video.rife_multiplier = preset["rife_multiplier"]
        config.pipeline.transition_style = state.transition_style
        config.pipeline.transition_duration_seconds = state.transition_duration_seconds
        timing = compute_transition_plan(
            transition_style=state.transition_style,
            transition_duration_seconds=state.transition_duration_seconds,
            fps_output=config.video.fps_output,
            rife_multiplier=config.video.rife_multiplier,
            num_images=max(2, len(images)),
            turbo_mode=state.turbo_mode,
            output_width=config.video.output_width,
            output_height=config.video.output_height,
        )
        config.pipeline.frames_per_transition = int(timing["keyframes_per_transition"])
        device = DeviceManager.get_device()
        if (
            config.video.prefer_gpu_encoder
            and device.type == "cuda"
            and _ffmpeg_supports_encoder("h264_nvenc")
        ):
            config.video.codec = "h264_nvenc"
            config.video.preset = preset["ffmpeg_gpu_preset"]
        else:
            config.video.codec = "libx264"
            config.video.preset = preset["ffmpeg_preset"]

        pipeline = GrowingUpPipeline(config)

        def progress_cb(p: float, msg: str):
            state.progress = p
            state.status_message = msg

        if len(images) > 1:
            output_path = pipeline.run_multi_image(
                images,
                captions=captions,
                fade_in_out=state.fade_enabled,
                music_path=state.music_path,
                progress_callback=progress_cb,
            )
        else:
            output_path = pipeline.run(images[0], progress_callback=progress_cb)

        state.output_path = str(output_path)
        state.preview_frames = pipeline.stage_previews
        state.stage_timings = getattr(pipeline, "last_stage_timings", {})
    except Exception as exc:
        import traceback
        state.error_message = f"{str(exc)}\n\n{traceback.format_exc()}"
    finally:
        state.is_processing = False


def main():
    config = get_config()
    state = get_state()

    _sidebar(state)

    st.markdown('<p class="main-title">AI Growing Up Generator</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">'
        'Upload one photo for AI-generated age progression, '
        'or multiple photos for authentic chronological transitions.'
        '</p>',
        unsafe_allow_html=True,
    )

    col_upload, col_info = st.columns([1, 1], gap="large")

    with col_upload:
        st.subheader("Upload Photos")
        st.caption("📸 Upload photos in chronological order (youngest → oldest)")

        uploaded_files = st.file_uploader(
            "Choose clear, frontal face photos",
            type=["jpg", "jpeg", "png", "webp"],
            help="Best results: well-lit, no sunglasses, facing camera. Upload in chronological order.",
            disabled=state.is_processing,
            accept_multiple_files=True,
        )

        if uploaded_files:
            state.uploaded_images = []
            successful_loads = 0

            for idx, uploaded_file in enumerate(uploaded_files[:120]):
                try:
                    uploaded_file.seek(0)
                    file_bytes = io.BytesIO(uploaded_file.read())
                    image = ImageOps.exif_transpose(Image.open(file_bytes)).convert("RGB")
                    w, h = image.size
                    max_edge = max(w, h)
                    if max_edge > MAX_UPLOAD_EDGE:
                        scale = MAX_UPLOAD_EDGE / max_edge
                        image = image.resize(
                            (max(1, int(w * scale)), max(1, int(h * scale))),
                            Image.LANCZOS,
                        )
                    state.uploaded_images.append(image)
                    successful_loads += 1
                except (Image.UnidentifiedImageError, IOError) as e:
                    st.error(f"❌ Could not load image {idx + 1} ({uploaded_file.name}): {e}")
                except Exception as e:
                    st.error(f"❌ Unexpected error with image {idx + 1}: {e}")

            if state.uploaded_images:
                state.uploaded_image = state.uploaded_images[0]
                st.success(f"✅ Loaded {successful_loads} photo(s)")

                num = len(state.uploaded_images)

                # ── Image reorder + captions section ──────────────
                if num > 1:
                    st.markdown("**📋 Image Order & Captions**")
                    st.caption("Change the number in 'Order' to rearrange. Add optional captions.")

                    # Ensure captions list matches
                    while len(state.image_captions) < num:
                        state.image_captions.append("")
                    state.image_captions = state.image_captions[:num]

                    new_order = [None] * num
                    new_captions = [""] * num

                    cols_per_row = min(num, 4)
                    for row_start in range(0, num, cols_per_row):
                        row_end = min(row_start + cols_per_row, num)
                        cols = st.columns(cols_per_row)
                        for ci, i in enumerate(range(row_start, row_end)):
                            with cols[ci]:
                                st.image(state.uploaded_images[i], use_column_width=True)
                                pos = st.number_input(
                                    "Order",
                                    min_value=1, max_value=num,
                                    value=i + 1,
                                    key=f"order_{i}",
                                    disabled=state.is_processing,
                                )
                                cap = st.text_input(
                                    "Caption",
                                    value=state.image_captions[i],
                                    placeholder=f"e.g. Age 3, Bar Mitzvah…",
                                    key=f"cap_{i}",
                                    disabled=state.is_processing,
                                )
                                new_order[i] = pos - 1
                                new_captions[i] = cap

                    # Validate no duplicate orders
                    if len(set(new_order)) == num:
                        sorted_indices = sorted(range(num), key=lambda x: new_order[x])
                        state.uploaded_images = [state.uploaded_images[j] for j in sorted_indices]
                        state.image_captions = [new_captions[j] for j in sorted_indices]
                    else:
                        st.warning("⚠️ Duplicate order numbers — fix to reorder")
                        state.image_captions = new_captions

                else:
                    st.image(state.uploaded_images[0], caption="Uploaded photo", use_column_width=True)
                    state.image_captions = [""]

                # Resolution info
                W, H = state.uploaded_images[0].size
                if W < 256 or H < 256:
                    st.warning("⚠️ Low resolution — results may lack detail.")
                elif W >= 512 and H >= 512:
                    st.success(f"✓ Good resolution: {W}×{H}")

    with col_info:
        num_images = len(state.uploaded_images)
        res_w, res_h = RESOLUTION_MAP[state.resolution]

        if num_images == 0:
            st.subheader("How It Works")
            st.markdown("""
            **Option 1: Single Photo (AI Generation)**
            - Upload one clear face photo
            - AI generates age progression stages
            - Creates cinematic aging video

            **Option 2: Multiple Photos (Authentic)**
            - Upload 2-120 photos in chronological order
            - Creates smooth transitions between your real photos
            - More authentic and personal result
            """)
        elif num_images == 1:
            st.subheader("Age Progression Stages")
            ages = config.pipeline.age_stages
            st.write("  ".join([f"**{a}y**" for a in ages]))
            st.caption("AI will generate these age stages from your photo")
        else:
            st.subheader(f"Video from {num_images} Photos")
            c1, c2, c3 = st.columns(3)
            c1.metric("Transitions", f"{num_images - 1}")
            c2.metric("Style", state.transition_style)
            c3.metric("Duration", f"{state.transition_duration_seconds:.2f}s")
            preset = PERFORMANCE_PRESETS[state.performance_mode]
            profile = compute_transition_plan(
                transition_style=state.transition_style,
                transition_duration_seconds=state.transition_duration_seconds,
                fps_output=config.video.fps_output,
                rife_multiplier=max(1, int(preset["rife_multiplier"])),
                turbo_mode=state.turbo_mode,
                num_images=max(2, num_images),
                output_width=res_w,
                output_height=res_h,
            )
            transition_frames = int(profile["transition_output_frames"])
            transition_sec = transition_frames / max(config.video.fps_output, 1)
            st.caption(
                f"Estimated transition duration: ~{transition_sec:.2f}s "
                f"({transition_frames} frames @ {config.video.fps_output}fps)"
            )
            if state.turbo_mode:
                st.caption(
                    f"Turbo profile: morph {int(profile['morph_w'])}x{int(profile['morph_h'])}, "
                    f"hold {int(profile['hold_frames'])} frames, "
                    f"smoothing window {int(profile['smoothing_window'])}, "
                    f"workers {int(state.transition_process_workers)}, "
                    f"chunked={'on' if state.chunked_parallel else 'off'}"
                )
            st.success(f"Smooth transitions between your {num_images} photos")

        st.subheader("Output Spec")
        c1, c2, c3 = st.columns(3)
        c1.metric("Resolution", f"{res_w}×{res_h}")
        c2.metric("Frame Rate", "60 fps")
        c3.metric("Format", "H.264 MP4")

        extras = []
        if state.fade_enabled:
            extras.append("🌑 Fade In/Out")
        if state.music_path:
            extras.append("🎵 Music")
        if any(c.strip() for c in state.image_captions):
            extras.append("📝 Captions")
        if extras:
            st.info("Features: " + " · ".join(extras))

        st.subheader("Pipeline")
        if num_images > 1:
            steps = [
                "Face detection & alignment",
                "Landmark extraction",
                "Low-res Delaunay morphing (fast)",
                "RIFE 4× frame interpolation (GPU)",
                "Upscale to output resolution",
                "Ken Burns cinematic zoom",
            ]
            if any(c.strip() for c in state.image_captions):
                steps.append("Text caption overlay")
            if state.fade_enabled:
                steps.append("Fade in / fade out")
            steps.append("FFmpeg pipe encoding")
            if state.music_path:
                steps.append("Add background music")
        else:
            steps = [
                "Face detection & alignment",
                "Identity embedding (ArcFace)",
                "SDXL age-stage generation",
                "GFPGAN face refinement",
                "Delaunay landmark morphing",
                "RIFE 4× frame interpolation",
                "Ken Burns cinematic zoom",
                "FFmpeg encoding",
            ]
        for s in steps:
            st.markdown(f"- {s}")

    st.divider()

    generate_disabled = not state.uploaded_images or state.is_processing

    if st.button(
        "🎬 Generate Growing Up Video" if len(state.uploaded_images) <= 1
        else f"🎬 Generate Video from {len(state.uploaded_images)} Photos",
        disabled=generate_disabled,
        type="primary",
        use_container_width=True,
    ):
        images_to_process = state.uploaded_images.copy()
        captions_to_process = state.image_captions.copy()
        saved_resolution = state.resolution
        saved_style = state.transition_style
        saved_duration = state.transition_duration_seconds
        saved_workers = state.transition_process_workers
        saved_chunked = state.chunked_parallel
        saved_chunk_size = state.transition_chunk_size
        saved_fade = state.fade_enabled
        saved_music = state.music_path
        saved_turbo = state.turbo_mode

        reset_state()
        state = get_state()
        state.is_processing = True
        state.progress = 0.0
        state.status_message = "Starting pipeline…"
        state.uploaded_images = images_to_process
        state.uploaded_image = images_to_process[0] if images_to_process else None
        state.image_captions = captions_to_process
        state.resolution = saved_resolution
        state.transition_style = saved_style
        state.transition_duration_seconds = saved_duration
        state.transition_process_workers = saved_workers
        state.chunked_parallel = saved_chunked
        state.transition_chunk_size = saved_chunk_size
        state.fade_enabled = saved_fade
        state.music_path = saved_music
        state.turbo_mode = saved_turbo

        thread = threading.Thread(
            target=_run_pipeline,
            args=(images_to_process, captions_to_process, config, state),
            daemon=True,
        )
        thread.start()

    if state.is_processing:
        st.subheader("Generating…")
        st.progress(state.progress)
        st.info(state.status_message)
        time.sleep(2)
        _safe_rerun()

    if state.error_message:
        st.error("Pipeline failed")
        st.code(state.error_message)

    if state.output_path and not state.is_processing:
        st.success("Video generated successfully!")
        if state.stage_timings:
            st.subheader("Stage Timings")
            st.json({k: round(float(v), 2) for k, v in state.stage_timings.items()})
        st.subheader("Preview")
        st.video(state.output_path)

        with open(state.output_path, "rb") as f:
            video_bytes = f.read()

        res_label = state.resolution
        st.download_button(
            label=f"Download MP4 ({res_label})",
            data=video_bytes,
            file_name="growing_up.mp4",
            mime="video/mp4",
            type="primary",
            use_container_width=True,
        )

        if state.preview_frames:
            st.subheader("Stage Previews")
            cols = st.columns(min(len(state.preview_frames), 8))
            for col, (age, img) in zip(cols, state.preview_frames):
                col.image(img, caption=f"Stage {age}", use_column_width=True)


if __name__ == "__main__":
    main()
