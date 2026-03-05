from __future__ import annotations

import io
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from functools import lru_cache
from pathlib import Path

import cv2
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
    page_icon="AI",
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

# â”€â”€ Resolution presets â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
RESOLUTION_MAP = {
    "720p":  (1280, 720),
    "1080p": (1920, 1080),
    "4K":    (3840, 2160),
}
MAX_UPLOAD_EDGE = 2048
MAX_TIMELINE_ITEMS = 400
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
VIDEO_EXTENSIONS = {"mp4", "mov", "m4v", "webm", "avi"}
TRANSITION_STYLE_OPTIONS = ["Emotional", "Balanced", "Fast"]
TRANSITION_DURATION_OPTIONS = perceptual_duration_options()

PERFORMANCE_PRESETS = {
    "fast": {
        "sdxl_steps": 18,
        "refiner_steps": 0,
        "rife_multiplier": 1,
        "ffmpeg_preset": "veryfast",
        "ffmpeg_gpu_preset": "p4",
        "enable_gfpgan": False,
        "enable_refiner": False,
        "use_fast_crossdissolve": False,
    },
    "balanced": {
        "sdxl_steps": 25,
        "refiner_steps": 15,
        "rife_multiplier": 2,
        "ffmpeg_preset": "medium",
        "ffmpeg_gpu_preset": "p5",
        "enable_gfpgan": True,
        "enable_refiner": True,
        "use_fast_crossdissolve": False,
    },
    "quality": {
        "sdxl_steps": 35,
        "refiner_steps": 25,
        "rife_multiplier": 4,
        "ffmpeg_preset": "slow",
        "ffmpeg_gpu_preset": "p6",
        "enable_gfpgan": True,
        "enable_refiner": True,
        "use_fast_crossdissolve": False,
    },
    "extreme_speed_3090": {
        "sdxl_steps": 6,
        "refiner_steps": 0,
        "rife_multiplier": 1,
        "ffmpeg_preset": "ultrafast",
        "ffmpeg_gpu_preset": "p1",
        "enable_gfpgan": False,
        "enable_refiner": False,
        "use_fast_crossdissolve": False,
    },
    "extreme_quality_3090": {
        "sdxl_steps": 24,
        "refiner_steps": 18,
        "rife_multiplier": 2,
        "ffmpeg_preset": "slow",
        "ffmpeg_gpu_preset": "p5",
        "enable_gfpgan": True,
        "enable_refiner": True,
        "use_fast_crossdissolve": False,
    },
}
PERFORMANCE_PRESET_ORDER = [
    "extreme_speed_3090",
    "fast",
    "balanced",
    "quality",
    "extreme_quality_3090",
]


def _available_cpu_count() -> int:
    try:
        if hasattr(os, "sched_getaffinity"):
            return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        pass
    return max(1, int(os.cpu_count() or 1))


def _upload_ext(filename: str) -> str:
    return Path(filename).suffix.lower().lstrip(".")


def _save_bytes_to_temp(payload: bytes, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(payload)
    tmp.close()
    return tmp.name


def _file_signature(file_name: str, payload: bytes) -> str:
    digest = hashlib.sha1(payload).hexdigest()
    return f"{file_name}:{digest}"


def _resolve_ffprobe_binary() -> str | None:
    env = os.environ.get("FFPROBE_BINARY")
    if env:
        return env
    probe = shutil.which("ffprobe")
    if probe:
        return probe
    try:
        ffmpeg_bin = resolve_ffmpeg_binary()
        candidate = Path(ffmpeg_bin).with_name("ffprobe")
        if candidate.exists():
            return str(candidate)
    except Exception:
        return None
    return None


def _read_video_rotation_degrees(video_path: str) -> int:
    ffprobe_bin = _resolve_ffprobe_binary()
    if ffprobe_bin is None:
        return 0
    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream_tags=rotate:stream_side_data=rotation",
        "-of", "json",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=8)
        if result.returncode != 0 or not result.stdout.strip():
            return 0
        payload = json.loads(result.stdout)
        streams = payload.get("streams") or []
        if not streams:
            return 0
        stream = streams[0]
        rotation_val = None
        tags = stream.get("tags") or {}
        if "rotate" in tags:
            rotation_val = tags.get("rotate")
        side_data = stream.get("side_data_list") or stream.get("side_data") or []
        if isinstance(side_data, list):
            for entry in side_data:
                if isinstance(entry, dict) and "rotation" in entry:
                    rotation_val = entry.get("rotation")
                    break
        if rotation_val is None:
            return 0
        degrees = int(round(float(rotation_val))) % 360
        snapped = min((0, 90, 180, 270), key=lambda d: abs(d - degrees))
        return int(snapped)
    except Exception:
        return 0


def _rotate_frame_bgr(frame, rotation_deg: int):
    if rotation_deg == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation_deg == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation_deg == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def _load_video_thumbnail(video_path: str) -> Image.Image | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        rotation_deg = _read_video_rotation_degrees(video_path)
        if rotation_deg == 0 and hasattr(cv2, "CAP_PROP_ORIENTATION_META"):
            try:
                meta = float(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0.0)
                if abs(meta) >= 1.0:
                    meta_deg = int(round(meta)) % 360
                    if meta_deg in (90, 180, 270):
                        rotation_deg = meta_deg
            except Exception:
                pass
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        frame = _rotate_frame_bgr(frame, rotation_deg)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    finally:
        cap.release()


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
    st.sidebar.header("Settings")

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
    st.sidebar.subheader("Performance")
    state.performance_mode = st.sidebar.selectbox(
        "Render Preset",
        options=PERFORMANCE_PRESET_ORDER,
        index=PERFORMANCE_PRESET_ORDER.index(state.performance_mode)
        if state.performance_mode in PERFORMANCE_PRESET_ORDER
        else 0,
        help="Fast = quick preview. Quality = best detail, slower render.",
        disabled=state.is_processing,
    )
    state.turbo_mode = st.sidebar.checkbox(
        "Super Turbo (large albums)",
        value=state.turbo_mode,
        help="Aggressively reduces morph workload for much faster renders.",
        disabled=state.is_processing,
    )
    cpu_budget = _available_cpu_count()
    max_worker_ui = max(1, min(64, cpu_budget))
    worker_value = max(1, min(int(state.transition_process_workers), max_worker_ui))
    state.transition_process_workers = st.sidebar.slider(
        "Transition Workers (Processes)",
        min_value=1,
        max_value=max_worker_ui,
        value=worker_value,
        help=f"Parallel transition generation. Container CPU budget detected: {cpu_budget}.",
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
        max_value=24,
        value=max(1, min(int(state.transition_chunk_size), 24)),
        help="How many transitions each process computes per chunk.",
        disabled=state.is_processing or not state.chunked_parallel,
    )
    # Background music upload (up to 3 tracks: beginning, middle, end)
    st.sidebar.divider()
    st.sidebar.subheader("Background Music")
    _music_labels = [
        ("🎵 Track 1 — Beginning", "music_begin"),
        ("🎵 Track 2 — Middle", "music_middle"),
        ("🎵 Track 3 — End", "music_end"),
    ]
    new_music_paths: list[str] = []
    for label, key in _music_labels:
        mf = st.sidebar.file_uploader(
            label,
            type=["mp3", "wav", "m4a", "ogg"],
            help="Optional — tracks are concatenated in order",
            disabled=state.is_processing,
            key=key,
        )
        if mf is not None:
            music_bytes = mf.read()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tmp.write(music_bytes)
            tmp.close()
            new_music_paths.append(tmp.name)
            st.sidebar.caption(f"✓ {mf.name}")
    state.music_paths = new_music_paths
    # Backward compat: set music_path to first track if any
    state.music_path = new_music_paths[0] if new_music_paths else None

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
            st.sidebar.warning("No GPU - CPU mode (slower)")

        st.sidebar.divider()
        st.sidebar.caption(
            "All processing happens locally.  \n"
            "No data is sent to any server."
        )


def _run_pipeline(media_items, captions, config, state):
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
        config.pipeline.use_fast_crossdissolve = preset.get("use_fast_crossdissolve", False)
        config.pipeline.transition_style = state.transition_style
        config.pipeline.transition_duration_seconds = state.transition_duration_seconds
        timing = compute_transition_plan(
            transition_style=state.transition_style,
            transition_duration_seconds=state.transition_duration_seconds,
            fps_output=config.video.fps_output,
            rife_multiplier=config.video.rife_multiplier,
            num_images=max(2, len(media_items)),
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

        all_images = all(str(item.get("kind", "")).lower() == "image" for item in media_items)
        if all_images and len(media_items) == 1:
            output_path = pipeline.run(media_items[0]["image"], progress_callback=progress_cb)
        elif all_images:
            output_path = pipeline.run_multi_image(
                [item["image"] for item in media_items],
                captions=captions,
                fade_in_out=state.fade_enabled,
                music_path=state.music_path,
                music_paths=state.music_paths,
                progress_callback=progress_cb,
            )
        else:
            output_path = pipeline.run_mixed_timeline(
                media_items=media_items,
                captions=captions,
                fade_in_out=state.fade_enabled,
                music_path=state.music_path,
                music_paths=state.music_paths,
                progress_callback=progress_cb,
            )

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
        "Upload one photo for AI age progression, "
        "or build a timeline from photos and MP4 videos."
        '</p>',
        unsafe_allow_html=True,
    )

    col_upload, col_info = st.columns([1, 1], gap="large")

    with col_upload:
        st.subheader("Upload Timeline Media")
        st.caption("Upload in chronological order (youngest -> oldest)")

        uploaded_files = st.file_uploader(
            "Choose photos and videos",
            type=sorted(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS),
            help="Supported: JPG/PNG/WEBP + MP4/MOV/M4V/WEBM/AVI",
            disabled=state.is_processing,
            accept_multiple_files=True,
        )

        if uploaded_files:
            limited_files = uploaded_files[:MAX_TIMELINE_ITEMS]
            upload_token = "|".join(
                f"{f.name}:{getattr(f, 'size', 0)}:{_upload_ext(f.name)}"
                for f in limited_files
            )
            if upload_token != state.uploads_token:
                prev_caps: dict[str, str] = {}
                prev_texts: dict[str, str] = {}
                prev_slow: dict[str, float] = {}
                for i, item in enumerate(state.uploaded_media_items):
                    sig = str(item.get("signature", ""))
                    if not sig:
                        continue
                    if i < len(state.image_captions):
                        prev_caps[sig] = state.image_captions[i]
                    if i < len(state.image_text_overlays):
                        prev_texts[sig] = state.image_text_overlays[i]
                    if i < len(state.timeline_video_slow_factors):
                        prev_slow[sig] = float(state.timeline_video_slow_factors[i])

                new_items: list[dict] = []
                new_captions: list[str] = []
                new_texts: list[str] = []
                new_slow_factors: list[float] = []
                seen_signatures: set[str] = set()
                successful_loads = 0
                duplicate_count = 0

                for idx, uploaded_file in enumerate(limited_files):
                    try:
                        ext = _upload_ext(uploaded_file.name)
                        payload = uploaded_file.getvalue()
                        sig = _file_signature(uploaded_file.name, payload)
                        if sig in seen_signatures:
                            duplicate_count += 1
                            continue
                        seen_signatures.add(sig)

                        if ext in IMAGE_EXTENSIONS:
                            image = ImageOps.exif_transpose(Image.open(io.BytesIO(payload))).convert("RGB")
                            w, h = image.size
                            max_edge = max(w, h)
                            if max_edge > MAX_UPLOAD_EDGE:
                                scale = MAX_UPLOAD_EDGE / max_edge
                                image = image.resize(
                                    (max(1, int(w * scale)), max(1, int(h * scale))),
                                    Image.LANCZOS,
                                )
                            new_items.append(
                                {
                                    "kind": "image",
                                    "image": image,
                                    "name": uploaded_file.name,
                                    "signature": sig,
                                }
                            )
                            new_slow_factors.append(1.0)
                        elif ext in VIDEO_EXTENSIONS:
                            temp_path = _save_bytes_to_temp(payload, suffix=f".{ext or 'mp4'}")
                            thumb = _load_video_thumbnail(temp_path)
                            if thumb is None:
                                raise ValueError("Could not decode video")
                            new_items.append(
                                {
                                    "kind": "video",
                                    "video_path": temp_path,
                                    "thumbnail": thumb,
                                    "name": uploaded_file.name,
                                    "signature": sig,
                                }
                            )
                            new_slow_factors.append(float(prev_slow.get(sig, 1.0)))
                        else:
                            raise ValueError(f"Unsupported file type: {ext}")

                        new_captions.append(prev_caps.get(sig, ""))
                        new_texts.append(prev_texts.get(sig, ""))
                        successful_loads += 1
                    except Exception as e:
                        st.error(f"Could not load file {idx + 1} ({uploaded_file.name}): {e}")

                state.uploaded_media_items = new_items
                state.image_captions = new_captions
                state.image_text_overlays = new_texts
                state.timeline_video_slow_factors = new_slow_factors
                state.uploaded_images = [
                    item["image"] for item in state.uploaded_media_items if item["kind"] == "image"
                ]
                state.uploaded_image = next(
                    (item["image"] for item in state.uploaded_media_items if item["kind"] == "image"),
                    None,
                )
                state.uploads_token = upload_token
                if state.uploaded_media_items:
                    msg = f"Loaded {successful_loads} timeline item(s)"
                    if duplicate_count > 0:
                        msg += f" ({duplicate_count} duplicate file(s) skipped)"
                    st.success(msg)
            elif state.uploaded_media_items:
                st.caption(f"Loaded {len(state.uploaded_media_items)} timeline item(s)")

            if state.uploaded_media_items:
                state.uploaded_image = next(
                    (item["image"] for item in state.uploaded_media_items if item["kind"] == "image"),
                    None,
                )

                num = len(state.uploaded_media_items)
                while len(state.image_captions) < num:
                    state.image_captions.append("")
                state.image_captions = state.image_captions[:num]
                while len(state.image_text_overlays) < num:
                    state.image_text_overlays.append("")
                state.image_text_overlays = state.image_text_overlays[:num]
                while len(state.timeline_video_slow_factors) < num:
                    state.timeline_video_slow_factors.append(1.0)
                state.timeline_video_slow_factors = state.timeline_video_slow_factors[:num]
                for i, item in enumerate(state.uploaded_media_items):
                    if item["kind"] != "video":
                        state.timeline_video_slow_factors[i] = 1.0

                st.markdown("**Timeline Order + Captions/Text**")
                st.caption("Caption appears on screen. If Caption is empty, Text is used.")

                new_order = [None] * num
                new_captions = [""] * num
                new_texts = [""] * num
                new_slow_factors = [1.0] * num

                cols_per_row = min(num, 4)
                remove_index = None
                for row_start in range(0, num, cols_per_row):
                    row_end = min(row_start + cols_per_row, num)
                    cols = st.columns(cols_per_row)
                    for ci, i in enumerate(range(row_start, row_end)):
                        with cols[ci]:
                            item = state.uploaded_media_items[i]
                            if st.button("X", key=f"remove_item_{i}", disabled=state.is_processing):
                                remove_index = i
                            st.caption(f"{item['kind'].capitalize()}: {item.get('name', '')}")
                            if item["kind"] == "image":
                                st.image(item["image"], use_column_width=True)
                                slow_factor = 1.0
                            else:
                                st.image(item["thumbnail"], use_column_width=True)
                                slow_factor = float(
                                    st.slider(
                                        "Video Slow Motion",
                                        min_value=1.0,
                                        max_value=10.0,
                                        value=float(state.timeline_video_slow_factors[i]),
                                        step=0.1,
                                        key=f"video_slow_{i}",
                                        disabled=state.is_processing,
                                    )
                                )

                            pos = st.number_input(
                                "Order",
                                min_value=1,
                                max_value=num,
                                value=i + 1,
                                key=f"order_{i}",
                                disabled=state.is_processing,
                            )
                            cap = st.text_input(
                                "Caption",
                                value=state.image_captions[i],
                                placeholder="e.g. Age 3, Bar Mitzvah...",
                                key=f"cap_{i}",
                                disabled=state.is_processing,
                            )
                            txt = st.text_input(
                                "Text",
                                value=state.image_text_overlays[i],
                                placeholder="Optional fallback text",
                                key=f"text_{i}",
                                disabled=state.is_processing,
                            )
                            new_order[i] = pos - 1
                            new_captions[i] = cap
                            new_texts[i] = txt
                            new_slow_factors[i] = slow_factor

                if remove_index is not None:
                    state.uploaded_media_items.pop(remove_index)
                    state.image_captions.pop(remove_index)
                    state.image_text_overlays.pop(remove_index)
                    state.timeline_video_slow_factors.pop(remove_index)
                    state.uploaded_images = [
                        item["image"] for item in state.uploaded_media_items if item["kind"] == "image"
                    ]
                    state.uploaded_image = next(
                        (item["image"] for item in state.uploaded_media_items if item["kind"] == "image"),
                        None,
                    )
                    _safe_rerun()

                if len(set(new_order)) == num:
                    sorted_indices = sorted(range(num), key=lambda x: new_order[x])
                    state.uploaded_media_items = [state.uploaded_media_items[j] for j in sorted_indices]
                    state.image_captions = [new_captions[j] for j in sorted_indices]
                    state.image_text_overlays = [new_texts[j] for j in sorted_indices]
                    state.timeline_video_slow_factors = [new_slow_factors[j] for j in sorted_indices]
                else:
                    st.warning("Duplicate order numbers detected. Fix them to reorder.")
                    state.image_captions = new_captions
                    state.image_text_overlays = new_texts
                    state.timeline_video_slow_factors = new_slow_factors

                state.uploaded_images = [
                    item["image"] for item in state.uploaded_media_items if item["kind"] == "image"
                ]
                lead_image = next(
                    (item["image"] for item in state.uploaded_media_items if item["kind"] == "image"),
                    None,
                )
                if lead_image is not None:
                    w, h = lead_image.size
                    if w < 256 or h < 256:
                        st.warning("Low-resolution image detected; quality may drop.")
                    elif w >= 512 and h >= 512:
                        st.success(f"Good image resolution: {w}x{h}")

    with col_info:
        num_items = len(state.uploaded_media_items)
        num_images = sum(1 for item in state.uploaded_media_items if item["kind"] == "image")
        num_videos = sum(1 for item in state.uploaded_media_items if item["kind"] == "video")
        slow_video_factors = [
            float(state.timeline_video_slow_factors[i])
            for i, item in enumerate(state.uploaded_media_items)
            if (
                item["kind"] == "video"
                and i < len(state.timeline_video_slow_factors)
            )
        ]
        slow_videos = [f for f in slow_video_factors if f > 1.0]
        res_w, res_h = RESOLUTION_MAP[state.resolution]

        if num_items == 0:
            st.subheader("How It Works")
            st.markdown("""
            **Option 1: Single Photo (AI Generation)**
            - Upload one clear face photo
            - AI generates age progression stages
            - Creates cinematic aging video

            **Option 2: Timeline (Photos + Videos)**
            - Upload photos and MP4/MOV clips in order
            - Videos are played directly in the timeline
            - Transitions connect from each item end to the next item start
            """)
        elif num_items == 1 and num_videos == 0:
            st.subheader("Age Progression Stages")
            ages = config.pipeline.age_stages
            st.write("  ".join([f"**{a}y**" for a in ages]))
            st.caption("AI will generate these age stages from your photo")
        else:
            st.subheader(f"Timeline: {num_items} Item(s)")
            c1, c2, c3 = st.columns(3)
            c1.metric("Transitions", f"{max(0, num_items - 1)}")
            c2.metric("Style", state.transition_style)
            c3.metric("Duration", f"{state.transition_duration_seconds:.2f}s")
            st.caption(f"Media mix: {num_images} image(s), {num_videos} video(s)")
            if slow_videos:
                st.caption(f"Video slow motion: {len(slow_videos)} clip(s), up to {max(slow_videos):.1f}x")
            preset = PERFORMANCE_PRESETS[state.performance_mode]
            profile = compute_transition_plan(
                transition_style=state.transition_style,
                transition_duration_seconds=state.transition_duration_seconds,
                fps_output=config.video.fps_output,
                rife_multiplier=max(1, int(preset["rife_multiplier"])),
                turbo_mode=state.turbo_mode,
                num_images=max(2, num_items),
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
            st.success("Smooth transitions between timeline items")

        st.subheader("Output Spec")
        c1, c2, c3 = st.columns(3)
        c1.metric("Resolution", f"{res_w}x{res_h}")
        c2.metric("Frame Rate", "60 fps")
        c3.metric("Format", "H.264 MP4")

        overlay_texts = [
            (cap or "").strip() or (txt or "").strip()
            for cap, txt in zip(state.image_captions, state.image_text_overlays)
        ]
        extras = []
        if state.fade_enabled:
            extras.append("Fade In/Out")
        if state.music_paths:
            extras.append(f"Music ({len(state.music_paths)} track{'s' if len(state.music_paths) > 1 else ''})")
        elif state.music_path:
            extras.append("Music")
        if slow_videos:
            extras.append(f"Video Slow x{max(slow_videos):.1f}")
        if any(t for t in overlay_texts):
            extras.append("Text Overlay")
        if extras:
            st.info("Features: " + " | ".join(extras))

        st.subheader("Pipeline")
        if num_items > 1 or num_videos > 0:
            steps = [
                "Timeline media normalization",
                "Landmark extraction",
                "Delaunay morphing for transitions",
                "RIFE interpolation (if enabled)",
                "Upscale to output resolution",
                "Ken Burns cinematic zoom",
            ]
            if any(t for t in overlay_texts):
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
                "RIFE frame interpolation",
                "Ken Burns cinematic zoom",
                "FFmpeg encoding",
            ]
        for s in steps:
            st.markdown(f"- {s}")

    st.divider()

    generate_disabled = not state.uploaded_media_items or state.is_processing
    only_single_image = (
        len(state.uploaded_media_items) == 1 and
        state.uploaded_media_items[0]["kind"] == "image"
    )

    if st.button(
        "Generate Growing Up Video" if only_single_image
        else f"Generate Timeline Video ({len(state.uploaded_media_items)} items)",
        disabled=generate_disabled,
        type="primary",
        use_container_width=True,
    ):
        media_to_process = []
        for i, item in enumerate(state.uploaded_media_items):
            slow_factor = (
                float(state.timeline_video_slow_factors[i])
                if i < len(state.timeline_video_slow_factors)
                else 1.0
            )
            if item["kind"] == "image":
                media_to_process.append(
                    {
                        "kind": "image",
                        "image": item["image"],
                        "name": item.get("name", "image"),
                    }
                )
            else:
                media_to_process.append(
                    {
                        "kind": "video",
                        "video_path": item["video_path"],
                        "name": item.get("name", "video"),
                        "slow_motion_factor": max(1.0, slow_factor),
                    }
                )

        captions_to_process = [
            (cap or "").strip() or (txt or "").strip()
            for cap, txt in zip(state.image_captions, state.image_text_overlays)
        ]
        saved_resolution = state.resolution
        saved_style = state.transition_style
        saved_duration = state.transition_duration_seconds
        saved_workers = state.transition_process_workers
        saved_chunked = state.chunked_parallel
        saved_chunk_size = state.transition_chunk_size
        saved_fade = state.fade_enabled
        saved_music = state.music_path
        saved_music_paths = list(state.music_paths)
        saved_turbo = state.turbo_mode
        saved_video_slow_factors = list(state.timeline_video_slow_factors)

        reset_state()
        state = get_state()
        state.is_processing = True
        state.progress = 0.0
        state.status_message = "Starting pipeline..."
        state.uploaded_media_items = media_to_process
        state.uploaded_images = [item["image"] for item in media_to_process if item["kind"] == "image"]
        state.uploaded_image = state.uploaded_images[0] if state.uploaded_images else None
        state.image_captions = captions_to_process
        state.image_text_overlays = [""] * len(captions_to_process)
        state.resolution = saved_resolution
        state.transition_style = saved_style
        state.transition_duration_seconds = saved_duration
        state.transition_process_workers = saved_workers
        state.chunked_parallel = saved_chunked
        state.transition_chunk_size = saved_chunk_size
        state.fade_enabled = saved_fade
        state.music_path = saved_music
        state.music_paths = saved_music_paths
        state.turbo_mode = saved_turbo
        state.timeline_video_slow_factors = saved_video_slow_factors

        thread = threading.Thread(
            target=_run_pipeline,
            args=(media_to_process, captions_to_process, config, state),
            daemon=True,
        )
        thread.start()

    if state.is_processing:
        st.subheader("Generating...")
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

        video_size_bytes = os.path.getsize(state.output_path)
        video_size_mb = video_size_bytes / (1024 * 1024)

        with open(state.output_path, "rb") as f:
            video_bytes = f.read()

        res_label = state.resolution
        st.download_button(
            label=f"Download MP4 ({res_label}) — {video_size_mb:.0f} MB",
            data=video_bytes,
            file_name="growing_up.mp4",
            mime="video/mp4",
            type="primary",
            use_container_width=True,
            key="auto_download_btn",
        )
        del video_bytes

        # --- Auto-click the download button via JS (works for any file size) ---
        import streamlit.components.v1 as components
        auto_click_js = """
        <script>
        (function() {
            function clickDownload() {
                // Find the Streamlit download button by its key or by type=primary
                var buttons = window.parent.document.querySelectorAll(
                    'button[kind="primary"], button[data-testid="stDownloadButton"], button.st-emotion-cache-1ny7cjd'
                );
                for (var i = 0; i < buttons.length; i++) {
                    var btn = buttons[i];
                    if (btn.innerText && btn.innerText.indexOf('Download MP4') !== -1) {
                        btn.click();
                        return true;
                    }
                }
                // Fallback: try any primary button with Download text
                var allBtns = window.parent.document.querySelectorAll('button');
                for (var j = 0; j < allBtns.length; j++) {
                    if (allBtns[j].innerText && allBtns[j].innerText.indexOf('Download MP4') !== -1) {
                        allBtns[j].click();
                        return true;
                    }
                }
                return false;
            }
            // Retry a few times — button may not be rendered yet
            var attempts = 0;
            var interval = setInterval(function() {
                attempts++;
                if (clickDownload() || attempts > 15) {
                    clearInterval(interval);
                }
            }, 500);
        })();
        </script>
        """
        components.html(auto_click_js, height=0, width=0)

        if state.preview_frames:
            st.subheader("Stage Previews")
            cols = st.columns(min(len(state.preview_frames), 8))
            for col, (age, img) in zip(cols, state.preview_frames):
                col.image(img, caption=f"Stage {age}", use_column_width=True)


if __name__ == "__main__":
    main()
