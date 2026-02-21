from __future__ import annotations

import threading
import time
from pathlib import Path

import streamlit as st
from PIL import Image

from app.config import get_config
from app.state import get_state, reset_state
from utils.device import DeviceManager

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
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


def _sidebar():
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
        st.sidebar.warning("No GPU detected — running on CPU (very slow)")

    st.sidebar.divider()
    st.sidebar.header("About")
    st.sidebar.caption(
        "All processing happens locally on your machine.  \n"
        "No data is sent to any server."
    )


def _run_pipeline(image: Image.Image, config, state):
    try:
        from pipeline.orchestrator import GrowingUpPipeline

        pipeline = GrowingUpPipeline(config)

        def progress_cb(p: float, msg: str):
            state.progress = p
            state.status_message = msg

        output_path = pipeline.run(image, progress_callback=progress_cb)
        state.output_path = str(output_path)
        state.preview_frames = pipeline.stage_previews
    except Exception as exc:
        state.error_message = str(exc)
    finally:
        state.is_processing = False


def main():
    config = get_config()
    state = get_state()

    _sidebar()

    st.markdown('<p class="main-title">AI Growing Up Generator</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Upload a clear frontal face photo — '
        "the AI will create a cinematic aging video from birth to old age.</p>",
        unsafe_allow_html=True,
    )

    col_upload, col_info = st.columns([1, 1], gap="large")

    with col_upload:
        st.subheader("Upload Photo")
        uploaded_file = st.file_uploader(
            "Choose a clear, frontal face photo",
            type=["jpg", "jpeg", "png", "webp"],
            help="Best results: well-lit, no sunglasses, facing camera",
            disabled=state.is_processing,
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            state.uploaded_image = image
            st.image(image, caption="Uploaded photo", use_column_width=True)
            W, H = image.size
            if W < 256 or H < 256:
                st.warning("Image resolution is low — results may lack detail.")
            elif W >= 512 and H >= 512:
                st.success(f"Good resolution: {W}×{H}")

    with col_info:
        st.subheader("Age Progression Stages")
        ages = config.pipeline.age_stages
        st.write("  ".join([f"**{a}y**" for a in ages]))

        st.subheader("Output Spec")
        c1, c2, c3 = st.columns(3)
        c1.metric("Resolution", "1920×1080")
        c2.metric("Frame Rate", "60 fps")
        c3.metric("Format", "H.264 MP4")

        st.subheader("Pipeline")
        steps = [
            "Face detection & alignment",
            "Identity embedding (ArcFace)",
            "SDXL age-stage generation",
            "GFPGAN face refinement",
            "Delaunay landmark morphing",
            "RIFE 4× frame interpolation",
            "Ken Burns cinematic zoom",
            "FFmpeg 1080p encoding",
        ]
        for s in steps:
            st.markdown(f"- {s}")

    st.divider()

    generate_disabled = state.uploaded_image is None or state.is_processing

    if st.button(
        "Generate Growing Up Video",
        disabled=generate_disabled,
        type="primary",
        use_container_width=True,
    ):
        reset_state()
        state = get_state()
        state.is_processing = True
        state.progress = 0.0
        state.status_message = "Starting pipeline…"
        state.uploaded_image = image  # re-assign after reset

        thread = threading.Thread(
            target=_run_pipeline,
            args=(state.uploaded_image, config, state),
            daemon=True,
        )
        thread.start()

    if state.is_processing:
        st.subheader("Generating…")
        st.progress(state.progress)
        st.info(state.status_message)
        time.sleep(2)
        st.rerun()

    if state.error_message:
        st.error("Pipeline failed")
        st.code(state.error_message)

    if state.output_path and not state.is_processing:
        st.success("Video generated successfully!")
        st.subheader("Preview")
        st.video(state.output_path)

        with open(state.output_path, "rb") as f:
            video_bytes = f.read()

        st.download_button(
            label="Download MP4 (1080p)",
            data=video_bytes,
            file_name="growing_up.mp4",
            mime="video/mp4",
            type="primary",
            use_container_width=True,
        )

        if state.preview_frames:
            st.subheader("Age Stage Previews")
            cols = st.columns(len(state.preview_frames))
            for col, (age, img) in zip(cols, state.preview_frames):
                col.image(img, caption=f"Age {age}", use_column_width=True)


if __name__ == "__main__":
    main()
