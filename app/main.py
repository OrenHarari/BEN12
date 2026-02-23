from __future__ import annotations

import io
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


def _run_pipeline(images: list[Image.Image], config, state):
    try:
        from pipeline.orchestrator import GrowingUpPipeline

        pipeline = GrowingUpPipeline(config)

        def progress_cb(p: float, msg: str):
            state.progress = p
            state.status_message = msg

        # Use multi-image pipeline if multiple images provided
        if len(images) > 1:
            output_path = pipeline.run_multi_image(images, progress_callback=progress_cb)
        else:
            output_path = pipeline.run(images[0], progress_callback=progress_cb)
        
        state.output_path = str(output_path)
        state.preview_frames = pipeline.stage_previews
    except Exception as exc:
        import traceback
        state.error_message = f"{str(exc)}\n\n{traceback.format_exc()}"
    finally:
        state.is_processing = False


def main():
    config = get_config()
    state = get_state()

    _sidebar()

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
        st.caption("📸 Upload photos of the person at different ages (chronologically ordered)")
        
        uploaded_files = st.file_uploader(
            "Choose clear, frontal face photos",
            type=["jpg", "jpeg", "png", "webp"],
            help="Best results: well-lit, no sunglasses, facing camera. Upload in chronological order (youngest to oldest).",
            disabled=state.is_processing,
            accept_multiple_files=True,
        )
        
        if uploaded_files:
            state.uploaded_images = []
            successful_loads = 0
            
            for idx, uploaded_file in enumerate(uploaded_files[:30]):  # Max 30 images
                try:
                    uploaded_file.seek(0)
                    file_bytes = io.BytesIO(uploaded_file.read())
                    image = Image.open(file_bytes).convert("RGB")
                    state.uploaded_images.append(image)
                    successful_loads += 1
                except (Image.UnidentifiedImageError, IOError) as e:
                    st.error(f"❌ Could not load image {idx + 1} ({uploaded_file.name}): {str(e)}")
                except Exception as e:
                    st.error(f"❌ Unexpected error with image {idx + 1}: {str(e)}")
            
            if state.uploaded_images:
                state.uploaded_image = state.uploaded_images[0]  # For compatibility
                st.success(f"✅ Successfully loaded {successful_loads} photo(s)")
                
                # Show thumbnails
                if len(state.uploaded_images) > 1:
                    cols = st.columns(min(len(state.uploaded_images), 5))
                    for i, img in enumerate(state.uploaded_images[:5]):
                        with cols[i % 5]:
                            st.image(img, caption=f"Photo {i + 1}", use_column_width=True)
                    if len(state.uploaded_images) > 5:
                        st.caption(f"...and {len(state.uploaded_images) - 5} more")
                else:
                    st.image(state.uploaded_images[0], caption="Uploaded photo", use_column_width=True)
                
                # Show resolution info
                W, H = state.uploaded_images[0].size
                if W < 256 or H < 256:
                    st.warning("⚠️ Image resolution is low — results may lack detail.")
                elif W >= 512 and H >= 512:
                    st.success(f"✓ Good resolution: {W}×{H}")

    with col_info:
        # Dynamic info based on number of uploaded images
        num_images = len(state.uploaded_images)
        
        if num_images == 0:
            st.subheader("How It Works")
            st.markdown("""
            **Option 1: Single Photo (AI Generation)**
            - Upload one clear face photo
            - AI generates age progression stages
            - Creates cinematic aging video
            
            **Option 2: Multiple Photos (Authentic)**
            - Upload 2-30 photos in chronological order
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
            st.metric("Transitions", f"{num_images - 1}")
            st.metric("Mode", "Authentic Timeline")
            st.success(f"✨ Creating smooth transitions between your {num_images} photos")

        st.subheader("Output Spec")
        c1, c2, c3 = st.columns(3)
        c1.metric("Resolution", "1920×1080")
        c2.metric("Frame Rate", "60 fps")
        c3.metric("Format", "H.264 MP4")

        st.subheader("Pipeline")
        if num_images > 1:
            steps = [
                "Face detection & alignment",
                "GFPGAN face enhancement",
                "Landmark extraction",
                "Delaunay morphing",
                "RIFE 4× interpolation",
                "Ken Burns cinematic zoom",
                "FFmpeg 1080p encoding",
            ]
        else:
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

    generate_disabled = not state.uploaded_images or state.is_processing

    if st.button(
        "🎬 Generate Growing Up Video" if len(state.uploaded_images) <= 1 else f"🎬 Generate Video from {len(state.uploaded_images)} Photos",
        disabled=generate_disabled,
        type="primary",
        use_container_width=True,
    ):
        # Save images before reset
        images_to_process = state.uploaded_images.copy()
        
        reset_state()
        state = get_state()
        state.is_processing = True
        state.progress = 0.0
        state.status_message = "Starting pipeline…"
        state.uploaded_images = images_to_process  # Restore images after reset
        state.uploaded_image = images_to_process[0] if images_to_process else None

        thread = threading.Thread(
            target=_run_pipeline,
            args=(images_to_process, config, state),
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
