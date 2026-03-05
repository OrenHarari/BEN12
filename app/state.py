from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from PIL import Image

import streamlit as st


@dataclass
class SessionState:
    uploaded_media_items: list[dict] = field(default_factory=list)
    uploaded_images: list[Image.Image] = field(default_factory=list)
    uploaded_image: Optional[Image.Image] = None  # for backward compatibility
    job_id: Optional[str] = None
    is_processing: bool = False
    progress: float = 0.0
    status_message: str = ""
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    preview_frames: list = field(default_factory=list)
    stage_timings: dict = field(default_factory=dict)
    uploads_token: str = ""
    # New feature settings
    image_captions: list[str] = field(default_factory=list)
    image_text_overlays: list[str] = field(default_factory=list)
    transition_style: str = "Balanced"  # Emotional / Balanced / Fast
    transition_duration_seconds: float = 1.5
    transition_process_workers: int = 12
    chunked_parallel: bool = True
    transition_chunk_size: int = 6
    resolution: str = "720p"  # 720p / 1080p / 4K
    performance_mode: str = "extreme_speed_3090"  # includes extreme_* presets
    turbo_mode: bool = True
    timeline_video_slow_factors: list[float] = field(default_factory=list)
    music_path: Optional[str] = None
    music_paths: list[str] = field(default_factory=list)  # up to 3 music files
    fade_enabled: bool = True


def get_state() -> SessionState:
    if "app_state" not in st.session_state:
        st.session_state.app_state = SessionState()
    return st.session_state.app_state


def reset_state() -> SessionState:
    st.session_state.app_state = SessionState()
    return st.session_state.app_state
