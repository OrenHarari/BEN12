from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from PIL import Image

import streamlit as st


@dataclass
class SessionState:
    uploaded_images: list[Image.Image] = field(default_factory=list)
    uploaded_image: Optional[Image.Image] = None  # for backward compatibility
    job_id: Optional[str] = None
    is_processing: bool = False
    progress: float = 0.0
    status_message: str = ""
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    preview_frames: list = field(default_factory=list)
    # New feature settings
    image_captions: list[str] = field(default_factory=list)
    transition_speed: str = "normal"  # slow / normal / fast
    resolution: str = "1080p"  # 720p / 1080p / 4K
    music_path: Optional[str] = None
    fade_enabled: bool = True


def get_state() -> SessionState:
    if "app_state" not in st.session_state:
        st.session_state.app_state = SessionState()
    return st.session_state.app_state


def reset_state() -> SessionState:
    st.session_state.app_state = SessionState()
    return st.session_state.app_state
