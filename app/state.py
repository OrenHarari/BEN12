from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from PIL import Image

import streamlit as st


@dataclass
class SessionState:
    uploaded_image: Optional[Image.Image] = None
    job_id: Optional[str] = None
    is_processing: bool = False
    progress: float = 0.0
    status_message: str = ""
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    preview_frames: list = field(default_factory=list)


def get_state() -> SessionState:
    if "app_state" not in st.session_state:
        st.session_state.app_state = SessionState()
    return st.session_state.app_state


def reset_state() -> SessionState:
    st.session_state.app_state = SessionState()
    return st.session_state.app_state
