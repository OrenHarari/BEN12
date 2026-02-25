from __future__ import annotations

import os
import shutil


def resolve_ffmpeg_binary() -> str:
    """
    Resolve FFmpeg executable path.

    Order:
      1) FFMPEG_BINARY env var
      2) ffmpeg from PATH
      3) imageio-ffmpeg bundled binary (if installed)
    """
    env_bin = os.environ.get("FFMPEG_BINARY")
    if env_bin:
        return env_bin

    path_bin = shutil.which("ffmpeg")
    if path_bin:
        return path_bin

    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        return get_ffmpeg_exe()
    except Exception as exc:
        raise FileNotFoundError(
            "FFmpeg executable not found. Install FFmpeg (system PATH) or "
            "install imageio-ffmpeg: `python -m pip install imageio-ffmpeg`."
        ) from exc
