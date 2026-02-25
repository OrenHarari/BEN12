"""FFmpeg pipe writer — streams raw BGR frames directly into an H.264 MP4.

Eliminates the PNG-encode → disk-write → FFmpeg-read round-trip that was
adding ~100 s and ~7 GB temp disk usage for a 14-image project at 1080p.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


class FFmpegPipeWriter:
    """
    Streams raw BGR uint8 frames to FFmpeg via stdin pipe.

    Usage::

        writer = FFmpegPipeWriter(output_path, width=1920, height=1080, fps=60)
        writer.open()
        for frame in frames:
            writer.write(frame)
        writer.close()

    Or as a context manager::

        with FFmpegPipeWriter(output_path, ...) as w:
            for frame in frames:
                w.write(frame)
    """

    def __init__(
        self,
        output_path: Path | str,
        width: int,
        height: int,
        fps: int = 60,
        crf: int = 18,
        codec: str = "libx264",
        preset: str = "medium",
        pixel_format: str = "yuv420p",
    ):
        self.output_path = Path(output_path)
        self.width = width
        self.height = height
        self.fps = fps
        self.crf = crf
        self.codec = codec
        self.preset = preset
        self.pixel_format = pixel_format
        self._process: subprocess.Popen | None = None
        self._frame_count = 0

    def open(self) -> None:
        """Start the FFmpeg subprocess with stdin pipe."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "pipe:0",
            "-c:v", self.codec,
            "-preset", self.preset,
            "-crf", str(self.crf),
            "-pix_fmt", self.pixel_format,
            "-movflags", "+faststart",
            str(self.output_path),
        ]
        logger.info("FFmpeg pipe cmd: %s", " ".join(cmd))

        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._frame_count = 0

    def write(self, frame_bgr: np.ndarray) -> None:
        """Write one BGR uint8 frame.  Raises if pipe is broken."""
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("FFmpegPipeWriter is not open")

        # Ensure correct resolution
        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            import cv2
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        self._process.stdin.write(frame_bgr.tobytes())
        self._frame_count += 1

    def close(self) -> None:
        """Flush stdin and wait for FFmpeg to finish."""
        if self._process is None:
            return

        if self._process.stdin:
            try:
                self._process.stdin.close()
            except BrokenPipeError:
                pass

        # Read stderr and wait — do NOT use communicate() since stdin is already closed
        stderr_bytes = b""
        if self._process.stderr:
            stderr_bytes = self._process.stderr.read()
            self._process.stderr.close()

        self._process.wait()
        rc = self._process.returncode

        if rc != 0:
            stderr_text = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
            logger.error("FFmpeg pipe failed (rc=%d): %s", rc, stderr_text[-1000:])
            raise RuntimeError(f"FFmpeg pipe exited with code {rc}")

        logger.info("FFmpeg pipe finished — %d frames written to %s", self._frame_count, self.output_path)
        self._process = None

    @property
    def frames_written(self) -> int:
        return self._frame_count

    # ── Context manager ────────────────────────────────────────────────
    def __enter__(self) -> "FFmpegPipeWriter":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            self.close()
        except RuntimeError:
            if exc_type is None:
                raise  # only re-raise if no other exception
