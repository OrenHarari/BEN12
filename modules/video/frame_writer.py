from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class FrameWriter:
    """Writes a list of BGR numpy frames as a PNG sequence to disk."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

    def write_all(self, frames: list[np.ndarray]) -> None:
        for idx, frame in enumerate(frames):
            path = self.output_dir / f"{idx:06d}.png"
            cv2.imwrite(str(path), frame)

    def write_frame(self, frame: np.ndarray, index: int) -> None:
        path = self.output_dir / f"{index:06d}.png"
        cv2.imwrite(str(path), frame)

    # Alias for convenience
    write_single = write_frame
