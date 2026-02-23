from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class KenBurnsKeyframe:
    time: float    # Normalised position in clip [0, 1]
    zoom: float    # 1.0 = no zoom, 1.08 = 8% zoom
    pan_x: float   # Horizontal centre offset (fraction of width)
    pan_y: float   # Vertical centre offset (fraction of height)


class KenBurnsEffect:
    """
    Subtle Ken Burns zoom-and-pan applied per frame.

    Strategy: over the full duration of the video, zoom linearly from 1.0 to
    zoom_max (default 1.08, i.e. 8%) while drifting the frame centre slightly
    right and down.  Easing is cubic in/out so the start and end feel settled.
    """

    def __init__(self, output_size: tuple[int, int] = (1920, 1080)):
        self.W, self.H = output_size

    @staticmethod
    def _ease_in_out_cubic(t: float) -> float:
        if t < 0.5:
            return 4.0 * t * t * t
        p = 2.0 * t - 2.0
        return 0.5 * p * p * p + 1.0

    def apply(
        self,
        frame: np.ndarray,
        zoom: float,
        pan_x: float,
        pan_y: float,
    ) -> np.ndarray:
        """
        Crop-and-resize frame to simulate zoom and pan.

        Args:
            frame:  uint8 BGR [H, W, 3]
            zoom:   scale factor ≥ 1.0
            pan_x:  centre shift as fraction of frame width  (positive = right)
            pan_y:  centre shift as fraction of frame height (positive = down)
        Returns:
            uint8 BGR [self.H, self.W, 3]
        """
        H, W = frame.shape[:2]
        crop_w = int(W / zoom)
        crop_h = int(H / zoom)

        cx = W // 2 + int(pan_x * W)
        cy = H // 2 + int(pan_y * H)

        x1 = cx - crop_w // 2
        y1 = cy - crop_h // 2
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        # Clamp
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > W:
            x1 -= x2 - W
            x2 = W
        if y2 > H:
            y1 -= y2 - H
            y2 = H

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        cropped = frame[y1:y2, x1:x2]
        return cv2.resize(cropped, (self.W, self.H), interpolation=cv2.INTER_LANCZOS4)

    def apply_single(
        self,
        frame: np.ndarray,
        frame_index: int,
        total_frames: int,
        zoom_start: float = 1.0,
        zoom_end: float = 1.08,
        pan_x_end: float = 0.02,
        pan_y_end: float = 0.01,
    ) -> np.ndarray:
        """Apply Ken Burns to a single frame based on its position in the sequence."""
        t = frame_index / max(total_frames - 1, 1)
        t_e = self._ease_in_out_cubic(t)
        zoom = zoom_start + (zoom_end - zoom_start) * t_e
        px = pan_x_end * t_e
        py = pan_y_end * t_e
        return self.apply(frame, zoom, px, py)

    def apply_sequence(
        self,
        frames: list[np.ndarray],
        zoom_start: float = 1.0,
        zoom_end: float = 1.08,
        pan_x_end: float = 0.02,
        pan_y_end: float = 0.01,
    ) -> list[np.ndarray]:
        """
        Apply progressive Ken Burns to an entire frame sequence.
        Returns a new list of frames the same length as input.
        """
        n = len(frames)
        result = []
        for i, frame in enumerate(frames):
            t = i / max(n - 1, 1)
            t_e = self._ease_in_out_cubic(t)
            zoom = zoom_start + (zoom_end - zoom_start) * t_e
            px = pan_x_end * t_e
            py = pan_y_end * t_e
            result.append(self.apply(frame, zoom, px, py))
        return result
