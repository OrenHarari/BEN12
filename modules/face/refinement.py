from __future__ import annotations

import numpy as np
import torch


class FaceRefiner:
    """
    GFPGANv1.4 face restoration.
    Applied to each SDXL-generated age-stage image before morphing.
    """

    def __init__(self, weights_path: str, device: torch.device, upscale: int = 1):
        from gfpgan import GFPGANer

        self.restorer = GFPGANer(
            model_path=weights_path,
            upscale=upscale,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )

    def enhance(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Args:
            image_bgr: uint8 BGR [H, W, 3]
        Returns:
            restored: uint8 BGR [H, W, 3]
        """
        _, _, restored = self.restorer.enhance(
            image_bgr,
            has_aligned=False,
            only_center_face=True,
            paste_back=True,
        )
        if restored is None:
            return image_bgr
        return restored
