from __future__ import annotations

import numpy as np


def alpha_blend(img_a: np.ndarray, img_b: np.ndarray, alpha: float) -> np.ndarray:
    """
    Simple alpha cross-dissolve between two uint8 images.
    alpha=0.0 → img_a, alpha=1.0 → img_b.
    """
    blended = img_a.astype(np.float32) * (1.0 - alpha) + img_b.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)
