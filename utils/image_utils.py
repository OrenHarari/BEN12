from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def resize_keep_aspect(image: np.ndarray, long_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = long_side / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def pad_to_size(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = image.shape[:2]
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT)


def numpy_to_tensor(
    image: np.ndarray,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    t = torch.from_numpy(image).float() / 255.0
    t = t.permute(2, 0, 1).unsqueeze(0)
    return t.to(device=device, dtype=dtype)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    t = tensor.squeeze(0).permute(1, 2, 0)
    return (t.cpu().float().clamp(0, 1).numpy() * 255).astype(np.uint8)


def resize_pil(image: Image.Image, width: int, height: int) -> Image.Image:
    return image.resize((width, height), Image.LANCZOS)
