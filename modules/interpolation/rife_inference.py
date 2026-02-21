from __future__ import annotations

import gc
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from modules.interpolation.rife_model import IFNet


class RIFEInterpolator:
    """
    4× frame interpolation using RIFE IFNet v4.6.

    For each consecutive pair of morph-keyframes (A, B) the recursive
    bisection strategy generates 3 intermediate frames at t=0.25, 0.5, 0.75:

        step 1: interpolate(A, B, t=0.5)  → M
        step 2: interpolate(A, M, t=0.5)  → Q1  (= A…B at 0.25)
        step 3: interpolate(M, B, t=0.5)  → Q3  (= A…B at 0.75)

    Using t=0.5 throughout matches what the pretrained weights were optimised for.
    """

    def __init__(self, weights_path: str, device: torch.device):
        self.device = device
        self.model = IFNet().to(device)
        state = torch.load(weights_path, map_location=device)
        # Accept both raw state_dicts and checkpoint dicts
        if "model" in state:
            state = state["model"]
        elif "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        if device.type == "cuda":
            self.model = self.model.half()

    @staticmethod
    def _pad32(tensor: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Pad H and W up to the next multiple of 32."""
        _, _, H, W = tensor.shape
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32
        padded = F.pad(tensor, [0, pad_w, 0, pad_h])
        return padded, H, W

    def _frame_to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(frame).float().div(255.0)
        t = t.permute(2, 0, 1).unsqueeze(0).to(self.device)
        if self.device.type == "cuda":
            t = t.half()
        return t

    def _tensor_to_frame(self, t: torch.Tensor, H: int, W: int) -> np.ndarray:
        t = t[:, :, :H, :W].squeeze(0).permute(1, 2, 0)
        return (t.float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    @torch.inference_mode()
    def _interp_once(self, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
        """Interpolate at t=0.5 between two padded tensors."""
        x = torch.cat([t0, t1], dim=1)
        x_pad, H, W = self._pad32(x)
        mid = self.model(x_pad, timestep=0.5)
        return mid[:, :, :H, :W].float()

    def _recursive(
        self,
        t0: torch.Tensor,
        t1: torch.Tensor,
        multiplier: int,
    ) -> list[torch.Tensor]:
        """Return multiplier-1 intermediate tensors between t0 and t1."""
        if multiplier <= 1:
            return []
        mid = self._interp_once(t0, t1)
        # Promote to fp32 for recursion to avoid accumulation errors
        mid_fp32 = mid.float()
        t0_fp32 = t0.float()
        t1_fp32 = t1.float()
        left = self._recursive(t0_fp32, mid_fp32, multiplier // 2)
        right = self._recursive(mid_fp32, t1_fp32, multiplier // 2)
        return left + [mid_fp32] + right

    def interpolate_pair(
        self,
        frame0: np.ndarray,
        frame1: np.ndarray,
        multiplier: int = 4,
    ) -> list[np.ndarray]:
        """
        Returns (multiplier - 1) intermediate numpy frames between frame0 and frame1.
        """
        H, W = frame0.shape[:2]
        t0 = self._frame_to_tensor(frame0)
        t1 = self._frame_to_tensor(frame1)
        intermediates = self._recursive(t0, t1, multiplier)
        return [self._tensor_to_frame(t, H, W) for t in intermediates]

    def interpolate_sequence(
        self,
        frames: list[np.ndarray],
        multiplier: int = 4,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[np.ndarray]:
        """
        Interpolate an entire sequence.

        Input:  N frames
        Output: (N-1)*multiplier + 1 frames
        """
        if len(frames) < 2:
            return frames

        output: list[np.ndarray] = [frames[0]]
        n_pairs = len(frames) - 1

        for i in range(n_pairs):
            intermediates = self.interpolate_pair(frames[i], frames[i + 1], multiplier)
            output.extend(intermediates)
            output.append(frames[i + 1])

            if i % 50 == 49 and self.device.type == "cuda":
                torch.cuda.empty_cache()

            if progress_callback:
                progress_callback((i + 1) / n_pairs)

        return output
