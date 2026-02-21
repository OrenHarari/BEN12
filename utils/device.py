from __future__ import annotations

import torch


class DeviceManager:
    @staticmethod
    def get_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def get_dtype(device: torch.device) -> torch.dtype:
        if device.type in ("cuda", "mps"):
            return torch.float16
        return torch.float32

    @staticmethod
    def vram_gb() -> float:
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1e9
        return 0.0

    @staticmethod
    def recommend_settings(device: torch.device) -> dict:
        vram = DeviceManager.vram_gb()
        if vram >= 16:
            return {"sdxl_steps": 40, "cpu_offload": False}
        if vram >= 8:
            return {"sdxl_steps": 30, "cpu_offload": False}
        if vram >= 6:
            return {"sdxl_steps": 25, "cpu_offload": True}
        return {"sdxl_steps": 20, "cpu_offload": True}
