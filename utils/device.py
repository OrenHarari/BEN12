from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def _torch() -> Any:
    import torch

    return torch


class DeviceManager:
    _device_cache: "torch.device | None" = None

    @staticmethod
    def get_device() -> "torch.device":
        if DeviceManager._device_cache is not None:
            return DeviceManager._device_cache

        torch = _torch()
        if torch.cuda.is_available():
            DeviceManager._device_cache = torch.device("cuda:0")
            return DeviceManager._device_cache
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            DeviceManager._device_cache = torch.device("mps")
            return DeviceManager._device_cache
        DeviceManager._device_cache = torch.device("cpu")
        return DeviceManager._device_cache

    @staticmethod
    def get_dtype(device: "torch.device") -> "torch.dtype":
        torch = _torch()
        if device.type in ("cuda", "mps"):
            return torch.float16
        return torch.float32

    @staticmethod
    def vram_gb() -> float:
        torch = _torch()
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1e9
        return 0.0

    @staticmethod
    def recommend_settings(device: "torch.device") -> dict:
        _ = device
        vram = DeviceManager.vram_gb()
        if vram >= 16:
            return {"sdxl_steps": 40, "cpu_offload": False}
        if vram >= 8:
            return {"sdxl_steps": 30, "cpu_offload": False}
        if vram >= 6:
            return {"sdxl_steps": 25, "cpu_offload": True}
        return {"sdxl_steps": 20, "cpu_offload": True}
