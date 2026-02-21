from __future__ import annotations

"""
Thin wrapper around the ip_adapter library's IPAdapterXL class.
Provides a consistent interface used by sdxl_pipeline.py.

The ip_adapter package must be installed from:
    pip install git+https://github.com/tencent-ailab/IP-Adapter.git
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from diffusers import StableDiffusionXLPipeline
    from PIL.Image import Image


def load_ip_adapter_xl(
    pipe: "StableDiffusionXLPipeline",
    image_encoder_path: str,
    ip_ckpt: str,
    device: "torch.device",
) -> object:
    """
    Load IPAdapterXL and attach to the given SDXL pipeline.

    Returns the IPAdapterXL instance which exposes a .generate() method.
    """
    from ip_adapter import IPAdapterXL

    return IPAdapterXL(
        pipe,
        image_encoder_path=image_encoder_path,
        ip_ckpt=ip_ckpt,
        device=device,
    )
