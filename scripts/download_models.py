#!/usr/bin/env python3
"""
One-shot model weight downloader.

Usage:
    python scripts/download_models.py

Downloads:
    - IP-Adapter SDXL weights  (h94/IP-Adapter on HuggingFace)
    - GFPGANv1.4               (TencentARC GitHub releases)
    - RIFE v4.6 flownet.pkl    (hzwer/Practical-RIFE GitHub releases)

InsightFace buffalo_l is downloaded automatically by the insightface
library on first use — no manual download required.

SDXL base + refiner + VAE are cached by HuggingFace diffusers on first use.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from tqdm import tqdm

WEIGHTS_DIR = Path(__file__).parent.parent / "weights"

GFPGAN_URL = (
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
)
RIFE_URL = (
    "https://github.com/hzwer/Practical-RIFE/releases/download/model-v4.6/flownet.pkl"
)


def _download_file(url: str, dest: Path, desc: str) -> None:
    if dest.exists():
        print(f"[skip] {desc} already exists at {dest}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {desc}")
    print(f"  URL : {url}")
    print(f"  Dest: {dest}")

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as fh, tqdm(
        total=total, unit="B", unit_scale=True, desc=desc
    ) as bar:
        for chunk in response.iter_content(chunk_size=65536):
            fh.write(chunk)
            bar.update(len(chunk))

    print(f"  [done] {dest}")


def download_gfpgan() -> None:
    dest = WEIGHTS_DIR / "gfpgan" / "GFPGANv1.4.pth"
    _download_file(GFPGAN_URL, dest, "GFPGANv1.4")


def download_rife() -> None:
    dest = WEIGHTS_DIR / "rife" / "flownet.pkl"
    _download_file(RIFE_URL, dest, "RIFE v4.6 flownet.pkl")


def download_ip_adapter() -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[error] huggingface_hub not installed. Run: pip install huggingface-hub")
        return

    dest_dir = WEIGHTS_DIR / "ip_adapter"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # The file will be placed at weights/ip_adapter/sdxl_models/ip-adapter_sdxl.bin
    local_path = dest_dir / "sdxl_models" / "ip-adapter_sdxl.bin"
    if local_path.exists():
        print(f"[skip] IP-Adapter SDXL weights already at {local_path}")
        return

    print("[download] IP-Adapter SDXL weights (h94/IP-Adapter on HuggingFace)")
    hf_hub_download(
        repo_id="h94/IP-Adapter",
        filename="sdxl_models/ip-adapter_sdxl.bin",
        local_dir=str(dest_dir),
    )
    print(f"  [done] {local_path}")


def main() -> None:
    print("=" * 60)
    print("AI Growing Up — Model Weight Downloader")
    print("=" * 60)

    download_gfpgan()
    download_rife()
    download_ip_adapter()

    print()
    print("All downloads complete.")
    print()
    print("Note: The following models are downloaded automatically on first use:")
    print("  • InsightFace buffalo_l   (by insightface library)")
    print("  • SDXL Base + Refiner     (by HuggingFace diffusers)")
    print("  • SDXL VAE FP16-fix       (by HuggingFace diffusers)")
    print("  • CLIP ViT-H-14           (by IP-Adapter / transformers)")


if __name__ == "__main__":
    main()
