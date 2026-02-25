# AI Growing Up Video Generator

A fully local, production-ready pipeline that takes a single childhood face photo and produces a cinematic 1080p MP4 showing the subject aging from birth to old age.

**No external APIs. All models are open-source. All inference runs locally.**

---

## Features

| Feature | Technology |
|---|---|
| Face detection & alignment | InsightFace `buffalo_l` (RetinaFace + ArcFace) |
| Identity preservation | IP-Adapter SDXL (face image conditioning) |
| Age progression | Stable Diffusion XL + SDXL Refiner |
| Face enhancement | GFPGANv1.4 |
| Landmark morphing | Delaunay triangulation + affine warp (OpenCV) |
| Frame interpolation | RIFE IFNet v4.6 (4× recursive bisection) |
| Cinematic motion | Ken Burns cubic-eased zoom (1.0 → 1.08×) |
| Final render | FFmpeg libx264 CRF 18 @ 60fps 1920×1080 |
| UI | Streamlit |

---

## Requirements

- Python 3.11+
- NVIDIA GPU with ≥8 GB VRAM (recommended), or Apple Silicon MPS, or CPU (very slow)
- FFmpeg installed system-wide
- ~30 GB disk space for model weights + outputs

---

## Quick Start

### 1. Install dependencies

```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. Download model weights

```bash
python scripts/download_models.py
```

This downloads:
- `GFPGANv1.4.pth` → `weights/gfpgan/`
- `flownet.pkl` (RIFE v4.6) → `weights/rife/`
- `ip-adapter_sdxl.bin` → `weights/ip_adapter/sdxl_models/`

InsightFace `buffalo_l` and all SDXL/diffusers models download automatically on first run.

### 3. Launch the UI

```bash
streamlit run app/main.py
```

Open http://localhost:8501 in your browser.

---

## Docker

### Build and run with GPU

```bash
docker compose up --build
```

Open http://localhost:8501.

Weights are persisted in `./weights/` on the host via Docker volume mount.

---

## Project Structure

```
BEN12/
├── app/                    # Streamlit UI + config
│   ├── main.py             # Entry point
│   ├── config.py           # All configuration
│   └── state.py            # Session state management
├── pipeline/
│   └── orchestrator.py     # 10-stage pipeline sequencer
├── modules/
│   ├── face/               # Alignment, embedding, GFPGAN
│   ├── generation/         # SDXL + IP-Adapter age progression
│   ├── morph/              # Delaunay morphing
│   ├── interpolation/      # RIFE IFNet frame interpolation
│   └── video/              # Ken Burns, frame writer, FFmpeg
├── utils/                  # Device detection, image conversion
├── scripts/
│   └── download_models.py  # One-shot weight downloader
├── weights/                # Model weights (gitignored)
├── outputs/                # Generated MP4s (gitignored)
└── tmp/                    # Intermediate frames (gitignored)
```

---

## Pipeline Stages

| # | Stage | Progress |
|---|---|---|
| 1 | Face detection + 5-pt affine alignment | 2% |
| 2 | ArcFace identity embedding | 4% |
| 3 | SDXL + IP-Adapter age generation (10 stages) | 5–40% |
| 4 | GFPGANv1.4 face refinement | 40–48% |
| 5 | InsightFace 68-pt landmark extraction | 48–50% |
| 6 | Delaunay morph (48 frames/transition) | 50–65% |
| 7 | RIFE 4× frame interpolation | 65–80% |
| 8 | Ken Burns cubic-eased zoom | 80–85% |
| 9 | PNG frame sequence write | 85–90% |
| 10 | FFmpeg 1080p H.264 encode | 90–100% |

---

## GPU Memory Usage

| Model | VRAM | Strategy |
|---|---|---|
| SDXL Base FP16 | ~5.5 GB | xformers, VAE slicing/tiling |
| SDXL Refiner FP16 | ~4.8 GB | Sequential (not simultaneous with Base) |
| IP-Adapter | +0.3 GB | Attached to Base only during generation |
| GFPGANv1.4 | ~0.8 GB | Loaded/deleted after Stage 4 |
| RIFE IFNet FP16 | ~0.4 GB | Loaded/deleted after Stage 7 |

**Critical**: SDXL is fully unloaded before RIFE loads. Never both in VRAM simultaneously.

---

## Configuration

Edit `app/config.py` to adjust:

```python
PipelineConfig(
    age_stages=[0, 3, 7, 12, 16, 25, 35, 50, 65, 80],  # ages to generate
    frames_per_transition=48,    # morph frames between each stage
    sdxl_steps=30,               # SDXL inference steps
    ip_adapter_scale=0.6,        # face identity strength (0.4–0.8)
)

VideoConfig(
    fps_output=60,               # output frame rate
    rife_multiplier=4,           # RIFE interpolation factor (2 or 4)
    ken_burns_zoom_max=1.08,     # max zoom (1.0 = none, 1.08 = 8%)
    crf=18,                      # FFmpeg quality (lower = better)
    prefer_gpu_encoder=True,     # auto-use h264_nvenc on CUDA when available
)
```

In the Streamlit UI, transition speed is controlled via a numeric slider `1..10`:
- `1` = slowest transition (most in-between frames)
- `10` = fastest transition
