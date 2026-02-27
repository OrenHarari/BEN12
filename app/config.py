from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


@dataclass
class ModelConfig:
    sdxl_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    sdxl_refiner_id: str = "stabilityai/stable-diffusion-xl-refiner-1.0"
    sdxl_vae_id: str = "madebyollin/sdxl-vae-fp16-fix"
    ip_adapter_repo: str = "h94/IP-Adapter"
    ip_adapter_weights: str = "sdxl_models/ip-adapter_sdxl.bin"
    ip_adapter_image_encoder: str = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    insightface_model: str = "buffalo_l"
    gfpgan_weights: str = "GFPGANv1.4.pth"
    rife_weights: str = "flownet.pkl"


@dataclass
class VideoConfig:
    output_width: int = 1920
    output_height: int = 1080
    fps_output: int = 30
    rife_multiplier: int = 1
    ken_burns_zoom_max: float = 1.08
    ken_burns_pan_x: float = 0.02
    ken_burns_pan_y: float = 0.01
    crf: int = 18
    codec: str = "libx264"
    preset: str = "p1"
    pixel_format: str = "yuv420p"
    prefer_gpu_encoder: bool = True


def transition_speed_to_frames(
    speed_level: int,
    fps_output: int = 60,
    rife_multiplier: int = 2,
    slowest_seconds: float = 3.2,
    fastest_seconds: float = 0.5,
    min_keyframes: int = 6,
    max_keyframes: int = 140,
) -> int:
    """
    Map UI speed slider 1..10 to morph keyframes using target transition duration.

    1 = slowest transition (longest duration), 10 = fastest (shortest duration).
    """
    speed = max(1, min(10, int(speed_level)))
    ratio = (speed - 1) / 9.0
    # Non-linear curve: make low-speed settings noticeably slower.
    eased = ratio ** 1.35
    duration_sec = slowest_seconds - eased * (slowest_seconds - fastest_seconds)

    mult = max(1, int(rife_multiplier))
    target_out_frames = max(2, int(round(duration_sec * fps_output)))
    keyframes = int(round((target_out_frames - 1) / mult)) + 1
    return max(min_keyframes, min(max_keyframes, keyframes))


def perceptual_duration_options(
    min_sec: float = 0.2,
    max_sec: float = 12.0,
    steps: int = 80,
) -> list[float]:
    """
    Return log-spaced duration options for more perceptually uniform control.
    """
    if steps < 2:
        return [round(float(min_sec), 2)]
    lo = math.log(min_sec)
    hi = math.log(max_sec)
    vals: list[float] = []
    for i in range(steps):
        t = i / (steps - 1)
        v = math.exp(lo + t * (hi - lo))
        vals.append(round(v, 2))
    # Preserve ordering and unique rounded values
    uniq = []
    seen = set()
    for v in vals:
        if v not in seen:
            uniq.append(v)
            seen.add(v)
    return uniq


def _style_params(transition_style: str) -> dict:
    style = transition_style.strip().lower()
    if style == "emotional":
        return {
            "pause_fraction": 0.08,
            "smoothing_window": 7,
            "hold_ratio": 0.22,
            "fade_ratio": 0.22,
            "camera_motion_scale": 0.78,
        }
    if style == "fast":
        return {
            "pause_fraction": 0.0,
            "smoothing_window": 1,
            "hold_ratio": 0.06,
            "fade_ratio": 0.10,
            "camera_motion_scale": 1.18,
        }
    # Balanced
    return {
        "pause_fraction": 0.0,
        "smoothing_window": 3,
        "hold_ratio": 0.12,
        "fade_ratio": 0.14,
        "camera_motion_scale": 1.0,
    }


def compute_transition_plan(
    transition_style: str,
    transition_duration_seconds: float,
    fps_output: int,
    rife_multiplier: int,
    num_images: int,
    turbo_mode: bool,
    output_width: int,
    output_height: int,
) -> dict[str, float | int | str]:
    """
    Compute synchronized transition timing plan.

    Controls morph count, interpolation frame count, fade cadence, and camera motion.
    """
    style = transition_style.strip().title()
    params = _style_params(style)
    fps = max(1, int(fps_output))
    rife = max(1, int(rife_multiplier))
    n = max(2, int(num_images))
    duration_sec = max(0.2, min(12.0, float(transition_duration_seconds)))
    hold_scale = 1.0
    fade_scale = 1.0
    keyframes_cap: int | None = None

    if turbo_mode:
        if n >= 100:
            morph_h = 480
            hold_scale = 0.45
            fade_scale = 0.50
            keyframes_cap = 30
        elif n >= 70:
            morph_h = 512
            hold_scale = 0.50
            fade_scale = 0.60
            keyframes_cap = 35
        elif n >= 40:
            morph_h = 576
            hold_scale = 0.55
            fade_scale = 0.65
            keyframes_cap = 40
        elif n >= 25:
            morph_h = 640
            hold_scale = 0.65
            fade_scale = 0.70
            keyframes_cap = 45
        else:
            morph_h = 640
    else:
        morph_h = 768

    transition_output_frames = max(2, int(round(duration_sec * fps)))
    keyframes = max(2, int(round((transition_output_frames - 1) / rife)) + 1)
    if keyframes_cap is not None:
        keyframes = min(keyframes, keyframes_cap)

    if output_height > 0:
        morph_h = min(morph_h, int(output_height))
    morph_h = max(320, (morph_h // 32) * 32)
    morph_w = int(morph_h * output_width / max(1, output_height))
    morph_w = max(320, (morph_w // 32) * 32)

    hold_frames = max(5, int(round(transition_output_frames * params["hold_ratio"] * hold_scale)))
    fade_frames = max(3, int(round(transition_output_frames * params["fade_ratio"] * fade_scale)))

    return {
        "style": style,
        "transition_duration_seconds": duration_sec,
        "transition_output_frames": transition_output_frames,
        "keyframes_per_transition": keyframes,
        "hold_frames": hold_frames,
        "fade_frames": fade_frames,
        "pause_fraction": float(params["pause_fraction"]),
        "smoothing_window": int(params["smoothing_window"]),
        "camera_motion_scale": float(params["camera_motion_scale"]),
        "morph_w": morph_w,
        "morph_h": morph_h,
        "rife_multiplier": rife,
    }


def compute_multi_image_profile(
    speed_level: int,
    num_images: int,
    fps_output: int,
    rife_multiplier: int,
    turbo_mode: bool,
    output_width: int,
    output_height: int,
) -> dict[str, int]:
    """
    Compute runtime profile for multi-image generation.
    """
    n = max(2, int(num_images))
    fps = max(1, int(fps_output))
    rife = max(1, int(rife_multiplier))

    base_keyframes = transition_speed_to_frames(
        speed_level=speed_level,
        fps_output=fps,
        rife_multiplier=rife,
    )

    if turbo_mode:
        if n >= 100:
            key_scale, hold_scale, morph_h = 0.28, 0.20, 448
        elif n >= 70:
            key_scale, hold_scale, morph_h = 0.34, 0.24, 480
        elif n >= 40:
            key_scale, hold_scale, morph_h = 0.45, 0.30, 512
        elif n >= 25:
            key_scale, hold_scale, morph_h = 0.58, 0.40, 576
        else:
            key_scale, hold_scale, morph_h = 0.78, 0.65, 640
        keyframes = max(6, int(round(base_keyframes * key_scale)))
        hold_frames = max(4, int(round(fps * 0.35 * hold_scale)))
    else:
        keyframes = base_keyframes
        hold_frames = max(8, int(round(fps * 0.35)))
        morph_h = 768

    if output_height > 0:
        morph_h = min(morph_h, output_height)
    morph_h = max(320, (morph_h // 32) * 32)
    morph_w = int(morph_h * output_width / max(1, output_height))
    morph_w = max(320, (morph_w // 32) * 32)

    transition_output_frames = (keyframes - 1) * rife + 1
    return {
        "morph_w": morph_w,
        "morph_h": morph_h,
        "keyframes_per_transition": keyframes,
        "hold_frames": hold_frames,
        "transition_output_frames": transition_output_frames,
    }


@dataclass
class PipelineConfig:
    age_stages: list = field(default_factory=lambda: [0, 10, 25, 50, 75])
    frames_per_transition: int = 6
    sdxl_steps: int = 8
    use_fast_crossdissolve: bool = False
    sdxl_guidance: float = 7.5
    sdxl_refiner_steps: int = 0
    sdxl_refiner_strength: float = 0.25
    sdxl_refiner_guidance: float = 5.0
    ip_adapter_scale: float = 0.6
    use_half_precision: bool = True
    face_align_size: int = 512
    identity_sim_threshold: float = 0.20
    enable_refiner: bool = False
    enable_gfpgan: bool = False
    enable_turbo_mode: bool = True
    transition_process_workers: int = 12
    enable_chunked_parallel: bool = True
    transition_chunk_size: int = 6
    transition_style: str = "Balanced"  # Emotional / Balanced / Fast
    transition_duration_seconds: float = 1.5


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    weights_dir: Path = field(default_factory=lambda: BASE_DIR / "weights")
    output_dir: Path = field(default_factory=lambda: BASE_DIR / "outputs")
    tmp_dir: Path = field(default_factory=lambda: BASE_DIR / "tmp")

    def __post_init__(self):
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    @property
    def gfpgan_weights_path(self) -> Path:
        return self.weights_dir / "gfpgan" / self.model.gfpgan_weights

    @property
    def rife_weights_path(self) -> Path:
        return self.weights_dir / "rife" / self.model.rife_weights

    @property
    def ip_adapter_weights_path(self) -> Path:
        return self.weights_dir / "ip_adapter" / "sdxl_models" / "ip-adapter_sdxl.bin"


_config: AppConfig | None = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = AppConfig()
    return _config
