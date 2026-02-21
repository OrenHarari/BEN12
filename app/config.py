from __future__ import annotations

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
    ip_adapter_image_encoder: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    insightface_model: str = "buffalo_l"
    gfpgan_weights: str = "GFPGANv1.4.pth"
    rife_weights: str = "flownet.pkl"


@dataclass
class VideoConfig:
    output_width: int = 1920
    output_height: int = 1080
    fps_output: int = 60
    rife_multiplier: int = 4
    ken_burns_zoom_max: float = 1.08
    ken_burns_pan_x: float = 0.02
    ken_burns_pan_y: float = 0.01
    crf: int = 18
    codec: str = "libx264"
    preset: str = "slow"
    pixel_format: str = "yuv420p"


@dataclass
class PipelineConfig:
    age_stages: list = field(default_factory=lambda: [0, 3, 7, 12, 16, 25, 35, 50, 65, 80])
    frames_per_transition: int = 48
    sdxl_steps: int = 30
    sdxl_guidance: float = 7.5
    ip_adapter_scale: float = 0.6
    use_half_precision: bool = True
    face_align_size: int = 512
    identity_sim_threshold: float = 0.20


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
