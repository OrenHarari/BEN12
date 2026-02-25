from __future__ import annotations

import gc

import torch
from PIL import Image

from modules.generation.prompt_builder import PromptBuilder


class AgeProgressionPipeline:
    """
    SDXL + IP-Adapter age progression pipeline.

    Memory layout:
    - SDXL Base (fp16) ~5.5 GB VRAM
    - SDXL Refiner (fp16) ~4.8 GB VRAM  (loaded after base)
    - IP-Adapter overhead ~0.3 GB
    - VAE shared between base and refiner

    Call load() once, then generate_age_stage() N times, then unload().
    """

    def __init__(self, config, device: torch.device):
        self.config = config
        self.device = device
        self.dtype = (
            torch.float16
            if device.type in ("cuda", "mps")
            else torch.float32
        )
        self._base_pipe = None
        self._refiner_pipe = None
        self._ip_adapter = None

    def load(self) -> None:
        from diffusers import (
            AutoencoderKL,
            DPMSolverMultistepScheduler,
            StableDiffusionXLImg2ImgPipeline,
            StableDiffusionXLPipeline,
        )

        from modules.generation.ip_adapter import load_ip_adapter_xl

        mc = self.config.model
        pc = self.config.pipeline

        # Shared FP16-safe VAE (prevents NaN issues with standard SDXL VAE at fp16)
        vae = AutoencoderKL.from_pretrained(
            mc.sdxl_vae_id,
            torch_dtype=self.dtype,
        )

        # SDXL base
        self._base_pipe = StableDiffusionXLPipeline.from_pretrained(
            mc.sdxl_model_id,
            vae=vae,
            torch_dtype=self.dtype,
            variant="fp16",
            use_safetensors=True,
        )
        self._base_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self._base_pipe.scheduler.config,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
        )
        self._base_pipe.enable_xformers_memory_efficient_attention()
        self._base_pipe.enable_vae_slicing()
        self._base_pipe.enable_vae_tiling()
        self._base_pipe = self._base_pipe.to(self.device)

        # IP-Adapter SDXL for face identity conditioning
        self._ip_adapter = load_ip_adapter_xl(
            self._base_pipe,
            image_encoder_path=mc.ip_adapter_image_encoder,
            ip_ckpt=str(self.config.ip_adapter_weights_path),
            device=self.device,
        )

        # SDXL refiner (shares VAE, loaded to same device)
        self._refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            mc.sdxl_refiner_id,
            vae=vae,
            torch_dtype=self.dtype,
            variant="fp16",
            use_safetensors=True,
        )
        self._refiner_pipe.enable_xformers_memory_efficient_attention()
        self._refiner_pipe = self._refiner_pipe.to(self.device)

    def generate_age_stage(
        self,
        face_pil: Image.Image,
        age: int,
        seed: int = 42,
        suffix: str = "",
    ) -> Image.Image:
        """
        Generate one age-progressed portrait.

        Args:
            face_pil: Aligned PIL face image (512×512, RGB)
            age: Target age in years
            seed: RNG seed for reproducibility
            suffix: Optional extra style text appended to positive prompt

        Returns:
            PIL Image (512×512, RGB)
        """
        positive, negative = PromptBuilder.build(age, suffix)
        pc = self.config.pipeline
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # IP-Adapter generate
        images = self._ip_adapter.generate(
            pil_image=face_pil,
            num_samples=1,
            num_inference_steps=pc.sdxl_steps,
            seed=seed,
            prompt=positive,
            negative_prompt=negative,
            guidance_scale=pc.sdxl_guidance,
            scale=pc.ip_adapter_scale,
        )
        base_img = images[0]
        if not pc.enable_refiner:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            return base_img

        # Light refiner pass to add detail without changing structure
        refined = self._refiner_pipe(
            prompt=positive,
            negative_prompt=negative,
            image=base_img,
            num_inference_steps=pc.sdxl_refiner_steps,
            strength=pc.sdxl_refiner_strength,
            guidance_scale=pc.sdxl_refiner_guidance,
            generator=generator,
        ).images[0]

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return refined

    def unload(self) -> None:
        """Release all VRAM held by this pipeline."""
        del self._base_pipe
        del self._refiner_pipe
        del self._ip_adapter
        self._base_pipe = None
        self._refiner_pipe = None
        self._ip_adapter = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
