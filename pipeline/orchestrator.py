from __future__ import annotations

import gc
import logging
import uuid
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from PIL import Image

from app.config import AppConfig
from modules.face.alignment import FaceAligner
from modules.face.embedding import IdentityEmbedder
from modules.face.refinement import FaceRefiner
from modules.generation.sdxl_pipeline import AgeProgressionPipeline
from modules.interpolation.rife_inference import RIFEInterpolator
from modules.morph.landmarks import DelaunayMorpher, LandmarkExtractor
from modules.morph.warp import TriangleWarper
from modules.video.frame_writer import FrameWriter
from modules.video.ken_burns import KenBurnsEffect
from modules.video.renderer import VideoRenderer
from utils.device import DeviceManager
from utils.image_utils import bgr_to_pil, pil_to_bgr

logger = logging.getLogger(__name__)


class GrowingUpPipeline:
    """
    End-to-end pipeline: face photo → cinematic aging video.

    Stages:
      1.  Face detection + alignment (InsightFace)
      2.  Identity embedding (ArcFace)
      3.  SDXL + IP-Adapter age-stage generation
      4.  GFPGAN face refinement
      5.  Landmark extraction (68-pt)
      6.  Delaunay morph between consecutive stages
      7.  RIFE 4× frame interpolation
      8.  Ken Burns cinematic zoom
      9.  PNG frame sequence write
      10. FFmpeg 1080p H.264 render

    Public attributes after run():
        stage_previews: list of (age, PIL.Image) thumbnails for the UI
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.device = DeviceManager.get_device()
        self.job_id = uuid.uuid4().hex[:8]
        self.stage_previews: list[tuple[int, Image.Image]] = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        input_image: Image.Image,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> Path:
        """
        Args:
            input_image: PIL RGB face photo
            progress_callback: fn(progress: float [0,1], message: str)

        Returns:
            Path to the rendered MP4 file
        """
        cfg = self.config
        pc = cfg.pipeline
        vc = cfg.video

        output_path = cfg.output_dir / f"growing_up_{self.job_id}.mp4"
        frame_dir = cfg.tmp_dir / self.job_id
        frame_dir.mkdir(parents=True, exist_ok=True)

        def report(p: float, msg: str) -> None:
            logger.info("[%.0f%%] %s", p * 100, msg)
            if progress_callback:
                progress_callback(p, msg)

        # ── Stage 1: Face alignment ────────────────────────────────────
        report(0.02, "Detecting and aligning face…")
        aligner = FaceAligner(cfg.model.insightface_model)
        img_bgr = pil_to_bgr(input_image)
        aligned_bgr, _, _ = aligner.align_face(img_bgr, target_size=pc.face_align_size)
        aligned_pil = bgr_to_pil(aligned_bgr)

        # ── Stage 2: Identity embedding ───────────────────────────────
        report(0.04, "Extracting identity embedding…")
        embedder = IdentityEmbedder(aligner)
        try:
            identity_emb = embedder.extract(img_bgr)
        except ValueError:
            identity_emb = None

        # ── Stage 3: SDXL age-stage generation ────────────────────────
        report(0.05, "Loading SDXL + IP-Adapter…")
        sdxl = AgeProgressionPipeline(cfg, self.device)
        sdxl.load()

        age_stages = pc.age_stages
        stage_images: list[Image.Image] = []
        self.stage_previews = []

        for i, age in enumerate(age_stages):
            p_frac = 0.05 + 0.35 * (i / len(age_stages))
            report(p_frac, f"Generating age {age}…")
            gen = sdxl.generate_age_stage(aligned_pil, age, seed=42 + i)
            stage_images.append(gen)
            thumb = gen.resize((128, 128), Image.LANCZOS)
            self.stage_previews.append((age, thumb))

        report(0.40, "Unloading SDXL…")
        sdxl.unload()
        _free_memory()

        # ── Stage 4: GFPGAN face refinement ───────────────────────────
        report(0.42, "Refining faces with GFPGANv1.4…")
        if cfg.gfpgan_weights_path.exists():
            refiner = FaceRefiner(str(cfg.gfpgan_weights_path), self.device)
            refined_images: list[Image.Image] = []
            for img in stage_images:
                refined_bgr = refiner.enhance(pil_to_bgr(img))
                refined_images.append(bgr_to_pil(refined_bgr))
            del refiner
        else:
            logger.warning("GFPGANv1.4.pth not found — skipping face refinement.")
            refined_images = stage_images
        _free_memory()

        # ── Stage 5: Landmark extraction ──────────────────────────────
        report(0.48, "Extracting facial landmarks…")
        out_w, out_h = vc.output_width, vc.output_height
        lm_extractor = LandmarkExtractor(aligner)
        all_landmarks: list[np.ndarray] = []

        for img in refined_images:
            img_full = np.array(img.resize((out_w, out_h), Image.LANCZOS))
            img_bgr_full = cv2.cvtColor(img_full, cv2.COLOR_RGB2BGR)
            try:
                lm = lm_extractor.extract_landmarks(img_bgr_full)
            except ValueError:
                # Fallback: use zero-landmark placeholder; morphing degrades gracefully
                lm = np.zeros((68, 2), dtype=np.float32)
            all_landmarks.append(lm)

        # ── Stage 6: Delaunay morphing ────────────────────────────────
        report(0.50, "Computing Delaunay morph transitions…")
        morpher = DelaunayMorpher()
        warper = TriangleWarper()
        n_trans = len(refined_images) - 1
        morph_keyframes: list[np.ndarray] = []

        for i in range(n_trans):
            img_src = np.array(refined_images[i].resize((out_w, out_h), Image.LANCZOS))
            img_dst = np.array(refined_images[i + 1].resize((out_w, out_h), Image.LANCZOS))
            img_src_bgr = cv2.cvtColor(img_src, cv2.COLOR_RGB2BGR)
            img_dst_bgr = cv2.cvtColor(img_dst, cv2.COLOR_RGB2BGR)

            pts_src = all_landmarks[i]
            pts_dst = all_landmarks[i + 1]

            triangles = morpher.compute_triangulation(pts_src, pts_dst, (out_h, out_w))

            for f in range(pc.frames_per_transition):
                alpha = f / max(pc.frames_per_transition - 1, 1)
                frame = warper.morph_frame(
                    img_src_bgr, img_dst_bgr, pts_src, pts_dst, triangles, alpha
                )
                morph_keyframes.append(frame)

            p_frac = 0.50 + 0.15 * ((i + 1) / n_trans)
            report(p_frac, f"Morphing transition {i + 1}/{n_trans}…")

        # ── Stage 7: RIFE interpolation ───────────────────────────────
        rife_weights = cfg.rife_weights_path
        if rife_weights.exists():
            report(0.65, "Loading RIFE interpolator…")
            rife = RIFEInterpolator(str(rife_weights), self.device)

            report(0.66, "Running RIFE 4× interpolation…")
            interpolated = rife.interpolate_sequence(
                morph_keyframes,
                multiplier=vc.rife_multiplier,
                progress_callback=lambda p: report(0.66 + 0.14 * p, "Interpolating frames…"),
            )
            del rife
            _free_memory()
        else:
            logger.warning("RIFE weights not found — skipping interpolation.")
            interpolated = morph_keyframes

        # ── Stage 8: Ken Burns effect ─────────────────────────────────
        report(0.80, "Applying Ken Burns cinematic zoom…")
        kb = KenBurnsEffect(output_size=(out_w, out_h))
        kb_frames = kb.apply_sequence(
            interpolated,
            zoom_start=1.0,
            zoom_end=vc.ken_burns_zoom_max,
            pan_x_end=vc.ken_burns_pan_x,
            pan_y_end=vc.ken_burns_pan_y,
        )

        # ── Stage 9: Write frames ─────────────────────────────────────
        report(0.85, "Writing frames to disk…")
        writer = FrameWriter(frame_dir)
        writer.write_all(kb_frames)

        # ── Stage 10: FFmpeg render ────────────────────────────────────
        report(0.90, "Rendering 1080p MP4 with FFmpeg…")
        renderer = VideoRenderer(cfg)
        renderer.render(
            frame_dir=frame_dir,
            output_path=output_path,
            fps=vc.fps_output,
            total_frames=len(kb_frames),
            progress_callback=lambda p: report(0.90 + 0.10 * p, "Encoding video…"),
        )

        report(1.00, "Done!")
        return output_path


def _free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
