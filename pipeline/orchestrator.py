from __future__ import annotations

import gc
import logging
import uuid
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

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

    def run_multi_image(
        self,
        input_images: list[Image.Image],
        captions: list[str] | None = None,
        fade_in_out: bool = True,
        music_path: str | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> Path:
        """
        Multi-image pipeline: Creates a video from multiple photos of the same person.
        Uses FULL original images (not just face crops) with smooth transitions
        that preserve backgrounds and surroundings.
        
        Args:
            input_images: List of PIL RGB face photos in chronological order
            progress_callback: fn(progress: float [0,1], message: str)

        Returns:
            Path to the rendered MP4 file
        """
        cfg = self.config
        vc = cfg.video

        output_path = cfg.output_dir / f"growing_up_{self.job_id}.mp4"
        frame_dir = cfg.tmp_dir / self.job_id
        frame_dir.mkdir(parents=True, exist_ok=True)

        def report(p: float, msg: str) -> None:
            logger.info("[%.0f%%] %s", p * 100, msg)
            if progress_callback:
                progress_callback(p, msg)

        out_w, out_h = vc.output_width, vc.output_height

        # ── Stage 1: Resize full images to output resolution ──────────
        report(0.05, f"Preparing {len(input_images)} full-resolution images…")
        full_images: list[Image.Image] = []
        for i, img in enumerate(input_images):
            # Resize to output resolution while keeping aspect ratio, then pad/crop
            resized = self._fit_to_canvas(img, out_w, out_h)
            full_images.append(resized)

        # Store previews from full images
        self.stage_previews = []
        for i, img in enumerate(full_images):
            thumb = img.resize((128, 128), Image.LANCZOS)
            self.stage_previews.append((i, thumb))

        # ── Stage 2: Face detection & landmark extraction ─────────────
        report(0.10, "Detecting faces and extracting landmarks…")
        aligner = FaceAligner(cfg.model.insightface_model)
        lm_extractor = LandmarkExtractor(aligner)
        all_landmarks: list[np.ndarray] = []
        face_detected: list[bool] = []

        for i, img in enumerate(full_images):
            img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            try:
                lm = lm_extractor.extract_landmarks(img_bgr)
                all_landmarks.append(lm)
                face_detected.append(True)
            except ValueError:
                logger.warning(f"No face detected in image {i} — using cross-dissolve")
                all_landmarks.append(np.zeros((68, 2), dtype=np.float32))
                face_detected.append(False)

        # ── Stage 3: Generate transitions and write frames directly ────
        report(0.20, "Creating smooth transitions…")
        morpher = DelaunayMorpher()
        warper = TriangleWarper()
        
        # Frames per transition (user-selected speed)
        frames_per_transition = max(30, cfg.pipeline.frames_per_transition)
        # Hold frames to show each image for a moment
        hold_frames = 30  # ~0.5 seconds at 60fps
        
        # Build caption text for each image
        cap_list: list[str] = []
        if captions:
            cap_list = [c.strip() if c else "" for c in captions]
        while len(cap_list) < len(input_images):
            cap_list.append("")
        
        # Pre-convert all full images to BGR numpy arrays, then free PIL images
        full_bgr: list[np.ndarray] = []
        for img in full_images:
            full_bgr.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        del full_images
        _free_memory()

        # Write frames directly to disk to save memory
        kb = KenBurnsEffect(output_size=(out_w, out_h))
        writer = FrameWriter(frame_dir)
        frame_idx = 0
        n_trans = len(full_bgr) - 1
        
        # Fade settings
        fade_frames = 30 if fade_in_out else 0  # 0.5 sec at 60fps
        
        # Calculate total frames for Ken Burns progress
        total_frames = hold_frames  # first image hold
        for i in range(n_trans):
            total_frames += frames_per_transition + hold_frames

        def _apply_caption(frame_bgr: np.ndarray, text: str, opacity: float = 1.0) -> np.ndarray:
            """Overlay text caption at the bottom of the frame."""
            if not text:
                return frame_bgr
            h, w = frame_bgr.shape[:2]
            # Convert to PIL for text rendering
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(pil_img)
            
            font_size = max(24, h // 25)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except (IOError, OSError):
                font = ImageFont.load_default()
            
            # Measure text
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            tx = (w - tw) // 2
            ty = h - th - max(30, h // 20)
            
            # Semi-transparent background bar
            bar_pad = 12
            overlay = pil_img.copy()
            draw_ov = ImageDraw.Draw(overlay)
            draw_ov.rectangle(
                [tx - bar_pad, ty - bar_pad, tx + tw + bar_pad, ty + th + bar_pad],
                fill=(0, 0, 0),
            )
            pil_img = Image.blend(pil_img, overlay, alpha=0.5 * opacity)
            
            # Draw text
            draw2 = ImageDraw.Draw(pil_img)
            text_color = (255, 255, 255, int(255 * opacity))
            draw2.text((tx, ty), text, font=font, fill=text_color[:3])
            
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        def _apply_fade(frame_bgr: np.ndarray, fade_factor: float) -> np.ndarray:
            """Apply fade: 0.0 = black, 1.0 = fully visible."""
            if fade_factor >= 1.0:
                return frame_bgr
            return (frame_bgr.astype(np.float32) * fade_factor).astype(np.uint8)

        def _write_frame(frame_bgr: np.ndarray, idx: int, caption: str = "") -> int:
            """Apply KB + caption + fade, write to disk, return next idx."""
            f = kb.apply_single(
                frame_bgr, idx, total_frames,
                zoom_start=1.0,
                zoom_end=vc.ken_burns_zoom_max,
                pan_x_end=vc.ken_burns_pan_x,
                pan_y_end=vc.ken_burns_pan_y,
            )
            if caption:
                f = _apply_caption(f, caption)
            # Fade in at start
            if fade_in_out and idx < fade_frames:
                f = _apply_fade(f, idx / max(fade_frames, 1))
            # Fade out at end
            if fade_in_out and idx >= total_frames - fade_frames:
                remaining = total_frames - 1 - idx
                f = _apply_fade(f, remaining / max(fade_frames, 1))
            writer.write_single(f, idx)
            return idx + 1

        for i in range(n_trans):
            img_src_bgr = full_bgr[i]
            img_dst_bgr = full_bgr[i + 1]
            src_caption = cap_list[i]
            dst_caption = cap_list[i + 1]
            
            # Hold on source image for a moment (only for first image)
            if i == 0:
                for _ in range(hold_frames):
                    frame_idx = _write_frame(img_src_bgr, frame_idx, src_caption)

            pts_src = all_landmarks[i]
            pts_dst = all_landmarks[i + 1]
            both_have_faces = face_detected[i] and face_detected[i + 1]
            
            # Compute triangulation once per transition
            triangles = None
            if both_have_faces:
                triangles = morpher.compute_triangulation(
                    pts_src, pts_dst, (out_h, out_w)
                )
            
            for f in range(frames_per_transition):
                alpha = f / max(frames_per_transition - 1, 1)
                alpha = self._ease_in_out(alpha)
                
                if triangles is not None:
                    frame = warper.morph_frame(
                        img_src_bgr, img_dst_bgr,
                        pts_src, pts_dst, triangles, alpha,
                    )
                else:
                    frame = cv2.addWeighted(
                        img_src_bgr, 1.0 - alpha,
                        img_dst_bgr, alpha, 0,
                    )
                
                # Cross-fade captions during transition
                cap = src_caption if alpha < 0.5 else dst_caption
                frame_idx = _write_frame(frame, frame_idx, cap)
            
            # Hold on destination image
            for _ in range(hold_frames):
                frame_idx = _write_frame(img_dst_bgr, frame_idx, dst_caption)
            
            # Free source image if no longer needed
            if i > 0:
                full_bgr[i] = None  # type: ignore[assignment]
                _free_memory()

            p_frac = 0.20 + 0.55 * ((i + 1) / n_trans)
            report(p_frac, f"Transition {i + 1}/{n_trans}…")

        del full_bgr
        _free_memory()

        # ── Stage 4: FFmpeg render ────────────────────────────────────
        report(0.80, "Rendering video…")
        renderer = VideoRenderer(cfg)
        renderer.render(
            frame_dir=frame_dir,
            output_path=output_path,
            fps=vc.fps_output,
            total_frames=frame_idx,
            progress_callback=lambda p: report(0.80 + 0.10 * p, "Encoding video…"),
        )

        # ── Stage 5: Add background music if provided ─────────────────
        if music_path:
            report(0.92, "Adding background music…")
            renderer.mux_audio(
                video_path=output_path,
                audio_path=Path(music_path),
            )

        report(1.00, "Done!")
        return output_path

    @staticmethod
    def _ease_in_out(t: float) -> float:
        """Cubic ease-in-out for smoother transitions."""
        if t < 0.5:
            return 4.0 * t * t * t
        p = 2.0 * t - 2.0
        return 0.5 * p * p * p + 1.0

    @staticmethod
    def _fit_to_canvas(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
        """
        Resize image to fit target dimensions, preserving aspect ratio.
        Fills remaining space with blurred version of the image (no black bars).
        """
        w, h = img.size
        target_ratio = target_w / target_h
        img_ratio = w / h

        if img_ratio > target_ratio:
            # Image is wider — fit by width
            new_w = target_w
            new_h = int(target_w / img_ratio)
        else:
            # Image is taller — fit by height
            new_h = target_h
            new_w = int(target_h * img_ratio)

        # Create blurred background from the image itself
        bg = img.resize((target_w, target_h), Image.LANCZOS)
        bg_np = np.array(bg)
        bg_blurred = cv2.GaussianBlur(bg_np, (51, 51), 30)
        # Darken the blurred background slightly
        bg_blurred = (bg_blurred * 0.4).astype(np.uint8)
        background = Image.fromarray(bg_blurred)

        # Resize the main image
        foreground = img.resize((new_w, new_h), Image.LANCZOS)

        # Paste centered on the blurred background
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        background.paste(foreground, (x_offset, y_offset))

        return background


def _free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
