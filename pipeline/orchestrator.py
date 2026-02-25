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
from modules.video.ffmpeg_pipe import FFmpegPipeWriter
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
        if cfg.pipeline.enable_gfpgan and cfg.gfpgan_weights_path.exists():
            refiner = FaceRefiner(str(cfg.gfpgan_weights_path), self.device)
            refined_images: list[Image.Image] = []
            for img in stage_images:
                refined_bgr = refiner.enhance(pil_to_bgr(img))
                refined_images.append(bgr_to_pil(refined_bgr))
            del refiner
        else:
            if not cfg.pipeline.enable_gfpgan:
                logger.info("GFPGAN disabled — skipping face refinement.")
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
        if vc.rife_multiplier > 1 and rife_weights.exists():
            report(0.65, "Loading RIFE interpolator…")
            rife = RIFEInterpolator(str(rife_weights), self.device)

            report(0.66, f"Running RIFE {vc.rife_multiplier}× interpolation…")
            interpolated = rife.interpolate_sequence(
                morph_keyframes,
                multiplier=vc.rife_multiplier,
                progress_callback=lambda p: report(0.66 + 0.14 * p, "Interpolating frames…"),
            )
            del rife
            _free_memory()
        else:
            if vc.rife_multiplier <= 1:
                logger.info("RIFE disabled — skipping interpolation.")
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
        Multi-image pipeline — optimised for speed.

        Key optimisations vs. the original implementation:
          1. Morph at reduced resolution (MORPH_SIZE) instead of full 1080p
          2. Use RIFE 4× GPU interpolation to fill frames between morph keyframes
          3. Upscale to output resolution only after interpolation
          4. Pipe frames directly to FFmpeg (no PNG on disk)
          5. OpenCV-only caption rendering (no PIL round-trips)
          6. Hold frames written once + repeated via pipe

        Args:
            input_images: List of PIL RGB photos in chronological order
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

        # ── Morph resolution ──────────────────────────────────────────
        # Work at a smaller size for morphing; upscale afterwards.
        # 768 keeps good face detail while being ~6× cheaper than 1080p.
        MORPH_H = 768
        MORPH_W = int(MORPH_H * out_w / out_h)  # maintain aspect ratio
        # Ensure even dimensions for RIFE (needs multiples of 32 ideally)
        MORPH_W = (MORPH_W // 32) * 32 or 768
        MORPH_H = (MORPH_H // 32) * 32 or 768

        # ── Stage 1: Prepare images at BOTH resolutions ───────────────
        report(0.02, f"Preparing {len(input_images)} images…")
        full_images: list[Image.Image] = []
        for img in input_images:
            resized = self._fit_to_canvas(img, out_w, out_h)
            full_images.append(resized)

        # Store previews
        self.stage_previews = []
        for i, img in enumerate(full_images):
            thumb = img.resize((128, 128), Image.LANCZOS)
            self.stage_previews.append((i, thumb))

        # ── Stage 2: Face detection & landmark extraction (at morph res) ─
        report(0.05, "Detecting faces and extracting landmarks…")
        aligner = FaceAligner(cfg.model.insightface_model)
        lm_extractor = LandmarkExtractor(aligner)
        all_landmarks: list[np.ndarray] = []
        face_detected: list[bool] = []

        # Convert to morph-size BGR arrays for morphing
        morph_bgr: list[np.ndarray] = []
        for i, img in enumerate(full_images):
            morph_img = img.resize((MORPH_W, MORPH_H), Image.LANCZOS)
            bgr = cv2.cvtColor(np.array(morph_img), cv2.COLOR_RGB2BGR)
            morph_bgr.append(bgr)

            # Detect landmarks at morph resolution
            try:
                lm = lm_extractor.extract_landmarks(bgr)
                all_landmarks.append(lm)
                face_detected.append(True)
            except ValueError:
                logger.warning(f"No face detected in image {i} — using cross-dissolve")
                all_landmarks.append(np.zeros((68, 2), dtype=np.float32))
                face_detected.append(False)

        # Pre-convert full images to BGR for hold frames (at output res)
        full_bgr: list[np.ndarray] = []
        for img in full_images:
            full_bgr.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        del full_images
        _free_memory()

        # ── Stage 3: Morph keyframes at low resolution ────────────────
        report(0.10, "Morphing transitions at reduced resolution…")
        morpher = DelaunayMorpher()
        warper = TriangleWarper()

        keyframes_per_transition = max(8, cfg.pipeline.frames_per_transition)
        hold_count = 30  # ~0.5 s at 60fps

        # Build caption list
        cap_list: list[str] = []
        if captions:
            cap_list = [c.strip() if c else "" for c in captions]
        while len(cap_list) < len(input_images):
            cap_list.append("")

        n_trans = len(morph_bgr) - 1

        # Generate morph keyframes per transition (at MORPH_W × MORPH_H)
        # Each entry: list of BGR frames for that transition
        transition_keyframes: list[list[np.ndarray]] = []

        for i in range(n_trans):
            img_src_bgr = morph_bgr[i]
            img_dst_bgr = morph_bgr[i + 1]
            pts_src = all_landmarks[i]
            pts_dst = all_landmarks[i + 1]
            both_have_faces = face_detected[i] and face_detected[i + 1]

            frames_this: list[np.ndarray] = []

            if both_have_faces:
                triangles = morpher.compute_triangulation(
                    pts_src, pts_dst, (MORPH_H, MORPH_W)
                )
                for f in range(keyframes_per_transition):
                    alpha = f / max(keyframes_per_transition - 1, 1)
                    alpha = self._ease_in_out(alpha)
                    frame = warper.morph_frame(
                        img_src_bgr, img_dst_bgr,
                        pts_src, pts_dst, triangles, alpha,
                    )
                    frames_this.append(frame)
            else:
                # No face landmarks — fall back to cross-dissolve
                for f in range(keyframes_per_transition):
                    alpha = f / max(keyframes_per_transition - 1, 1)
                    alpha = self._ease_in_out(alpha)
                    frame = cv2.addWeighted(
                        img_src_bgr, 1.0 - alpha,
                        img_dst_bgr, alpha, 0,
                    )
                    frames_this.append(frame)

            transition_keyframes.append(frames_this)
            p_frac = 0.10 + 0.20 * ((i + 1) / n_trans)
            report(p_frac, f"Morphing keyframes {i + 1}/{n_trans}…")

        del morph_bgr
        _free_memory()

        # ── Stage 4: RIFE interpolation on morph keyframes ────────────
        rife_weights = cfg.rife_weights_path
        rife_mult = vc.rife_multiplier if vc.rife_multiplier > 1 else 4

        interpolated_transitions: list[list[np.ndarray]] = []

        if rife_weights.exists():
            report(0.30, "Loading RIFE interpolator…")
            rife = RIFEInterpolator(str(rife_weights), self.device)

            for i, kf_list in enumerate(transition_keyframes):
                report(
                    0.32 + 0.20 * (i / n_trans),
                    f"RIFE {rife_mult}× interpolation — transition {i + 1}/{n_trans}…",
                )
                interp = rife.interpolate_sequence(
                    kf_list,
                    multiplier=rife_mult,
                )
                interpolated_transitions.append(interp)

            del rife
            _free_memory()
        else:
            logger.warning("RIFE weights not found — skipping interpolation (slower output)")
            interpolated_transitions = transition_keyframes

        del transition_keyframes
        _free_memory()

        # ── Stage 5: Upscale + Ken Burns + captions → FFmpeg pipe ─────
        report(0.55, "Upscaling and encoding video…")

        # Calculate total output frames for Ken Burns / fade progress
        total_output_frames = hold_count  # first image hold
        for interp_frames in interpolated_transitions:
            total_output_frames += len(interp_frames) + hold_count

        kb = KenBurnsEffect(output_size=(out_w, out_h))
        fade_frames = 30 if fade_in_out else 0
        global_frame_idx = 0

        pipe = FFmpegPipeWriter(
            output_path,
            width=out_w,
            height=out_h,
            fps=vc.fps_output,
            crf=vc.crf,
            codec=vc.codec,
            preset=vc.preset,
            pixel_format=vc.pixel_format,
        )
        pipe.open()

        try:
            def _process_and_write(frame_bgr: np.ndarray, caption: str = "") -> None:
                """Apply KB + caption + fade, write to FFmpeg pipe."""
                nonlocal global_frame_idx

                # Ken Burns
                f = kb.apply_single(
                    frame_bgr, global_frame_idx, total_output_frames,
                    zoom_start=1.0,
                    zoom_end=vc.ken_burns_zoom_max,
                    pan_x_end=vc.ken_burns_pan_x,
                    pan_y_end=vc.ken_burns_pan_y,
                )

                # Caption (OpenCV only — no PIL round-trip)
                if caption:
                    f = self._apply_caption_cv2(f, caption)

                # Fade in
                if fade_in_out and global_frame_idx < fade_frames:
                    factor = global_frame_idx / max(fade_frames, 1)
                    f = (f.astype(np.float32) * factor).astype(np.uint8)
                # Fade out
                if fade_in_out and global_frame_idx >= total_output_frames - fade_frames:
                    remaining = total_output_frames - 1 - global_frame_idx
                    factor = remaining / max(fade_frames, 1)
                    f = (f.astype(np.float32) * factor).astype(np.uint8)

                pipe.write(f)
                global_frame_idx += 1

            def _write_hold(hold_bgr: np.ndarray, caption: str, count: int) -> None:
                """Write hold frames — reuses same source frame."""
                for _ in range(count):
                    _process_and_write(hold_bgr, caption)

            # --- Write frames per transition ---
            for i in range(n_trans):
                src_caption = cap_list[i]
                dst_caption = cap_list[i + 1]

                # Hold on first image
                if i == 0:
                    _write_hold(full_bgr[0], src_caption, hold_count)

                # Write interpolated transition frames (upscale from morph res)
                interp_frames = interpolated_transitions[i]
                for fi, morph_frame in enumerate(interp_frames):
                    # Upscale from MORPH_W×MORPH_H to output resolution
                    upscaled = cv2.resize(
                        morph_frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR
                    )
                    alpha = fi / max(len(interp_frames) - 1, 1)
                    cap = src_caption if alpha < 0.5 else dst_caption
                    _process_and_write(upscaled, cap)

                # Hold on destination image (full res — no upscale needed)
                _write_hold(full_bgr[i + 1], dst_caption, hold_count)

                # Free source if no longer needed
                if i > 0:
                    full_bgr[i] = None  # type: ignore[assignment]

                # Free transition data
                interpolated_transitions[i] = []  # type: ignore[assignment]
                _free_memory()

                p_frac = 0.55 + 0.30 * ((i + 1) / n_trans)
                report(p_frac, f"Encoding transition {i + 1}/{n_trans}…")

        finally:
            pipe.close()

        del full_bgr, interpolated_transitions
        _free_memory()

        # ── Stage 6: Add background music if provided ─────────────────
        if music_path:
            report(0.90, "Adding background music…")
            renderer = VideoRenderer(cfg)
            renderer.mux_audio(
                video_path=output_path,
                audio_path=Path(music_path),
            )

        report(1.00, "Done!")
        return output_path

    @staticmethod
    def _apply_caption_cv2(frame_bgr: np.ndarray, text: str) -> np.ndarray:
        """
        Overlay text caption at the bottom of the frame using OpenCV only.

        No PIL conversion round-trip — about 10× faster than the PIL path.
        """
        if not text:
            return frame_bgr
        h, w = frame_bgr.shape[:2]

        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = max(0.6, h / 1080.0)
        thickness = max(1, int(font_scale * 2))

        # Measure text
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        tx = (w - tw) // 2
        ty = h - max(30, h // 20)

        # Semi-transparent background bar
        bar_pad = 12
        overlay = frame_bgr.copy()
        cv2.rectangle(
            overlay,
            (tx - bar_pad, ty - th - bar_pad),
            (tx + tw + bar_pad, ty + baseline + bar_pad),
            (0, 0, 0),
            cv2.FILLED,
        )
        result = cv2.addWeighted(frame_bgr, 0.5, overlay, 0.5, 0)

        # Draw text
        cv2.putText(
            result, text, (tx, ty),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )
        return result

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
