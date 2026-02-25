from __future__ import annotations

import gc
import logging
import os
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Callable

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from app.config import AppConfig, compute_transition_plan
from modules.face.alignment import FaceAligner
from modules.face.embedding import IdentityEmbedder
from modules.face.refinement import FaceRefiner
from modules.generation.sdxl_pipeline import AgeProgressionPipeline
from modules.interpolation.rife_inference import RIFEInterpolator
from modules.morph.landmarks import DelaunayMorpher, LandmarkExtractor
from modules.morph.warp import TriangleWarper
from modules.video.ffmpeg_pipe import FFmpegPipeWriter
from modules.video.ken_burns import KenBurnsEffect
from modules.video.renderer import VideoRenderer
from utils.device import DeviceManager
from utils.image_utils import bgr_to_pil, pil_to_bgr

logger = logging.getLogger(__name__)


def _generate_transition_keyframes_worker(
    src: np.ndarray,
    dst: np.ndarray,
    pts_src: np.ndarray,
    pts_dst: np.ndarray,
    both_have_faces: bool,
    alpha_schedule: list[float],
) -> list[np.ndarray]:
    """
    Process-safe worker for one transition morph sequence.
    """
    if not alpha_schedule:
        return [src]

    if both_have_faces:
        morpher = DelaunayMorpher()
        warper = TriangleWarper()
        h, w = src.shape[:2]
        triangles = morpher.compute_triangulation(pts_src, pts_dst, (h, w))
        pts_all_src, pts_all_dst = warper._build_border_pts(pts_src, pts_dst, h, w)
        src_f = src.astype(np.float32)
        dst_f = dst.astype(np.float32)
        return [
            warper.morph_frame_precomputed(
                img_src_f=src_f,
                img_dst_f=dst_f,
                pts_all_src=pts_all_src,
                pts_all_dst=pts_all_dst,
                triangles=triangles,
                alpha=float(alpha),
            )
            for alpha in alpha_schedule
        ]

    return [
        cv2.addWeighted(src, 1.0 - float(alpha), dst, float(alpha), 0)
        for alpha in alpha_schedule
    ]


def _generate_transition_chunk_worker(
    tasks: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]],
    alpha_schedule: list[float],
) -> dict[int, list[np.ndarray]]:
    """
    Process-safe worker for a chunk of transitions.
    """
    out: dict[int, list[np.ndarray]] = {}
    for idx, src, dst, pts_src, pts_dst, both_have_faces in tasks:
        out[idx] = _generate_transition_keyframes_worker(
            src=src,
            dst=dst,
            pts_src=pts_src,
            pts_dst=pts_dst,
            both_have_faces=both_have_faces,
            alpha_schedule=alpha_schedule,
        )
    return out


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
        self.cpu_workers = max(2, min(8, (os.cpu_count() or 4) // 2))
        self.last_stage_timings: dict[str, float] = {}
        cv2.setUseOptimized(True)
        cv2.setNumThreads(max(1, self.cpu_workers))

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
        single_plan = compute_transition_plan(
            transition_style=pc.transition_style,
            transition_duration_seconds=pc.transition_duration_seconds,
            fps_output=vc.fps_output,
            rife_multiplier=vc.rife_multiplier,
            num_images=max(2, len(refined_images)),
            turbo_mode=pc.enable_turbo_mode,
            output_width=out_w,
            output_height=out_h,
        )
        alpha_schedule_single = self._alpha_schedule(
            count=pc.frames_per_transition,
            style=str(single_plan["style"]),
            pause_fraction=float(single_plan["pause_fraction"]),
            smoothing_window=int(single_plan["smoothing_window"]),
        )

        for i in range(n_trans):
            img_src = np.array(refined_images[i].resize((out_w, out_h), Image.LANCZOS))
            img_dst = np.array(refined_images[i + 1].resize((out_w, out_h), Image.LANCZOS))
            img_src_bgr = cv2.cvtColor(img_src, cv2.COLOR_RGB2BGR)
            img_dst_bgr = cv2.cvtColor(img_dst, cv2.COLOR_RGB2BGR)

            pts_src = all_landmarks[i]
            pts_dst = all_landmarks[i + 1]

            triangles = morpher.compute_triangulation(pts_src, pts_dst, (out_h, out_w))

            for f in range(pc.frames_per_transition):
                alpha = alpha_schedule_single[f]
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
        single_motion_scale = float(single_plan["camera_motion_scale"])
        zoom_end = 1.0 + (vc.ken_burns_zoom_max - 1.0) * single_motion_scale
        pan_x_end = vc.ken_burns_pan_x * single_motion_scale
        pan_y_end = vc.ken_burns_pan_y * single_motion_scale
        kb_frames = kb.apply_sequence(
            interpolated,
            zoom_start=1.0,
            zoom_end=zoom_end,
            pan_x_end=pan_x_end,
            pan_y_end=pan_y_end,
        )

        # ── Stage 9/10: Stream frames directly to FFmpeg ─────────────
        report(0.85, "Streaming frames to FFmpeg...")
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
        total_frames = max(len(kb_frames), 1)
        pipe.open()
        try:
            for i, frame in enumerate(kb_frames):
                pipe.write(frame)
                if i % 30 == 0 or i == total_frames - 1:
                    p = (i + 1) / total_frames
                    report(0.85 + 0.15 * p, "Encoding video...")
        finally:
            pipe.close()

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
        Multi-image pipeline optimized for throughput.

        Key optimizations:
          1. Dynamic morph resolution and hold frames (turbo profile)
          2. Transition-by-transition processing (no giant in-memory frame lists)
          3. Direct FFmpeg pipe with optional upscale in FFmpeg
          4. Optional RIFE interpolation per transition
        """
        cfg = self.config
        vc = cfg.video

        output_path = cfg.output_dir / f"growing_up_{self.job_id}.mp4"
        n_images = len(input_images)
        if n_images < 2:
            raise ValueError("run_multi_image requires at least 2 images")

        def report(p: float, msg: str) -> None:
            logger.info("[%.0f%%] %s", p * 100, msg)
            if progress_callback:
                progress_callback(p, msg)

        stage_times: dict[str, float] = {}
        t_prepare = perf_counter()
        out_w, out_h = vc.output_width, vc.output_height
        profile = compute_transition_plan(
            transition_style=cfg.pipeline.transition_style,
            transition_duration_seconds=cfg.pipeline.transition_duration_seconds,
            fps_output=vc.fps_output,
            rife_multiplier=vc.rife_multiplier,
            num_images=n_images,
            turbo_mode=cfg.pipeline.enable_turbo_mode,
            output_width=out_w,
            output_height=out_h,
        )

        morph_w = int(profile["morph_w"])
        morph_h = int(profile["morph_h"])
        keyframes_per_transition = max(2, int(profile["keyframes_per_transition"]))
        hold_count = max(2, int(profile["hold_frames"]))
        style_name = str(profile["style"])
        pause_fraction = float(profile["pause_fraction"])
        smoothing_window = max(1, int(profile["smoothing_window"]))
        camera_motion_scale = float(profile["camera_motion_scale"])
        stage_times["transition_duration_s"] = float(profile["transition_duration_seconds"])
        stage_times["transition_frames"] = float(profile["transition_output_frames"])

        report(0.02, f"Preparing {n_images} images...")
        morph_bgr: list[np.ndarray] = []
        self.stage_previews = []

        for i, img in enumerate(input_images):
            fitted = self._fit_to_canvas(img, out_w, out_h)
            self.stage_previews.append((i, fitted.resize((128, 128), Image.LANCZOS)))
            morph_img = fitted.resize((morph_w, morph_h), Image.LANCZOS)
            morph_bgr.append(cv2.cvtColor(np.array(morph_img), cv2.COLOR_RGB2BGR))
        stage_times["prepare_images"] = perf_counter() - t_prepare

        t_landmarks = perf_counter()
        report(0.05, "Detecting faces and extracting landmarks...")
        aligner = FaceAligner(cfg.model.insightface_model)
        lm_extractor = LandmarkExtractor(aligner)
        all_landmarks: list[np.ndarray] = []
        face_detected: list[bool] = []

        for i, bgr in enumerate(morph_bgr):
            try:
                lm = lm_extractor.extract_landmarks(bgr)
                all_landmarks.append(lm)
                face_detected.append(True)
            except ValueError:
                logger.warning("No face detected in image %d - using cross-dissolve", i)
                all_landmarks.append(np.zeros((68, 2), dtype=np.float32))
                face_detected.append(False)
        stage_times["landmarks"] = perf_counter() - t_landmarks

        cap_list: list[str] = []
        if captions:
            cap_list = [c.strip() if c else "" for c in captions]
        while len(cap_list) < n_images:
            cap_list.append("")

        n_trans = n_images - 1
        rife_weights = cfg.rife_weights_path
        rife_mult = vc.rife_multiplier if vc.rife_multiplier > 1 else 1
        transition_out_frames = (
            int(profile["transition_output_frames"]) if rife_weights.exists() and rife_mult > 1
            else keyframes_per_transition
        )

        total_output_frames = hold_count
        total_output_frames += n_trans * (transition_out_frames + hold_count)

        kb = KenBurnsEffect(output_size=(morph_w, morph_h))
        fade_frames = min(int(profile["fade_frames"]), hold_count) if fade_in_out else 0
        zoom_end = 1.0 + (vc.ken_burns_zoom_max - 1.0) * camera_motion_scale
        pan_x_end = vc.ken_burns_pan_x * camera_motion_scale
        pan_y_end = vc.ken_burns_pan_y * camera_motion_scale
        global_frame_idx = 0

        pipe = FFmpegPipeWriter(
            output_path,
            width=morph_w,
            height=morph_h,
            fps=vc.fps_output,
            crf=vc.crf,
            codec=vc.codec,
            preset=vc.preset,
            pixel_format=vc.pixel_format,
            scale_width=out_w,
            scale_height=out_h,
        )
        pipe.open()

        morpher = DelaunayMorpher()
        warper = TriangleWarper()

        rife: RIFEInterpolator | None = None
        if rife_weights.exists() and rife_mult > 1:
            report(0.18, "Loading RIFE interpolator...")
            try:
                rife = RIFEInterpolator(str(rife_weights), self.device)
            except Exception as exc:
                logger.warning("RIFE unavailable (%s). Falling back to non-interpolated transitions.", exc)
                rife = None

        report(0.20, "Generating transitions...")
        alpha_schedule = self._alpha_schedule(
            count=keyframes_per_transition,
            style=style_name,
            pause_fraction=pause_fraction,
            smoothing_window=smoothing_window,
        )
        t_morph_total = 0.0
        t_rife_total = 0.0
        t_encode = perf_counter()

        try:
            process_workers = max(1, int(cfg.pipeline.transition_process_workers))
            use_process_pool = process_workers > 1 and n_trans >= (process_workers + 1)
            chunked_parallel = bool(cfg.pipeline.enable_chunked_parallel) and use_process_pool
            chunk_size = max(1, int(cfg.pipeline.transition_chunk_size))
            stage_times["process_workers"] = float(process_workers)
            stage_times["multiprocess_mode"] = 1.0 if use_process_pool else 0.0
            stage_times["chunked_parallel"] = 1.0 if chunked_parallel else 0.0
            stage_times["chunk_size"] = float(chunk_size)
            process_pool: ProcessPoolExecutor | None = None
            pending_frames: dict[int, object] = {}
            pending_chunks: dict[int, object] = {}
            chunk_results: dict[int, dict[int, list[np.ndarray]]] = {}
            next_chunk_start = 0

            if use_process_pool:
                process_pool = ProcessPoolExecutor(max_workers=process_workers)
                if chunked_parallel:
                    in_flight = 0
                    while next_chunk_start < n_trans and in_flight < process_workers:
                        start = next_chunk_start
                        end = min(n_trans, start + chunk_size)
                        tasks = [
                            (
                                idx,
                                morph_bgr[idx],
                                morph_bgr[idx + 1],
                                all_landmarks[idx],
                                all_landmarks[idx + 1],
                                face_detected[idx] and face_detected[idx + 1],
                            )
                            for idx in range(start, end)
                        ]
                        pending_chunks[start] = process_pool.submit(
                            _generate_transition_chunk_worker,
                            tasks,
                            alpha_schedule,
                        )
                        next_chunk_start = end
                        in_flight += 1
                else:
                    prefetch = min(n_trans, process_workers * 2)
                    for idx in range(prefetch):
                        pending_frames[idx] = process_pool.submit(
                            _generate_transition_keyframes_worker,
                            morph_bgr[idx],
                            morph_bgr[idx + 1],
                            all_landmarks[idx],
                            all_landmarks[idx + 1],
                            face_detected[idx] and face_detected[idx + 1],
                            alpha_schedule,
                        )

            def _process_and_write(frame_bgr: np.ndarray, caption: str = "") -> None:
                nonlocal global_frame_idx

                f = kb.apply_single(
                    frame_bgr,
                    global_frame_idx,
                    total_output_frames,
                    zoom_start=1.0,
                    zoom_end=zoom_end,
                    pan_x_end=pan_x_end,
                    pan_y_end=pan_y_end,
                )

                if caption:
                    f = self._apply_caption_cv2(f, caption)

                if fade_in_out and global_frame_idx < fade_frames:
                    factor = global_frame_idx / max(fade_frames, 1)
                    f = (f.astype(np.float32) * factor).astype(np.uint8)
                if fade_in_out and global_frame_idx >= total_output_frames - fade_frames:
                    remaining = total_output_frames - 1 - global_frame_idx
                    factor = remaining / max(fade_frames, 1)
                    f = (f.astype(np.float32) * factor).astype(np.uint8)

                pipe.write(f)
                global_frame_idx += 1

            def _write_hold(frame_bgr: np.ndarray, caption: str, count: int) -> None:
                for _ in range(count):
                    _process_and_write(frame_bgr, caption)

            for i in range(n_trans):
                src = morph_bgr[i]
                dst = morph_bgr[i + 1]
                src_caption = cap_list[i]
                dst_caption = cap_list[i + 1]

                if i == 0:
                    _write_hold(src, src_caption, hold_count)

                both_have_faces = face_detected[i] and face_detected[i + 1]
                frames_this: list[np.ndarray] = []

                if use_process_pool and process_pool is not None:
                    if chunked_parallel:
                        chunk_start = (i // chunk_size) * chunk_size
                        if chunk_start not in chunk_results:
                            t0 = perf_counter()
                            future = pending_chunks.pop(chunk_start)
                            chunk_results[chunk_start] = future.result()  # type: ignore[assignment]
                            t_morph_total += perf_counter() - t0

                            if next_chunk_start < n_trans:
                                start = next_chunk_start
                                end = min(n_trans, start + chunk_size)
                                tasks = [
                                    (
                                        idx,
                                        morph_bgr[idx],
                                        morph_bgr[idx + 1],
                                        all_landmarks[idx],
                                        all_landmarks[idx + 1],
                                        face_detected[idx] and face_detected[idx + 1],
                                    )
                                    for idx in range(start, end)
                                ]
                                pending_chunks[start] = process_pool.submit(
                                    _generate_transition_chunk_worker,
                                    tasks,
                                    alpha_schedule,
                                )
                                next_chunk_start = end

                        frames_this = chunk_results[chunk_start].pop(i)
                        if not chunk_results[chunk_start]:
                            del chunk_results[chunk_start]
                    else:
                        t0 = perf_counter()
                        future = pending_frames.pop(i)
                        frames_this = future.result()  # type: ignore[assignment]
                        t_morph_total += perf_counter() - t0
                        next_idx = i + (process_workers * 2)
                        if next_idx < n_trans:
                            pending_frames[next_idx] = process_pool.submit(
                                _generate_transition_keyframes_worker,
                                morph_bgr[next_idx],
                                morph_bgr[next_idx + 1],
                                all_landmarks[next_idx],
                                all_landmarks[next_idx + 1],
                                face_detected[next_idx] and face_detected[next_idx + 1],
                                alpha_schedule,
                            )
                else:
                    if both_have_faces:
                        t0 = perf_counter()
                        triangles = morpher.compute_triangulation(
                            all_landmarks[i],
                            all_landmarks[i + 1],
                            (morph_h, morph_w),
                        )
                        pts_all_src, pts_all_dst = warper._build_border_pts(
                            all_landmarks[i],
                            all_landmarks[i + 1],
                            morph_h,
                            morph_w,
                        )
                        src_f = src.astype(np.float32)
                        dst_f = dst.astype(np.float32)

                        def _make_morph_frame(fidx: int) -> np.ndarray:
                            alpha = alpha_schedule[fidx]
                            return warper.morph_frame_precomputed(
                                img_src_f=src_f,
                                img_dst_f=dst_f,
                                pts_all_src=pts_all_src,
                                pts_all_dst=pts_all_dst,
                                triangles=triangles,
                                alpha=alpha,
                            )

                        if keyframes_per_transition >= 14 and self.cpu_workers > 1:
                            with ThreadPoolExecutor(max_workers=self.cpu_workers) as ex:
                                frames_this = list(ex.map(_make_morph_frame, range(keyframes_per_transition)))
                        else:
                            frames_this = [_make_morph_frame(fidx) for fidx in range(keyframes_per_transition)]
                        t_morph_total += perf_counter() - t0
                    else:
                        t0 = perf_counter()
                        for fidx in range(keyframes_per_transition):
                            alpha = alpha_schedule[fidx]
                            frames_this.append(
                                cv2.addWeighted(src, 1.0 - alpha, dst, alpha, 0)
                            )
                        t_morph_total += perf_counter() - t0

                report(0.20 + 0.30 * ((i + 1) / n_trans), f"Morphing keyframes {i + 1}/{n_trans}...")

                if rife is not None:
                    t0 = perf_counter()
                    interp_frames = rife.interpolate_sequence(frames_this, multiplier=rife_mult)
                    t_rife_total += perf_counter() - t0
                else:
                    interp_frames = frames_this

                for fi, frame in enumerate(interp_frames):
                    alpha = fi / max(len(interp_frames) - 1, 1)
                    cap = src_caption if alpha < 0.5 else dst_caption
                    _process_and_write(frame, cap)

                _write_hold(dst, dst_caption, hold_count)

                report(0.55 + 0.35 * ((i + 1) / n_trans), f"Encoding transition {i + 1}/{n_trans}...")
                _free_memory()

        finally:
            if 'process_pool' in locals() and process_pool is not None:
                process_pool.shutdown(wait=True, cancel_futures=True)
            if rife is not None:
                del rife
            pipe.close()
            _free_memory()
        stage_times["morph_total"] = t_morph_total
        stage_times["rife_total"] = t_rife_total
        stage_times["encode_pipe"] = perf_counter() - t_encode

        if music_path:
            t_mux = perf_counter()
            report(0.93, "Adding background music...")
            renderer = VideoRenderer(cfg)
            renderer.mux_audio(
                video_path=output_path,
                audio_path=Path(music_path),
            )
            stage_times["mux_audio"] = perf_counter() - t_mux

        self.last_stage_timings = stage_times
        logger.info("Stage timings (s): %s", {k: round(v, 2) for k, v in stage_times.items()})

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
    def _style_ease(t: float, style: str) -> float:
        t = max(0.0, min(1.0, float(t)))
        s = style.strip().lower()
        if s == "emotional":
            # Smoother, more dramatic curve.
            return t * t * t * (t * (6 * t - 15) + 10)  # smootherstep
        if s == "fast":
            # Snappier and more direct.
            return t * t * (3 - 2 * t) if t < 0.5 else t
        # Balanced
        return GrowingUpPipeline._ease_in_out(t)

    @staticmethod
    def _alpha_schedule(
        count: int,
        style: str,
        pause_fraction: float,
        smoothing_window: int,
    ) -> list[float]:
        if count <= 1:
            return [0.0]
        n = max(2, int(count))
        pause = max(0.0, min(0.2, float(pause_fraction)))
        vals: list[float] = []
        for i in range(n):
            t = i / (n - 1)
            if pause > 0.0:
                if t <= pause:
                    remapped = 0.0
                elif t >= 1.0 - pause:
                    remapped = 1.0
                else:
                    remapped = (t - pause) / max(1e-6, (1.0 - 2.0 * pause))
            else:
                remapped = t
            vals.append(GrowingUpPipeline._style_ease(remapped, style))

        win = max(1, int(smoothing_window))
        if win <= 1:
            vals[0], vals[-1] = 0.0, 1.0
            return vals
        kernel = np.ones(win, dtype=np.float32) / float(win)
        pad = win // 2
        padded = np.pad(np.array(vals, dtype=np.float32), (pad, pad), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid").tolist()
        smoothed[0], smoothed[-1] = 0.0, 1.0
        return [float(max(0.0, min(1.0, v))) for v in smoothed]

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
