from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Callable

from modules.video.ffmpeg_pipe import _video_encode_args
from utils.ffmpeg import resolve_ffmpeg_binary

log = logging.getLogger(__name__)


class VideoRenderer:
    """
    FFmpeg-based 1080p H.264 MP4 renderer.

    Reads a PNG frame sequence from frame_dir/%06d.png and encodes to MP4.
    Uses CRF 18 (near-lossless), preset slow, yuv420p, faststart for streaming.
    """

    def __init__(self, config):
        self.config = config

    def render(
        self,
        frame_dir: Path,
        output_path: Path,
        fps: int | None = None,
        progress_callback: Callable[[float], None] | None = None,
        total_frames: int | None = None,
    ) -> Path:
        fps = fps or self.config.video.fps_output
        vc = self.config.video

        ffmpeg_bin = resolve_ffmpeg_binary()
        cmd = [
            ffmpeg_bin, "-y",
            "-framerate", str(fps),
            "-i", str(frame_dir / "%06d.png"),
            *_video_encode_args(vc.codec, vc.preset, vc.crf),
            "-pix_fmt", vc.pixel_format,
            "-vf", f"scale={vc.output_width}:{vc.output_height}",
            "-movflags", "+faststart",
            str(output_path),
        ]

        process = subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

        _frame_re = re.compile(r"frame=\s*(\d+)")
        for line in process.stderr:  # type: ignore[union-attr]
            if progress_callback and total_frames:
                m = _frame_re.search(line)
                if m:
                    done = int(m.group(1))
                    progress_callback(min(done / total_frames, 1.0))

        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg exited with code {process.returncode}")

        return output_path

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _probe_duration(ffmpeg_bin: str, file_path: Path) -> float:
        """Return duration of a media file in seconds using ffmpeg -i.

        Uses ffmpeg itself (not ffprobe) so it works with the bundled
        imageio-ffmpeg binary that does not include ffprobe.
        """
        try:
            res = subprocess.run(
                [ffmpeg_bin, "-i", str(file_path)],
                capture_output=True, text=True, timeout=15,
            )
            # ffmpeg -i always exits non-zero when no output is given,
            # but prints duration in stderr like:
            #   Duration: 00:03:45.12, start: ...
            m = re.search(
                r"Duration:\s*(\d+):(\d+):(\d+)\.(\d+)",
                res.stderr or "",
            )
            if m:
                h, mi, s, frac = m.groups()
                return int(h) * 3600 + int(mi) * 60 + int(s) + float(f"0.{frac}")
        except Exception:
            pass
        return 0.0

    # ------------------------------------------------------------------
    # audio muxing
    # ------------------------------------------------------------------

    def mux_audio(
        self,
        video_path: Path,
        audio_paths: list[Path] | None = None,
        audio_path: Path | None = None,
        crossfade_sec: float = 3.0,
        fade_in_sec: float = 2.0,
        fade_out_sec: float = 3.0,
    ) -> Path:
        """
        Merge one or more audio tracks into the video with crossfade
        transitions between tracks and fade-in / fade-out at the edges.

        **Video length is always determined by the images — never by the
        music.**  If the combined music is longer than the video it is
        trimmed (with a nice fade-out).  If it is shorter the video
        continues silently.
        """
        ffmpeg_bin = resolve_ffmpeg_binary()

        # ---- collect valid audio files ----
        paths: list[Path] = []
        if audio_paths:
            paths = [p for p in audio_paths if p and p.exists()]
        if not paths and audio_path and audio_path.exists():
            paths = [audio_path]
        if not paths:
            log.warning("mux_audio: no valid audio files — skipping")
            return video_path

        # ---- probe video duration (this is the MASTER length) ----
        video_dur = self._probe_duration(ffmpeg_bin, video_path)
        if video_dur <= 0:
            log.warning("mux_audio: could not probe video duration — skipping")
            return video_path
        log.info("mux_audio: video duration = %.2fs, %d audio track(s)", video_dur, len(paths))

        # ---- build combined audio ----
        combined_audio = video_path.with_suffix(".combined_audio.m4a")

        if len(paths) == 1:
            # Single track — apply fade-in / fade-out, trim to video length
            dur = self._probe_duration(ffmpeg_bin, paths[0])
            effective_dur = min(dur, video_dur) if dur > 0 else video_dur
            fo_start = max(effective_dur - fade_out_sec, 0)

            af = (
                f"afade=t=in:d={fade_in_sec},"
                f"afade=t=out:st={fo_start:.2f}:d={fade_out_sec}"
            )
            cmd_audio = [
                ffmpeg_bin, "-y",
                "-i", str(paths[0]),
                "-af", af,
                "-t", f"{video_dur:.2f}",
                "-c:a", "aac", "-b:a", "192k",
                str(combined_audio),
            ]
        else:
            # Multiple tracks — acrossfade chain, then edge fades, trim
            durations = [self._probe_duration(ffmpeg_bin, p) for p in paths]
            log.info("mux_audio: track durations = %s", durations)

            # Clamp crossfade to half the shortest track
            pos_durs = [d for d in durations if d > 0]
            min_dur = min(pos_durs) if pos_durs else 999
            xfade = min(crossfade_sec, min_dur / 2)

            inputs: list[str] = []
            for p in paths:
                inputs += ["-i", str(p)]

            # Build filter_complex: chain acrossfade
            n = len(paths)
            filters: list[str] = []
            prev_label = "[0:a]"
            for i in range(1, n):
                out_label = f"[a{i}]" if i < n - 1 else "[mixed]"
                filters.append(
                    f"{prev_label}[{i}:a]acrossfade=d={xfade:.2f}"
                    f":c1=tri:c2=tri{out_label}"
                )
                prev_label = out_label

            # Compute total music duration after crossfades
            total_music = sum(durations) - (n - 1) * xfade
            effective_dur = min(total_music, video_dur)
            fo_start = max(effective_dur - fade_out_sec, 0)

            # Fade-in at start, fade-out before trim
            filters.append(
                f"[mixed]afade=t=in:d={fade_in_sec},"
                f"afade=t=out:st={fo_start:.2f}:d={fade_out_sec}[out]"
            )

            filter_complex = "; ".join(filters)
            cmd_audio = [
                ffmpeg_bin, "-y",
                *inputs,
                "-filter_complex", filter_complex,
                "-map", "[out]",
                "-t", f"{video_dur:.2f}",
                "-c:a", "aac", "-b:a", "192k",
                str(combined_audio),
            ]

        log.info("mux_audio: audio cmd = %s", " ".join(cmd_audio))
        res = subprocess.run(cmd_audio, capture_output=True, text=True)
        if res.returncode != 0:
            log.warning(
                "mux_audio: audio combine failed (rc=%d): %s",
                res.returncode, res.stderr[-600:] if res.stderr else "unknown",
            )
            # Fallback: copy first track raw (still works, just no fades)
            combined_audio = paths[0]

        # ---- mux combined audio into video ----
        tmp_out = video_path.with_suffix(".tmp.mp4")
        cmd = [
            ffmpeg_bin, "-y",
            "-i", str(video_path),
            "-i", str(combined_audio),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(tmp_out),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Clean up temp audio
        if combined_audio != paths[0] and combined_audio.exists():
            combined_audio.unlink(missing_ok=True)

        if result.returncode != 0:
            log.warning(
                "mux_audio: mux failed: %s",
                result.stderr[-500:] if result.stderr else "unknown",
            )
            tmp_out.unlink(missing_ok=True)
            return video_path

        # Replace original with muxed version
        tmp_out.replace(video_path)
        return video_path
