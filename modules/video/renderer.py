from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Callable


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

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(frame_dir / "%06d.png"),
            "-c:v", vc.codec,
            "-preset", vc.preset,
            "-crf", str(vc.crf),
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

    def mux_audio(
        self,
        video_path: Path,
        audio_path: Path,
    ) -> Path:
        """
        Merge an audio track into the video, trimmed to video length.
        Overwrites the original video file.
        """
        tmp_out = video_path.with_suffix(".tmp.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(tmp_out),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            import logging
            logging.getLogger(__name__).warning(
                "Audio mux failed: %s", result.stderr[-500:] if result.stderr else "unknown"
            )
            tmp_out.unlink(missing_ok=True)
            return video_path

        # Replace original with muxed version
        tmp_out.replace(video_path)
        return video_path
