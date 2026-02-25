from __future__ import annotations

import unittest

from app.config import (
    compute_multi_image_profile,
    compute_transition_plan,
    transition_speed_to_frames,
)
from modules.video.ffmpeg_pipe import _video_encode_args


class TransitionSpeedMappingTests(unittest.TestCase):
    def test_slider_edges_order(self) -> None:
        self.assertGreater(
            transition_speed_to_frames(1),
            transition_speed_to_frames(10),
        )

    def test_slider_clamps_out_of_range_values(self) -> None:
        self.assertEqual(
            transition_speed_to_frames(0),
            transition_speed_to_frames(1),
        )
        self.assertEqual(
            transition_speed_to_frames(999),
            transition_speed_to_frames(10),
        )

    def test_speed_2_is_meaningfully_slower_than_speed_5(self) -> None:
        s2 = transition_speed_to_frames(2, fps_output=60, rife_multiplier=2)
        s5 = transition_speed_to_frames(5, fps_output=60, rife_multiplier=2)
        self.assertGreaterEqual(s2 - s5, 15)

    def test_monotonic_decreasing(self) -> None:
        vals = [transition_speed_to_frames(i, fps_output=60, rife_multiplier=2) for i in range(1, 11)]
        self.assertEqual(vals, sorted(vals, reverse=True))


class VideoEncodingArgsTests(unittest.TestCase):
    def test_libx264_uses_crf(self) -> None:
        args = _video_encode_args("libx264", "slow", 18)
        self.assertIn("-crf", args)
        self.assertNotIn("-cq", args)

    def test_nvenc_uses_cq(self) -> None:
        args = _video_encode_args("h264_nvenc", "p5", 18)
        self.assertIn("-cq", args)
        self.assertNotIn("-crf", args)


class TurboProfileTests(unittest.TestCase):
    def test_large_album_reduces_morph_resolution_in_turbo(self) -> None:
        p_small = compute_multi_image_profile(
            speed_level=5,
            num_images=10,
            fps_output=60,
            rife_multiplier=2,
            turbo_mode=True,
            output_width=1920,
            output_height=1080,
        )
        p_large = compute_multi_image_profile(
            speed_level=5,
            num_images=100,
            fps_output=60,
            rife_multiplier=2,
            turbo_mode=True,
            output_width=1920,
            output_height=1080,
        )
        self.assertLess(p_large["morph_h"], p_small["morph_h"])
        self.assertLessEqual(p_large["hold_frames"], p_small["hold_frames"])


class TransitionTimingModelTests(unittest.TestCase):
    def test_duration_controls_output_frames(self) -> None:
        p1 = compute_transition_plan(
            transition_style="Balanced",
            transition_duration_seconds=1.0,
            fps_output=60,
            rife_multiplier=2,
            num_images=12,
            turbo_mode=False,
            output_width=1920,
            output_height=1080,
        )
        p2 = compute_transition_plan(
            transition_style="Balanced",
            transition_duration_seconds=8.0,
            fps_output=60,
            rife_multiplier=2,
            num_images=12,
            turbo_mode=False,
            output_width=1920,
            output_height=1080,
        )
        self.assertLess(int(p1["transition_output_frames"]), int(p2["transition_output_frames"]))

    def test_styles_change_smoothing_and_motion(self) -> None:
        emo = compute_transition_plan(
            transition_style="Emotional",
            transition_duration_seconds=3.0,
            fps_output=60,
            rife_multiplier=2,
            num_images=12,
            turbo_mode=False,
            output_width=1920,
            output_height=1080,
        )
        fast = compute_transition_plan(
            transition_style="Fast",
            transition_duration_seconds=3.0,
            fps_output=60,
            rife_multiplier=2,
            num_images=12,
            turbo_mode=False,
            output_width=1920,
            output_height=1080,
        )
        self.assertGreater(int(emo["smoothing_window"]), int(fast["smoothing_window"]))
        self.assertLess(float(emo["camera_motion_scale"]), float(fast["camera_motion_scale"]))


if __name__ == "__main__":
    unittest.main()
