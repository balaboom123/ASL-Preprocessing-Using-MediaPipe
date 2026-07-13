"""Tests for FPSSampler (video.py)."""

from signdata.utils import video as video_utils
from signdata.utils.video import FPSSampler, get_video_duration


def test_get_video_duration_from_frame_metadata(monkeypatch):
    class FakeCapture:
        def isOpened(self):
            return True

        def get(self, prop):
            if prop == video_utils.cv2.CAP_PROP_FPS:
                return 25.0
            if prop == video_utils.cv2.CAP_PROP_FRAME_COUNT:
                return 50.0
            return 0.0

        def release(self):
            pass

    monkeypatch.setattr(video_utils.cv2, "VideoCapture", lambda _: FakeCapture())

    assert get_video_duration("video.mp4") == 2.0


class TestFPSSamplerFPSMode:
    def test_30_to_24_fps(self):
        """30→24 fps yields ~80% frames sampled."""
        sampler = FPSSampler(src_fps=30.0, sample_rate=24.0)

        total = 300
        taken = sum(1 for _ in range(total) if sampler.take())
        ratio = taken / total
        assert 0.75 <= ratio <= 0.85, f"Expected ~80%, got {ratio:.2%}"

    def test_target_gte_source_keeps_all(self):
        """target >= source keeps every frame."""
        sampler = FPSSampler(src_fps=24.0, sample_rate=30.0)

        total = 100
        taken = sum(1 for _ in range(total) if sampler.take())
        assert taken == total

    def test_exact_halving(self):
        """30→15 fps yields ~50% frames sampled."""
        sampler = FPSSampler(src_fps=30.0, sample_rate=15.0)
        total = 300
        taken = sum(1 for _ in range(total) if sampler.take())
        ratio = taken / total
        assert 0.45 <= ratio <= 0.55, f"Expected ~50%, got {ratio:.2%}"


class TestFPSSamplerRatioMode:
    def test_ratio_0_5_yields_every_other_on_average(self):
        """ratio=0.5 yields ~50% of frames."""
        sampler = FPSSampler(src_fps=30.0, sample_rate=0.5)

        total = 300
        taken = sum(1 for _ in range(total) if sampler.take())
        ratio = taken / total
        assert 0.45 <= ratio <= 0.55, f"Expected ~50%, got {ratio:.2%}"

    def test_ratio_0_75_yields_three_quarters(self):
        sampler = FPSSampler(src_fps=40.0, sample_rate=0.75)

        total = 400
        taken = sum(1 for _ in range(total) if sampler.take())
        ratio = taken / total
        assert 0.70 <= ratio <= 0.80, f"Expected ~75%, got {ratio:.2%}"


class TestFPSSamplerNativeRate:
    def test_none_keeps_all_frames(self):
        sampler = FPSSampler(src_fps=30.0, sample_rate=None)

        total = 100
        taken = sum(1 for _ in range(total) if sampler.take())
        assert taken == total

    def test_reset_restarts_accumulator(self):
        sampler = FPSSampler(src_fps=30.0, sample_rate=15.0)
        assert [sampler.take() for _ in range(3)] == [False, True, False]
        sampler.reset()
        assert [sampler.take() for _ in range(3)] == [False, True, False]
