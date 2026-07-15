"""Tests for video utilities."""

from signdata.utils import video as video_utils
from signdata.utils.video import get_video_duration, resolve_effective_sample_fps


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


def test_resolve_effective_sample_fps():
    assert resolve_effective_sample_fps(30.0, None) is None
    assert resolve_effective_sample_fps(30.0, 0.5) == 15.0
    assert resolve_effective_sample_fps(30.0, 15.0) == 15.0
    assert resolve_effective_sample_fps(24.0, 30.0) == 24.0
