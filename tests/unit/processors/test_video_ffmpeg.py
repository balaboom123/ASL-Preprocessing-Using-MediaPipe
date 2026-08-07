"""Tests for shared video decoding helpers."""

import numpy as np

from signdata.processors.video import ffmpeg


class _FakeCapture:
    def __init__(self, frames, fps=10.0):
        self.frames = frames
        self.fps = fps
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        assert prop == ffmpeg.cv2.CAP_PROP_FPS
        return self.fps

    def set(self, prop, value):
        assert prop == ffmpeg.cv2.CAP_PROP_POS_FRAMES
        self.index = int(value)
        return True

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def release(self):
        self.released = True


def test_ffmpeg_pipe_frames_uses_opencv_when_binary_is_missing(monkeypatch):
    frames = [
        np.full((2, 3, 3), value, dtype=np.uint8)
        for value in range(6)
    ]
    capture = _FakeCapture(frames)
    monkeypatch.setattr(ffmpeg.shutil, "which", lambda _name: None)
    monkeypatch.setattr(ffmpeg.cv2, "VideoCapture", lambda _path: capture)

    def unexpected_popen(*_args, **_kwargs):
        raise AssertionError("FFmpeg must not be launched")

    monkeypatch.setattr(ffmpeg.subprocess, "Popen", unexpected_popen)

    decoded = ffmpeg.ffmpeg_pipe_frames("video.mp4", 0.0, 0.5, 0.5)

    # Same frames ffmpeg's `fps=5` filter would emit for [0.0, 0.5) at 10 fps:
    # the window starts at frame 0, and frame 5 is outside it.
    assert [int(frame[0, 0, 0]) for frame in decoded] == [0, 2, 4]
    assert capture.released
