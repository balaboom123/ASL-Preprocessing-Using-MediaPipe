"""Tests for shared frame materialization helpers."""

import subprocess

from signdata.datasets._ingestion import media as ingestion_media


def test_get_video_fps_releases_failed_capture(monkeypatch):
    class Capture:
        released = False

        def isOpened(self):
            return False

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(ingestion_media.cv2, "VideoCapture", lambda _: capture)

    assert ingestion_media.get_video_fps("missing.mp4") == 0.0
    assert capture.released


class TestMaterializeFramesToVideo:
    def test_merges_multiple_patterns_into_one_ordered_frame_list(
        self, tmp_path, monkeypatch
    ):
        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()
        (frame_dir / "000001.jpg").touch()
        (frame_dir / "000002.JPG").touch()
        (frame_dir / "000003.jpg").touch()
        output_path = tmp_path / "clip.mp4"
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["input"] = kwargs["input"]
            output_path.touch()
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(ingestion_media.subprocess, "run", fake_run)

        result = ingestion_media.materialize_frames_to_video(
            frame_dir,
            output_path,
            fps=30.0,
            pattern=("*.jpg", "*.JPG"),
            overwrite=True,
        )

        assert result == output_path
        assert output_path.exists()
        concat_input = captured["input"]
        assert "000001.jpg" in concat_input
        assert "000002.JPG" in concat_input
        assert "000003.jpg" in concat_input
        assert concat_input.index("000001.jpg") < concat_input.index("000002.JPG")
        assert concat_input.index("000002.JPG") < concat_input.index("000003.jpg")
