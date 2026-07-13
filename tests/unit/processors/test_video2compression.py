"""Tests for Video2CompressionProcessor.

Covers:
  - segment-level manifest deduplication to one task per source video
  - whole-video processing semantics (0.0 -> duration)
  - video-level output naming
  - registration
"""

from unittest.mock import patch

import numpy as np
import pandas as pd

from signdata.config.schema import Config
from signdata.datasets.how2sign import How2SignDataset
from signdata.pipeline.context import PipelineContext
from signdata.processors.detection.base import Detection
from signdata.processors.video import ffmpeg
from signdata.processors.video2compression import Video2CompressionProcessor
import signdata.processors  # noqa: F401 – trigger registrations
from signdata.registry import PROCESSOR_REGISTRY


def test_video_metadata_releases_capture(monkeypatch):
    class Capture:
        released = False

        def isOpened(self):
            return True

        def get(self, prop):
            return {
                ffmpeg.cv2.CAP_PROP_FRAME_WIDTH: 640,
                ffmpeg.cv2.CAP_PROP_FRAME_HEIGHT: 480,
                ffmpeg.cv2.CAP_PROP_FPS: 30,
            }[prop]

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(ffmpeg.cv2, "VideoCapture", lambda _: capture)

    assert ffmpeg._video_metadata("video.mp4") == (640, 480, 30.0)
    assert capture.released


class _FakeDetector:
    def __init__(self):
        self.closed = False
        self.batch_sizes = []

    def detect_batch(self, frames):
        self.batch_sizes.append(len(frames))
        return [
            [Detection(bbox=(1.0, 2.0, 20.0, 30.0), confidence=0.9)]
            for _ in frames
        ]

    def close(self):
        self.closed = True


class TestVideo2CompressionProcessor:
    def _make_config(self, tmp_path):
        return Config(
            dataset={"name": "how2sign"},
            processing={
                "enabled": True,
                "processor": "video2compression",
                "detection": "yolo",
                "detection_config": {
                    "model": "yolov8n.pt",
                    "device": "cpu",
                },
                "video_config": {
                    "codec": "libx264",
                    "padding": 0.2,
                },
            },
            paths={"root": str(tmp_path)},
        )

    def test_deduplicates_segment_rows_and_uses_video_level_output(self, tmp_path):
        videos_dir = tmp_path / "videos"
        videos_dir.mkdir()
        (videos_dir / "cam_1.mp4").touch()

        df = pd.DataFrame({
            "VIDEO_ID": ["vid_a", "vid_a"],
            "VIDEO_NAME": ["cam_1", "cam_1"],
            "SAMPLE_ID": ["seg_000", "seg_001"],
            "START": [0.0, 1.0],
            "END": [0.5, 1.5],
        })

        cfg = self._make_config(tmp_path)
        processor = Video2CompressionProcessor(cfg)
        ctx = PipelineContext(
            config=cfg,
            dataset=How2SignDataset(),
            manifest_df=df,
            videos_dir=videos_dir,
            output_dir=tmp_path / "output" / "compression",
        )

        fake_detector = _FakeDetector()
        frame = np.zeros((8, 8, 3), dtype=np.uint8)

        with patch(
            "signdata.processors.video2compression.create_detector",
            return_value=fake_detector,
        ), patch(
            "signdata.processors.video2compression.get_video_duration",
            return_value=12.5,
        ), patch(
            "signdata.processors.video2compression.iter_ffmpeg_frame_batches",
            return_value=iter([[frame], [frame]]),
        ) as frame_iter_mock, patch(
            "signdata.processors.video2compression.clip_and_crop",
            return_value=True,
        ) as crop_mock:
            result = processor.run(ctx)

        assert result.stats["processing"]["source_rows"] == 2
        assert result.stats["processing"]["total"] == 1
        assert result.stats["processing"]["processed"] == 1
        assert result.stats["processing"]["skipped"] == 0
        assert result.stats["processing"]["errors"] == 0
        assert fake_detector.closed
        assert fake_detector.batch_sizes == [1, 1]

        frame_iter_args = frame_iter_mock.call_args.args
        assert frame_iter_args[0] == str(videos_dir / "cam_1.mp4")
        assert frame_iter_args[1] == 0.0
        assert frame_iter_args[2] == 12.5
        assert frame_iter_args[3] is None

        crop_args = crop_mock.call_args.args
        assert crop_args[0] == str(videos_dir / "cam_1.mp4")
        assert crop_args[1] == 0.0
        assert crop_args[2] == 12.5
        assert crop_args[4] is None
        assert crop_args[6].endswith("compressed/cam_1.mp4")

    def test_rel_path_output_preserves_relative_structure(self, tmp_path):
        videos_dir = tmp_path / "videos"
        input_dir = videos_dir / "split_a" / "session_1"
        input_dir.mkdir(parents=True)
        (input_dir / "clip_source.mp4").touch()

        df = pd.DataFrame({
            "VIDEO_ID": ["vid_a"],
            "SAMPLE_ID": ["seg_000"],
            "REL_PATH": ["split_a/session_1/clip_source.mp4"],
        })

        cfg = self._make_config(tmp_path)
        processor = Video2CompressionProcessor(cfg)
        ctx = PipelineContext(
            config=cfg,
            dataset=How2SignDataset(),
            manifest_df=df,
            videos_dir=videos_dir,
            output_dir=tmp_path / "output" / "compression",
        )

        fake_detector = _FakeDetector()
        frame = np.zeros((8, 8, 3), dtype=np.uint8)

        with patch(
            "signdata.processors.video2compression.create_detector",
            return_value=fake_detector,
        ), patch(
            "signdata.processors.video2compression.get_video_duration",
            return_value=5.0,
        ), patch(
            "signdata.processors.video2compression.iter_ffmpeg_frame_batches",
            return_value=iter([[frame]]),
        ), patch(
            "signdata.processors.video2compression.clip_and_crop",
            return_value=True,
        ) as crop_mock:
            processor.run(ctx)

        assert crop_mock.call_args.args[6].endswith(
            "compressed/split_a/session_1/clip_source.mp4"
        )

    def test_streaming_batches_are_folded_into_one_union_bbox(self, tmp_path):
        videos_dir = tmp_path / "videos"
        videos_dir.mkdir()
        (videos_dir / "cam_1.mp4").touch()

        df = pd.DataFrame({
            "VIDEO_ID": ["vid_a"],
            "VIDEO_NAME": ["cam_1"],
            "SAMPLE_ID": ["seg_000"],
        })

        cfg = self._make_config(tmp_path)
        processor = Video2CompressionProcessor(cfg)
        ctx = PipelineContext(
            config=cfg,
            dataset=How2SignDataset(),
            manifest_df=df,
            videos_dir=videos_dir,
            output_dir=tmp_path / "output" / "compression",
        )

        class BatchSensitiveDetector(_FakeDetector):
            def detect_batch(self, frames):
                self.batch_sizes.append(len(frames))
                if len(self.batch_sizes) == 1:
                    return [[Detection(bbox=(10.0, 20.0, 30.0, 40.0), confidence=0.9)]]
                return [[Detection(bbox=(1.0, 2.0, 50.0, 60.0), confidence=0.9)]]

        fake_detector = BatchSensitiveDetector()
        frame = np.zeros((8, 8, 3), dtype=np.uint8)

        with patch(
            "signdata.processors.video2compression.create_detector",
            return_value=fake_detector,
        ), patch(
            "signdata.processors.video2compression.get_video_duration",
            return_value=5.0,
        ), patch(
            "signdata.processors.video2compression.iter_ffmpeg_frame_batches",
            return_value=iter([[frame], [frame]]),
        ), patch(
            "signdata.processors.video2compression.clip_and_crop",
            return_value=True,
        ) as crop_mock:
            processor.run(ctx)

        assert fake_detector.batch_sizes == [1, 1]
        assert crop_mock.call_args.args[3] == (1.0, 2.0, 50.0, 60.0)


class TestVideo2CompressionRegistration:
    def test_registered(self):
        assert "video2compression" in PROCESSOR_REGISTRY
