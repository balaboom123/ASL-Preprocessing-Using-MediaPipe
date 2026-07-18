"""Tests for the shot-aware Video2CompressionProcessor."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from signdata.config.schema import Config, VideoProcessingConfig
from signdata.datasets.how2sign import How2SignDataset
from signdata.pipeline.context import PipelineContext
from signdata.processors.detection.base import Detection
from signdata.processors.video import ffmpeg
from signdata.processors.video.ffmpeg import CropPlan, CropWindow
from signdata.processors.video2compression import (
    Video2CompressionProcessor,
    _ShotAccumulator,
    _ShotRegion,
    _build_crop_plan,
    _scene_change_score,
    _write_crop_metadata,
)
import signdata.processors  # noqa: F401 – trigger registrations
from signdata.registry import PROCESSOR_REGISTRY


def test_probe_video_releases_capture(monkeypatch):
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

    assert ffmpeg.probe_video("video.mp4") == (640, 480, 30.0)
    assert capture.released


def test_probe_frame_count_releases_capture(monkeypatch):
    class Capture:
        released = False

        def isOpened(self):
            return True

        def get(self, prop):
            assert prop == ffmpeg.cv2.CAP_PROP_FRAME_COUNT
            return 42

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(ffmpeg.cv2, "VideoCapture", lambda _: capture)

    assert ffmpeg.probe_frame_count("video.mp4") == 42
    assert capture.released


def _detection(bbox):
    return Detection(bbox=bbox, confidence=0.9)


class TestShotAwareBboxLogic:
    def test_scene_score_detects_large_change(self):
        black = np.zeros((32, 32, 3), dtype=np.uint8)
        white = np.full((32, 32, 3), 255, dtype=np.uint8)

        assert _scene_change_score(black, black) == 0.0
        assert _scene_change_score(black, white) == 1.0

    def test_isolated_false_detection_is_not_added_to_crop(self):
        config = VideoProcessingConfig(
            padding=0.0,
            min_track_hits=3,
        )
        shot = _ShotAccumulator(0, 100, 80, config)

        for frame_index in range(5):
            detections = [_detection((10, 10, 30, 50))]
            if frame_index == 2:
                detections.append(_detection((80, 5, 95, 30)))
            shot.add(frame_index, detections)

        region = shot.finish(5)

        assert region.bbox == (10, 10, 30, 50)

    def test_two_persistent_people_are_both_preserved(self):
        config = VideoProcessingConfig(
            padding=0.0,
            min_track_hits=3,
        )
        shot = _ShotAccumulator(0, 120, 80, config)

        for frame_index in range(4):
            shot.add(
                frame_index,
                [
                    _detection((10, 10, 35, 60)),
                    _detection((75, 10, 105, 60)),
                ],
            )

        region = shot.finish(4)

        assert region.bbox == (10, 10, 105, 60)

    def test_no_reliable_track_falls_back_to_full_frame(self):
        config = VideoProcessingConfig(min_track_hits=3)
        shot = _ShotAccumulator(0, 100, 80, config)
        shot.add(0, [])
        shot.add(1, [])

        assert shot.finish(2).bbox == (0, 0, 100, 80)

    def test_crop_origin_moves_only_between_shots(self):
        config = VideoProcessingConfig(
            max_crop_area_ratio=0.9,
            max_crop_shift_ratio=1.0,
        )
        plan = _build_crop_plan(
            [
                _ShotRegion(0, 10, (0, 10, 40, 50)),
                _ShotRegion(10, 20, (60, 10, 100, 50)),
            ],
            frame_width=100,
            frame_height=60,
            config=config,
        )

        assert plan.width == 40
        assert plan.height == 40
        assert plan.windows == (
            CropWindow(0, 10, 0, 10),
            CropWindow(10, 20, 60, 10),
        )

    def test_large_crop_uses_full_frame(self):
        config = VideoProcessingConfig(max_crop_area_ratio=0.9)
        plan = _build_crop_plan(
            [_ShotRegion(0, 10, (1, 1, 99, 59))],
            frame_width=100,
            frame_height=60,
            config=config,
        )

        assert (plan.width, plan.height) == (100, 60)
        assert plan.windows == (CropWindow(0, 10, 0, 0),)


def test_transcode_crop_plan_keeps_native_fps_and_uses_config(tmp_path):
    config = VideoProcessingConfig(crf=20, preset="medium")
    plan = CropPlan(
        width=80,
        height=80,
        windows=(
            CropWindow(0, 30, 0, 0),
            CropWindow(30, 60, 80, 40),
        ),
    )

    with patch(
        "signdata.processors.video.ffmpeg.subprocess.run",
        return_value=SimpleNamespace(returncode=0, stderr=b""),
    ) as run_mock:
        assert ffmpeg.transcode_with_crop_plan(
            "source.mp4",
            plan,
            config,
            str(tmp_path / "output.mp4"),
        )

    cmd = run_mock.call_args.args[0]
    assert cmd[cmd.index("-crf") + 1] == "20"
    assert cmd[cmd.index("-preset") + 1] == "medium"
    assert cmd[cmd.index("-fps_mode") + 1] == "passthrough"
    assert not any(part.startswith("fps=") for part in cmd)
    crop_filter = cmd[cmd.index("-vf") + 1]
    assert "gte(n\\,30)" in crop_filter


def test_crop_metadata_reports_post_resize_dimensions(tmp_path):
    config = VideoProcessingConfig(resize=[64, 48])
    plan = CropPlan(80, 60, (CropWindow(0, 10, 4, 6),))
    output_path = tmp_path / "output.mp4"

    _write_crop_metadata(
        output_path, "source.mp4", 160, 120, 30.0, plan, config
    )
    metadata = json.loads(
        output_path.with_suffix(".mp4.crop.json").read_text(
            encoding="utf-8"
        )
    )

    assert metadata["output_width"] == 64
    assert metadata["output_height"] == 48
    assert metadata["windows"][0]["width"] == 80
    assert metadata["windows"][0]["height"] == 60


class _FakeDetector:
    def __init__(self, frame_detections):
        self.frame_detections = frame_detections
        self.closed = False
        self.batch_sizes = []

    def detect_batch(self, frames):
        self.batch_sizes.append(len(frames))
        return self.frame_detections[: len(frames)]

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
                    "padding": 0.0,
                    "crf": 20,
                    "preset": "medium",
                    "scene_cut_threshold": 0.35,
                    "max_crop_shift_ratio": 1.0,
                    "min_track_hits": 2,
                },
            },
            paths={"root": str(tmp_path)},
        )

    def _make_context(self, tmp_path, df):
        videos_dir = tmp_path / "videos"
        videos_dir.mkdir(exist_ok=True)
        config = self._make_config(tmp_path)
        context = PipelineContext(
            config=config,
            dataset=How2SignDataset(),
            manifest_df=df,
            videos_dir=videos_dir,
            output_dir=tmp_path / "output" / "compression",
        )
        return config, context, videos_dir

    def test_deduplicates_and_resets_tracking_at_scene_cut(self, tmp_path):
        df = pd.DataFrame({
            "VIDEO_ID": ["vid_a", "vid_a"],
            "VIDEO_NAME": ["cam_1", "cam_1"],
            "SAMPLE_ID": ["seg_000", "seg_001"],
        })
        config, context, videos_dir = self._make_context(tmp_path, df)
        source = videos_dir / "cam_1.mp4"
        source.write_bytes(b"s" * 200)

        black = np.zeros((60, 100, 3), dtype=np.uint8)
        white = np.full((60, 100, 3), 255, dtype=np.uint8)
        frames = [black, black, white, white]
        detector = _FakeDetector([
            [_detection((5, 10, 30, 50))],
            [_detection((5, 10, 30, 50))],
            [
                _detection((50, 10, 70, 50)),
                _detection((75, 10, 95, 50)),
            ],
            [
                _detection((50, 10, 70, 50)),
                _detection((75, 10, 95, 50)),
            ],
        ])

        def write_smaller_output(_, crop_plan, __, output_path):
            assert crop_plan.windows[0].end_frame == 2
            assert crop_plan.windows[1].start_frame == 2
            Path(output_path).write_bytes(b"o" * 100)
            return True

        with patch(
            "signdata.processors.video2compression.create_detector",
            return_value=detector,
        ), patch(
            "signdata.processors.video2compression.get_video_duration",
            return_value=4 / 30,
        ), patch(
            "signdata.processors.video2compression.probe_video",
            side_effect=[(100, 60, 30.0), (46, 40, 30.0)],
        ), patch(
            "signdata.processors.video2compression.probe_frame_count",
            return_value=4,
        ), patch(
            "signdata.processors.video2compression.iter_ffmpeg_frame_batches",
            return_value=iter([frames]),
        ) as frame_iter_mock, patch(
            "signdata.processors.video2compression.transcode_with_crop_plan",
            side_effect=write_smaller_output,
        ):
            result = Video2CompressionProcessor(config).run(context)

        assert result.stats["processing"] == {
            "total": 1,
            "source_rows": 2,
            "processed": 1,
            "skipped": 0,
            "not_smaller": 0,
            "errors": 0,
        }
        assert detector.closed
        assert frame_iter_mock.call_args.args[3] is None

        output = (
            tmp_path
            / "output"
            / "compression"
            / "compressed"
            / "cam_1.mp4"
        )
        assert output.stat().st_size == 100
        metadata = json.loads(
            output.with_suffix(".mp4.crop.json").read_text(encoding="utf-8")
        )
        assert metadata["fps_changed"] is False
        assert metadata["source_fps"] == 30.0
        assert len(metadata["windows"]) == 2

    def test_larger_output_is_rejected(self, tmp_path):
        df = pd.DataFrame({
            "VIDEO_ID": ["vid_a"],
            "VIDEO_NAME": ["cam_1"],
            "SAMPLE_ID": ["seg_000"],
        })
        config, context, videos_dir = self._make_context(tmp_path, df)
        source = videos_dir / "cam_1.mp4"
        source.write_bytes(b"s" * 100)
        frame = np.zeros((60, 100, 3), dtype=np.uint8)
        detector = _FakeDetector([
            [_detection((10, 10, 40, 50))],
            [_detection((10, 10, 40, 50))],
        ])

        def write_larger_output(_, __, ___, output_path):
            Path(output_path).write_bytes(b"o" * 150)
            return True

        with patch(
            "signdata.processors.video2compression.create_detector",
            return_value=detector,
        ), patch(
            "signdata.processors.video2compression.get_video_duration",
            return_value=2 / 30,
        ), patch(
            "signdata.processors.video2compression.probe_video",
            side_effect=[(100, 60, 30.0), (30, 40, 30.0)],
        ), patch(
            "signdata.processors.video2compression.probe_frame_count",
            return_value=2,
        ), patch(
            "signdata.processors.video2compression.iter_ffmpeg_frame_batches",
            return_value=iter([[frame, frame]]),
        ), patch(
            "signdata.processors.video2compression.transcode_with_crop_plan",
            side_effect=write_larger_output,
        ):
            result = Video2CompressionProcessor(config).run(context)

        assert result.stats["processing"]["processed"] == 0
        assert result.stats["processing"]["skipped"] == 1
        assert result.stats["processing"]["not_smaller"] == 1
        output = (
            tmp_path
            / "output"
            / "compression"
            / "compressed"
            / "cam_1.mp4"
        )
        assert not output.exists()


class TestVideo2CompressionRegistration:
    def test_registered(self):
        assert "video2compression" in PROCESSOR_REGISTRY
