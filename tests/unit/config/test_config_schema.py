"""Tests for Pydantic configuration models (schema.py)."""

import pytest
from pydantic import ValidationError

from signdata.config.schema import (
    Config,
    DatasetConfig,
    MMDetDetectionConfig,
    MMPosePoseConfig,
    MediaPipeDetectionConfig,
    MediaPipePoseConfig,
    NormalizeConfig,
    OutputConfig,
    PartsProcessingConfig,
    PathsConfig,
    PostProcessingConfig,
    ProcessingConfig,
    VideoProcessingConfig,
    YOLODetectionConfig,
)


class TestConfigMinimal:
    """Config builds with minimal fields."""

    def test_dataset_required(self):
        with pytest.raises(ValidationError):
            Config()

    def test_minimal_config(self):
        cfg = Config(dataset={"name": "youtube_asl"})
        assert cfg.dataset.name == "youtube_asl"

    def test_all_sub_configs_have_defaults(self):
        cfg = Config(dataset={"name": "test"})
        assert cfg.run_name == "default"
        assert isinstance(cfg.dataset, DatasetConfig)
        assert isinstance(cfg.processing, ProcessingConfig)
        assert isinstance(cfg.post_processing, PostProcessingConfig)
        assert isinstance(cfg.output, OutputConfig)
        assert isinstance(cfg.paths, PathsConfig)

    def test_dataset_config_from_dict(self):
        cfg = Config(dataset={"name": "test", "download": False})
        assert cfg.dataset.name == "test"
        assert cfg.dataset.download is False

    def test_dataset_config_from_model(self):
        ds = DatasetConfig(name="test", manifest=False)
        cfg = Config(dataset=ds)
        assert cfg.dataset.manifest is False


class TestDatasetConfig:
    def test_defaults(self):
        ds = DatasetConfig(name="test")
        assert ds.download is True
        assert ds.manifest is True
        assert ds.source == {}

    def test_with_source(self):
        ds = DatasetConfig(
            name="youtube_asl",
            source={"video_ids_file": "ids.txt"},
        )
        assert ds.source["video_ids_file"] == "ids.txt"


class TestProcessingConfig:
    """ProcessingConfig defaults and model_validator."""

    def test_defaults(self):
        # video2pose requires pose + pose_config, so use video2crop for defaults test
        p = ProcessingConfig(processor="video2crop", detection="yolo",
                             detection_config={"model": "yolov8n.pt"})
        assert p.enabled is True
        assert p.sample_rate == 0.5
        assert p.max_workers == 1

    def test_video2pose_requires_pose(self):
        with pytest.raises(ValidationError, match="pose is required"):
            ProcessingConfig(
                processor="video2pose",
                detection="null",
            )

    def test_video2pose_requires_pose_config(self):
        with pytest.raises(ValidationError, match="pose_config"):
            ProcessingConfig(
                processor="video2pose",
                detection="null",
                pose="mmpose",
                # missing pose_config
            )

    def test_video2pose_valid(self):
        p = ProcessingConfig(
            processor="video2pose",
            detection="null",
            pose="mediapipe",
            pose_config={"model_complexity": 1},
        )
        assert isinstance(p.pose_config, MediaPipePoseConfig)
        assert p.pose_config.model_complexity == 1

    def test_video2parts_valid(self):
        p = ProcessingConfig(
            processor="video2parts",
            detection="null",
            pose="mediapipe",
            pose_config={"model_complexity": 1},
        )
        assert isinstance(p.pose_config, MediaPipePoseConfig)
        assert isinstance(p.parts_config, PartsProcessingConfig)
        assert p.parts_config.crop_size == 224

    def test_video2parts_requires_mediapipe(self):
        with pytest.raises(ValidationError, match="requires pose=mediapipe"):
            ProcessingConfig(
                processor="video2parts",
                detection="null",
                pose="mmpose",
                pose_config={
                    "pose_model_config": "pose.py",
                    "pose_model_checkpoint": "pose.pth",
                },
            )

    def test_video2parts_rejects_unknown_parts_option(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ProcessingConfig(
                processor="video2parts",
                detection="null",
                pose="mediapipe",
                pose_config={"model_complexity": 1},
                parts_config={"future_option": True},
            )

    def test_detection_config_validated_yolo(self):
        p = ProcessingConfig(
            processor="video2crop",
            detection="yolo",
            detection_config={"model": "yolov8s.pt", "device": "cuda:0"},
        )
        assert isinstance(p.detection_config, YOLODetectionConfig)
        assert p.detection_config.model == "yolov8s.pt"

    def test_detection_config_validated_mmdet(self):
        p = ProcessingConfig(
            processor="video2pose",
            detection="mmdet",
            pose="mmpose",
            detection_config={
                "det_model_config": "config.py",
                "det_model_checkpoint": "ckpt.pth",
            },
            pose_config={
                "pose_model_config": "pose.py",
                "pose_model_checkpoint": "pose.pth",
            },
        )
        assert isinstance(p.detection_config, MMDetDetectionConfig)
        assert isinstance(p.pose_config, MMPosePoseConfig)

    def test_detection_null_no_config_needed(self):
        p = ProcessingConfig(
            processor="video2pose",
            detection="null",
            pose="mediapipe",
            pose_config={},
        )
        assert p.detection_config is None

    def test_detection_requires_config(self):
        """Non-null detection with missing detection_config raises."""
        with pytest.raises(ValidationError, match="detection_config"):
            ProcessingConfig(
                processor="video2crop",
                detection="yolo",
                # missing detection_config
            )

    def test_video2crop_creates_default_video_config(self):
        p = ProcessingConfig(
            processor="video2crop",
            detection="yolo",
            detection_config={"model": "yolov8n.pt"},
        )
        assert isinstance(p.video_config, VideoProcessingConfig)
        assert p.video_config.codec == "libx264"
        assert p.video_config.padding == 0.0
        assert p.video_config.crf == 20
        assert p.video_config.preset == "medium"
        assert p.video_config.aq_strength == 8
        assert p.video_config.max_bitrate_ratio == 0.8
        assert p.video_config.min_video_reduction_ratio == 0.1

    def test_video_config_rejects_invalid_compression_controls(self):
        with pytest.raises(ValidationError):
            VideoProcessingConfig(crf=64)
        with pytest.raises(ValidationError):
            VideoProcessingConfig(aq_strength=0)
        with pytest.raises(ValidationError):
            VideoProcessingConfig(max_bitrate_ratio=0.0)
        with pytest.raises(ValidationError):
            VideoProcessingConfig(min_video_reduction_ratio=1.0)

    def test_video_config_rejects_codec_preset_mismatch(self):
        """A silently-mapped preset is how the wrong encoder settings ship."""
        with pytest.raises(ValidationError, match="p1-p7"):
            VideoProcessingConfig(codec="hevc_nvenc", preset="medium")
        with pytest.raises(ValidationError, match="NVENC-only"):
            VideoProcessingConfig(codec="libx265", preset="p6")

        nvenc = VideoProcessingConfig(codec="hevc_nvenc", preset="p6")
        assert nvenc.preset == "p6"

    def test_invalid_processor_rejected(self):
        with pytest.raises(ValidationError):
            ProcessingConfig(processor="invalid")

    def test_invalid_detection_rejected(self):
        with pytest.raises(ValidationError):
            ProcessingConfig(detection="invalid")

    def test_removed_sampling_keys_are_rejected(self):
        for key in ("frame_skip", "target_fps"):
            with pytest.raises(ValidationError, match="use processing.sample_rate"):
                ProcessingConfig(**{key: 2})


class TestNormalizeConfig:
    def test_defaults(self):
        n = NormalizeConfig()
        assert n.mode == "xy_isotropic_z_minmax"
        assert n.remove_z is False
        assert n.select_keypoints is True
        assert n.mask_empty_frames is True
        assert n.mask_low_confidence is False
        assert n.visibility_threshold == 0.3
        assert n.missing_value == -999.0

    def test_keypoint_preset_none(self):
        n = NormalizeConfig()
        assert n.keypoint_preset is None

    def test_keypoint_preset_set(self):
        n = NormalizeConfig(keypoint_preset="mediapipe_553_to_85")
        assert n.keypoint_preset == "mediapipe_553_to_85"

    def test_keypoint_indices_none(self):
        n = NormalizeConfig()
        assert n.keypoint_indices is None

    def test_keypoint_indices_set(self):
        n = NormalizeConfig(keypoint_indices=[0, 1, 2])
        assert n.keypoint_indices == [0, 1, 2]

    def test_both_preset_and_indices_coexist(self):
        n = NormalizeConfig(
            keypoint_preset="mediapipe_553_to_85",
            keypoint_indices=[0, 1, 2],
        )
        assert n.keypoint_preset == "mediapipe_553_to_85"
        assert n.keypoint_indices == [0, 1, 2]


class TestPostProcessingConfig:
    def test_defaults(self):
        pp = PostProcessingConfig()
        assert pp.enabled is False
        assert pp.normalize is None

    def test_with_normalize(self):
        pp = PostProcessingConfig(
            enabled=True,
            normalize={"mode": "isotropic_3d"},
        )
        assert pp.enabled is True
        assert pp.normalize.mode == "isotropic_3d"


class TestOutputConfig:
    def test_defaults(self):
        o = OutputConfig()
        assert o.enabled is True
        assert o.max_shard_count == 10_000
        assert o.max_shard_size is None

    def test_with_limits(self):
        o = OutputConfig(max_shard_count=5000, max_shard_size=1_000_000)
        assert o.max_shard_count == 5000
        assert o.max_shard_size == 1_000_000


class TestPathsConfig:
    def test_defaults(self):
        p = PathsConfig()
        assert p.root == ""
        assert p.videos == ""
        assert p.transcripts == ""
        assert p.manifest == ""
        assert p.output == ""
        assert p.webdataset == ""


class TestTypedDetectionConfigs:
    def test_yolo_defaults(self):
        c = YOLODetectionConfig()
        assert c.model == "yolo11m.pt"
        assert c.device == "cpu"
        assert c.confidence_threshold == 0.5

    def test_mmdet_required_fields(self):
        c = MMDetDetectionConfig(
            det_model_config="config.py",
            det_model_checkpoint="ckpt.pth",
        )
        assert c.device == "cuda:0"

    def test_mediapipe_defaults(self):
        c = MediaPipeDetectionConfig()
        assert c.min_detection_confidence == 0.5


class TestTypedPoseConfigs:
    def test_mediapipe_defaults(self):
        c = MediaPipePoseConfig()
        assert c.model_complexity == 1
        assert c.refine_face_landmarks is True
        assert c.batch_size == 16

    def test_mmpose_required_fields(self):
        c = MMPosePoseConfig(
            pose_model_config="pose.py",
            pose_model_checkpoint="pose.pth",
        )
        assert c.device == "cuda:0"
        assert c.batch_size == 16


class TestTypeCoercion:
    def test_int_field(self):
        p = ProcessingConfig(
            processor="video2crop",
            detection="yolo",
            detection_config={"model": "yolov8n.pt"},
            max_workers=8,
        )
        assert p.max_workers == 8
        assert isinstance(p.max_workers, int)

        with pytest.raises(ValidationError):
            ProcessingConfig(
                processor="video2crop",
                detection="yolo",
                detection_config={"model": "yolov8n.pt"},
                max_workers=0,
            )

    def test_bool_field(self):
        n = NormalizeConfig(select_keypoints=False)
        assert n.select_keypoints is False
