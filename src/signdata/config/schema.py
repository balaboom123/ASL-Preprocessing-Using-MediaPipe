"""Pydantic configuration models."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- Detection configs (one per backend, no `type` field) ---

class YOLODetectionConfig(StrictModel):
    model: str = "yolo11m.pt"
    device: str = "cpu"
    confidence_threshold: float = 0.5
    min_bbox_area: float = 0.05
    batch_size: int = Field(default=16, gt=0)
    allow_download: bool = True
    weights_dir: str | None = None


class MMDetDetectionConfig(StrictModel):
    det_model_config: str
    det_model_checkpoint: str
    device: str = "cuda:0"
    batch_size: int = Field(default=16, gt=0)


class MediaPipeDetectionConfig(StrictModel):
    min_detection_confidence: float = 0.5


# --- Pose configs (one per backend, no `type` field) ---

class MediaPipePoseConfig(StrictModel):
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    refine_face_landmarks: bool = True
    batch_size: int = 16


class MMPosePoseConfig(StrictModel):
    pose_model_config: str
    pose_model_checkpoint: str
    device: str = "cuda:0"
    batch_size: int = 16


# --- Config selector maps (used by model_validator) ---

DETECTION_CONFIG_MAP = {
    "yolo": YOLODetectionConfig,
    "mmdet": MMDetDetectionConfig,
    "mediapipe": MediaPipeDetectionConfig,
    "null": None,
}

POSE_CONFIG_MAP = {
    "mediapipe": MediaPipePoseConfig,
    "mmpose": MMPosePoseConfig,
}


# --- Video processing config ---

class VideoProcessingConfig(StrictModel):
    codec: str = "libx264"
    padding: float = 0.0
    resize: list[int] | None = None
    crf: int = Field(default=20, ge=0, le=51)
    preset: str = "medium"
    scene_cut_threshold: float = Field(default=0.35, gt=0.0, le=1.0)
    crop_window_seconds: float = Field(default=5.0, gt=0.0)
    min_track_seconds: float = Field(default=1.0, gt=0.0)
    min_track_hits: int = Field(default=2, gt=0)
    max_track_gap: int = Field(default=2, ge=0)
    track_iou_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    track_center_distance_ratio: float = Field(default=1.5, gt=0.0)
    max_crop_area_ratio: float = Field(default=0.8, gt=0.0, le=1.0)
    max_crop_shift_ratio: float = Field(default=0.25, gt=0.0, le=1.0)
    min_video_reduction_ratio: float = Field(default=0.10, ge=0.0, lt=1.0)
    pilot_segment_seconds: float = Field(default=10.0, gt=0.0)
    encoding_timeout_seconds: int = Field(default=3600, gt=0)


class PartsProcessingConfig(StrictModel):
    crop_size: int = Field(default=224, gt=0)
    bbox_scale: float = Field(default=1.2, gt=0)
    codec: str = "mp4v"
    missing_value: float = -1.0


# --- Stage configs ---

class DatasetConfig(StrictModel):
    name: str
    download: bool = True
    manifest: bool = True
    source: dict[str, Any] = Field(default_factory=dict)


class ProcessingConfig(StrictModel):
    enabled: bool = True
    processor: Literal[
        "video2pose",
        "video2crop",
        "video2compression",
        "video2parts",
    ] = "video2pose"
    detection: Literal["yolo", "mediapipe", "mmdet", "null"] = "null"
    pose: Literal["mediapipe", "mmpose"] | None = None

    # Shared sampling param: native / ratio / absolute FPS
    sample_rate: float | None = 0.5
    max_workers: int = Field(default=1, gt=0)

    # Backend-specific configs — raw dicts from YAML, validated into typed
    # models by model_validator and stored back as public attributes.
    detection_config: (
        YOLODetectionConfig
        | MMDetDetectionConfig
        | MediaPipeDetectionConfig
        | dict
        | None
    ) = None
    pose_config: MediaPipePoseConfig | MMPosePoseConfig | dict | None = None
    video_config: VideoProcessingConfig | None = None
    parts_config: PartsProcessingConfig | dict | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_removed_sampling_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            removed = {"frame_skip", "target_fps"}.intersection(data)
            if removed:
                raise ValueError(
                    f"Removed processing option(s): {', '.join(sorted(removed))}; "
                    "use processing.sample_rate"
                )
        return data

    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError("sample_rate must be positive or null")
        return v

    @model_validator(mode="after")
    def validate_backend_configs(self):
        """Parse raw dicts into typed config models."""
        # Skip validation when processing is disabled (allows default construction)
        if not self.enabled:
            return self

        # Validate + replace detection_config
        det_cls = DETECTION_CONFIG_MAP.get(self.detection)
        if det_cls and self.detection_config is not None:
            if isinstance(self.detection_config, dict):
                self.detection_config = det_cls(**self.detection_config)
        elif det_cls and self.detection_config is None:
            raise ValueError(
                f"detection={self.detection!r} requires detection_config"
            )

        # video2pose/video2parts require pose + pose_config
        if self.processor in ("video2pose", "video2parts"):
            if not self.pose:
                raise ValueError(f"processing.pose is required for {self.processor}")
            if self.processor == "video2parts" and self.pose != "mediapipe":
                raise ValueError("video2parts currently requires pose=mediapipe")
            pose_cls = POSE_CONFIG_MAP.get(self.pose)
            if pose_cls and self.pose_config is not None:
                if isinstance(self.pose_config, dict):
                    self.pose_config = pose_cls(**self.pose_config)
            elif pose_cls and self.pose_config is None:
                raise ValueError(
                    f"pose={self.pose!r} requires pose_config"
                )

        if self.processor == "video2parts":
            if isinstance(self.parts_config, dict):
                self.parts_config = PartsProcessingConfig(**self.parts_config)
            elif self.parts_config is None:
                self.parts_config = PartsProcessingConfig()

        # video2crop / video2compression require video_config (defaults if omitted)
        if self.processor in ("video2crop", "video2compression") and not self.video_config:
            self.video_config = VideoProcessingConfig()

        return self


class NormalizeConfig(StrictModel):
    mode: Literal["isotropic_3d", "xy_isotropic_z_minmax"] = "xy_isotropic_z_minmax"
    remove_z: bool = False
    select_keypoints: bool = True
    keypoint_preset: str | None = None
    keypoint_indices: list[int] | None = None
    mask_empty_frames: bool = True
    mask_low_confidence: bool = False
    visibility_threshold: float = 0.3
    missing_value: float = -999.0

    @field_validator("keypoint_preset")
    @classmethod
    def validate_keypoint_preset(cls, v: str | None) -> str | None:
        if v is not None:
            from signdata.processors.pose import KEYPOINT_PRESETS
            if v not in KEYPOINT_PRESETS:
                raise ValueError(
                    f"Unknown keypoint preset '{v}'. "
                    f"Available: {sorted(KEYPOINT_PRESETS.keys())}"
                )
        return v


class PostProcessingConfig(StrictModel):
    enabled: bool = False
    normalize: NormalizeConfig | None = None


class OutputConfig(StrictModel):
    enabled: bool = True
    max_shard_count: int = Field(default=10_000, gt=0)
    max_shard_size: int | None = Field(default=None, gt=0)


class PathsConfig(StrictModel):
    root: str = ""
    videos: str = ""
    transcripts: str = ""
    manifest: str = ""
    output: str = ""
    webdataset: str = ""


# --- Top-level ---

class Config(StrictModel):
    dataset: DatasetConfig
    processing: ProcessingConfig = Field(
        default_factory=lambda: ProcessingConfig(enabled=False)
    )
    post_processing: PostProcessingConfig = Field(
        default_factory=PostProcessingConfig
    )
    output: OutputConfig = Field(default_factory=OutputConfig)
    run_name: str = "default"
    paths: PathsConfig = Field(default_factory=PathsConfig)
