"""Pydantic configuration models."""

import warnings
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# --- Detection configs (one per backend, no `type` field) ---

class YOLODetectionConfig(BaseModel):
    model: str = "yolo11m.pt"
    device: str = "cpu"
    confidence_threshold: float = 0.5
    min_bbox_area: float = 0.05
    batch_size: int = Field(default=16, gt=0)
    allow_download: bool = True
    weights_dir: Optional[str] = None


class MMDetDetectionConfig(BaseModel):
    det_model_config: str
    det_model_checkpoint: str
    device: str = "cuda:0"
    batch_size: int = Field(default=16, gt=0)


class MediaPipeDetectionConfig(BaseModel):
    min_detection_confidence: float = 0.5


# --- Pose configs (one per backend, no `type` field) ---

class MediaPipePoseConfig(BaseModel):
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    refine_face_landmarks: bool = True
    batch_size: int = 16


class MMPosePoseConfig(BaseModel):
    pose_model_config: str
    pose_model_checkpoint: str
    device: str = "cuda:0"
    bbox_threshold: float = 0.5
    keypoint_threshold: float = 0.3
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

class VideoProcessingConfig(BaseModel):
    codec: str = "libx264"
    padding: float = 0.0
    resize: Optional[List[int]] = None


class PartsProcessingConfig(BaseModel):
    crop_size: int = Field(default=224, gt=0)
    bbox_scale: float = Field(default=1.2, gt=0)
    codec: str = "mp4v"
    missing_value: float = -1.0
    store_crops: Literal["mp4"] = "mp4"
    store_features: bool = False

    @model_validator(mode="after")
    def validate_supported_modes(self):
        if self.store_features:
            raise ValueError(
                "parts_config.store_features is reserved for a future feature-cache stage"
            )
        return self


# --- Stage configs ---

class DatasetConfig(BaseModel):
    name: str
    download: bool = True
    manifest: bool = True
    source: Dict[str, Any] = {}


class ProcessingConfig(BaseModel):
    enabled: bool = True
    processor: Literal[
        "video2pose",
        "video2crop",
        "video2compression",
        "video2parts",
    ] = "video2pose"
    detection: Literal["yolo", "mediapipe", "mmdet", "null"] = "null"
    pose: Optional[Literal["mediapipe", "mmpose"]] = None

    # Shared sampling param: native / ratio / absolute FPS
    sample_rate: Optional[float] = 0.5
    max_workers: int = 1

    # Backend-specific configs — raw dicts from YAML, validated into typed
    # models by model_validator and stored back as public attributes.
    detection_config: Optional[Union[
        YOLODetectionConfig, MMDetDetectionConfig,
        MediaPipeDetectionConfig, dict,
    ]] = None
    pose_config: Optional[Union[
        MediaPipePoseConfig, MMPosePoseConfig, dict,
    ]] = None
    video_config: Optional[VideoProcessingConfig] = None
    parts_config: Optional[Union[PartsProcessingConfig, dict]] = None

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_sampling_keys(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        has_sample_rate = "sample_rate" in data
        has_frame_skip = "frame_skip" in data
        has_target_fps = "target_fps" in data

        if not (has_frame_skip or has_target_fps):
            return data

        migrated = dict(data)
        processor = migrated.get("processor", "video2pose")

        if has_sample_rate:
            warnings.warn(
                "processing.frame_skip and processing.target_fps are deprecated "
                "and ignored when processing.sample_rate is set.",
                FutureWarning,
                stacklevel=2,
            )
            migrated.pop("frame_skip", None)
            migrated.pop("target_fps", None)
            return migrated

        frame_skip = migrated.pop("frame_skip", None)
        target_fps = migrated.pop("target_fps", None)

        warning_suffix = (
            "Use processing.sample_rate instead: null keeps native FPS, "
            "0<sample_rate<1 keeps that fraction of frames, and "
            "sample_rate>=1 downsamples to that FPS."
        )

        if frame_skip is not None:
            frame_skip = int(frame_skip)
            if frame_skip <= 0:
                raise ValueError("processing.frame_skip must be a positive integer")

        if target_fps is not None and target_fps <= 0:
            raise ValueError(
                "processing.target_fps (deprecated) must be positive or null — "
                "use processing.sample_rate instead"
            )

        if target_fps is not None:
            migrated["sample_rate"] = float(target_fps)
            if frame_skip is not None:
                if processor == "video2crop":
                    warnings.warn(
                        "processing.frame_skip and processing.target_fps are "
                        "deprecated. They were mapped to "
                        f"processing.sample_rate={float(target_fps)!r}. "
                        "This preserves the old target FPS but no longer keeps a "
                        "separate detection-only stride for video2crop. "
                        + warning_suffix,
                        FutureWarning,
                        stacklevel=2,
                    )
                else:
                    warnings.warn(
                        "processing.frame_skip and processing.target_fps are "
                        "deprecated. They were mapped to "
                        f"processing.sample_rate={float(target_fps)!r}. "
                        "This matches the old video2pose behavior where "
                        "target_fps took precedence over frame_skip. "
                        + warning_suffix,
                        FutureWarning,
                        stacklevel=2,
                    )
            else:
                warnings.warn(
                    "processing.target_fps is deprecated and was mapped to "
                    f"processing.sample_rate={float(target_fps)!r}. "
                    + warning_suffix,
                    FutureWarning,
                    stacklevel=2,
                )
            return migrated

        sample_rate = None if frame_skip == 1 else (1.0 / frame_skip)
        migrated["sample_rate"] = sample_rate
        if processor == "video2crop":
            warnings.warn(
                "processing.frame_skip is deprecated and was mapped to "
                f"processing.sample_rate={sample_rate!r}. "
                "This preserves the approximate keep ratio, but video2crop no "
                "longer applies frame skipping only to detection. "
                + warning_suffix,
                FutureWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                "processing.frame_skip is deprecated and was mapped to "
                f"processing.sample_rate={sample_rate!r}. "
                + warning_suffix,
                FutureWarning,
                stacklevel=2,
            )
        return migrated

    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, v: Optional[float]) -> Optional[float]:
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


class NormalizeConfig(BaseModel):
    mode: Literal["isotropic_3d", "xy_isotropic_z_minmax"] = "xy_isotropic_z_minmax"
    remove_z: bool = False
    select_keypoints: bool = True
    keypoint_preset: Optional[str] = None
    keypoint_indices: Optional[List[int]] = None
    mask_empty_frames: bool = True
    mask_low_confidence: bool = False
    visibility_threshold: float = 0.3
    missing_value: float = -999.0

    @field_validator("keypoint_preset")
    @classmethod
    def validate_keypoint_preset(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            from signdata.processors.pose import KEYPOINT_PRESETS
            if v not in KEYPOINT_PRESETS:
                raise ValueError(
                    f"Unknown keypoint preset '{v}'. "
                    f"Available: {sorted(KEYPOINT_PRESETS.keys())}"
                )
        return v


class PostProcessingConfig(BaseModel):
    enabled: bool = True
    recipes: List[str] = []
    normalize: Optional[NormalizeConfig] = None


class OutputConfig(BaseModel):
    enabled: bool = True
    type: Literal["webdataset"] = "webdataset"
    config: Dict[str, Any] = {}


class PathsConfig(BaseModel):
    root: str = ""
    videos: str = ""
    transcripts: str = ""
    manifest: str = ""
    output: str = ""
    webdataset: str = ""


# --- Top-level ---

class Config(BaseModel):
    dataset: DatasetConfig
    processing: ProcessingConfig = ProcessingConfig(enabled=False)
    post_processing: PostProcessingConfig = PostProcessingConfig()
    output: OutputConfig = OutputConfig()
    run_name: str = "default"
    paths: PathsConfig = PathsConfig()
