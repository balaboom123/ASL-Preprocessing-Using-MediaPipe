"""Pose extractors, factory helpers, and keypoint presets."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np


MEDIAPIPE_543_TO_83 = {
    "description": (
        "MediaPipe unrefined: 6 pose + 35 face + 21 left hand + 21 right hand"
    ),
    "source_keypoints": 543,
    "target_keypoints": 83,
    "indices": (
        [11, 12, 13, 14, 23, 24]
        + [
            33, 37, 46, 47, 50, 66, 70, 72, 79, 85, 88, 94, 97,
            114, 115, 126, 166, 184, 185, 192, 205, 211, 214,
            296, 302, 309, 315, 318, 324, 327, 344, 356,
            395, 419, 430,
        ]
        + list(range(501, 522))
        + list(range(522, 543))
    ),
}

MEDIAPIPE_553_TO_85 = {
    "description": (
        "MediaPipe refined: 6 pose + 37 face + 21 left hand + 21 right hand"
    ),
    "source_keypoints": 553,
    "target_keypoints": 85,
    "indices": (
        [11, 12, 13, 14, 23, 24]
        + [
            33, 37, 46, 47, 50, 66, 70, 72, 79, 85, 88, 94, 97,
            114, 115, 126, 166, 184, 185, 192, 205, 211, 214,
            296, 302, 309, 315, 318, 324, 327, 344, 356,
            395, 419, 430, 501, 506,
        ]
        + list(range(511, 532))
        + list(range(532, 553))
    ),
}

MMPOSE_133_TO_85 = {
    "description": (
        "COCO WholeBody 133 to 85 selected upper body, face, and hand keypoints"
    ),
    "source_keypoints": 133,
    "target_keypoints": 85,
    "indices": (
        [5, 6, 7, 8, 11, 12]
        + [23, 25, 27, 29, 31, 33, 35, 37, 39]
        + [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
        + [52, 54, 56, 58]
        + [71, 73, 75, 77, 79, 81, 83, 84, 85, 86, 87, 88, 89, 90]
        + list(range(91, 133))
    ),
}

KEYPOINT_PRESETS: Dict[str, dict] = {
    "mediapipe_553_to_85": MEDIAPIPE_553_TO_85,
    "mediapipe_543_to_83": MEDIAPIPE_543_TO_83,
    "mmpose_133_to_85": MMPOSE_133_TO_85,
}


def resolve_keypoint_indices(
    preset_name: Optional[str] = None,
    manual_indices: Optional[List[int]] = None,
) -> Optional[List[int]]:
    """Resolve keypoint indices from a preset name or manual list."""
    if preset_name is not None:
        if preset_name not in KEYPOINT_PRESETS:
            raise ValueError(
                f"Unknown keypoint preset '{preset_name}'. "
                f"Available: {sorted(KEYPOINT_PRESETS.keys())}"
            )
        return list(KEYPOINT_PRESETS[preset_name]["indices"])

    if manual_indices is not None:
        return list(manual_indices)

    return None


def list_presets() -> Dict[str, str]:
    """Return a mapping of preset names to their descriptions."""
    return {
        name: info["description"]
        for name, info in KEYPOINT_PRESETS.items()
    }


class LandmarkExtractor(ABC):
    """Abstract base class for landmark extraction."""

    # Whether the extractor supports true batch inference (GPU batching)
    supports_batch_inference: bool = False

    @abstractmethod
    def process_frame(
        self, frame: np.ndarray, bbox: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Process a single frame and extract landmarks.

        Args:
            frame: Input video frame (BGR format from OpenCV).
            bbox: Optional bounding box array of shape (N, 4) with
                [x1, y1, x2, y2] in pixel coordinates, provided by
                an upstream detector. Backends that perform their own
                detection (e.g. MediaPipe Holistic) may ignore this.

        Returns:
            Numpy array of landmarks, or None if extraction fails.
        """
        pass

    def process_batch(
        self,
        frames: List[np.ndarray],
        bboxes: Optional[List[Optional[np.ndarray]]] = None,
        fallback_on_error: bool = True,
    ) -> List[Optional[np.ndarray]]:
        """Process a batch of frames and extract landmarks.

        Default implementation processes frames sequentially.
        Subclasses may override for true batch inference.

        Args:
            frames: List of input video frames (BGR format from OpenCV).
            bboxes: Optional per-frame bounding boxes from upstream
                detector. Each element is an array of shape (N, 4) or
                None. Backends that perform their own detection may
                ignore this.
            fallback_on_error: If True, continue processing remaining frames
                when one fails. If False, raise the exception.

        Returns:
            List of numpy arrays of landmarks (or None for failed frames).
        """
        results = []
        for i, frame in enumerate(frames):
            try:
                bbox = bboxes[i] if bboxes else None
                landmarks = self.process_frame(frame, bbox=bbox)
                results.append(landmarks)
            except Exception:
                if fallback_on_error:
                    results.append(None)
                else:
                    raise
        return results

    @abstractmethod
    def close(self):
        """Release resources (models, GPU memory, etc.)."""
        pass

    # Number of landmarks this extractor produces. Set in subclass __init__.
    num_landmarks: int = 0


def create_estimator(pose_type: str, pose_config) -> LandmarkExtractor:
    """Factory: create a pose estimator from the pose type and config.

    Args:
        pose_type: One of "mediapipe", "mmpose".
        pose_config: Typed config model (MediaPipePoseConfig or MMPosePoseConfig).

    Returns:
        A LandmarkExtractor instance.
    """
    if pose_type == "mediapipe":
        from .mediapipe import MediaPipeExtractor

        from ...config.schema import MediaPipePoseConfig

        cfg = (
            pose_config
            if isinstance(pose_config, MediaPipePoseConfig)
            else MediaPipePoseConfig(**pose_config)
        )

        class _Cfg:
            name = "mediapipe"
            model_complexity = cfg.model_complexity
            min_detection_confidence = cfg.min_detection_confidence
            min_tracking_confidence = cfg.min_tracking_confidence
            refine_face_landmarks = cfg.refine_face_landmarks
            batch_size = cfg.batch_size

        return MediaPipeExtractor(_Cfg())

    elif pose_type == "mmpose":
        from .mmpose import MMPoseExtractor
        from mmpose.apis import init_model

        pose_estimator = init_model(
            pose_config.pose_model_config,
            pose_config.pose_model_checkpoint,
            device=pose_config.device,
        )
        try:
            if pose_estimator.cfg.model.get("type") == "TopdownPoseEstimator3D":
                pose_estimator.cfg.model.test_cfg.mode = "3d"
        except AttributeError:
            pass

        class _Cfg:
            name = "mmpose"
            bbox_threshold = pose_config.bbox_threshold
            keypoint_threshold = pose_config.keypoint_threshold
            add_visible = True
            batch_size = pose_config.batch_size

        return MMPoseExtractor(
            config=_Cfg(),
            detector=None,
            pose_estimator=pose_estimator,
        )

    else:
        raise ValueError(f"Unknown pose type: {pose_type!r}")


__all__ = [
    "KEYPOINT_PRESETS",
    "LandmarkExtractor",
    "create_estimator",
    "list_presets",
    "resolve_keypoint_indices",
]
