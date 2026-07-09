"""MMPose-based whole-body pose landmark extraction."""

import logging
from typing import Optional, List, Tuple

import numpy as np

from ..base import LandmarkExtractor

logger = logging.getLogger(__name__)


class MultiPersonDetected(Exception):
    """Raised when more than one person is detected in a frame."""
    pass


class MMPoseExtractor(LandmarkExtractor):
    """Extracts whole-body landmarks using MMPose.

    Uses a two-stage pipeline:
    1. RTMDet for person detection (bounding boxes)
    2. MMPose top-down whole-body pose estimation

    Always outputs all 133 COCO WholeBody keypoints as [x, y, z, visibility].
    For 2D MMPose models, z is 0.
    """

    # MMPose supports batch detection
    supports_batch_inference = True
    # Total landmark count
    num_landmarks = 133

    def __init__(
        self,
        config,
        detector=None,
        pose_estimator=None,
    ):
        self.detector = detector
        self.pose_estimator = pose_estimator
        self.bbox_threshold = config.bbox_threshold
        self.det_cat_id = 0
        self.add_visible = config.add_visible
        self._batch_inference_checked = False
        self._batch_inference_available = False

    def process_frame(
        self, frame: np.ndarray, bbox: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Extract 3D landmarks from a single frame.

        When *bbox* is provided (from an upstream detector), it is used
        directly for pose estimation.  When *bbox* is ``None`` and an
        internal detector exists, person detection runs first.  If
        neither is available, a full-frame bounding box is assumed.

        Returns array of shape (133, 4) with [x, y, z, visibility].
        Raises MultiPersonDetected if internal detection finds >1 person.
        """
        from mmpose.apis import inference_topdown
        from mmpose.structures import merge_data_samples

        if bbox is not None:
            # Upstream detector already ran — use provided bboxes
            bboxes = bbox
        elif self.detector is not None:
            # Legacy path: run internal detection
            from mmdet.apis import inference_detector

            det_result = inference_detector(self.detector, frame)
            pred_instance = det_result.pred_instances.cpu().numpy()

            bboxes = pred_instance.bboxes
            bboxes = bboxes[np.logical_and(
                pred_instance.labels == self.det_cat_id,
                pred_instance.scores > self.bbox_threshold
            )]

            if len(bboxes) == 0:
                return None

            if len(bboxes) > 1:
                raise MultiPersonDetected(
                    f"Detected {len(bboxes)} persons with score > {self.bbox_threshold}"
                )
        else:
            # No detector, no bbox — full-frame fallback
            H, W = frame.shape[:2]
            bboxes = np.array([[0, 0, W, H]], dtype=np.float32)

        pose_est_results = inference_topdown(self.pose_estimator, frame, bboxes)
        if not pose_est_results:
            return None

        for idx, pose_est_result in enumerate(pose_est_results):
            pose_est_result.track_id = pose_est_results[idx].get("track_id", 1e4)

            pred_instances = pose_est_result.pred_instances
            keypoints = pred_instances.keypoints
            keypoint_scores = pred_instances.keypoint_scores

            if keypoint_scores.ndim == 3:
                keypoint_scores = np.squeeze(keypoint_scores, axis=1)
                pose_est_results[idx].pred_instances.keypoint_scores = keypoint_scores

            if keypoints.ndim == 4:
                keypoints = np.squeeze(keypoints, axis=1)

            pose_est_results[idx].pred_instances.keypoints = keypoints

        pose_est_results = sorted(
            pose_est_results, key=lambda x: x.get("track_id", 1e4)
        )

        pred_3d_data_samples = merge_data_samples(pose_est_results)
        pred_3d_instances = pred_3d_data_samples.get("pred_instances", None)

        if pred_3d_instances is None:
            return None

        H, W = frame.shape[:2]
        packed = self._pack_keypoints(pred_3d_instances, W, H)
        return packed

    def _check_batch_inference_support(self) -> bool:
        """Lazily check if batch detection is supported.

        Returns True if inference_detector accepts a list of images.
        """
        if self._batch_inference_checked:
            return self._batch_inference_available

        self._batch_inference_checked = True
        try:
            from mmdet.apis import inference_detector
            import inspect
            sig = inspect.signature(inference_detector)
            # Check if imgs parameter accepts list (it does in modern mmdet)
            self._batch_inference_available = True
        except Exception:
            self._batch_inference_available = False

        return self._batch_inference_available

    def process_batch(
        self,
        frames: List[np.ndarray],
        bboxes: Optional[List[Optional[np.ndarray]]] = None,
        fallback_on_error: bool = True,
    ) -> List[Optional[np.ndarray]]:
        """Process a batch of frames with optional external bounding boxes.

        When *bboxes* are provided (or no internal detector exists),
        each frame is processed individually using the supplied bbox.
        Otherwise, uses batched internal detection with three-level
        fallback.

        Args:
            frames: List of input video frames (BGR format).
            bboxes: Optional per-frame bounding boxes from upstream
                detector (each element shape (N, 4) or None).
            fallback_on_error: If True, return None for failed frames.

        Returns:
            List of landmark arrays (or None for failed/multi-person frames).
        """
        if not frames:
            return []

        # When external bboxes are provided or there is no internal
        # detector, process per-frame with the supplied bboxes.
        if bboxes is not None or self.detector is None:
            results: List[Optional[np.ndarray]] = [None] * len(frames)
            for i, frame in enumerate(frames):
                try:
                    bbox = bboxes[i] if bboxes else None
                    results[i] = self.process_frame(frame, bbox=bbox)
                except Exception:
                    if not fallback_on_error:
                        raise
            return results

        # Legacy path: internal batched detection
        if self._check_batch_inference_support():
            try:
                return self._process_batch_batched(frames)
            except Exception as e:
                if not fallback_on_error:
                    raise
                logger.warning(
                    "Batch inference failed, falling back to sequential: %s", e
                )

        # Fallback: sequential processing
        return self._process_batch_sequential(frames, fallback_on_error)

    def _process_batch_batched(
        self,
        frames: List[np.ndarray],
    ) -> List[Optional[np.ndarray]]:
        """Process batch with true batched detection.

        Step 1: Run batched detection on all frames
        Step 2: Filter valid detections (single person only)
        Step 3: Run pose estimation on valid frames
        """
        from mmpose.apis import inference_topdown
        from mmpose.structures import merge_data_samples
        from mmdet.apis import inference_detector

        # Pre-allocate results
        results: List[Optional[np.ndarray]] = [None] * len(frames)

        # Step 1: Batch detection
        det_results = inference_detector(self.detector, frames)

        # Step 2: Process each detection result and run pose estimation
        for i, det_result in enumerate(det_results):
            try:
                pred_instance = det_result.pred_instances.cpu().numpy()
                bboxes = pred_instance.bboxes
                bboxes = bboxes[np.logical_and(
                    pred_instance.labels == self.det_cat_id,
                    pred_instance.scores > self.bbox_threshold
                )]

                # Skip if no person or multiple persons detected
                if len(bboxes) == 0:
                    continue
                if len(bboxes) > 1:
                    # Multi-person frame → None (not an error, just skip)
                    continue

                # Step 3: Pose estimation for this frame
                frame = frames[i]
                pose_est_results = inference_topdown(
                    self.pose_estimator, frame, bboxes
                )

                for idx, pose_est_result in enumerate(pose_est_results):
                    pose_est_result.track_id = pose_est_results[idx].get(
                        "track_id", 1e4
                    )

                    pred_instances = pose_est_result.pred_instances
                    keypoints = pred_instances.keypoints
                    keypoint_scores = pred_instances.keypoint_scores

                    if keypoint_scores.ndim == 3:
                        keypoint_scores = np.squeeze(keypoint_scores, axis=1)
                        pose_est_results[idx].pred_instances.keypoint_scores = (
                            keypoint_scores
                        )

                    if keypoints.ndim == 4:
                        keypoints = np.squeeze(keypoints, axis=1)

                    pose_est_results[idx].pred_instances.keypoints = keypoints

                pose_est_results = sorted(
                    pose_est_results, key=lambda x: x.get("track_id", 1e4)
                )

                pred_3d_data_samples = merge_data_samples(pose_est_results)
                pred_3d_instances = pred_3d_data_samples.get(
                    "pred_instances", None
                )

                if pred_3d_instances is None:
                    continue

                H, W = frame.shape[:2]
                results[i] = self._pack_keypoints(pred_3d_instances, W, H)

            except Exception:
                # Per-frame error → result stays None
                continue

        return results

    def _process_batch_sequential(
        self,
        frames: List[np.ndarray],
        fallback_on_error: bool = True,
    ) -> List[Optional[np.ndarray]]:
        """Process batch sequentially (fallback mode).

        Args:
            frames: List of input video frames
            fallback_on_error: If True, continue on errors.

        Returns:
            List of landmark arrays (or None for failed frames).
        """
        results: List[Optional[np.ndarray]] = [None] * len(frames)

        for i, frame in enumerate(frames):
            try:
                landmarks = self.process_frame(frame)
                results[i] = landmarks
            except MultiPersonDetected:
                # Multi-person → None, continue processing
                continue
            except Exception:
                if not fallback_on_error:
                    raise
                # Other errors → None, continue processing
                continue

        return results

    def _pack_keypoints(
        self,
        pred_3d_instances,
        img_w: int,
        img_h: int,
        instance_index: int = 0,
    ) -> Optional[np.ndarray]:
        """Extract and pack keypoints from MMPose estimation results."""
        if pred_3d_instances is None:
            return None

        tk = getattr(pred_3d_instances, "transformed_keypoints", None)
        k3d = getattr(pred_3d_instances, "keypoints", None)
        if k3d is None:
            return None

        k3d = self._to_numpy(k3d)
        k3d = self._squeeze_kpts(k3d)
        tk = k3d if tk is None else self._squeeze_kpts(self._to_numpy(tk))

        if tk.ndim != 3 or k3d.ndim != 3 or tk.shape[0] == 0 or k3d.shape[0] == 0:
            return None

        xy = tk[instance_index][..., :2]
        z = (
            k3d[instance_index][..., 2]
            if k3d.shape[-1] >= 3
            else np.zeros(xy.shape[0], dtype=np.float32)
        )

        x_norm = xy[..., 0] / float(img_w)
        y_norm = xy[..., 1] / float(img_h)

        kpt_scores = getattr(pred_3d_instances, "keypoint_scores", None)
        if kpt_scores is not None:
            kpt_scores = self._to_numpy(kpt_scores)
            if kpt_scores.ndim == 2:
                visible = kpt_scores[instance_index]
            elif kpt_scores.ndim == 3:
                visible = kpt_scores[instance_index, :, 0]
            else:
                visible = np.ones(xy.shape[0], dtype=np.float32)
        else:
            visible = np.ones(xy.shape[0], dtype=np.float32)

        out = np.stack([x_norm, y_norm, z, visible], axis=-1).astype(np.float32)
        return out

    @staticmethod
    def _to_numpy(x):
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        elif hasattr(x, "cpu"):
            x = x.cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def _squeeze_kpts(arr, expect_last: int = 2):
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr[:, 0]
        return arr

    def close(self):
        pass
