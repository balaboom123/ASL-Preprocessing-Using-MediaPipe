"""MMPose-based whole-body pose landmark extraction."""

from typing import Optional

import numpy as np

from ..base import LandmarkExtractor


class MMPoseExtractor(LandmarkExtractor):
    """Extracts whole-body landmarks using MMPose.

    Uses bounding boxes supplied by the upstream detection stage.

    Always outputs all 133 COCO WholeBody keypoints as [x, y, z, visibility].
    For 2D MMPose models, z is 0.
    """

    num_landmarks = 133

    def __init__(self, pose_estimator):
        self.pose_estimator = pose_estimator

    def process_frame(
        self, frame: np.ndarray, bbox: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Extract 3D landmarks from a single frame.

        When *bbox* is omitted, a full-frame bounding box is assumed.

        Returns array of shape (133, 4) with [x, y, z, visibility].
        """
        from mmpose.apis import inference_topdown
        from mmpose.structures import merge_data_samples

        if bbox is not None:
            bboxes = bbox
        else:
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
    def _squeeze_kpts(arr):
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr[:, 0]
        return arr
