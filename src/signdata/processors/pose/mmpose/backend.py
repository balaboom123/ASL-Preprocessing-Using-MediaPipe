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

        if bbox is None:
            height, width = frame.shape[:2]
            bbox = np.array([[0, 0, width, height]], dtype=np.float32)

        pose_est_results = inference_topdown(self.pose_estimator, frame, bbox)
        if not pose_est_results:
            return None

        for pose_est_result in pose_est_results:
            pose_est_result.track_id = pose_est_result.get("track_id", 1e4)
            pred_instances = pose_est_result.pred_instances
            pred_instances.keypoints = self._squeeze_instance_axis(
                pred_instances.keypoints, expected_ndim=3
            )
            pred_instances.keypoint_scores = self._squeeze_instance_axis(
                pred_instances.keypoint_scores, expected_ndim=2
            )

        pose_est_results = sorted(
            pose_est_results, key=lambda x: x.get("track_id", 1e4)
        )

        pred_3d_data_samples = merge_data_samples(pose_est_results)
        pred_3d_instances = pred_3d_data_samples.get("pred_instances")

        if pred_3d_instances is None:
            return None

        height, width = frame.shape[:2]
        return self._pack_keypoints(pred_3d_instances, width, height)

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
        k3d = self._squeeze_instance_axis(k3d, expected_ndim=3)
        tk = (
            k3d
            if tk is None
            else self._squeeze_instance_axis(self._to_numpy(tk), expected_ndim=3)
        )

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

        return np.stack([x_norm, y_norm, z, visible], axis=-1).astype(np.float32)

    @staticmethod
    def _to_numpy(x):
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        elif hasattr(x, "cpu"):
            x = x.cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def _squeeze_instance_axis(arr, expected_ndim: int):
        if arr.ndim == expected_ndim + 1 and arr.shape[1] == 1:
            return arr[:, 0]
        return arr
