"""Tests for MMPoseExtractor.__init__ parameter handling.

Only tests __init__ parameter handling — no GPU, mmpose, or real frames needed.
"""

import numpy as np

from signdata.processors.pose.mmpose import MMPoseExtractor


class _Cfg:
    bbox_threshold = 0.5
    keypoint_threshold = 0.3
    add_visible = True
    batch_size = 16


class TestMMPoseInit:
    def test_default_config(self):
        """Duck-typed config creates extractor with expected attributes."""

        ext = MMPoseExtractor(_Cfg())
        assert ext.bbox_threshold == 0.5
        assert ext.add_visible is True
        assert ext.det_cat_id == 0

    def test_custom_bbox_threshold(self):
        """Custom bbox_threshold is stored."""

        class _Cfg:
            bbox_threshold = 0.7
            keypoint_threshold = 0.3
            add_visible = True
            batch_size = 16

        ext = MMPoseExtractor(_Cfg())
        assert ext.bbox_threshold == 0.7

    def test_no_reduction_attributes(self):
        """Extractor no longer has reduction-related attributes."""

        ext = MMPoseExtractor(_Cfg())
        assert not hasattr(ext, "apply_reduction")
        assert not hasattr(ext, "keypoint_indices")

    def test_pack_2d_mmpose_keypoints(self):
        """Pip MMPose whole-body 2D models pack as x, y, z=0, score."""

        class _Pred:
            keypoints = [[[10.0, 20.0], [30.0, 40.0]]]
            keypoint_scores = [[0.9, 0.8]]

        packed = MMPoseExtractor(_Cfg())._pack_keypoints(
            _Pred(),
            img_w=100,
            img_h=200,
        )
        assert packed.shape == (2, 4)
        np.testing.assert_allclose(
            packed,
            [
                [0.1, 0.1, 0.0, 0.9],
                [0.3, 0.2, 0.0, 0.8],
            ],
        )
