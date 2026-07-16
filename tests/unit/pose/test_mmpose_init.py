"""Tests for MMPose extraction helpers without loading MMPose."""

import numpy as np

from signdata.processors.pose.mmpose import MMPoseExtractor


class TestMMPoseExtractor:
    def test_process_batch_forwards_upstream_bboxes(self):
        ext = MMPoseExtractor(object())
        seen = []
        ext.process_frame = lambda frame, bbox=None: seen.append(bbox)
        bboxes = [np.ones((1, 4)), np.ones((1, 4)) * 2]

        ext.process_batch([np.zeros((1, 1, 3))] * 2, bboxes=bboxes)

        assert seen[0] is bboxes[0]
        assert seen[1] is bboxes[1]

    def test_process_batch_continues_after_frame_error(self):
        ext = MMPoseExtractor(object())

        def fail(frame, bbox=None):
            raise RuntimeError("bad frame")

        ext.process_frame = fail

        assert ext.process_batch([np.zeros((1, 1, 3))]) == [None]

    def test_pack_2d_mmpose_keypoints(self):
        """Pip MMPose whole-body 2D models pack as x, y, z=0, score."""

        class _Pred:
            keypoints = [[[10.0, 20.0], [30.0, 40.0]]]
            keypoint_scores = [[0.9, 0.8]]

        packed = MMPoseExtractor(object())._pack_keypoints(
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

    def test_squeeze_instance_axis_uses_expected_output_rank(self):
        ext = MMPoseExtractor(object())

        scores = np.zeros((2, 1, 3))
        keypoints = np.zeros((2, 1, 3, 2))
        one_keypoint = np.zeros((2, 1, 3))

        assert ext._squeeze_instance_axis(scores, expected_ndim=2).shape == (2, 3)
        assert ext._squeeze_instance_axis(keypoints, expected_ndim=3).shape == (2, 3, 2)
        assert ext._squeeze_instance_axis(one_keypoint, expected_ndim=3).shape == (2, 1, 3)
