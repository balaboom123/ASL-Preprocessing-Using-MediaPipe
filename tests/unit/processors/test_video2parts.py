"""Tests for SignMusketeers-style video2parts helpers."""

import numpy as np

import signdata.processors  # noqa: F401 - trigger registrations
from signdata.processors.video2parts import (
    _bbox_from_landmarks,
    _normalize_upper_body_pose,
)
from signdata.registry import PROCESSOR_REGISTRY


def test_bbox_from_landmarks_returns_clipped_square():
    landmarks = np.array([
        [0.4, 0.2, 0.0, 1.0],
        [0.6, 0.4, 0.0, 1.0],
    ], dtype=np.float32)

    assert _bbox_from_landmarks(landmarks, (100, 200, 3), 1.0) == (80, 10, 120, 50)


def test_normalize_upper_body_pose_reuses_previous_when_missing():
    previous = np.arange(14, dtype=np.float32)
    body, valid = _normalize_upper_body_pose(
        np.zeros((33, 4), dtype=np.float32),
        missing_value=-1.0,
        previous=previous,
    )

    assert valid is False
    assert np.array_equal(body, previous)


def test_normalize_upper_body_pose_returns_14_values():
    pose = np.zeros((33, 4), dtype=np.float32)
    pose[0, :2] = [0.5, 0.2]
    pose[2, :2] = [0.48, 0.18]
    pose[5, :2] = [0.52, 0.18]
    pose[11, :2] = [0.4, 0.4]
    pose[12, :2] = [0.6, 0.4]
    pose[13, :2] = [0.35, 0.55]
    pose[14, :2] = [0.65, 0.55]
    pose[15, :2] = [0.3, 0.7]
    pose[16, :2] = [0.7, 0.7]

    body, valid = _normalize_upper_body_pose(pose, missing_value=-1.0)

    assert valid is True
    assert body.shape == (14,)
    assert np.isfinite(body).all()


def test_video2parts_registered():
    assert "video2parts" in PROCESSOR_REGISTRY
