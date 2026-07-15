"""Tests for normalization math in the canonical post-processor module.

This is the most important test file — covers core numerical logic.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from signdata.config.schema import Config
from signdata.post_processors import normalize as normalize_module
from signdata.post_processors.normalize import (
    NormalizePostProcessor,
    _apply_keypoint_reduction,
    _apply_visibility_mask,
    _normalize_clip_xyz,
    _process_single_file,
)


def test_single_worker_normalizes_without_process_pool(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    np.save(raw_dir / "sample.npy", np.ones((2, 85, 4), dtype=np.float32))
    config = Config(
        dataset={"name": "test"},
        processing={"enabled": False, "max_workers": 1},
        post_processing={"enabled": True, "normalize": {}},
    )
    context = SimpleNamespace(output_dir=tmp_path, force_all=False, stats={})
    monkeypatch.setattr(
        normalize_module,
        "ProcessPoolExecutor",
        lambda **_: pytest.fail("single worker should run in-process"),
    )

    NormalizePostProcessor(config).run(context)

    assert (tmp_path / "normalized" / "sample.npy").exists()
    assert context.stats["post_processing.normalize"]["success"] == 1


# ── _apply_keypoint_reduction ───────────────────────────────────────────────

class TestApplyKeypointReduction:
    def test_empty_indices_raises(self):
        arr = np.zeros((5, 10, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="empty"):
            _apply_keypoint_reduction(arr, [])

    def test_select_indices(self):
        arr = np.arange(5 * 10 * 4, dtype=np.float32).reshape(5, 10, 4)
        indices = [0, 3, 7]
        result = _apply_keypoint_reduction(arr, indices)
        assert result.shape == (5, 3, 4)
        np.testing.assert_array_equal(result[:, 0, :], arr[:, 0, :])
        np.testing.assert_array_equal(result[:, 1, :], arr[:, 3, :])
        np.testing.assert_array_equal(result[:, 2, :], arr[:, 7, :])

    def test_output_shape(self):
        arr = np.zeros((10, 133, 4), dtype=np.float32)
        indices = list(range(85))
        result = _apply_keypoint_reduction(arr, indices)
        assert result.shape == (10, 85, 4)

    def test_out_of_bounds_mentions_preset(self):
        arr = np.zeros((5, 10, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="keypoint_preset"):
            _apply_keypoint_reduction(arr, [0, 5, 15])

    def test_wrong_ndim_raises(self):
        arr = np.zeros((5, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="3D array"):
            _apply_keypoint_reduction(arr, [0, 1])


# ── _apply_visibility_mask ──────────────────────────────────────────────────

SENTINEL = -999.0


class TestApplyVisibilityMask:
    def test_frame_level_all_zero(self):
        """All-zero frame → sentinel -999.0."""
        arr = np.zeros((3, 4, 4), dtype=np.float32)
        # Set frame 1 to have some data
        arr[1, :, :3] = 1.0
        arr[1, :, 3] = 0.9

        result = _apply_visibility_mask(
            arr,
            mask_empty_frames=True,
            mask_low_confidence=False,
            visibility_threshold=0.3,
            missing_value=SENTINEL,
        )
        assert result.shape == (3, 4, 3)
        # Frame 0 and 2 are all-zero → sentinel
        np.testing.assert_array_equal(result[0], SENTINEL)
        np.testing.assert_array_equal(result[2], SENTINEL)
        # Frame 1 has valid data
        np.testing.assert_array_equal(result[1], arr[1, :, :3])

    def test_landmark_level_low_visibility(self):
        """Low visibility → sentinel -999.0."""
        arr = np.ones((2, 3, 4), dtype=np.float32)
        arr[:, :, 3] = 0.8  # All visible
        arr[0, 1, 3] = 0.1  # Landmark 1 in frame 0 → low visibility

        result = _apply_visibility_mask(
            arr,
            mask_empty_frames=False,
            mask_low_confidence=True,
            visibility_threshold=0.3,
            missing_value=SENTINEL,
        )
        assert result.shape == (2, 3, 3)
        # Landmark with low vis → sentinel
        np.testing.assert_array_equal(result[0, 1], SENTINEL)
        # Other landmarks intact
        np.testing.assert_array_equal(result[0, 0], arr[0, 0, :3])

    def test_both_disabled_no_masking(self):
        """No masking when both flags are False."""
        arr = np.zeros((2, 3, 4), dtype=np.float32)
        result = _apply_visibility_mask(
            arr,
            mask_empty_frames=False,
            mask_low_confidence=False,
            visibility_threshold=0.3,
            missing_value=SENTINEL,
        )
        assert result.shape == (2, 3, 3)
        np.testing.assert_array_equal(result, arr[:, :, :3])

    def test_output_strips_visibility_channel(self):
        """Output shape is (T, K, 3) — visibility channel stripped."""
        arr = np.ones((5, 10, 4), dtype=np.float32)
        result = _apply_visibility_mask(
            arr,
            mask_empty_frames=True,
            mask_low_confidence=True,
            visibility_threshold=0.3,
            missing_value=SENTINEL,
        )
        assert result.shape == (5, 10, 3)


# ── _normalize_clip_xyz ─────────────────────────────────────────────────────

class TestNormalizeClipXYZ:
    def test_isotropic_3d_known_input(self):
        """isotropic_3d: hand-computed expected output."""
        # 2 frames, 2 keypoints
        xyz = np.array([
            [[0.0, 0.0, 0.0],
             [1.0, 0.5, 0.25]],
            [[0.5, 0.25, 0.125],
             [1.0, 1.0, 0.5]],
        ], dtype=np.float32)

        result = _normalize_clip_xyz(xyz, "isotropic_3d", SENTINEL)

        # min = [0, 0, 0], max = [1, 1, 0.5], range = [1, 1, 0.5]
        # max_range = 1.0
        # scaled = (xyz - min) / 1.0 = xyz
        expected = xyz.copy()
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_xy_isotropic_z_minmax_known_input(self):
        """xy_isotropic_z_minmax: hand-computed expected output."""
        xyz = np.array([
            [[0.0, 0.0, 0.0],
             [2.0, 1.0, 10.0]],
            [[1.0, 0.5, 5.0],
             [2.0, 2.0, 10.0]],
        ], dtype=np.float32)

        result = _normalize_clip_xyz(xyz, "xy_isotropic_z_minmax", SENTINEL)

        # x: min=0, max=2 → range=2
        # y: min=0, max=2 → range=2
        # max_xy = 2
        # x_s = (x - 0) / 2, y_s = (y - 0) / 2
        # z: min=0, max=10 → z_s = (z - 0) / 10
        expected = np.array([
            [[0.0, 0.0, 0.0],
             [1.0, 0.5, 1.0]],
            [[0.5, 0.25, 0.5],
             [1.0, 1.0, 1.0]],
        ], dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_all_invalid_returns_unchanged(self):
        """All-invalid clip → returns unchanged."""
        xyz = np.full((3, 4, 3), SENTINEL, dtype=np.float32)
        result = _normalize_clip_xyz(xyz, "isotropic_3d", SENTINEL)
        np.testing.assert_array_equal(result, xyz)

    def test_zero_range_no_division_error(self):
        """Zero-range edge case → all zeros (no division error)."""
        # All valid points at the same location
        xyz = np.full((3, 2, 3), 0.5, dtype=np.float32)
        result = _normalize_clip_xyz(xyz, "isotropic_3d", SENTINEL)
        # zero range → scaled to zeros
        np.testing.assert_array_equal(result, np.zeros_like(xyz))

    def test_zero_range_xy_isotropic(self):
        """Zero range in xy_isotropic_z_minmax → all zeros."""
        xyz = np.full((3, 2, 3), 0.5, dtype=np.float32)
        result = _normalize_clip_xyz(xyz, "xy_isotropic_z_minmax", SENTINEL)
        np.testing.assert_array_equal(result, np.zeros_like(xyz))

    def test_sentinel_values_preserved(self):
        """Sentinel values preserved after normalization."""
        xyz = np.array([
            [[0.0, 0.0, 0.0],
             [SENTINEL, SENTINEL, SENTINEL]],
            [[1.0, 1.0, 1.0],
             [0.5, 0.5, 0.5]],
        ], dtype=np.float32)

        result = _normalize_clip_xyz(xyz, "isotropic_3d", SENTINEL)

        # Sentinel point should remain sentinel
        np.testing.assert_array_equal(result[0, 1], [SENTINEL, SENTINEL, SENTINEL])
        # Valid points should be normalized (not sentinel)
        assert not np.any(result[1, 0] == SENTINEL)

    def test_unknown_mode_raises(self):
        xyz = np.zeros((2, 3, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown mode"):
            _normalize_clip_xyz(xyz, "unknown_mode", SENTINEL)

    def test_wrong_shape_raises(self):
        xyz = np.zeros((2, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected shape"):
            _normalize_clip_xyz(xyz, "isotropic_3d", SENTINEL)


# ── _process_single_file integration ──────────────────────────────────────

class TestProcessSingleFileIntegration:
    """End-to-end tests for preset resolution inside _process_single_file."""

    def _make_config(self, **overrides):
        config = {
            "select_keypoints": True,
            "keypoint_preset": None,
            "keypoint_indices": None,
            "mask_empty_frames": True,
            "mask_low_confidence": False,
            "visibility_threshold": 0.3,
            "missing_value": SENTINEL,
            "mode": "xy_isotropic_z_minmax",
            "remove_z": False,
            "skip_existing": False,
        }
        config.update(overrides)
        return config

    def test_preset_reduces_keypoints(self, tmp_path):
        """Preset name resolves and reduces 553 → 85 keypoints."""
        arr = np.random.rand(10, 553, 4).astype(np.float32)
        inp = str(tmp_path / "input.npy")
        out = str(tmp_path / "output.npy")
        np.save(inp, arr)

        config = self._make_config(keypoint_preset="mediapipe_553_to_85")
        fname, ok, msg = _process_single_file((inp, out, config))

        assert ok, f"Failed: {msg}"
        result = np.load(out)
        # 85 keypoints * 3 coords (xyz, no z removal)
        assert result.shape == (10, 85 * 3)

    def test_manual_indices_still_work(self, tmp_path):
        """Manual keypoint_indices without preset still works."""
        arr = np.random.rand(5, 10, 4).astype(np.float32)
        inp = str(tmp_path / "input.npy")
        out = str(tmp_path / "output.npy")
        np.save(inp, arr)

        config = self._make_config(keypoint_indices=[0, 2, 4])
        fname, ok, msg = _process_single_file((inp, out, config))

        assert ok, f"Failed: {msg}"
        result = np.load(out)
        assert result.shape == (5, 3 * 3)

    def test_no_preset_no_indices_passes_through(self, tmp_path):
        """Neither preset nor indices → no reduction (all keypoints kept)."""
        arr = np.random.rand(5, 10, 4).astype(np.float32)
        inp = str(tmp_path / "input.npy")
        out = str(tmp_path / "output.npy")
        np.save(inp, arr)

        config = self._make_config()
        fname, ok, msg = _process_single_file((inp, out, config))

        assert ok, f"Failed: {msg}"
        result = np.load(out)
        assert result.shape == (5, 10 * 3)

    def test_preset_mismatch_reports_error(self, tmp_path):
        """553-preset applied to 133-keypoint file → clear error."""
        arr = np.random.rand(5, 133, 4).astype(np.float32)
        inp = str(tmp_path / "input.npy")
        out = str(tmp_path / "output.npy")
        np.save(inp, arr)

        config = self._make_config(keypoint_preset="mediapipe_553_to_85")
        fname, ok, msg = _process_single_file((inp, out, config))

        assert not ok
        assert "keypoint_preset" in msg

    def test_mmpose_preset(self, tmp_path):
        """MMPose preset reduces 133 → 85."""
        arr = np.random.rand(8, 133, 4).astype(np.float32)
        inp = str(tmp_path / "input.npy")
        out = str(tmp_path / "output.npy")
        np.save(inp, arr)

        config = self._make_config(keypoint_preset="mmpose_133_to_85")
        fname, ok, msg = _process_single_file((inp, out, config))

        assert ok, f"Failed: {msg}"
        result = np.load(out)
        assert result.shape == (8, 85 * 3)
