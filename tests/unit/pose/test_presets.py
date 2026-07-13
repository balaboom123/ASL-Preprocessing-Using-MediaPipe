"""Tests for the canonical pose keypoint preset registry."""

import pytest

from signdata.processors.pose import (
    KEYPOINT_PRESETS,
    list_presets,
    resolve_keypoint_indices,
)


# -----------------------------------------------------------------------
# KEYPOINT_PRESETS data integrity
# -----------------------------------------------------------------------

class TestKeypointPresetsData:
    def test_mediapipe_553_to_85_count(self):
        preset = KEYPOINT_PRESETS["mediapipe_553_to_85"]
        assert len(preset["indices"]) == 85

    def test_mediapipe_553_to_85_bounds(self):
        indices = KEYPOINT_PRESETS["mediapipe_553_to_85"]["indices"]
        assert all(0 <= i < 553 for i in indices)

    def test_mediapipe_543_to_83_count(self):
        preset = KEYPOINT_PRESETS["mediapipe_543_to_83"]
        assert len(preset["indices"]) == 83

    def test_mediapipe_543_to_83_bounds(self):
        indices = KEYPOINT_PRESETS["mediapipe_543_to_83"]["indices"]
        assert all(0 <= i < 543 for i in indices)

    def test_mmpose_133_to_85_count(self):
        preset = KEYPOINT_PRESETS["mmpose_133_to_85"]
        assert len(preset["indices"]) == 85

    def test_mmpose_133_to_85_bounds(self):
        indices = KEYPOINT_PRESETS["mmpose_133_to_85"]["indices"]
        assert all(0 <= i < 133 for i in indices)

    def test_no_duplicate_indices(self):
        """Each preset should have unique indices."""
        for name, preset in KEYPOINT_PRESETS.items():
            indices = preset["indices"]
            assert len(indices) == len(set(indices)), (
                f"Preset '{name}' has duplicate indices"
            )

    def test_all_presets_have_required_keys(self):
        required = {"description", "indices"}
        for name, preset in KEYPOINT_PRESETS.items():
            assert required <= set(preset.keys()), (
                f"Preset '{name}' missing keys: {required - set(preset.keys())}"
            )


# -----------------------------------------------------------------------
# resolve_keypoint_indices
# -----------------------------------------------------------------------

class TestResolveKeypointIndices:
    def test_preset_returns_indices(self):
        indices = resolve_keypoint_indices("mediapipe_553_to_85")
        assert indices is not None
        assert len(indices) == 85

    def test_preset_returns_copy(self):
        """Should return a copy, not a reference to the preset data."""
        a = resolve_keypoint_indices("mediapipe_553_to_85")
        b = resolve_keypoint_indices("mediapipe_553_to_85")
        assert a == b
        assert a is not b

    def test_manual_indices_returned(self):
        manual = [0, 1, 2, 3]
        indices = resolve_keypoint_indices(manual_indices=manual)
        assert indices == [0, 1, 2, 3]

    def test_manual_indices_returns_copy(self):
        manual = [0, 1, 2]
        result = resolve_keypoint_indices(manual_indices=manual)
        assert result == manual
        assert result is not manual

    def test_preset_takes_priority_over_manual(self):
        """When both are set, preset wins."""
        indices = resolve_keypoint_indices(
            preset_name="mmpose_133_to_85",
            manual_indices=[0, 1, 2],
        )
        assert len(indices) == 85

    def test_neither_returns_none(self):
        assert resolve_keypoint_indices() is None

    def test_none_preset_none_manual(self):
        assert resolve_keypoint_indices(None, None) is None

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown keypoint preset"):
            resolve_keypoint_indices("nonexistent_preset")

    def test_error_lists_available_presets(self):
        with pytest.raises(ValueError, match="mediapipe_553_to_85"):
            resolve_keypoint_indices("bad_name")


# -----------------------------------------------------------------------
# list_presets
# -----------------------------------------------------------------------

class TestListPresets:
    def test_returns_all_presets(self):
        presets = list_presets()
        assert set(presets.keys()) == set(KEYPOINT_PRESETS.keys())

    def test_values_are_strings(self):
        for name, desc in list_presets().items():
            assert isinstance(desc, str), f"Preset '{name}' description is not str"

    def test_descriptions_non_empty(self):
        for name, desc in list_presets().items():
            assert len(desc) > 0, f"Preset '{name}' has empty description"
