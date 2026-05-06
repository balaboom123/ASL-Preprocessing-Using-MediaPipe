"""Tests for shared availability policy helpers.

Covers:
  - get_existing_video_ids (multi-extension scanning)
  - apply_availability_policy (all three policies)
  - filter_available (AVAILABLE column filtering)
  - write_acquire_report (report file generation)
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from signdata.datasets._ingestion.availability import (
    apply_availability_policy,
    get_existing_video_ids,
    write_acquire_report,
)
from signdata.utils.availability import filter_available


# ── get_existing_video_ids ─────────────────────────────────────────────

class TestGetExistingVideoIds:
    def test_mp4(self, tmp_path):
        (tmp_path / "v1.mp4").touch()
        assert get_existing_video_ids(str(tmp_path)) == {"v1"}

    def test_webm(self, tmp_path):
        (tmp_path / "v1.webm").touch()
        assert "v1" in get_existing_video_ids(str(tmp_path))

    def test_mkv(self, tmp_path):
        (tmp_path / "v1.mkv").touch()
        assert "v1" in get_existing_video_ids(str(tmp_path))

    def test_non_video_ignored(self, tmp_path):
        (tmp_path / "v1.txt").touch()
        (tmp_path / "v2.json").touch()
        assert get_existing_video_ids(str(tmp_path)) == set()

    def test_multiple_extensions_same_stem(self, tmp_path):
        (tmp_path / "v1.mp4").touch()
        (tmp_path / "v1.webm").touch()
        assert get_existing_video_ids(str(tmp_path)) == {"v1"}

    def test_multiple_videos(self, tmp_path):
        (tmp_path / "a.mp4").touch()
        (tmp_path / "b.webm").touch()
        (tmp_path / "c.mkv").touch()
        assert get_existing_video_ids(str(tmp_path)) == {"a", "b", "c"}

    def test_empty_directory(self, tmp_path):
        assert get_existing_video_ids(str(tmp_path)) == set()

    def test_recursive_nested_directory(self, tmp_path):
        nested = tmp_path / "class0"
        nested.mkdir()
        (nested / "v1.mp4").touch()
        assert get_existing_video_ids(str(tmp_path), recursive=True) == {"v1"}


# ── apply_availability_policy ──────────────────────────────────────────

class TestApplyAvailabilityPolicy:
    def _make_df(self, video_ids):
        return pd.DataFrame({
            "SAMPLE_ID": [f"s{i}" for i in range(len(video_ids))],
            "VIDEO_ID": video_ids,
        })

    def test_drop_unavailable_removes_missing(self, tmp_path):
        (tmp_path / "v1.mp4").touch()
        df = self._make_df(["v1", "v2"])

        result = apply_availability_policy(df, str(tmp_path), "drop_unavailable")
        assert len(result) == 1
        assert result.iloc[0]["VIDEO_ID"] == "v1"
        assert "AVAILABLE" not in result.columns

    def test_mark_unavailable_adds_column(self, tmp_path):
        (tmp_path / "v1.mp4").touch()
        df = self._make_df(["v1", "v2"])

        result = apply_availability_policy(df, str(tmp_path), "mark_unavailable")
        assert len(result) == 2
        assert "AVAILABLE" in result.columns
        assert result.iloc[0]["AVAILABLE"] == True
        assert result.iloc[1]["AVAILABLE"] == False

    def test_mark_unavailable_all_present(self, tmp_path):
        (tmp_path / "v1.mp4").touch()
        (tmp_path / "v2.mp4").touch()
        df = self._make_df(["v1", "v2"])

        result = apply_availability_policy(df, str(tmp_path), "mark_unavailable")
        assert len(result) == 2
        assert all(result["AVAILABLE"])

    def test_fail_fast_raises(self, tmp_path):
        (tmp_path / "v1.mp4").touch()
        df = self._make_df(["v1", "v2"])

        with pytest.raises(RuntimeError, match="not found"):
            apply_availability_policy(df, str(tmp_path), "fail_fast")

    def test_fail_fast_all_present_no_error(self, tmp_path):
        (tmp_path / "v1.mp4").touch()
        (tmp_path / "v2.mp4").touch()
        df = self._make_df(["v1", "v2"])

        result = apply_availability_policy(df, str(tmp_path), "fail_fast")
        assert len(result) == 2

    def test_drop_does_not_mutate_original(self, tmp_path):
        (tmp_path / "v1.mp4").touch()
        df = self._make_df(["v1", "v2"])
        original_len = len(df)

        apply_availability_policy(df, str(tmp_path), "drop_unavailable")
        assert len(df) == original_len  # original unchanged

    def test_mark_does_not_mutate_original(self, tmp_path):
        (tmp_path / "v1.mp4").touch()
        df = self._make_df(["v1", "v2"])

        apply_availability_policy(df, str(tmp_path), "mark_unavailable")
        assert "AVAILABLE" not in df.columns  # original unchanged

    def test_extension_agnostic(self, tmp_path):
        """Finds videos with non-mp4 extensions."""
        (tmp_path / "v1.webm").touch()
        df = self._make_df(["v1", "v2"])

        result = apply_availability_policy(df, str(tmp_path), "drop_unavailable")
        assert len(result) == 1
        assert result.iloc[0]["VIDEO_ID"] == "v1"


# ── filter_available ───────────────────────────────────────────────────

class TestFilterAvailable:
    def test_no_available_column_passthrough(self):
        df = pd.DataFrame({"VIDEO_ID": ["v1", "v2"]})
        result = filter_available(df)
        assert len(result) == 2

    def test_filters_false_rows(self):
        df = pd.DataFrame({
            "VIDEO_ID": ["v1", "v2", "v3"],
            "AVAILABLE": [True, False, True],
        })
        result = filter_available(df)
        assert len(result) == 2
        assert list(result["VIDEO_ID"]) == ["v1", "v3"]

    def test_all_available_no_change(self):
        df = pd.DataFrame({
            "VIDEO_ID": ["v1", "v2"],
            "AVAILABLE": [True, True],
        })
        result = filter_available(df)
        assert len(result) == 2

    def test_resets_index(self):
        df = pd.DataFrame({
            "VIDEO_ID": ["v1", "v2", "v3"],
            "AVAILABLE": [False, True, True],
        })
        result = filter_available(df)
        assert list(result.index) == [0, 1]


# ── write_acquire_report ──────────────────────────────────────────────

class TestWriteAcquireReport:
    def test_writes_json_report(self, tmp_path):
        report_dir = str(tmp_path / "reports")
        stats = {"total": 10, "downloaded": 8, "errors": 2}
        write_acquire_report(report_dir, stats, missing=[])

        report = json.loads((tmp_path / "reports" / "download_report.json").read_text())
        assert report["total"] == 10
        assert report["errors"] == 2

    def test_writes_missing_csv_with_data(self, tmp_path):
        report_dir = str(tmp_path / "reports")
        missing = [
            {"VIDEO_ID": "v1", "REASON": "unavailable"},
            {"VIDEO_ID": "v2", "REASON": "blocked"},
        ]
        write_acquire_report(report_dir, {"total": 5}, missing)

        csv_path = tmp_path / "reports" / "missing_videos.csv"
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert "VIDEO_ID" in df.columns
        assert "REASON" in df.columns

    def test_writes_empty_csv_with_headers(self, tmp_path):
        report_dir = str(tmp_path / "reports")
        write_acquire_report(report_dir, {"total": 5}, missing=[])

        csv_path = tmp_path / "reports" / "missing_videos.csv"
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 0
        assert "VIDEO_ID" in df.columns
        assert "REASON" in df.columns

    def test_creates_directory(self, tmp_path):
        report_dir = str(tmp_path / "nested" / "reports")
        write_acquire_report(report_dir, {"total": 1}, missing=[])
        assert (tmp_path / "nested" / "reports" / "download_report.json").exists()
