"""Tests for OpenASL dataset adapter.

Covers:
  - Registration in DATASET_REGISTRY
  - OpenASLSourceConfig validation and defaults
  - validate_config
  - build_manifest (column mapping, text processing, optional columns)
  - Bounding-box JSON merging
  - acquire error paths (no yt-dlp needed)
  - [P1] Extension-agnostic video skip check
  - [P2] Availability policy (drop_unavailable, mark_unavailable, fail_fast)
  - [P2] Acquire report generation
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from signdata.config.schema import Config
from signdata.datasets.openasl import OpenASLDataset, OpenASLSourceConfig
from signdata.datasets.openasl import source as openasl_source
from signdata.pipeline.context import PipelineContext
from signdata.registry import DATASET_REGISTRY
from signdata.datasets._ingestion.availability import get_existing_video_ids


# ── Registration ──────────────────────────────────────────────────────────

class TestOpenASLRegistration:
    def test_registered(self):
        assert "openasl" in DATASET_REGISTRY

    def test_instance_has_name(self):
        adapter = OpenASLDataset()
        assert adapter.name == "openasl"


# ── validate_config ──────────────────────────────────────────────────────

class TestOpenASLValidateConfig:
    def test_valid_config_passes(self):
        cfg = Config(
            dataset={
                "name": "openasl",
                "source": {"manifest_tsv": "assets/openasl-v1.0.tsv"},
            },
        )
        OpenASLDataset.validate_config(cfg)

    def test_missing_manifest_tsv_raises(self):
        cfg = Config(dataset={"name": "openasl"})
        with pytest.raises(ValueError, match="manifest_tsv"):
            OpenASLDataset.validate_config(cfg)

    def test_empty_manifest_tsv_raises(self):
        cfg = Config(
            dataset={
                "name": "openasl",
                "source": {"manifest_tsv": ""},
            },
        )
        with pytest.raises(ValueError, match="manifest_tsv"):
            OpenASLDataset.validate_config(cfg)


# ── get_source_config ────────────────────────────────────────────────────

class TestOpenASLSourceConfig:
    def test_defaults(self):
        cfg = Config(
            dataset={
                "name": "openasl",
                "source": {"manifest_tsv": "data/openasl.tsv"},
            },
        )
        adapter = OpenASLDataset()
        source = adapter.get_source_config(cfg)

        assert isinstance(source, OpenASLSourceConfig)
        assert source.manifest_tsv == "data/openasl.tsv"
        assert source.text_column == "en"
        assert source.bbox_json == ""
        assert source.availability_policy == "drop_unavailable"
        assert source.text_processing.fix_encoding is True
        assert source.text_processing.lowercase is False

    def test_custom_text_column(self):
        cfg = Config(
            dataset={
                "name": "openasl",
                "source": {
                    "manifest_tsv": "data/openasl.tsv",
                    "text_column": "translation",
                },
            },
        )
        adapter = OpenASLDataset()
        source = adapter.get_source_config(cfg)
        assert source.text_column == "translation"

    def test_text_processing_config(self):
        cfg = Config(
            dataset={
                "name": "openasl",
                "source": {
                    "manifest_tsv": "data/openasl.tsv",
                    "text_processing": {"lowercase": True},
                },
            },
        )
        adapter = OpenASLDataset()
        source = adapter.get_source_config(cfg)
        assert source.text_processing.lowercase is True
        assert source.text_processing.fix_encoding is True  # default preserved

    def test_bbox_json_config(self):
        cfg = Config(
            dataset={
                "name": "openasl",
                "source": {
                    "manifest_tsv": "data/openasl.tsv",
                    "bbox_json": "data/bbox-v1.0.json",
                },
            },
        )
        adapter = OpenASLDataset()
        source = adapter.get_source_config(cfg)
        assert source.bbox_json == "data/bbox-v1.0.json"


# ── build_manifest ───────────────────────────────────────────────────────

class TestOpenASLBuildManifest:
    def _make_context(self, config):
        adapter = OpenASLDataset()
        return PipelineContext(config=config, dataset=adapter)

    def _write_tsv(self, path, data, columns=None):
        if columns is None:
            columns = ["vid", "yid", "start", "end", "en"]
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(path, sep="\t", index=False)

    def test_basic_manifest(self, tmp_path):
        tsv_path = tmp_path / "openasl.tsv"
        self._write_tsv(tsv_path, [
            ["seg001", "yt_abc", 0.0, 5.0, "Hello world"],
            ["seg002", "yt_abc", 5.0, 10.0, "Second sentence"],
            ["seg003", "yt_def", 0.0, 3.0, "Another video"],
        ])

        manifest_path = tmp_path / "manifest.csv"
        cfg = Config(
            dataset={
                "name": "openasl",
                "source": {"manifest_tsv": str(tsv_path)},
            },
            paths={"manifest": str(manifest_path)},
        )
        adapter = OpenASLDataset()
        context = self._make_context(cfg)
        context = adapter.build_manifest(cfg, context)

        assert context.manifest_path == manifest_path
        df = context.manifest_df
        assert len(df) == 3
        assert set(df.columns) >= {"SAMPLE_ID", "VIDEO_ID", "START", "END", "TEXT"}
        assert df.iloc[0]["SAMPLE_ID"] == "seg001"
        assert df.iloc[0]["VIDEO_ID"] == "yt_abc"
        assert df.iloc[0]["START"] == 0.0
        assert df.iloc[0]["END"] == 5.0
        assert df.iloc[0]["TEXT"] == "Hello world"
        assert context.stats["dataset.manifest"]["videos"] == 2
        assert context.stats["dataset.manifest"]["segments"] == 3

    def test_manifest_written_to_disk(self, tmp_path):
        """Manifest file is actually written as TSV."""
        tsv_path = tmp_path / "openasl.tsv"
        self._write_tsv(tsv_path, [
            ["s1", "yt1", 0.0, 1.0, "Text"],
        ])

        manifest_path = tmp_path / "out" / "manifest.csv"
        cfg = Config(
            dataset={
                "name": "openasl",
                "source": {"manifest_tsv": str(tsv_path)},
            },
            paths={"manifest": str(manifest_path)},
        )
        adapter = OpenASLDataset()
        context = self._make_context(cfg)
        adapter.build_manifest(cfg, context)

        assert manifest_path.exists()
        reloaded = pd.read_csv(manifest_path, delimiter="\t")
        assert "SAMPLE_ID" in reloaded.columns
        assert len(reloaded) == 1

    def test_missing_tsv_raises(self, tmp_path):
        manifest_path = tmp_path / "manifest.csv"
        cfg = Config(
            dataset={
                "name": "openasl",
                "source": {"manifest_tsv": str(tmp_path / "nope.tsv")},
            },
            paths={"manifest": str(manifest_path)},
        )
        adapter = OpenASLDataset()
        context = self._make_context(cfg)
        with pytest.raises(FileNotFoundError, match="OpenASL manifest TSV"):
            adapter.build_manifest(cfg, context)

    def test_missing_required_column_raises(self, tmp_path):
        """TSV missing 'yid' column raises ValueError."""
        tsv_path = tmp_path / "bad.tsv"
        pd.DataFrame({"vid": ["s1"], "start": [0.0], "end": [1.0]}).to_csv(
            tsv_path, sep="\t", index=False,
        )
        manifest_path = tmp_path / "manifest.csv"
        cfg = Config(
            dataset={
                "name": "openasl",
                "source": {"manifest_tsv": str(tsv_path)},
            },
            paths={"manifest": str(manifest_path)},
        )
        adapter = OpenASLDataset()
        context = self._make_context(cfg)
        with pytest.raises(ValueError, match="yid"):
            adapter.build_manifest(cfg, context)

    def test_custom_text_column(self, tmp_path):
        tsv_path = tmp_path / "openasl.tsv"
        pd.DataFrame({
            "vid": ["s1"], "yid": ["yt1"],
            "start": [0.0], "end": [1.0],
            "translation": ["Custom text"],
        }).to_csv(tsv_path, sep="\t", index=False)

        manifest_path = tmp_path / "manifest.csv"
        cfg = Config(
            dataset={
                "name": "openasl",
                "source": {
                    "manifest_tsv": str(tsv_path),
                    "text_column": "translation",
                },
            },
            paths={"manifest": str(manifest_path)},
        )
        adapter = OpenASLDataset()
        context = self._make_context(cfg)
        context = adapter.build_manifest(cfg, context)

        assert context.manifest_df.iloc[0]["TEXT"] == "Custom text"

    def test_missing_text_column_no_crash(self, tmp_path):
        """If the configured text column doesn't exist, build succeeds without TEXT."""
        tsv_path = tmp_path / "openasl.tsv"
        pd.DataFrame({
            "vid": ["s1"], "yid": ["yt1"],
            "start": [0.0], "end": [1.0],
        }).to_csv(tsv_path, sep="\t", index=False)

        manifest_path = tmp_path / "manifest.csv"
        cfg = Config(
            dataset={
                "name": "openasl",
                "source": {"manifest_tsv": str(tsv_path)},
            },
            paths={"manifest": str(manifest_path)},
        )
        adapter = OpenASLDataset()
        context = self._make_context(cfg)
        context = adapter.build_manifest(cfg, context)

        assert "TEXT" not in context.manifest_df.columns
        assert len(context.manifest_df) == 1

    def test_text_processing_applied(self, tmp_path):
        tsv_path = tmp_path / "openasl.tsv"
        self._write_tsv(tsv_path, [
            ["s1", "yt1", 0.0, 1.0, "Hello World!"],
        ])

        manifest_path = tmp_path / "manifest.csv"
        cfg = Config(
            dataset={
                "name": "openasl",
                "source": {
                    "manifest_tsv": str(tsv_path),
                    "text_processing": {"lowercase": True, "strip_punctuation": True},
                },
            },
            paths={"manifest": str(manifest_path)},
        )
        adapter = OpenASLDataset()
        context = self._make_context(cfg)
        context = adapter.build_manifest(cfg, context)

        assert context.manifest_df.iloc[0]["TEXT"] == "hello world"

    def test_optional_columns_passed_through(self, tmp_path):
        tsv_path = tmp_path / "openasl.tsv"
        pd.DataFrame({
            "vid": ["s1"], "yid": ["yt1"],
            "start": [0.0], "end": [1.0],
            "en": ["Text"],
            "split": ["train"],
            "signer_id": ["signer_42"],
        }).to_csv(tsv_path, sep="\t", index=False)

        manifest_path = tmp_path / "manifest.csv"
        cfg = Config(
            dataset={
                "name": "openasl",
                "source": {"manifest_tsv": str(tsv_path)},
            },
            paths={"manifest": str(manifest_path)},
        )
        adapter = OpenASLDataset()
        context = self._make_context(cfg)
        context = adapter.build_manifest(cfg, context)

        assert context.manifest_df.iloc[0]["SPLIT"] == "train"
        assert context.manifest_df.iloc[0]["SIGNER_ID"] == "signer_42"

    def test_multiple_segments_same_video(self, tmp_path):
        """Multiple segments from the same YouTube video are preserved."""
        tsv_path = tmp_path / "openasl.tsv"
        self._write_tsv(tsv_path, [
            ["s1", "yt1", 0.0, 5.0, "First"],
            ["s2", "yt1", 5.0, 10.0, "Second"],
            ["s3", "yt1", 10.0, 15.0, "Third"],
        ])

        manifest_path = tmp_path / "manifest.csv"
        cfg = Config(
            dataset={
                "name": "openasl",
                "source": {"manifest_tsv": str(tsv_path)},
            },
            paths={"manifest": str(manifest_path)},
        )
        adapter = OpenASLDataset()
        context = self._make_context(cfg)
        context = adapter.build_manifest(cfg, context)

        assert len(context.manifest_df) == 3
        assert context.stats["dataset.manifest"]["videos"] == 1
        assert context.stats["dataset.manifest"]["segments"] == 3


# ── Bounding-box merging ─────────────────────────────────────────────────

class TestOpenASLBboxMerge:
    def _make_context(self, config):
        adapter = OpenASLDataset()
        return PipelineContext(config=config, dataset=adapter)

    def test_bbox_merged_from_json(self, tmp_path):
        tsv_path = tmp_path / "openasl.tsv"
        pd.DataFrame({
            "vid": ["s1", "s2"], "yid": ["yt1", "yt1"],
            "start": [0.0, 5.0], "end": [5.0, 10.0],
            "en": ["A", "B"],
        }).to_csv(tsv_path, sep="\t", index=False)

        bbox_path = tmp_path / "bbox.json"
        bbox_path.write_text(json.dumps({
            "s1": [10, 20, 100, 200],
        }))

        manifest_path = tmp_path / "manifest.csv"
        cfg = Config(
            dataset={
                "name": "openasl",
                "source": {
                    "manifest_tsv": str(tsv_path),
                    "bbox_json": str(bbox_path),
                },
            },
            paths={"manifest": str(manifest_path)},
        )
        adapter = OpenASLDataset()
        context = self._make_context(cfg)
        context = adapter.build_manifest(cfg, context)

        df = context.manifest_df
        assert "BBOX_X1" in df.columns
        assert df.iloc[0]["BBOX_X1"] == 10.0
        assert df.iloc[0]["BBOX_Y2"] == 200.0
        assert df.iloc[0]["PERSON_DETECTED"] == True
        # s2 has no bbox
        assert pd.isna(df.iloc[1]["BBOX_X1"])
        assert df.iloc[1]["PERSON_DETECTED"] == False

    def test_bbox_dict_format(self, tmp_path):
        """Handle bbox JSON with nested dict format: {"vid": {"bbox": [...]}}."""
        tsv_path = tmp_path / "openasl.tsv"
        pd.DataFrame({
            "vid": ["s1"], "yid": ["yt1"],
            "start": [0.0], "end": [1.0],
            "en": ["Text"],
        }).to_csv(tsv_path, sep="\t", index=False)

        bbox_path = tmp_path / "bbox.json"
        bbox_path.write_text(json.dumps({
            "s1": {"bbox": [5, 10, 50, 80]},
        }))

        manifest_path = tmp_path / "manifest.csv"
        cfg = Config(
            dataset={
                "name": "openasl",
                "source": {
                    "manifest_tsv": str(tsv_path),
                    "bbox_json": str(bbox_path),
                },
            },
            paths={"manifest": str(manifest_path)},
        )
        adapter = OpenASLDataset()
        context = self._make_context(cfg)
        context = adapter.build_manifest(cfg, context)

        df = context.manifest_df
        assert df.iloc[0]["BBOX_X1"] == 5.0
        assert df.iloc[0]["BBOX_Y2"] == 80.0
        assert df.iloc[0]["PERSON_DETECTED"] == True

    def test_no_bbox_json_no_columns(self, tmp_path):
        """Without bbox_json, no BBOX columns are added."""
        tsv_path = tmp_path / "openasl.tsv"
        pd.DataFrame({
            "vid": ["s1"], "yid": ["yt1"],
            "start": [0.0], "end": [1.0],
            "en": ["Text"],
        }).to_csv(tsv_path, sep="\t", index=False)

        manifest_path = tmp_path / "manifest.csv"
        cfg = Config(
            dataset={
                "name": "openasl",
                "source": {"manifest_tsv": str(tsv_path)},
            },
            paths={"manifest": str(manifest_path)},
        )
        adapter = OpenASLDataset()
        context = self._make_context(cfg)
        context = adapter.build_manifest(cfg, context)

        assert "BBOX_X1" not in context.manifest_df.columns

    def test_nonexistent_bbox_json_ignored(self, tmp_path):
        """If bbox_json path doesn't exist, bboxes are silently skipped."""
        tsv_path = tmp_path / "openasl.tsv"
        pd.DataFrame({
            "vid": ["s1"], "yid": ["yt1"],
            "start": [0.0], "end": [1.0],
            "en": ["Text"],
        }).to_csv(tsv_path, sep="\t", index=False)

        manifest_path = tmp_path / "manifest.csv"
        cfg = Config(
            dataset={
                "name": "openasl",
                "source": {
                    "manifest_tsv": str(tsv_path),
                    "bbox_json": str(tmp_path / "nonexistent.json"),
                },
            },
            paths={"manifest": str(manifest_path)},
        )
        adapter = OpenASLDataset()
        context = self._make_context(cfg)
        context = adapter.build_manifest(cfg, context)

        assert "BBOX_X1" not in context.manifest_df.columns


# ── download ──────────────────────────────────────────────────────────────

class TestOpenASLDownload:
    def test_download_missing_tsv_raises(self, tmp_path):
        cfg = Config(
            dataset={
                "name": "openasl",
                "source": {"manifest_tsv": str(tmp_path / "nope.tsv")},
            },
            paths={"videos": str(tmp_path / "videos")},
        )
        adapter = OpenASLDataset()
        context = PipelineContext(config=cfg, dataset=adapter)
        with pytest.raises(FileNotFoundError, match="OpenASL manifest TSV"):
            adapter.download(cfg, context)

    def test_download_missing_yid_column_raises(self, tmp_path):
        tsv_path = tmp_path / "bad.tsv"
        pd.DataFrame({"vid": ["s1"], "start": [0.0]}).to_csv(
            tsv_path, sep="\t", index=False,
        )
        cfg = Config(
            dataset={
                "name": "openasl",
                "source": {"manifest_tsv": str(tsv_path)},
            },
            paths={"videos": str(tmp_path / "videos")},
        )
        adapter = OpenASLDataset()
        context = PipelineContext(config=cfg, dataset=adapter)
        with pytest.raises(ValueError, match="yid"):
            adapter.download(cfg, context)

    def test_download_all_downloaded_skips(self, tmp_path):
        """If all videos already exist, download is a no-op."""
        tsv_path = tmp_path / "openasl.tsv"
        pd.DataFrame({
            "vid": ["s1", "s2"], "yid": ["yt1", "yt1"],
            "start": [0.0, 5.0], "end": [5.0, 10.0],
        }).to_csv(tsv_path, sep="\t", index=False)

        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "yt1.mp4").touch()

        cfg = Config(
            dataset={
                "name": "openasl",
                "source": {"manifest_tsv": str(tsv_path)},
            },
            paths={"videos": str(video_dir)},
        )
        adapter = OpenASLDataset()
        context = PipelineContext(config=cfg, dataset=adapter)
        context = adapter.download(cfg, context)
        assert context.stats["dataset.download"]["downloaded"] == 0
        assert context.stats["dataset.download"]["total"] == 1  # 1 unique yid

    def test_download_skips_webm_extension(self, tmp_path, monkeypatch):
        """[P1] Extension-agnostic skip: .webm files are recognized."""
        tsv_path = tmp_path / "openasl.tsv"
        pd.DataFrame({
            "vid": ["s1", "s2"], "yid": ["yt1", "yt2"],
            "start": [0.0, 1.0], "end": [1.0, 2.0],
        }).to_csv(tsv_path, sep="\t", index=False)

        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "yt1.webm").touch()
        (video_dir / "unrelated.mp4").touch()
        monkeypatch.setattr(
            openasl_source,
            "download_youtube_videos",
            lambda video_ids, *_args, **_kwargs: {
                "downloaded": len(video_ids), "errors": 0, "missing": [],
            },
        )

        cfg = Config(
            dataset={
                "name": "openasl",
                "source": {"manifest_tsv": str(tsv_path)},
            },
            paths={"videos": str(video_dir)},
        )
        adapter = OpenASLDataset()
        context = PipelineContext(config=cfg, dataset=adapter)
        context = adapter.download(cfg, context)
        assert context.stats["dataset.download"] == {
            "total": 2, "downloaded": 1, "errors": 0, "skipped": 1,
        }

    def test_download_writes_report(self, tmp_path):
        """[P2] Download writes download_report.json even on skip."""
        tsv_path = tmp_path / "openasl.tsv"
        pd.DataFrame({
            "vid": ["s1"], "yid": ["yt1"],
            "start": [0.0], "end": [1.0],
        }).to_csv(tsv_path, sep="\t", index=False)

        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "yt1.mp4").touch()

        root_dir = tmp_path / "root"
        root_dir.mkdir()

        cfg = Config(
            dataset={
                "name": "openasl",
                "source": {"manifest_tsv": str(tsv_path)},
            },
            paths={"videos": str(video_dir), "root": str(root_dir)},
        )
        adapter = OpenASLDataset()
        context = PipelineContext(config=cfg, dataset=adapter)
        adapter.download(cfg, context)

        report_path = root_dir / "acquire_report" / "download_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert report["total"] == 1
        assert report["errors"] == 0

    def test_download_missing_videos_path_raises(self, tmp_path):
        tsv_path = tmp_path / "openasl.tsv"
        pd.DataFrame({
            "vid": ["s1"], "yid": ["yt1"],
            "start": [0.0], "end": [1.0],
        }).to_csv(tsv_path, sep="\t", index=False)

        cfg = Config(
            dataset={
                "name": "openasl",
                "source": {"manifest_tsv": str(tsv_path)},
            },
            paths={"videos": ""},
        )
        adapter = OpenASLDataset()
        context = PipelineContext(config=cfg, dataset=adapter)
        with pytest.raises(ValueError, match="paths.videos"):
            adapter.download(cfg, context)


# ── [P1] Extension-agnostic video ID scanning ─────────────────────────────

class TestExtensionAgnosticScan:
    def test_mp4_detected(self, tmp_path):
        (tmp_path / "abc.mp4").touch()
        assert "abc" in get_existing_video_ids(str(tmp_path))

    def test_webm_detected(self, tmp_path):
        (tmp_path / "abc.webm").touch()
        assert "abc" in get_existing_video_ids(str(tmp_path))

    def test_mkv_detected(self, tmp_path):
        (tmp_path / "abc.mkv").touch()
        assert "abc" in get_existing_video_ids(str(tmp_path))

    def test_non_video_ignored(self, tmp_path):
        (tmp_path / "abc.txt").touch()
        assert "abc" not in get_existing_video_ids(str(tmp_path))

    def test_multiple_extensions_same_id(self, tmp_path):
        """Same stem with multiple extensions counts as one ID."""
        (tmp_path / "abc.mp4").touch()
        (tmp_path / "abc.webm").touch()
        ids = get_existing_video_ids(str(tmp_path))
        assert ids == {"abc"}

    def test_empty_directory(self, tmp_path):
        assert get_existing_video_ids(str(tmp_path)) == set()


# ── [P2] Availability policy in build_manifest ─────────────────────────────

class TestAvailabilityPolicy:
    def _make_context(self, config):
        adapter = OpenASLDataset()
        return PipelineContext(config=config, dataset=adapter)

    def _write_tsv(self, path, data, columns=None):
        if columns is None:
            columns = ["vid", "yid", "start", "end", "en"]
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(path, sep="\t", index=False)

    def test_drop_unavailable_filters_rows(self, tmp_path):
        """drop_unavailable removes rows whose VIDEO_ID has no file."""
        tsv_path = tmp_path / "openasl.tsv"
        self._write_tsv(tsv_path, [
            ["s1", "yt1", 0.0, 5.0, "Present"],
            ["s2", "yt2", 0.0, 3.0, "Missing"],
        ])

        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "yt1.mp4").touch()  # yt2 is missing

        manifest_path = tmp_path / "manifest.csv"
        cfg = Config(
            dataset={
                "name": "openasl",
                "source": {
                    "manifest_tsv": str(tsv_path),
                    "availability_policy": "drop_unavailable",
                },
            },
            paths={
                "manifest": str(manifest_path),
                "videos": str(video_dir),
            },
        )
        adapter = OpenASLDataset()
        context = self._make_context(cfg)
        context = adapter.build_manifest(cfg, context)

        df = context.manifest_df
        assert len(df) == 1
        assert df.iloc[0]["VIDEO_ID"] == "yt1"
        assert "AVAILABLE" not in df.columns

    def test_mark_unavailable_adds_column(self, tmp_path):
        """mark_unavailable keeps all rows and adds AVAILABLE column."""
        tsv_path = tmp_path / "openasl.tsv"
        self._write_tsv(tsv_path, [
            ["s1", "yt1", 0.0, 5.0, "Present"],
            ["s2", "yt2", 0.0, 3.0, "Missing"],
        ])

        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "yt1.mp4").touch()

        manifest_path = tmp_path / "manifest.csv"
        cfg = Config(
            dataset={
                "name": "openasl",
                "source": {
                    "manifest_tsv": str(tsv_path),
                    "availability_policy": "mark_unavailable",
                },
            },
            paths={
                "manifest": str(manifest_path),
                "videos": str(video_dir),
            },
        )
        adapter = OpenASLDataset()
        context = self._make_context(cfg)
        context = adapter.build_manifest(cfg, context)

        df = context.manifest_df
        assert len(df) == 2
        assert "AVAILABLE" in df.columns
        assert df.iloc[0]["AVAILABLE"] == True
        assert df.iloc[1]["AVAILABLE"] == False

    def test_fail_fast_raises_on_missing(self, tmp_path):
        """fail_fast raises RuntimeError when videos are missing."""
        tsv_path = tmp_path / "openasl.tsv"
        self._write_tsv(tsv_path, [
            ["s1", "yt1", 0.0, 5.0, "Present"],
            ["s2", "yt2", 0.0, 3.0, "Missing"],
        ])

        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "yt1.mp4").touch()

        manifest_path = tmp_path / "manifest.csv"
        cfg = Config(
            dataset={
                "name": "openasl",
                "source": {
                    "manifest_tsv": str(tsv_path),
                    "availability_policy": "fail_fast",
                },
            },
            paths={
                "manifest": str(manifest_path),
                "videos": str(video_dir),
            },
        )
        adapter = OpenASLDataset()
        context = self._make_context(cfg)
        with pytest.raises(RuntimeError, match="not found"):
            adapter.build_manifest(cfg, context)

    def test_all_available_no_filter(self, tmp_path):
        """When all videos exist, no rows are dropped regardless of policy."""
        tsv_path = tmp_path / "openasl.tsv"
        self._write_tsv(tsv_path, [
            ["s1", "yt1", 0.0, 5.0, "A"],
            ["s2", "yt2", 0.0, 3.0, "B"],
        ])

        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "yt1.mp4").touch()
        (video_dir / "yt2.mp4").touch()

        manifest_path = tmp_path / "manifest.csv"
        cfg = Config(
            dataset={
                "name": "openasl",
                "source": {
                    "manifest_tsv": str(tsv_path),
                    "availability_policy": "drop_unavailable",
                },
            },
            paths={
                "manifest": str(manifest_path),
                "videos": str(video_dir),
            },
        )
        adapter = OpenASLDataset()
        context = self._make_context(cfg)
        context = adapter.build_manifest(cfg, context)

        assert len(context.manifest_df) == 2

    def test_no_video_dir_skips_policy(self, tmp_path):
        """If paths.videos is empty, availability policy is skipped."""
        tsv_path = tmp_path / "openasl.tsv"
        self._write_tsv(tsv_path, [
            ["s1", "yt1", 0.0, 5.0, "A"],
        ])

        manifest_path = tmp_path / "manifest.csv"
        cfg = Config(
            dataset={
                "name": "openasl",
                "source": {
                    "manifest_tsv": str(tsv_path),
                    "availability_policy": "fail_fast",
                },
            },
            paths={"manifest": str(manifest_path), "videos": ""},
        )
        adapter = OpenASLDataset()
        context = self._make_context(cfg)
        # Should NOT raise even with fail_fast — no video_dir to check
        context = adapter.build_manifest(cfg, context)
        assert len(context.manifest_df) == 1

    def test_availability_policy_config_values(self):
        """All three policy values are accepted by the source config."""
        for policy in ("fail_fast", "drop_unavailable", "mark_unavailable"):
            source = OpenASLSourceConfig(
                manifest_tsv="data.tsv",
                availability_policy=policy,
            )
            assert source.availability_policy == policy
