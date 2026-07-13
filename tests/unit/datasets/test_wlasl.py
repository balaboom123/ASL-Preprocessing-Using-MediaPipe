"""Tests for the WLASL dataset package."""

import json

import pandas as pd
import pytest

from signdata.config.schema import Config
from signdata.datasets.wlasl import WLASLDataset, WLASLSourceConfig
from signdata.datasets.wlasl import manifest as wlasl_manifest
from signdata.datasets.wlasl import source as wlasl_source
from signdata.pipeline.context import PipelineContext
from signdata.registry import DATASET_REGISTRY


def _write_metadata(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_config(metadata_path, video_dir, manifest_path, source=None, root_dir=None):
    dataset_source = {"metadata_json": str(metadata_path)}
    if source:
        dataset_source.update(source)

    paths = {
        "videos": str(video_dir),
        "manifest": str(manifest_path),
    }
    if root_dir is not None:
        paths["root"] = str(root_dir)

    return Config(
        dataset={
            "name": "wlasl",
            "source": dataset_source,
        },
        paths=paths,
    )


class TestWLASLRegistration:
    def test_registered(self):
        assert "wlasl" in DATASET_REGISTRY

    def test_instance_has_name(self):
        assert WLASLDataset().name == "wlasl"


class TestWLASLValidateConfig:
    def test_valid_config_passes(self, tmp_path):
        metadata_path = tmp_path / "WLASL_v0.3.json"
        _write_metadata(metadata_path, [])
        cfg = Config(
            dataset={
                "name": "wlasl",
                "source": {"metadata_json": str(metadata_path)},
            },
        )
        WLASLDataset.validate_config(cfg)

    def test_missing_metadata_json_raises(self):
        cfg = Config(dataset={"name": "wlasl"})
        with pytest.raises(ValueError, match="metadata_json"):
            WLASLDataset.validate_config(cfg)


class TestWLASLSourceConfig:
    def test_defaults(self, tmp_path):
        metadata_path = tmp_path / "WLASL_v0.3.json"
        _write_metadata(metadata_path, [])
        cfg = Config(
            dataset={
                "name": "wlasl",
                "source": {"metadata_json": str(metadata_path)},
            },
        )

        source = WLASLDataset().get_source_config(cfg)
        assert isinstance(source, WLASLSourceConfig)
        assert source.metadata_json == str(metadata_path)
        assert source.split == "all"
        assert source.subset == 0
        assert source.download_mode == "validate"
        assert source.availability_policy == "drop_unavailable"

    def test_custom_options(self, tmp_path):
        metadata_path = tmp_path / "WLASL_v0.3.json"
        _write_metadata(metadata_path, [])
        cfg = Config(
            dataset={
                "name": "wlasl",
                "source": {
                    "metadata_json": str(metadata_path),
                    "split": "train",
                    "subset": 100,
                    "download_mode": "validate",
                    "availability_policy": "mark_unavailable",
                },
            },
        )

        source = WLASLDataset().get_source_config(cfg)
        assert source.split == "train"
        assert source.subset == 100
        assert source.download_mode == "validate"
        assert source.availability_policy == "mark_unavailable"


class TestWLASLDownload:
    def test_download_validates_inputs(self, tmp_path):
        metadata_path = tmp_path / "WLASL_v0.3.json"
        _write_metadata(metadata_path, [])
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "00001.mp4").touch()
        manifest_path = tmp_path / "manifest.tsv"

        cfg = _make_config(metadata_path, video_dir, manifest_path)
        adapter = WLASLDataset()
        context = PipelineContext(config=cfg, dataset=adapter)

        context = adapter.download(cfg, context)
        assert context.stats["dataset.download"]["validated"] is True
        assert context.stats["dataset.download"]["videos_on_disk"] == 1

    def test_download_missing_metadata_raises(self, tmp_path):
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        manifest_path = tmp_path / "manifest.tsv"
        cfg = _make_config(tmp_path / "missing.json", video_dir, manifest_path)

        adapter = WLASLDataset()
        context = PipelineContext(config=cfg, dataset=adapter)
        with pytest.raises(FileNotFoundError, match="metadata JSON"):
            adapter.build_manifest(cfg, context)

    def test_download_missing_video_dir_raises(self, tmp_path):
        metadata_path = tmp_path / "WLASL_v0.3.json"
        _write_metadata(metadata_path, [])
        manifest_path = tmp_path / "manifest.tsv"
        cfg = _make_config(
            metadata_path,
            tmp_path / "missing-videos",
            manifest_path,
        )

        adapter = WLASLDataset()
        context = PipelineContext(config=cfg, dataset=adapter)
        with pytest.raises(FileNotFoundError, match="video directory"):
            adapter.download(cfg, context)

    def test_unknown_download_mode_raises(self, tmp_path):
        metadata_path = tmp_path / "WLASL_v0.3.json"
        _write_metadata(metadata_path, [])
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        manifest_path = tmp_path / "manifest.tsv"
        cfg = _make_config(
            metadata_path,
            video_dir,
            manifest_path,
            source={"download_mode": "unknown"},
        )

        adapter = WLASLDataset()
        context = PipelineContext(config=cfg, dataset=adapter)
        with pytest.raises(ValueError, match="download_mode"):
            adapter.download(cfg, context)

    def test_download_missing_uses_instance_urls(self, tmp_path, monkeypatch):
        metadata_path = tmp_path / "WLASL_v0.3.json"
        _write_metadata(metadata_path, [
            {
                "gloss": "book",
                "instances": [
                    {
                        "video_id": "00001",
                        "url": "https://example.com/book.mp4",
                    },
                    {
                        "video_id": "00002",
                        "url": "https://example.com/drink.mp4",
                    },
                ],
            },
        ])

        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "00001.mp4").touch()
        manifest_path = tmp_path / "manifest.tsv"
        root_dir = tmp_path / "root"
        root_dir.mkdir()

        captured = {}

        def fake_download_video_urls(video_urls, video_dir_arg, **kwargs):
            captured["video_urls"] = video_urls
            captured["video_dir"] = video_dir_arg
            captured["kwargs"] = kwargs
            return {"downloaded": 1, "errors": 0, "missing": []}

        monkeypatch.setattr(wlasl_source, "download_video_urls", fake_download_video_urls)

        cfg = _make_config(
            metadata_path,
            video_dir,
            manifest_path,
            source={"download_mode": "download_missing"},
            root_dir=root_dir,
        )

        adapter = WLASLDataset()
        context = PipelineContext(config=cfg, dataset=adapter)
        context = adapter.download(cfg, context)

        assert captured["video_urls"] == {
            "00002": "https://example.com/drink.mp4",
        }
        assert captured["video_dir"] == str(video_dir)
        assert context.stats["dataset.download"] == {
            "total": 2,
            "downloaded": 1,
            "errors": 0,
            "skipped": 1,
        }

    def test_download_missing_applies_split_and_subset_filters(self, tmp_path, monkeypatch):
        metadata_path = tmp_path / "WLASL_v0.3.json"
        _write_metadata(metadata_path, [
            {
                "gloss": "book",
                "instances": [
                    {
                        "video_id": "00001",
                        "split": "train",
                        "url": "https://example.com/book-train.mp4",
                    },
                    {
                        "video_id": "00002",
                        "split": "val",
                        "url": "https://example.com/book-val.mp4",
                    },
                ],
            },
            {
                "gloss": "drink",
                "instances": [
                    {
                        "video_id": "00003",
                        "split": "train",
                        "url": "https://example.com/drink-train.mp4",
                    },
                ],
            },
            {
                "gloss": "sleep",
                "instances": [
                    {
                        "video_id": "00004",
                        "split": "train",
                        "url": "https://example.com/sleep-train.mp4",
                    },
                ],
            },
        ])

        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "00001.mp4").touch()
        manifest_path = tmp_path / "manifest.tsv"
        root_dir = tmp_path / "root"
        root_dir.mkdir()

        captured = {}

        def fake_download_video_urls(video_urls, video_dir_arg, **kwargs):
            captured["video_urls"] = video_urls
            captured["video_dir"] = video_dir_arg
            captured["kwargs"] = kwargs
            return {"downloaded": 1, "errors": 0, "missing": []}

        monkeypatch.setattr(wlasl_source, "download_video_urls", fake_download_video_urls)

        cfg = _make_config(
            metadata_path,
            video_dir,
            manifest_path,
            source={
                "download_mode": "download_missing",
                "split": "train",
                "subset": 2,
            },
            root_dir=root_dir,
        )

        adapter = WLASLDataset()
        context = PipelineContext(config=cfg, dataset=adapter)
        context = adapter.download(cfg, context)

        assert captured["video_urls"] == {
            "00003": "https://example.com/drink-train.mp4",
        }
        assert captured["video_dir"] == str(video_dir)
        assert context.stats["dataset.download"] == {
            "total": 2,
            "downloaded": 1,
            "errors": 0,
            "skipped": 1,
        }


class TestWLASLBuildManifest:
    def _make_context(self, config):
        adapter = WLASLDataset()
        return PipelineContext(config=config, dataset=adapter)

    def test_build_manifest_from_metadata(self, tmp_path):
        metadata_path = tmp_path / "WLASL_v0.3.json"
        _write_metadata(metadata_path, [
            {
                "gloss": "book",
                "instances": [
                    {
                        "video_id": "00001",
                        "split": "train",
                        "fps": 25,
                        "frame_start": 0,
                        "frame_end": 50,
                        "signer_id": 3,
                        "variation_id": 0,
                        "source": "aslbrick",
                        "url": "https://example.com/book.mp4",
                        "bbox": [10, 20, 30, 40],
                    },
                    {
                        "video_id": "00002",
                        "split": "val",
                        "fps": 25,
                        "frame_start": 0,
                        "frame_end": 25,
                        "signer_id": 4,
                        "variation_id": 1,
                        "source": "aslbrick",
                        "url": "https://example.com/book-val.mp4",
                    },
                ],
            },
            {
                "gloss": "drink",
                "instances": [
                    {
                        "video_id": "00003",
                        "split": "train",
                        "fps": 25,
                        "frame_start": 10,
                        "frame_end": 35,
                        "signer_id": 9,
                        "variation_id": 2,
                        "source": "signschool",
                        "url": "https://example.com/drink.mp4",
                    },
                ],
            },
        ])

        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "00001.mp4").touch()
        (video_dir / "00003.mp4").touch()
        manifest_path = tmp_path / "manifest.tsv"

        cfg = _make_config(
            metadata_path,
            video_dir,
            manifest_path,
            source={"split": "train"},
        )
        context = self._make_context(cfg)

        context = WLASLDataset().build_manifest(cfg, context)

        assert context.manifest_path == manifest_path
        assert len(context.manifest_df) == 2
        assert list(context.manifest_df["VIDEO_ID"]) == ["00001", "00003"]
        assert list(context.manifest_df["REL_PATH"]) == ["00001.mp4", "00003.mp4"]
        assert list(context.manifest_df["GLOSS"]) == ["book", "drink"]
        assert list(context.manifest_df["CLASS_ID"]) == [0, 1]
        assert list(context.manifest_df["SPLIT"]) == ["train", "train"]
        assert list(context.manifest_df["START"]) == [0.0, 0.0]
        assert list(context.manifest_df["END"]) == [2.0, 1.0]
        assert list(context.manifest_df["VARIATION_ID"]) == [0, 2]
        assert list(context.manifest_df["SOURCE"]) == ["aslbrick", "signschool"]
        assert list(context.manifest_df["FRAME_START"]) == [0, 10]
        assert list(context.manifest_df["FRAME_END"]) == [50, 35]
        assert context.manifest_df.iloc[0]["BBOX_X1"] == 10.0
        assert context.manifest_df.iloc[0]["PERSON_DETECTED"] == True
        assert pd.isna(context.manifest_df.iloc[1]["BBOX_X1"])
        assert context.manifest_df.iloc[1]["PERSON_DETECTED"] == False
        assert context.stats["dataset.manifest"] == {"videos": 2, "segments": 2}

        reloaded = pd.read_csv(manifest_path, delimiter="\t")
        assert len(reloaded) == 2
        assert "REL_PATH" in reloaded.columns
        assert "VARIATION_ID" in reloaded.columns

    def test_build_manifest_marks_unavailable_rows(self, tmp_path):
        metadata_path = tmp_path / "WLASL_v0.3.json"
        _write_metadata(metadata_path, [
            {
                "gloss": "book",
                "instances": [
                    {
                        "video_id": "00001",
                        "split": "train",
                        "fps": 25,
                        "frame_start": 0,
                        "frame_end": 50,
                    },
                    {
                        "video_id": "00002",
                        "split": "train",
                        "fps": 25,
                        "frame_start": 0,
                        "frame_end": 25,
                    },
                ],
            },
        ])

        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "00001.mp4").touch()
        manifest_path = tmp_path / "manifest.tsv"
        cfg = _make_config(
            metadata_path,
            video_dir,
            manifest_path,
            source={"availability_policy": "mark_unavailable"},
        )

        context = self._make_context(cfg)
        context = WLASLDataset().build_manifest(cfg, context)

        assert len(context.manifest_df) == 2
        assert list(context.manifest_df["AVAILABLE"]) == [True, False]

    def test_build_manifest_applies_subset(self, tmp_path):
        metadata_path = tmp_path / "WLASL_v0.3.json"
        _write_metadata(metadata_path, [
            {
                "gloss": "book",
                "instances": [
                    {
                        "video_id": "00001",
                        "split": "train",
                        "fps": 25,
                        "frame_start": 0,
                        "frame_end": 50,
                    },
                ],
            },
            {
                "gloss": "drink",
                "instances": [
                    {
                        "video_id": "00002",
                        "split": "train",
                        "fps": 25,
                        "frame_start": 0,
                        "frame_end": 25,
                    },
                ],
            },
        ])

        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "00001.mp4").touch()
        (video_dir / "00002.mp4").touch()
        manifest_path = tmp_path / "manifest.tsv"
        cfg = _make_config(
            metadata_path,
            video_dir,
            manifest_path,
            source={"subset": 1},
        )

        context = self._make_context(cfg)
        context = WLASLDataset().build_manifest(cfg, context)

        assert list(context.manifest_df["VIDEO_ID"]) == ["00001"]
        assert list(context.manifest_df["CLASS_ID"]) == [0]

    def test_build_manifest_falls_back_to_clip_duration(self, tmp_path, monkeypatch):
        metadata_path = tmp_path / "WLASL_v0.3.json"
        _write_metadata(metadata_path, [
            {
                "gloss": "book",
                "instances": [
                    {
                        "video_id": "00001",
                        "split": "train",
                        "fps": 25,
                        "frame_start": 0,
                        "frame_end": -1,
                    },
                ],
            },
        ])

        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "00001.mp4").touch()
        manifest_path = tmp_path / "manifest.tsv"
        cfg = _make_config(metadata_path, video_dir, manifest_path)

        monkeypatch.setattr(wlasl_manifest, "get_video_duration", lambda _: 3.6)

        context = self._make_context(cfg)
        context = WLASLDataset().build_manifest(cfg, context)

        assert context.manifest_df.iloc[0]["START"] == 0.0
        assert context.manifest_df.iloc[0]["END"] == 3.6

    def test_build_manifest_uses_existing_video_extension_for_rel_path(self, tmp_path, monkeypatch):
        metadata_path = tmp_path / "WLASL_v0.3.json"
        _write_metadata(metadata_path, [
            {
                "gloss": "book",
                "instances": [
                    {
                        "video_id": "00001",
                        "split": "train",
                        "fps": 25,
                        "frame_start": 0,
                        "frame_end": 50,
                    },
                ],
            },
        ])

        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "00001.webm").touch()
        manifest_path = tmp_path / "manifest.tsv"
        cfg = _make_config(metadata_path, video_dir, manifest_path)

        captured = {}

        def fake_get_video_duration(path):
            captured["path"] = path
            return 3.6

        monkeypatch.setattr(wlasl_manifest, "get_video_duration", fake_get_video_duration)

        context = self._make_context(cfg)
        context = WLASLDataset().build_manifest(cfg, context)

        assert context.manifest_df.iloc[0]["REL_PATH"] == "00001.webm"
        assert context.manifest_df.iloc[0]["START"] == 0.0
        assert context.manifest_df.iloc[0]["END"] == 3.6
        assert captured["path"].endswith("00001.webm")

    def test_build_manifest_download_missing_uses_source_timing(self, tmp_path):
        metadata_path = tmp_path / "WLASL_v0.3.json"
        _write_metadata(metadata_path, [
            {
                "gloss": "drink",
                "instances": [
                    {
                        "video_id": "00003",
                        "split": "train",
                        "fps": 25,
                        "frame_start": 10,
                        "frame_end": 35,
                    },
                ],
            },
        ])

        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "00003.mp4").touch()
        manifest_path = tmp_path / "manifest.tsv"
        cfg = _make_config(
            metadata_path,
            video_dir,
            manifest_path,
            source={"download_mode": "download_missing"},
        )

        context = self._make_context(cfg)
        context = WLASLDataset().build_manifest(cfg, context)

        assert context.manifest_df.iloc[0]["START"] == 0.4
        assert context.manifest_df.iloc[0]["END"] == 1.4
