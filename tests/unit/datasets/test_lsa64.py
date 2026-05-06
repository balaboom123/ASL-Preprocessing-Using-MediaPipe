"""Tests for the LSA64 dataset package."""

import logging
from pathlib import Path

import pandas as pd
import pytest

from signdata.config.schema import Config
from signdata.datasets.lsa64 import LSA64Dataset, LSA64SourceConfig
from signdata.datasets.lsa64 import manifest as lsa64_manifest
from signdata.datasets.lsa64 import source as lsa64_source
from signdata.pipeline.context import PipelineContext
from signdata.registry import DATASET_REGISTRY


def _touch_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _write_class_map(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "CLASS_ID\tGLOSS\tHANDEDNESS\n"
        "1\tOpaque\tR\n"
        "2\tRed\tR\n"
        "3\tGreen\tR\n",
        encoding="utf-8",
    )


def _make_config(release_dir, manifest_path, source=None, paths=None):
    dataset_source = {"release_dir": str(release_dir)}
    if source:
        dataset_source.update(source)

    config_paths = {"manifest": str(manifest_path)}
    if paths:
        config_paths.update(paths)

    return Config(
        dataset={
            "name": "lsa64",
            "source": dataset_source,
        },
        paths=config_paths,
    )


class TestLSA64Registration:
    def test_registered(self):
        assert "lsa64" in DATASET_REGISTRY

    def test_instance_has_name(self):
        assert LSA64Dataset().name == "lsa64"


class TestLSA64ValidateConfig:
    def test_valid_config_passes_with_release_dir(self, tmp_path):
        cfg = Config(
            dataset={
                "name": "lsa64",
                "source": {"release_dir": str(tmp_path)},
            },
        )
        LSA64Dataset.validate_config(cfg)

    def test_valid_config_passes_with_paths_videos(self, tmp_path):
        cfg = Config(
            dataset={"name": "lsa64"},
            paths={"videos": str(tmp_path)},
        )
        LSA64Dataset.validate_config(cfg)

    def test_missing_release_dir_and_paths_videos_raises(self):
        cfg = Config(dataset={"name": "lsa64"})
        with pytest.raises(ValueError, match="release_dir"):
            LSA64Dataset.validate_config(cfg)

    def test_explicit_variant_dir_conflicting_with_variant_raises(self, tmp_path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        cfg = Config(
            dataset={
                "name": "lsa64",
                "source": {
                    "release_dir": str(raw_dir),
                    "variant": "cut",
                },
            },
        )
        with pytest.raises(ValueError, match="variant"):
            LSA64Dataset.validate_config(cfg)


class TestLSA64SourceConfig:
    def test_defaults(self, tmp_path):
        cfg = Config(
            dataset={
                "name": "lsa64",
                "source": {"release_dir": str(tmp_path)},
            },
        )

        source = LSA64Dataset().get_source_config(cfg)
        assert isinstance(source, LSA64SourceConfig)
        assert source.release_dir == str(tmp_path)
        assert source.variant == "cut"
        assert source.split == "all"
        assert source.split_strategy == "none"
        assert source.train_signers == [1, 2, 3, 4, 5, 6, 7, 8]
        assert source.val_signers == [9]
        assert source.test_signers == [10]
        assert source.availability_policy == "fail_fast"

    def test_resolve_video_dir_uses_variant_subdirectory(self, tmp_path):
        release_root = tmp_path / "release"
        cut_dir = release_root / "cut"
        raw_dir = release_root / "raw"
        _touch_video(cut_dir / "01_01_01.mp4")
        _touch_video(raw_dir / "01_01_01.mp4")

        cfg = _make_config(
            release_root,
            tmp_path / "manifest.tsv",
            source={"variant": "raw"},
            paths={"videos": str(release_root)},
        )
        source = LSA64Dataset().get_source_config(cfg)

        assert lsa64_source.resolve_video_dir(cfg, source) == raw_dir

    def test_resolve_video_dir_accepts_flat_directory(self, tmp_path):
        flat_dir = tmp_path / "flat"
        _touch_video(flat_dir / "01_01_01.mp4")

        cfg = _make_config(flat_dir, tmp_path / "manifest.tsv")
        source = LSA64Dataset().get_source_config(cfg)

        assert lsa64_source.resolve_video_dir(cfg, source) == flat_dir


class TestLSA64Download:
    def test_download_validates_release_and_sets_runtime_video_dir(self, tmp_path):
        release_root = tmp_path / "release"
        raw_dir = release_root / "raw"
        cut_dir = release_root / "cut"
        _touch_video(raw_dir / "01_01_01.mp4")
        _touch_video(cut_dir / "01_01_01.mp4")

        cfg = _make_config(
            release_root,
            tmp_path / "manifest.tsv",
            source={"variant": "raw"},
            paths={"videos": str(release_root)},
        )
        adapter = LSA64Dataset()
        context = PipelineContext(config=cfg, dataset=adapter)
        context.resolve_paths()

        context = adapter.download(cfg, context)

        assert context.stats["dataset.download"]["validated"] is True
        assert context.stats["dataset.download"]["variant"] == "raw"
        assert context.stats["dataset.download"]["video_dir"] == str(raw_dir)
        assert context.videos_dir == raw_dir


class TestLSA64ClassMap:
    def test_load_class_map_uses_bundled_asset(self):
        source = LSA64SourceConfig(release_dir="/tmp/lsa64")

        class_map = lsa64_source.load_lsa64_class_map(source, logging.getLogger(__name__))

        assert class_map is not None
        assert len(class_map) == 64
        first = class_map.iloc[0]
        assert int(first["CLASS_ID"]) == 1
        assert first["GLOSS"] == "Opaque"
        assert first["HANDEDNESS"] == "R"
        none_row = class_map[class_map["CLASS_ID"] == 38].iloc[0]
        assert none_row["GLOSS"] == "None"

    def test_allow_missing_class_map_returns_none(self, tmp_path, monkeypatch):
        missing_bundled = tmp_path / "missing.tsv"
        monkeypatch.setattr(lsa64_source, "_BUNDLED_CLASS_MAP", missing_bundled)

        source = LSA64SourceConfig(
            release_dir=str(tmp_path),
            class_map_file=str(tmp_path / "also-missing.tsv"),
            allow_missing_class_map=True,
        )

        class_map = lsa64_source.load_lsa64_class_map(
            source,
            logging.getLogger(__name__),
        )

        assert class_map is None


class TestLSA64BuildManifest:
    def _make_context(self, config):
        adapter = LSA64Dataset()
        return PipelineContext(config=config, dataset=adapter)

    def test_build_manifest_joins_labels_and_tracks_context(self, tmp_path, monkeypatch):
        release_root = tmp_path / "release"
        cut_dir = release_root / "cut"
        _touch_video(cut_dir / "01_01_01.mp4")
        _touch_video(cut_dir / "02_09_05.mp4")
        (cut_dir / "bad_name.mp4").touch()

        class_map_path = tmp_path / "lsa64_class_map.tsv"
        _write_class_map(class_map_path)
        manifest_path = tmp_path / "manifest.tsv"

        monkeypatch.setattr(lsa64_manifest, "get_video_duration", lambda _: 1.25)
        monkeypatch.setattr(lsa64_manifest, "get_video_fps", lambda _: 60.0)

        cfg = _make_config(
            release_root,
            manifest_path,
            source={
                "variant": "cut",
                "class_map_file": str(class_map_path),
            },
            paths={"videos": str(release_root)},
        )

        context = self._make_context(cfg)
        context = LSA64Dataset().build_manifest(cfg, context)

        df = context.manifest_df
        assert list(df["VIDEO_ID"]) == ["01_01_01", "02_09_05"]
        assert list(df["REL_PATH"]) == ["01_01_01.mp4", "02_09_05.mp4"]
        assert list(df["CLASS_ID"]) == [1, 2]
        assert list(df["GLOSS"]) == ["Opaque", "Red"]
        assert list(df["TEXT"]) == ["Opaque", "Red"]
        assert list(df["SPLIT"]) == ["all", "all"]
        assert list(df["SOURCE_VARIANT"]) == ["cut", "cut"]
        assert list(df["SIGNER_ID"]) == [1, 9]
        assert list(df["REPETITION_ID"]) == [1, 5]
        assert list(df["FPS"]) == [60.0, 60.0]
        assert list(df["START"]) == [0.0, 0.0]
        assert list(df["END"]) == [1.25, 1.25]
        assert list(df["HANDEDNESS"]) == ["R", "R"]
        assert context.manifest_path == manifest_path
        assert context.videos_dir == cut_dir
        assert context.stats["dataset.manifest"] == {
            "videos": 2,
            "segments": 2,
            "classes": 2,
            "signers": 2,
        }

    def test_build_manifest_applies_signer_split_and_filter(self, tmp_path, monkeypatch):
        release_root = tmp_path / "release"
        cut_dir = release_root / "cut"
        _touch_video(cut_dir / "01_01_01.mp4")
        _touch_video(cut_dir / "01_09_01.mp4")
        _touch_video(cut_dir / "01_10_01.mp4")

        class_map_path = tmp_path / "lsa64_class_map.tsv"
        _write_class_map(class_map_path)

        monkeypatch.setattr(lsa64_manifest, "get_video_duration", lambda _: 1.0)
        monkeypatch.setattr(lsa64_manifest, "get_video_fps", lambda _: 60.0)

        cfg = _make_config(
            release_root,
            tmp_path / "manifest.tsv",
            source={
                "class_map_file": str(class_map_path),
                "split_strategy": "community_signer_8_1_1",
                "split": "val",
            },
        )

        context = self._make_context(cfg)
        context = LSA64Dataset().build_manifest(cfg, context)

        df = context.manifest_df
        assert list(df["VIDEO_ID"]) == ["01_09_01"]
        assert list(df["SPLIT"]) == ["val"]


class TestLSA64LoadedManifestValidation:
    def test_accepts_legacy_manifest_when_sample_ids_match_variant(self, tmp_path):
        cfg = _make_config(
            tmp_path / "release",
            tmp_path / "manifest.tsv",
            source={"variant": "cut"},
        )
        context = PipelineContext(config=cfg, dataset=LSA64Dataset())
        context.manifest_path = tmp_path / "manifest.tsv"
        context.manifest_df = pd.DataFrame({
            "SAMPLE_ID": ["cut-01_01_01"],
            "VIDEO_ID": ["01_01_01"],
            "REL_PATH": ["01_01_01.mp4"],
        })

        LSA64Dataset().validate_loaded_manifest(cfg, context)

    def test_rejects_loaded_manifest_when_variant_differs(self, tmp_path):
        cfg = _make_config(
            tmp_path / "release",
            tmp_path / "manifest.tsv",
            source={"variant": "raw"},
        )
        context = PipelineContext(config=cfg, dataset=LSA64Dataset())
        context.manifest_path = tmp_path / "manifest.tsv"
        context.manifest_df = pd.DataFrame({
            "SAMPLE_ID": ["cut-01_01_01"],
            "VIDEO_ID": ["01_01_01"],
            "REL_PATH": ["01_01_01.mp4"],
        })

        with pytest.raises(ValueError, match="variant"):
            LSA64Dataset().validate_loaded_manifest(cfg, context)
