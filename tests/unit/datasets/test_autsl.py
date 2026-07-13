"""Tests for the AUTSL dataset package."""

from pathlib import Path

import pandas as pd
import pytest

from signdata.config.schema import Config
from signdata.datasets.autsl import AUTSLDataset, AUTSLSourceConfig
from signdata.datasets.autsl import manifest as autsl_manifest
from signdata.pipeline.context import PipelineContext
from signdata.registry import DATASET_REGISTRY


def _touch_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _write_class_map(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "0,merhaba,hello\n"
        "1,tesekkur,thanks\n",
        encoding="utf-8",
    )


def _write_labels(path: Path, rows: list[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{sample_key},{label_id}\n" for sample_key, label_id in rows),
        encoding="utf-8",
    )


def _make_config(
    release_dir,
    manifest_path,
    *,
    source=None,
    paths=None,
):
    dataset_source = {"release_dir": str(release_dir)}
    if source:
        dataset_source.update(source)

    config_paths = {"manifest": str(manifest_path)}
    if paths:
        config_paths.update(paths)

    return Config(
        dataset={
            "name": "autsl",
            "source": dataset_source,
        },
        paths=config_paths,
    )


def _write_release(root: Path) -> None:
    _touch_video(root / "train" / "signer0_sample1_color.mp4")
    _touch_video(root / "validation" / "signer1_sample2_color.mp4")
    _touch_video(root / "test" / "signer2_sample3_color.mp4")
    _write_class_map(root / "SignList_ClassId_TR_EN.csv")
    _write_labels(root / "train_labels.csv", [("signer0_sample1", 0)])
    _write_labels(root / "val_labels.csv", [("signer1_sample2", 1)])


class TestAUTSLRegistration:
    def test_registered(self):
        assert "autsl" in DATASET_REGISTRY

    def test_instance_has_name(self):
        assert AUTSLDataset().name == "autsl"


class TestAUTSLValidateConfig:
    def test_valid_config_passes_with_release_dir(self, tmp_path):
        cfg = Config(
            dataset={
                "name": "autsl",
                "source": {"release_dir": str(tmp_path)},
            },
        )

        AUTSLDataset.validate_config(cfg)

    def test_valid_config_passes_with_paths_videos(self, tmp_path):
        cfg = Config(
            dataset={"name": "autsl"},
            paths={"videos": str(tmp_path)},
        )

        AUTSLDataset.validate_config(cfg)

    def test_missing_release_dir_and_paths_videos_raises(self):
        cfg = Config(dataset={"name": "autsl"}, paths={"videos": ""})

        with pytest.raises(ValueError, match="release_dir|paths\\.videos"):
            AUTSLDataset.validate_config(cfg)

    def test_invalid_modality_raises(self, tmp_path):
        cfg = Config(
            dataset={
                "name": "autsl",
                "source": {
                    "release_dir": str(tmp_path),
                    "modality": "skeleton",
                },
            },
        )

        with pytest.raises(ValueError, match="modality"):
            AUTSLDataset.validate_config(cfg)

    def test_invalid_split_raises(self, tmp_path):
        cfg = Config(
            dataset={
                "name": "autsl",
                "source": {
                    "release_dir": str(tmp_path),
                    "split": "trainval",
                },
            },
        )

        with pytest.raises(ValueError, match="split"):
            AUTSLDataset.validate_config(cfg)


class TestAUTSLSourceConfig:
    def test_defaults(self, tmp_path):
        cfg = Config(
            dataset={
                "name": "autsl",
                "source": {"release_dir": str(tmp_path)},
            },
        )

        source = AUTSLDataset().get_source_config(cfg)
        assert isinstance(source, AUTSLSourceConfig)
        assert source.release_dir == str(tmp_path)
        assert source.split == "train"
        assert source.modality == "rgb"
        assert source.availability_policy == "fail_fast"
        assert source.allow_unlabeled is False

    def test_paths_videos_fallback_populates_release_dir(self, tmp_path):
        cfg = Config(
            dataset={"name": "autsl"},
            paths={"videos": str(tmp_path)},
        )

        source = AUTSLDataset().get_source_config(cfg)
        assert source.release_dir == str(tmp_path)


class TestAUTSLDownload:
    def test_download_validates_release_and_sets_runtime_video_dir(self, tmp_path):
        release_root = tmp_path / "autsl"
        _write_release(release_root)
        manifest_path = tmp_path / "manifest.tsv"

        cfg = _make_config(
            release_root,
            manifest_path,
            paths={"videos": str(tmp_path / "unused-videos-dir")},
        )

        adapter = AUTSLDataset()
        context = PipelineContext(config=cfg, dataset=adapter)
        context.resolve_paths()

        context = adapter.download(cfg, context)

        assert context.stats["dataset.download"]["validated"] is True
        assert context.stats["dataset.download"]["release_root"] == str(release_root)
        assert context.stats["dataset.download"]["class_id_file"].endswith(
            "SignList_ClassId_TR_EN.csv"
        )
        assert context.videos_dir == release_root


class TestAUTSLBuildManifest:
    def _make_context(self, config):
        adapter = AUTSLDataset()
        context = PipelineContext(config=config, dataset=adapter)
        context.resolve_paths()
        return context

    def test_build_manifest_for_rgb_labeled_splits(self, tmp_path, monkeypatch):
        release_root = tmp_path / "autsl"
        _write_release(release_root)
        manifest_path = tmp_path / "manifest.tsv"

        monkeypatch.setattr(autsl_manifest, "get_video_duration", lambda _: 1.5)
        monkeypatch.setattr(autsl_manifest, "get_video_fps", lambda _: 25.0)

        cfg = _make_config(
            release_root,
            manifest_path,
            source={"split": "all", "allow_unlabeled": False},
            paths={"videos": str(tmp_path / "unused-videos-dir")},
        )

        context = self._make_context(cfg)
        context = AUTSLDataset().build_manifest(cfg, context)

        df = context.manifest_df
        assert list(df["VIDEO_ID"]) == [
            "signer0_sample1_color",
            "signer1_sample2_color",
        ]
        assert list(df["REL_PATH"]) == [
            "train/signer0_sample1_color.mp4",
            "validation/signer1_sample2_color.mp4",
        ]
        assert list(df["CLASS_ID"]) == [0, 1]
        assert list(df["GLOSS"]) == ["merhaba", "tesekkur"]
        assert list(df["TEXT"]) == ["hello", "thanks"]
        assert list(df["SIGNER_ID"]) == ["0", "1"]
        assert list(df["FPS"]) == [25.0, 25.0]
        assert list(df["START"]) == [0.0, 0.0]
        assert list(df["END"]) == [1.5, 1.5]
        assert list(df["SPLIT"]) == ["train", "val"]
        assert context.manifest_path == manifest_path
        assert context.videos_dir == release_root
        assert context.stats["dataset.manifest"] == {
            "videos": 2,
            "segments": 2,
        }

    def test_build_manifest_includes_unlabeled_test_when_enabled(
        self,
        tmp_path,
        monkeypatch,
    ):
        release_root = tmp_path / "autsl"
        _write_release(release_root)
        manifest_path = tmp_path / "manifest.tsv"

        monkeypatch.setattr(autsl_manifest, "get_video_duration", lambda _: 2.0)
        monkeypatch.setattr(autsl_manifest, "get_video_fps", lambda _: 30.0)

        cfg = _make_config(
            release_root,
            manifest_path,
            source={"split": "all", "allow_unlabeled": True},
            paths={"videos": str(tmp_path / "unused-videos-dir")},
        )

        context = self._make_context(cfg)
        context = AUTSLDataset().build_manifest(cfg, context)

        df = context.manifest_df
        assert list(df["VIDEO_ID"]) == [
            "signer0_sample1_color",
            "signer1_sample2_color",
            "signer2_sample3_color",
        ]
        test_row = df[df["SPLIT"] == "test"].iloc[0]
        assert pd.isna(test_row["CLASS_ID"])
        assert test_row["GLOSS"] == ""
        assert test_row["TEXT"] == ""
        assert test_row["REL_PATH"] == "test/signer2_sample3_color.mp4"

    def test_build_manifest_respects_explicit_class_and_label_files(
        self,
        tmp_path,
        monkeypatch,
    ):
        release_root = tmp_path / "autsl"
        (release_root / "train").mkdir(parents=True)
        _touch_video(release_root / "train" / "signer0_sample1_color.mp4")
        manifest_path = tmp_path / "manifest.tsv"

        custom_class_map = tmp_path / "metadata" / "classes.csv"
        custom_train_labels = tmp_path / "labels" / "my_train_labels.csv"
        _write_class_map(custom_class_map)
        _write_labels(custom_train_labels, [("signer0_sample1", 1)])

        monkeypatch.setattr(autsl_manifest, "get_video_duration", lambda _: 3.0)
        monkeypatch.setattr(autsl_manifest, "get_video_fps", lambda _: 60.0)

        cfg = _make_config(
            release_root,
            manifest_path,
            source={
                "split": "train",
                "class_id_file": str(custom_class_map),
                "train_labels_file": str(custom_train_labels),
            },
            paths={"videos": str(tmp_path / "unused-videos-dir")},
        )

        context = self._make_context(cfg)
        context = AUTSLDataset().build_manifest(cfg, context)

        df = context.manifest_df
        assert list(df["CLASS_ID"]) == [1]
        assert list(df["GLOSS"]) == ["tesekkur"]
        assert list(df["TEXT"]) == ["thanks"]

    def test_build_manifest_rejects_missing_explicit_label_override(
        self,
        tmp_path,
    ):
        release_root = tmp_path / "autsl"
        (release_root / "train").mkdir(parents=True)
        _touch_video(release_root / "train" / "signer0_sample1_color.mp4")
        _write_class_map(release_root / "SignList_ClassId_TR_EN.csv")
        manifest_path = tmp_path / "manifest.tsv"

        cfg = _make_config(
            release_root,
            manifest_path,
            source={
                "split": "train",
                "train_labels_file": str(tmp_path / "missing_train_labels.csv"),
            },
            paths={"videos": str(tmp_path / "unused-videos-dir")},
        )

        context = self._make_context(cfg)
        with pytest.raises(FileNotFoundError, match="train_labels_file"):
            AUTSLDataset().build_manifest(cfg, context)

    def test_build_manifest_uses_depth_suffix_when_requested(self, tmp_path, monkeypatch):
        release_root = tmp_path / "autsl"
        (release_root / "train").mkdir(parents=True)
        _touch_video(release_root / "train" / "signer0_sample1_depth.mp4")
        _write_class_map(release_root / "SignList_ClassId_TR_EN.csv")
        _write_labels(release_root / "train_labels.csv", [("signer0_sample1", 0)])
        manifest_path = tmp_path / "manifest.tsv"

        monkeypatch.setattr(autsl_manifest, "get_video_duration", lambda _: 0.75)
        monkeypatch.setattr(autsl_manifest, "get_video_fps", lambda _: 15.0)

        cfg = _make_config(
            release_root,
            manifest_path,
            source={"modality": "depth"},
            paths={"videos": str(tmp_path / "unused-videos-dir")},
        )

        context = self._make_context(cfg)
        context = AUTSLDataset().build_manifest(cfg, context)

        df = context.manifest_df
        assert list(df["VIDEO_ID"]) == ["signer0_sample1_depth"]
        assert list(df["REL_PATH"]) == ["train/signer0_sample1_depth.mp4"]
