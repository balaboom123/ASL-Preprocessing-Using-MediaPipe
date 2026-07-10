"""Tests for the SLoVo dataset package."""

from pathlib import Path

import pandas as pd
import pytest

from signdata.config.loader import load_config
from signdata.config.schema import Config
from signdata.datasets.slovo import SlovoDataset, SlovoSourceConfig
from signdata.datasets.slovo import manifest as slovo_manifest
from signdata.datasets.slovo import source as slovo_source
from signdata.pipeline.context import PipelineContext
from signdata.registry import DATASET_REGISTRY


def _touch_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _write_annotations(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _make_release(root: Path) -> None:
    _write_annotations(
        root / "annotations.csv",
        [
            {
                "attachment_id": "clip_train",
                "user_id": "signer_a",
                "width": 1920,
                "height": 1080,
                "length": 48,
                "text": "hello",
                "train": True,
                "begin": 12,
                "end": 60,
            },
            {
                "attachment_id": "clip_test",
                "user_id": "signer_b",
                "width": 1280,
                "height": 720,
                "length": 32,
                "text": "thanks",
                "train": False,
                "begin": 7,
                "end": 39,
            },
        ],
    )
    _touch_video(root / "clip_train.mp4")
    _touch_video(root / "clip_test.mp4")


def _make_config(release_dir, manifest_path, source=None):
    dataset_source = {"release_dir": str(release_dir)}
    if source:
        dataset_source.update(source)
    return Config(
        dataset={"name": "slovo", "source": dataset_source},
        paths={
            "videos": str(release_dir),
            "manifest": str(manifest_path),
        },
    )


class TestSlovoRegistration:
    def test_registered(self):
        assert "slovo" in DATASET_REGISTRY

    def test_instance_has_name(self):
        assert SlovoDataset().name == "slovo"


class TestSlovoSourceConfig:
    def test_defaults(self, tmp_path):
        cfg = _make_config(tmp_path, tmp_path / "manifest.tsv")
        source = SlovoDataset().get_source_config(cfg)

        assert isinstance(source, SlovoSourceConfig)
        assert source.release_dir == str(tmp_path)
        assert source.split == "all"
        assert source.class_map_mode == "derive"
        assert source.availability_policy == "fail_fast"

    def test_missing_release_dir_raises(self):
        cfg = Config(dataset={"name": "slovo"}, paths={"videos": ""})

        with pytest.raises(ValueError, match="release_dir|paths.videos"):
            SlovoDataset.validate_config(cfg)

    def test_base_config_release_override_derives_annotations(self, tmp_path):
        release_dir = tmp_path / "custom_slovo"
        cfg = load_config(
            "configs/base/datasets/slovo.yaml",
            overrides=[f"dataset.source.release_dir={release_dir}"],
        )
        source = SlovoDataset().get_source_config(cfg)

        assert source.annotations_csv == ""
        assert slovo_source.resolve_annotations_csv(
            source, source.release_dir,
        ) == str(release_dir / "annotations.csv")

    def test_processing_video_root_follows_release_dir(self, tmp_path):
        release_dir = tmp_path / "custom_slovo"
        cfg = load_config(
            "configs/base/datasets/slovo.yaml",
            overrides=[f"dataset.source.release_dir={release_dir}"],
        )

        assert SlovoDataset().resolve_videos_dir(cfg) == release_dir


class TestSlovoDownload:
    def test_download_validates_release_and_annotations(self, tmp_path):
        release_dir = tmp_path / "slovo"
        _make_release(release_dir)
        cfg = _make_config(release_dir, tmp_path / "manifest.tsv")
        adapter = SlovoDataset()
        context = PipelineContext(config=cfg, dataset=adapter)

        context = adapter.download(cfg, context)

        assert context.stats["dataset.download"] == {
            "validated": True,
            "variant": "trimmed",
            "rows_found": 2,
        }

    def test_download_rejects_missing_required_annotation_columns(self, tmp_path):
        release_dir = tmp_path / "slovo"
        _write_annotations(
            release_dir / "annotations.csv",
            [{"attachment_id": "clip", "text": "hello"}],
        )
        cfg = _make_config(release_dir, tmp_path / "manifest.tsv")
        adapter = SlovoDataset()
        context = PipelineContext(config=cfg, dataset=adapter)

        with pytest.raises(ValueError, match="required columns"):
            adapter.download(cfg, context)


class TestSlovoBuildManifest:
    def test_build_manifest_from_annotations(self, tmp_path, monkeypatch):
        release_dir = tmp_path / "slovo"
        _make_release(release_dir)
        manifest_path = tmp_path / "manifest.tsv"
        cfg = _make_config(release_dir, manifest_path)

        monkeypatch.setattr(slovo_manifest, "get_video_duration", lambda _: 1.6)
        monkeypatch.setattr(slovo_manifest, "get_video_fps", lambda _: 30.0)

        adapter = SlovoDataset()
        context = PipelineContext(config=cfg, dataset=adapter)
        context = adapter.build_manifest(cfg, context)

        df = context.manifest_df
        assert list(df["VIDEO_ID"]) == ["clip_train", "clip_test"]
        assert list(df["REL_PATH"]) == ["clip_train.mp4", "clip_test.mp4"]
        assert list(df["SPLIT"]) == ["train", "test"]
        assert list(df["TEXT"]) == ["hello", "thanks"]
        assert list(df["SIGNER_ID"]) == ["signer_a", "signer_b"]
        assert list(df["END"]) == [1.6, 1.6]
        assert list(df["FRAME_START"]) == [12, 7]
        assert list(df["FRAME_END"]) == [60, 39]
        assert sorted(df["CLASS_ID"].tolist()) == [0, 1]
        assert manifest_path.exists()
