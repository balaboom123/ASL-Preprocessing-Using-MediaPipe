"""Tests for the BSL-1K compatibility dataset package."""

import json

import pandas as pd
import pytest

from signdata.config.schema import Config
from signdata.datasets.bsl1k import BSL1KDataset, BSL1KSourceConfig
from signdata.pipeline.context import PipelineContext
from signdata.registry import DATASET_REGISTRY


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_isolated_signs_csv(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _make_release(tmp_path):
    release_dir = tmp_path / "bobsl-release"
    _write_json(
        release_dir / "metadata" / "subset2episode.json",
        {
            "train": ["episode_train"],
            "val": ["episode_val"],
        },
    )
    _write_isolated_signs_csv(
        release_dir / "annotations" / "isolated_signs.csv",
        [
            {
                "episode": "episode_train",
                "start": 5.0,
                "end": 5.8,
                "gloss": "HELLO",
                "class_id": 10,
            },
            {
                "episode": "episode_val",
                "start": 8.0,
                "end": 8.5,
                "gloss": "THANK-YOU",
                "class_id": 11,
            },
        ],
    )
    return release_dir


def _make_config(
    release_dir,
    video_dir,
    manifest_path,
    source=None,
    root_dir=None,
):
    dataset_source = {"release_dir": str(release_dir)}
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
            "name": "bsl1k",
            "source": dataset_source,
        },
        paths=paths,
    )


class TestBSL1KRegistration:
    def test_registered(self):
        assert "bsl1k" in DATASET_REGISTRY

    def test_instance_has_name(self):
        assert BSL1KDataset().name == "bsl1k"


class TestBSL1KValidateConfig:
    def test_valid_config_passes(self, tmp_path):
        cfg = Config(
            dataset={
                "name": "bsl1k",
                "source": {"release_dir": str(tmp_path)},
            },
        )
        BSL1KDataset.validate_config(cfg)

    def test_missing_release_dir_raises(self):
        cfg = Config(dataset={"name": "bsl1k"})
        with pytest.raises(ValueError, match="release_dir"):
            BSL1KDataset.validate_config(cfg)

    def test_non_lexical_view_is_rejected(self, tmp_path):
        cfg = Config(
            dataset={
                "name": "bsl1k",
                "source": {
                    "release_dir": str(tmp_path),
                    "view": "subtitle_slt",
                },
            },
        )
        with pytest.raises(ValueError, match="isolated_signs"):
            BSL1KDataset.validate_config(cfg)


class TestBSL1KSourceConfig:
    def test_defaults_to_isolated_sign_view(self, tmp_path):
        cfg = Config(
            dataset={
                "name": "bsl1k",
                "source": {"release_dir": str(tmp_path)},
            },
        )

        source = BSL1KDataset().get_source_config(cfg)
        assert isinstance(source, BSL1KSourceConfig)
        assert source.view == "isolated_signs"


class TestBSL1KBuildManifest:
    def test_build_manifest_uses_lexical_rows(self, tmp_path):
        release_dir = _make_release(tmp_path)
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "episode_train.mp4").touch()
        (video_dir / "episode_val.mp4").touch()
        manifest_path = tmp_path / "manifest.tsv"
        cfg = _make_config(release_dir, video_dir, manifest_path)

        adapter = BSL1KDataset()
        context = PipelineContext(config=cfg, dataset=adapter)
        context = adapter.build_manifest(cfg, context)

        df = context.manifest_df.sort_values("CLASS_ID").reset_index(drop=True)
        assert list(df["GLOSS"]) == ["HELLO", "THANK-YOU"]
        assert list(df["TEXT"]) == ["HELLO", "THANK-YOU"]
        assert list(df["SPLIT"]) == ["train", "val"]
