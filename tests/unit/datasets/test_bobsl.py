"""Tests for the BOBSL dataset package."""

import json

import pandas as pd
import pytest

from signdata.config.schema import Config
from signdata.datasets.bobsl import BOBSLDataset, BOBSLSourceConfig
from signdata.datasets.bobsl import source as bobsl_source
from signdata.pipeline.context import PipelineContext
from signdata.registry import DATASET_REGISTRY


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_vtt(path, cues: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["WEBVTT", ""]
    for index, (start, end, text) in enumerate(cues, start=1):
        lines.extend([str(index), f"{start} --> {end}", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_subtitle_csv(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_isolated_signs_csv(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _make_release(tmp_path):
    release_dir = tmp_path / "bobsl-release"
    metadata_path = release_dir / "metadata" / "subset2episode.json"
    subtitles_manual = release_dir / "subtitles" / "manually-aligned"
    subtitles_original = release_dir / "subtitles" / "audio-aligned-heuristic-correction"
    annotations_dir = release_dir / "annotations"

    _write_json(
        metadata_path,
        {
            "train": ["episode_train"],
            "val": ["episode_val"],
            "test": ["episode_test"],
            "challenge": ["episode_challenge"],
        },
    )
    _write_vtt(
        subtitles_manual / "episode_train.vtt",
        [
            ("00:00:01.000", "00:00:02.500", "hello there"),
            ("00:00:03.000", "00:00:04.500", "general kenobi"),
        ],
    )
    _write_subtitle_csv(
        subtitles_original / "episode_train.csv",
        [
            {
                "start sub (after alignement heuristic 1)": 1.2,
                "end sub (after alignement heuristic 1)": 2.4,
                "english sentence": "audio aligned one",
            },
            {
                "start sub (after alignement heuristic 1)": 2.8,
                "end sub (after alignement heuristic 1)": 4.2,
                "english sentence": "audio aligned two",
            },
        ],
    )
    _write_isolated_signs_csv(
        annotations_dir / "isolated_signs.csv",
        [
            {
                "episode": "episode_train",
                "start": 5.0,
                "end": 5.8,
                "gloss": "HELLO",
                "class_id": 10,
                "signer_id": "Signer01",
            },
            {
                "episode": "episode_val",
                "start": 8.0,
                "end": 8.5,
                "gloss": "THANK-YOU",
                "class_id": 11,
                "signer_id": "Signer02",
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
            "name": "bobsl",
            "source": dataset_source,
        },
        paths=paths,
    )


class TestBOBSLRegistration:
    def test_registered(self):
        assert "bobsl" in DATASET_REGISTRY

    def test_instance_has_name(self):
        assert BOBSLDataset().name == "bobsl"


class TestBOBSLValidateConfig:
    def test_valid_config_passes(self, tmp_path):
        cfg = Config(
            dataset={
                "name": "bobsl",
                "source": {"release_dir": str(tmp_path)},
            },
        )
        BOBSLDataset.validate_config(cfg)

    def test_missing_release_dir_raises(self):
        cfg = Config(dataset={"name": "bobsl"})
        with pytest.raises(ValueError, match="release_dir"):
            BOBSLDataset.validate_config(cfg)

    @pytest.mark.parametrize("view", ["subtitle_slt", "isolated_signs"])
    def test_challenge_split_is_rejected_for_supported_views(self, tmp_path, view):
        cfg = Config(
            dataset={
                "name": "bobsl",
                "source": {
                    "release_dir": str(tmp_path),
                    "view": view,
                    "split": "challenge",
                },
            },
        )
        with pytest.raises(ValueError, match="challenge partition"):
            BOBSLDataset.validate_config(cfg)


class TestBOBSLSourceConfig:
    def test_defaults(self, tmp_path):
        cfg = Config(
            dataset={
                "name": "bobsl",
                "source": {"release_dir": str(tmp_path)},
            },
        )

        source = BOBSLDataset().get_source_config(cfg)
        assert isinstance(source, BOBSLSourceConfig)
        assert source.release_dir == str(tmp_path)
        assert source.view == "subtitle_slt"
        assert source.split == "all"
        assert source.subtitle_alignment == "manual"
        assert source.availability_policy == "drop_unavailable"

    def test_manual_subtitle_autodiscovery_rejects_generic_subtitles_root(self, tmp_path):
        release_dir = tmp_path / "bobsl-release"
        _write_json(
            release_dir / "metadata" / "subset2episode.json",
            {"train": ["episode_train"]},
        )
        _write_subtitle_csv(
            release_dir / "subtitles" / "audio-aligned-heuristic-correction" / "episode_train.csv",
            [
                {
                    "start sub (after alignement heuristic 1)": 1.2,
                    "end sub (after alignement heuristic 1)": 2.4,
                    "english sentence": "audio aligned one",
                },
            ],
        )

        source = BOBSLSourceConfig(
            release_dir=str(release_dir),
            subtitle_alignment="manual",
        )

        with pytest.raises(FileNotFoundError, match="subtitles_root explicitly"):
            bobsl_source.resolve_subtitles_root(source, release_dir)


class TestBOBSLDownload:
    def test_download_validates_release_and_videos(self, tmp_path):
        release_dir = _make_release(tmp_path)
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "episode_train.mp4").touch()
        manifest_path = tmp_path / "manifest.tsv"

        cfg = _make_config(release_dir, video_dir, manifest_path)
        adapter = BOBSLDataset()
        context = PipelineContext(config=cfg, dataset=adapter)

        context = adapter.download(cfg, context)
        assert context.stats["dataset.download"]["validated"] is True
        assert context.stats["dataset.download"]["videos_on_disk"] == 1


class TestBOBSLBuildManifest:
    def _make_context(self, config):
        adapter = BOBSLDataset()
        return PipelineContext(config=config, dataset=adapter)

    def test_build_manifest_from_manual_subtitles(self, tmp_path):
        release_dir = _make_release(tmp_path)
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "episode_train.mp4").touch()
        manifest_path = tmp_path / "manifest.tsv"

        cfg = _make_config(
            release_dir,
            video_dir,
            manifest_path,
            source={"view": "subtitle_slt", "split": "train"},
        )
        context = self._make_context(cfg)

        context = BOBSLDataset().build_manifest(cfg, context)

        df = context.manifest_df
        assert list(df["TEXT"]) == ["hello there", "general kenobi"]
        assert list(df["SPLIT"]) == ["train", "train"]
        assert list(df["REL_PATH"]) == ["episode_train.mp4", "episode_train.mp4"]
        assert list(df["START"]) == [1.0, 3.0]
        assert list(df["END"]) == [2.5, 4.5]

    def test_build_manifest_from_original_subtitles_csv(self, tmp_path):
        release_dir = _make_release(tmp_path)
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "episode_train.mp4").touch()
        manifest_path = tmp_path / "manifest.tsv"

        cfg = _make_config(
            release_dir,
            video_dir,
            manifest_path,
            source={
                "view": "subtitle_slt",
                "split": "train",
                "subtitle_alignment": "original",
            },
        )
        context = self._make_context(cfg)

        context = BOBSLDataset().build_manifest(cfg, context)

        df = context.manifest_df
        assert list(df["TEXT"]) == ["audio aligned one", "audio aligned two"]
        assert list(df["START"]) == [1.2, 2.8]
        assert list(df["END"]) == [2.4, 4.2]

    def test_build_manifest_from_isolated_sign_annotations(self, tmp_path):
        release_dir = _make_release(tmp_path)
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "episode_train.mp4").touch()
        (video_dir / "episode_val.mp4").touch()
        manifest_path = tmp_path / "manifest.tsv"

        cfg = _make_config(
            release_dir,
            video_dir,
            manifest_path,
            source={"view": "isolated_signs", "split": "all"},
        )
        context = self._make_context(cfg)

        context = BOBSLDataset().build_manifest(cfg, context)

        df = context.manifest_df.sort_values("CLASS_ID").reset_index(drop=True)
        assert list(df["GLOSS"]) == ["HELLO", "THANK-YOU"]
        assert list(df["TEXT"]) == ["HELLO", "THANK-YOU"]
        assert list(df["CLASS_ID"]) == [10, 11]
        assert list(df["SPLIT"]) == ["train", "val"]
        assert list(df["REL_PATH"]) == ["episode_train.mp4", "episode_val.mp4"]

    def test_mark_unavailable_keeps_rows(self, tmp_path):
        release_dir = _make_release(tmp_path)
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "episode_train.mp4").touch()
        manifest_path = tmp_path / "manifest.tsv"

        cfg = _make_config(
            release_dir,
            video_dir,
            manifest_path,
            source={
                "view": "isolated_signs",
                "availability_policy": "mark_unavailable",
            },
        )
        context = self._make_context(cfg)

        context = BOBSLDataset().build_manifest(cfg, context)

        df = context.manifest_df.sort_values("CLASS_ID").reset_index(drop=True)
        assert list(df["AVAILABLE"]) == [True, False]

    def test_missing_class_ids_are_assigned_after_existing_ids(self, tmp_path):
        release_dir = _make_release(tmp_path)
        _write_isolated_signs_csv(
            release_dir / "annotations" / "isolated_signs.csv",
            [
                {
                    "episode": "episode_train",
                    "start": 1.0,
                    "end": 1.5,
                    "gloss": "HELLO",
                },
                {
                    "episode": "episode_train",
                    "start": 2.0,
                    "end": 2.5,
                    "gloss": "THANK-YOU",
                    "class_id": 0,
                },
                {
                    "episode": "episode_train",
                    "start": 3.0,
                    "end": 3.5,
                    "gloss": "HELLO",
                },
                {
                    "episode": "episode_train",
                    "start": 4.0,
                    "end": 4.5,
                    "gloss": "YES",
                    "class_id": 5,
                },
                {
                    "episode": "episode_train",
                    "start": 5.0,
                    "end": 5.5,
                    "gloss": "NO",
                },
            ],
        )

        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "episode_train.mp4").touch()
        manifest_path = tmp_path / "manifest.tsv"
        cfg = _make_config(
            release_dir,
            video_dir,
            manifest_path,
            source={"view": "isolated_signs", "split": "train"},
        )
        context = self._make_context(cfg)

        context = BOBSLDataset().build_manifest(cfg, context)

        df = context.manifest_df.sort_values("START").reset_index(drop=True)
        class_ids_by_gloss = (
            df.groupby("GLOSS")["CLASS_ID"].apply(lambda series: sorted(set(series))).to_dict()
        )
        assert class_ids_by_gloss == {
            "HELLO": [6],
            "THANK-YOU": [0],
            "YES": [5],
            "NO": [7],
        }
