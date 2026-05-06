"""Tests for the MS-ASL dataset package."""

import json

import pandas as pd
import pytest

from signdata.config.schema import Config
from signdata.datasets.msasl import MSASLDataset, MSASLSourceConfig
from signdata.datasets.msasl import source as msasl_source
from signdata.pipeline.context import PipelineContext
from signdata.registry import DATASET_REGISTRY


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_annotations_dir(base, train=None, val=None, test=None, classes=None) -> None:
    base.mkdir(parents=True, exist_ok=True)
    _write_json(base / "MSASL_train.json", train or [])
    _write_json(base / "MSASL_val.json", val or [])
    _write_json(base / "MSASL_test.json", test or [])
    _write_json(base / "MSASL_classes.json", classes or ["hello", "thanks"])


def _make_config(
    annotations_dir,
    video_dir,
    manifest_path,
    source=None,
    root_dir=None,
):
    dataset_source = {"annotations_dir": str(annotations_dir)}
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
            "name": "msasl",
            "source": dataset_source,
        },
        paths=paths,
    )


class TestMSASLRegistration:
    def test_registered(self):
        assert "msasl" in DATASET_REGISTRY

    def test_instance_has_name(self):
        assert MSASLDataset().name == "msasl"


class TestMSASLValidateConfig:
    def test_valid_config_passes(self, tmp_path):
        ann_dir = tmp_path / "annotations"
        _write_annotations_dir(ann_dir)

        cfg = Config(
            dataset={
                "name": "msasl",
                "source": {"annotations_dir": str(ann_dir)},
            },
        )

        MSASLDataset.validate_config(cfg)

    def test_missing_annotations_dir_raises(self):
        cfg = Config(dataset={"name": "msasl"})
        with pytest.raises(ValueError, match="annotations_dir"):
            MSASLDataset.validate_config(cfg)


class TestMSASLSourceConfig:
    def test_defaults(self, tmp_path):
        ann_dir = tmp_path / "annotations"
        _write_annotations_dir(ann_dir)
        cfg = Config(
            dataset={
                "name": "msasl",
                "source": {"annotations_dir": str(ann_dir)},
            },
        )

        source = MSASLDataset().get_source_config(cfg)
        assert isinstance(source, MSASLSourceConfig)
        assert source.annotations_dir == str(ann_dir)
        assert source.split == "all"
        assert source.subset == 1000
        assert source.download_mode == "validate"
        assert source.availability_policy == "drop_unavailable"

    def test_custom_options(self, tmp_path):
        ann_dir = tmp_path / "annotations"
        _write_annotations_dir(ann_dir)
        cfg = Config(
            dataset={
                "name": "msasl",
                "source": {
                    "annotations_dir": str(ann_dir),
                    "split": "train",
                    "subset": 5,
                    "download_mode": "download_missing",
                    "availability_policy": "mark_unavailable",
                },
            },
        )

        source = MSASLDataset().get_source_config(cfg)
        assert source.split == "train"
        assert source.subset == 5
        assert source.download_mode == "download_missing"
        assert source.availability_policy == "mark_unavailable"


class TestMSASLDownload:
    def test_download_validates_inputs(self, tmp_path):
        ann_dir = tmp_path / "annotations"
        _write_annotations_dir(ann_dir)
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "abcdefghijk.mp4").touch()
        manifest_path = tmp_path / "manifest.tsv"

        cfg = _make_config(ann_dir, video_dir, manifest_path)
        adapter = MSASLDataset()
        context = PipelineContext(config=cfg, dataset=adapter)

        context = adapter.download(cfg, context)
        assert context.stats["dataset.download"]["validated"] is True

    def test_download_missing_class_file_raises(self, tmp_path):
        ann_dir = tmp_path / "annotations"
        _write_annotations_dir(ann_dir)
        (ann_dir / "MSASL_classes.json").unlink()
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        manifest_path = tmp_path / "manifest.tsv"

        cfg = _make_config(ann_dir, video_dir, manifest_path)
        adapter = MSASLDataset()
        context = PipelineContext(config=cfg, dataset=adapter)

        with pytest.raises(FileNotFoundError, match="MSASL_classes.json"):
            adapter.download(cfg, context)

    def test_download_missing_uses_selected_splits(self, tmp_path, monkeypatch):
        ann_dir = tmp_path / "annotations"
        _write_annotations_dir(
            ann_dir,
            train=[
                {"url": "https://youtu.be/aaaaaaaaaaa", "label": 0},
                {"url": "https://youtu.be/bbbbbbbbbbb", "label": 1},
            ],
            val=[
                {"url": "https://youtu.be/ccccccccccc", "label": 0},
            ],
            test=[
                {"url": "https://youtu.be/ddddddddddd", "label": 1},
            ],
        )
        video_dir = tmp_path / "videos"
        (video_dir / "class0").mkdir(parents=True)
        (video_dir / "class0" / "aaaaaaaaaaa.mp4").touch()
        manifest_path = tmp_path / "manifest.tsv"
        root_dir = tmp_path / "root"
        root_dir.mkdir()

        captured = {}

        def fake_download(video_ids, video_dir_arg, **kwargs):
            captured["video_ids"] = video_ids
            captured["video_dir"] = video_dir_arg
            captured["kwargs"] = kwargs
            return {"downloaded": 1, "errors": 0, "missing": []}

        monkeypatch.setattr(msasl_source, "download_youtube_videos", fake_download)

        cfg = _make_config(
            ann_dir,
            video_dir,
            manifest_path,
            source={"download_mode": "download_missing", "split": "train"},
            root_dir=root_dir,
        )
        adapter = MSASLDataset()
        context = PipelineContext(config=cfg, dataset=adapter)

        context = adapter.download(cfg, context)

        assert captured["video_ids"] == ["bbbbbbbbbbb"]
        assert captured["video_dir"] == str(video_dir)
        assert context.stats["dataset.download"] == {
            "total": 2,
            "downloaded": 1,
            "errors": 0,
            "skipped": 1,
        }


class TestMSASLBuildManifest:
    def _make_context(self, config):
        adapter = MSASLDataset()
        return PipelineContext(config=config, dataset=adapter)

    def test_build_manifest_joins_classes_and_rel_paths(self, tmp_path):
        ann_dir = tmp_path / "annotations"
        _write_annotations_dir(
            ann_dir,
            train=[
                {
                    "url": "https://youtu.be/abcdefghijk",
                    "start_time": 0.0,
                    "end_time": 1.25,
                    "label": 0,
                    "text": "HELLO",
                    "signer_id": 12,
                    "fps": 30,
                    "box": [1, 2, 11, 12],
                },
                {
                    "url": "https://youtu.be/lmnopqrstuv",
                    "start_time": 2.0,
                    "end_time": 3.0,
                    "label": 1,
                    "text": "",
                    "signer_id": 15,
                    "fps": 24,
                },
            ],
            val=[],
            test=[],
            classes=[
                {"label": 0, "clean_text": "hello"},
                {"label": 1, "clean_text": "thanks"},
            ],
        )
        video_dir = tmp_path / "videos"
        (video_dir / "hello").mkdir(parents=True)
        (video_dir / "thanks").mkdir(parents=True)
        (video_dir / "hello" / "abcdefghijk.mp4").touch()
        (video_dir / "thanks" / "lmnopqrstuv.webm").touch()
        manifest_path = tmp_path / "manifest.tsv"

        cfg = _make_config(
            ann_dir,
            video_dir,
            manifest_path,
            source={"split": "train"},
        )
        context = self._make_context(cfg)

        context = MSASLDataset().build_manifest(cfg, context)

        df = context.manifest_df
        assert list(df["VIDEO_ID"]) == ["abcdefghijk", "lmnopqrstuv"]
        assert list(df["REL_PATH"]) == ["hello/abcdefghijk.mp4", "thanks/lmnopqrstuv.webm"]
        assert list(df["CLASS_ID"]) == [0, 1]
        assert list(df["GLOSS"]) == ["hello", "thanks"]
        assert list(df["TEXT"]) == ["HELLO", "thanks"]
        assert list(df["SPLIT"]) == ["train", "train"]
        assert list(df["SIGNER_ID"]) == ["12", "15"]
        assert df.iloc[0]["BBOX_X1"] == 2.0
        assert df.iloc[0]["BBOX_Y1"] == 1.0
        assert df.iloc[0]["BBOX_X2"] == 12.0
        assert df.iloc[0]["BBOX_Y2"] == 11.0
        assert df.iloc[0]["PERSON_DETECTED"] == True
        assert context.stats["dataset.manifest"] == {"videos": 2, "segments": 2}

    def test_build_manifest_all_splits_and_subset(self, tmp_path):
        ann_dir = tmp_path / "annotations"
        _write_annotations_dir(
            ann_dir,
            train=[
                {
                    "url": "https://youtu.be/abcdefghijk",
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "label": 0,
                },
            ],
            val=[
                {
                    "url": "https://youtu.be/bbbbbbbbbbb",
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "label": 1,
                },
            ],
            test=[
                {
                    "url": "https://youtu.be/ccccccccccc",
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "label": 2,
                },
            ],
            classes=["zero", "one", "two"],
        )
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "abcdefghijk.mp4").touch()
        (video_dir / "bbbbbbbbbbb.mp4").touch()
        (video_dir / "ccccccccccc.mp4").touch()
        manifest_path = tmp_path / "manifest.tsv"

        cfg = _make_config(
            ann_dir,
            video_dir,
            manifest_path,
            source={"split": "all", "subset": 2},
        )
        context = self._make_context(cfg)

        context = MSASLDataset().build_manifest(cfg, context)

        df = context.manifest_df
        assert list(df["CLASS_ID"]) == [0, 1]
        assert list(df["GLOSS"]) == ["zero", "one"]
        assert list(df["SPLIT"]) == ["train", "val"]

    def test_build_manifest_marks_unavailable_with_rel_path(self, tmp_path):
        ann_dir = tmp_path / "annotations"
        _write_annotations_dir(
            ann_dir,
            train=[
                {
                    "url": "https://youtu.be/abcdefghijk",
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "label": 0,
                },
                {
                    "url": "https://youtu.be/lmnopqrstuv",
                    "start_time": 1.0,
                    "end_time": 2.0,
                    "label": 1,
                },
            ],
            val=[],
            test=[],
            classes=["hello", "thanks"],
        )
        video_dir = tmp_path / "videos"
        (video_dir / "nested").mkdir(parents=True)
        (video_dir / "nested" / "abcdefghijk.webm").touch()
        manifest_path = tmp_path / "manifest.tsv"

        cfg = _make_config(
            ann_dir,
            video_dir,
            manifest_path,
            source={"split": "train", "availability_policy": "mark_unavailable"},
        )
        context = self._make_context(cfg)

        context = MSASLDataset().build_manifest(cfg, context)

        df = context.manifest_df
        assert list(df["REL_PATH"]) == ["nested/abcdefghijk.webm", "lmnopqrstuv.mp4"]
        assert list(df["AVAILABLE"]) == [True, False]

    def test_build_manifest_uses_sample_paths_for_duplicate_video_ids(self, tmp_path):
        ann_dir = tmp_path / "annotations"
        _write_annotations_dir(
            ann_dir,
            train=[
                {
                    "url": "https://youtu.be/abcdefghijk",
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "label": 0,
                },
                {
                    "url": "https://youtu.be/abcdefghijk",
                    "start_time": 1.0,
                    "end_time": 2.0,
                    "label": 1,
                },
            ],
            val=[],
            test=[],
            classes=["hello", "thanks"],
        )
        video_dir = tmp_path / "videos"
        (video_dir / "hello").mkdir(parents=True)
        (video_dir / "thanks").mkdir(parents=True)
        (video_dir / "hello" / "train-abcdefghijk-0-1000-0.mp4").touch()
        (video_dir / "thanks" / "train-abcdefghijk-1000-2000-1.webm").touch()
        manifest_path = tmp_path / "manifest.tsv"

        cfg = _make_config(
            ann_dir,
            video_dir,
            manifest_path,
            source={"split": "train", "availability_policy": "mark_unavailable"},
        )
        context = self._make_context(cfg)

        context = MSASLDataset().build_manifest(cfg, context)

        df = context.manifest_df
        assert list(df["REL_PATH"]) == [
            "hello/train-abcdefghijk-0-1000-0.mp4",
            "thanks/train-abcdefghijk-1000-2000-1.webm",
        ]
        assert list(df["AVAILABLE"]) == [True, True]

    def test_build_manifest_keeps_shared_video_path_for_download_missing(self, tmp_path):
        ann_dir = tmp_path / "annotations"
        _write_annotations_dir(
            ann_dir,
            train=[
                {
                    "url": "https://youtu.be/abcdefghijk",
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "label": 0,
                },
                {
                    "url": "https://youtu.be/abcdefghijk",
                    "start_time": 1.0,
                    "end_time": 2.0,
                    "label": 1,
                },
            ],
            val=[],
            test=[],
            classes=["hello", "thanks"],
        )
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "abcdefghijk.mp4").touch()
        manifest_path = tmp_path / "manifest.tsv"

        cfg = _make_config(
            ann_dir,
            video_dir,
            manifest_path,
            source={"split": "train", "download_mode": "download_missing"},
        )
        context = self._make_context(cfg)

        context = MSASLDataset().build_manifest(cfg, context)

        df = context.manifest_df
        assert list(df["REL_PATH"]) == ["abcdefghijk.mp4", "abcdefghijk.mp4"]

    def test_build_manifest_empty_selected_split_returns_empty_manifest(self, tmp_path):
        ann_dir = tmp_path / "annotations"
        _write_annotations_dir(
            ann_dir,
            train=[],
            val=[
                {
                    "url": "https://youtu.be/abcdefghijk",
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "label": 0,
                },
            ],
            test=[],
            classes=["hello"],
        )
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        manifest_path = tmp_path / "manifest.tsv"

        cfg = _make_config(
            ann_dir,
            video_dir,
            manifest_path,
            source={"split": "train"},
        )
        context = self._make_context(cfg)

        context = MSASLDataset().build_manifest(cfg, context)

        df = context.manifest_df
        assert df.empty
        assert "VIDEO_ID" in df.columns
        assert "CLASS_ID" in df.columns
        assert context.stats["dataset.manifest"] == {"videos": 0, "segments": 0}
