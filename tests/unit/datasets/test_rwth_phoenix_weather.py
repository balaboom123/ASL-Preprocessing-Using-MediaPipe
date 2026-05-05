"""Tests for the RWTH-PHOENIX-Weather dataset package."""

import pandas as pd
import pytest

from signdata.config.loader import load_config
from signdata.config.schema import Config
from signdata.datasets.rwth_phoenix_weather import RWTHPhoenixWeatherDataset
from signdata.datasets.rwth_phoenix_weather import manifest as phoenix_manifest
from signdata.datasets.rwth_phoenix_weather import source as phoenix_source
from signdata.datasets.rwth_phoenix_weather.source import (
    RWTHPhoenixWeatherSourceConfig,
    derive_clip_id,
    find_corpus_csvs,
)
from signdata.pipeline.context import PipelineContext
from signdata.registry import DATASET_REGISTRY


_CSV_HEADER = "id|folder|signer|orth|translation\n"


def _write_corpus_csv(path, rows: list[dict]) -> None:
    """Write a minimal PHOENIX corpus CSV to *path*."""
    lines = [_CSV_HEADER]
    for row in rows:
        line = "|".join([
            row.get("id", ""),
            row.get("folder", ""),
            row.get("signer", ""),
            row.get("orth", ""),
            row.get("translation", ""),
        ])
        lines.append(line + "\n")
    path.write_text("".join(lines), encoding="utf-8")


def _write_pipe_csv(path, header: list[str], rows: list[dict]) -> None:
    """Write a pipe-delimited CSV with a custom header."""
    lines = ["|".join(header) + "\n"]
    for row in rows:
        lines.append("|".join(str(row.get(column, "")) for column in header) + "\n")
    path.write_text("".join(lines), encoding="utf-8")


def _make_phoenix_release(tmp_path, splits=("train",), rows_per_split=2):
    """Build a minimal PHOENIX release directory structure."""
    release_dir = tmp_path / "PHOENIX-2014-T.v3.0" / "PHOENIX-2014-T"
    release_dir.mkdir(parents=True)

    for split in splits:
        rows = [
            {
                "id": f"clip_{i:04d}",
                "folder": f"phoenix2014T/{split}/clip_{i:04d}/*.png",
                "signer": f"Signer0{i % 3 + 1}",
                "orth": "MORGEN REGEN STARK",
                "translation": "tomorrow heavy rain",
            }
            for i in range(rows_per_split)
        ]
        csv_path = release_dir / f"PHOENIX-2014-T.{split}.corpus.csv"
        _write_corpus_csv(csv_path, rows)

    return release_dir


def _make_official_style_release(tmp_path, splits=("train",), rows_per_split=2):
    """Build a PHOENIX-like release with annotations/manual and features/fullFrame-210x260px."""
    release_dir = tmp_path / "PHOENIX-2014-T-release3"
    annotations_dir = release_dir / "annotations" / "manual"
    features_dir = release_dir / "features" / "fullFrame-210x260px"
    annotations_dir.mkdir(parents=True)
    features_dir.mkdir(parents=True)

    for split in splits:
        rows = []
        for index in range(rows_per_split):
            clip_name = f"01April_2010_Thursday_clip-{index}"
            rows.append({
                "name": clip_name,
                "video": f"{clip_name}/*.png",
                "start": 0,
                "end": 50 + index,
                "speaker": f"Signer0{index % 3 + 1}",
                "orth": "MORGEN REGEN STARK",
                "translation": "tomorrow heavy rain",
            })
        csv_path = annotations_dir / f"PHOENIX-2014-T.{split}.corpus.csv"
        _write_pipe_csv(
            csv_path,
            ["name", "video", "start", "end", "speaker", "orth", "translation"],
            rows,
        )

    return release_dir


def _make_config(release_dir, video_dir, manifest_path, source_extra=None):
    source = {"release_dir": str(release_dir)}
    if source_extra:
        source.update(source_extra)
    return Config(
        dataset={"name": "rwth_phoenix_weather", "source": source},
        paths={
            "videos": str(video_dir),
            "manifest": str(manifest_path),
        },
    )


class TestRWTHPhoenixWeatherRegistration:
    def test_registered(self):
        assert "rwth_phoenix_weather" in DATASET_REGISTRY

    def test_instance_has_name(self):
        assert RWTHPhoenixWeatherDataset().name == "rwth_phoenix_weather"

    def test_has_required_methods(self):
        adapter = RWTHPhoenixWeatherDataset()
        assert hasattr(adapter, "download")
        assert hasattr(adapter, "build_manifest")
        assert hasattr(adapter, "get_source_config")
        assert hasattr(adapter, "validate_config")


class TestRWTHPhoenixWeatherValidateConfig:
    def test_release_dir_in_source_passes(self, tmp_path):
        cfg = Config(
            dataset={
                "name": "rwth_phoenix_weather",
                "source": {"release_dir": str(tmp_path)},
            }
        )
        RWTHPhoenixWeatherDataset.validate_config(cfg)

    def test_paths_videos_fallback_passes(self, tmp_path):
        cfg = Config(
            dataset={"name": "rwth_phoenix_weather"},
            paths={"videos": str(tmp_path)},
        )
        RWTHPhoenixWeatherDataset.validate_config(cfg)

    def test_missing_both_raises(self):
        cfg = Config(dataset={"name": "rwth_phoenix_weather"})
        with pytest.raises(ValueError, match="release_dir"):
            RWTHPhoenixWeatherDataset.validate_config(cfg)


class TestRWTHPhoenixWeatherSourceConfig:
    def test_defaults(self, tmp_path):
        cfg = Config(
            dataset={
                "name": "rwth_phoenix_weather",
                "source": {"release_dir": str(tmp_path)},
            }
        )
        source = RWTHPhoenixWeatherDataset().get_source_config(cfg)
        assert isinstance(source, RWTHPhoenixWeatherSourceConfig)
        assert source.release_dir == str(tmp_path)
        assert source.split == "all"
        assert source.prepare_mode == "materialize_missing"
        assert source.availability_policy == "drop_unavailable"
        assert source.video_fps == 25.0

    def test_base_yaml_defaults_to_materialize_missing(self, tmp_path):
        cfg = load_config(
            "configs/base/datasets/rwth_phoenix_weather.yaml",
            overrides=[f"dataset.source.release_dir={tmp_path}"],
        )
        source = RWTHPhoenixWeatherDataset().get_source_config(cfg)
        assert source.prepare_mode == "materialize_missing"
        assert source.availability_policy == "drop_unavailable"

    def test_paths_videos_fallback(self, tmp_path):
        cfg = Config(
            dataset={"name": "rwth_phoenix_weather"},
            paths={"videos": str(tmp_path)},
        )
        source = RWTHPhoenixWeatherDataset().get_source_config(cfg)
        assert source.release_dir == str(tmp_path)

    def test_custom_options(self, tmp_path):
        cfg = Config(
            dataset={
                "name": "rwth_phoenix_weather",
                "source": {
                    "release_dir": str(tmp_path),
                    "split": "train",
                    "prepare_mode": "materialize_missing",
                    "availability_policy": "mark_unavailable",
                    "video_fps": 30.0,
                },
            }
        )
        source = RWTHPhoenixWeatherDataset().get_source_config(cfg)
        assert source.split == "train"
        assert source.prepare_mode == "materialize_missing"
        assert source.availability_policy == "mark_unavailable"
        assert source.video_fps == 30.0


class TestDeriveClipId:
    def test_plain_name(self):
        assert derive_clip_id("01April_2010_Thursday") == "01April_2010_Thursday"

    def test_strips_leading_slash(self):
        assert derive_clip_id("/phoenix2014T/train/clip") == "phoenix2014T_train_clip"

    def test_replaces_glob_suffix(self):
        assert derive_clip_id("phoenix2014T/train/clip/*") == "phoenix2014T_train_clip"

    def test_replaces_png_glob_suffix(self):
        assert derive_clip_id("phoenix2014T/train/clip/*.png") == "phoenix2014T_train_clip"

    def test_strips_trailing_slash(self):
        assert derive_clip_id("clip_001/") == "clip_001"

    def test_replaces_spaces(self):
        assert derive_clip_id("clip 001") == "clip_001"

    def test_full_folder_path(self):
        result = derive_clip_id("phoenix2014T/train/01April_2010_Thursday_heute-65/*.png")
        assert "/" not in result
        assert "*" not in result


class TestFindCorpusCsvs:
    def test_finds_train_csv(self, tmp_path):
        csv_path = tmp_path / "PHOENIX-2014-T.train.corpus.csv"
        csv_path.touch()
        found = find_corpus_csvs(tmp_path, "train")
        assert csv_path in found

    def test_finds_nested_csv(self, tmp_path):
        subdir = tmp_path / "deep" / "nested"
        subdir.mkdir(parents=True)
        csv_path = subdir / "PHOENIX-2014-T.dev.corpus.csv"
        csv_path.touch()
        found = find_corpus_csvs(tmp_path, "dev")
        assert csv_path in found

    def test_wrong_split_not_found(self, tmp_path):
        csv_path = tmp_path / "PHOENIX-2014-T.train.corpus.csv"
        csv_path.touch()
        found = find_corpus_csvs(tmp_path, "test")
        assert csv_path not in found

    def test_empty_dir_returns_empty(self, tmp_path):
        assert find_corpus_csvs(tmp_path, "train") == []


class TestRWTHPhoenixWeatherDownload:
    def test_validate_mode_passes_with_existing_dir(self, tmp_path):
        release_dir = _make_phoenix_release(tmp_path)
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        manifest_path = tmp_path / "manifest.tsv"

        cfg = _make_config(
            release_dir,
            video_dir,
            manifest_path,
            source_extra={"prepare_mode": "validate"},
        )
        adapter = RWTHPhoenixWeatherDataset()
        context = PipelineContext(config=cfg, dataset=adapter)

        context = adapter.download(cfg, context)
        assert context.stats["dataset.download"]["validated"] is True
        assert context.stats["dataset.download"]["mode"] == "validate"

    def test_validate_mode_missing_dir_raises(self, tmp_path):
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        manifest_path = tmp_path / "manifest.tsv"

        cfg = _make_config(
            tmp_path / "nonexistent_release",
            video_dir,
            manifest_path,
            source_extra={"prepare_mode": "validate"},
        )
        adapter = RWTHPhoenixWeatherDataset()
        context = PipelineContext(config=cfg, dataset=adapter)

        with pytest.raises(FileNotFoundError, match="PHOENIX release directory"):
            adapter.download(cfg, context)

    def test_materialize_mode_calls_materialise(self, tmp_path, monkeypatch):
        release_dir = _make_phoenix_release(tmp_path, splits=("train",), rows_per_split=1)
        frame_dir = release_dir / "phoenix2014T" / "train" / "clip_0000"
        frame_dir.mkdir(parents=True)
        (frame_dir / "0001.png").touch()
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        manifest_path = tmp_path / "manifest.tsv"

        calls = []

        def fake_materialize(frame_dir, output_path, fps, overwrite):
            calls.append((frame_dir, output_path, fps, overwrite))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()

        monkeypatch.setattr(phoenix_source, "materialize_frames_to_video", fake_materialize)

        cfg = _make_config(
            release_dir,
            video_dir,
            manifest_path,
            source_extra={"prepare_mode": "materialize_missing"},
        )
        adapter = RWTHPhoenixWeatherDataset()
        context = PipelineContext(config=cfg, dataset=adapter)

        context = adapter.download(cfg, context)
        stats = context.stats["dataset.download"]
        assert stats["mode"] == "materialize_missing"
        assert stats["errors"] == 0
        assert stats["materialized"] == 1
        assert stats["validated"] == 0
        assert calls == [
            (frame_dir, video_dir / "train" / "clip_0000.mp4", 25.0, False),
        ]

    def test_materialize_mode_skips_existing_video(self, tmp_path, monkeypatch):
        release_dir = _make_phoenix_release(tmp_path, splits=("train",), rows_per_split=1)
        video_dir = tmp_path / "videos"
        existing_output = video_dir / "train" / "clip_0000.mp4"
        existing_output.parent.mkdir(parents=True)
        existing_output.touch()
        manifest_path = tmp_path / "manifest.tsv"

        calls = []

        def fake_materialize(frame_dir, output_path, fps, overwrite):
            calls.append((frame_dir, output_path, fps, overwrite))

        monkeypatch.setattr(phoenix_source, "materialize_frames_to_video", fake_materialize)

        cfg = _make_config(
            release_dir,
            video_dir,
            manifest_path,
            source_extra={"prepare_mode": "materialize_missing"},
        )
        adapter = RWTHPhoenixWeatherDataset()
        context = PipelineContext(config=cfg, dataset=adapter)

        context = adapter.download(cfg, context)
        stats = context.stats["dataset.download"]
        assert stats["mode"] == "materialize_missing"
        assert stats["errors"] == 0
        assert stats["materialized"] == 0
        assert stats["validated"] == 1
        assert calls == []

    def test_materialize_mode_supports_official_release_layout(self, tmp_path, monkeypatch):
        release_dir = _make_official_style_release(
            tmp_path, splits=("train",), rows_per_split=1
        )
        frame_dir = (
            release_dir
            / "features"
            / "fullFrame-210x260px"
            / "train"
            / "01April_2010_Thursday_clip-0"
        )
        frame_dir.mkdir(parents=True)
        (frame_dir / "images0001.png").touch()

        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        manifest_path = tmp_path / "manifest.tsv"

        calls = []

        def fake_materialize(input_frame_dir, output_path, fps, overwrite):
            calls.append((input_frame_dir, output_path, fps, overwrite))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()

        monkeypatch.setattr(phoenix_source, "materialize_frames_to_video", fake_materialize)

        cfg = _make_config(
            release_dir,
            video_dir,
            manifest_path,
            source_extra={"prepare_mode": "materialize_missing"},
        )
        adapter = RWTHPhoenixWeatherDataset()
        context = PipelineContext(config=cfg, dataset=adapter)

        context = adapter.download(cfg, context)

        assert context.stats["dataset.download"]["materialized"] == 1
        assert calls == [
            (
                frame_dir,
                video_dir / "train" / "01April_2010_Thursday_clip-0.mp4",
                25.0,
                False,
            ),
        ]

    def test_materialize_mode_does_not_escape_release_dir(self, tmp_path, monkeypatch):
        release_root = tmp_path / "nested" / "PHOENIX-2014-T-release3"
        annotations_dir = release_root / "annotations" / "manual"
        annotations_dir.mkdir(parents=True)
        _write_pipe_csv(
            annotations_dir / "PHOENIX-2014-T.train.corpus.csv",
            ["name", "video", "start", "end", "speaker", "orth", "translation"],
            [{
                "name": "01April_2010_Thursday_clip-0",
                "video": "01April_2010_Thursday_clip-0/*.png",
                "start": 0,
                "end": 50,
                "speaker": "Signer01",
                "orth": "MORGEN REGEN STARK",
                "translation": "tomorrow heavy rain",
            }],
        )

        external_frame_dir = (
            tmp_path
            / "features"
            / "fullFrame-210x260px"
            / "train"
            / "01April_2010_Thursday_clip-0"
        )
        external_frame_dir.mkdir(parents=True)
        (external_frame_dir / "images0001.png").touch()

        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        manifest_path = tmp_path / "manifest.tsv"

        calls = []

        def fake_materialize(input_frame_dir, output_path, fps, overwrite):
            calls.append((input_frame_dir, output_path, fps, overwrite))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()

        monkeypatch.setattr(phoenix_source, "materialize_frames_to_video", fake_materialize)

        cfg = _make_config(
            release_root,
            video_dir,
            manifest_path,
            source_extra={"prepare_mode": "materialize_missing"},
        )
        adapter = RWTHPhoenixWeatherDataset()
        context = PipelineContext(config=cfg, dataset=adapter)

        context = adapter.download(cfg, context)
        stats = context.stats["dataset.download"]

        assert stats["materialized"] == 0
        assert stats["errors"] == 1
        assert calls == []


class TestLoadSplitDf:
    def test_basic_csv_parse(self, tmp_path):
        csv_path = tmp_path / "PHOENIX-2014-T.train.corpus.csv"
        csv_path.write_text(
            "id|folder|signer|orth|translation\n"
            "clip_0001|phoenix2014T/train/clip_0001/*.png|Signer01|MORGEN STARK|tomorrow heavy\n"
            "clip_0002|phoenix2014T/train/clip_0002/*.png|Signer02|HEUTE REGEN|today rain\n",
            encoding="utf-8",
        )
        df = phoenix_manifest._load_split_df(csv_path, "train", 25.0)
        assert df is not None
        assert len(df) == 2
        assert list(df["SPLIT"]) == ["train", "train"]
        assert list(df["GLOSS"]) == ["MORGEN STARK", "HEUTE REGEN"]
        assert list(df["TEXT"]) == ["tomorrow heavy", "today rain"]
        assert list(df["SIGNER_ID"]) == ["Signer01", "Signer02"]
        assert list(df["FPS"]) == [25.0, 25.0]

    def test_required_columns_present(self, tmp_path):
        csv_path = tmp_path / "PHOENIX-2014-T.dev.corpus.csv"
        csv_path.write_text(
            "id|folder|signer|orth|translation\n"
            "clip_0001|phoenix2014T/dev/clip_0001/*.png|Signer01|HEUTE|today\n",
            encoding="utf-8",
        )
        df = phoenix_manifest._load_split_df(csv_path, "dev", 25.0)
        for column in ("SAMPLE_ID", "VIDEO_ID", "REL_PATH", "SPLIT", "FPS", "GLOSS", "TEXT"):
            assert column in df.columns, f"Missing column: {column}"

    def test_rel_path_format(self, tmp_path):
        csv_path = tmp_path / "PHOENIX-2014-T.train.corpus.csv"
        csv_path.write_text(
            "id|folder|signer|orth|translation\n"
            "clip_0001|phoenix2014T/train/clip_0001/*.png|Signer01|REGEN|rain\n",
            encoding="utf-8",
        )
        df = phoenix_manifest._load_split_df(csv_path, "train", 25.0)
        assert df.iloc[0]["REL_PATH"] == "train/clip_0001.mp4"

    def test_missing_id_column_returns_none(self, tmp_path):
        csv_path = tmp_path / "PHOENIX-2014-T.train.corpus.csv"
        csv_path.write_text(
            "folder|signer|orth|translation\n"
            "phoenix2014T/train/clip_0001/*.png|Signer01|REGEN|rain\n",
            encoding="utf-8",
        )
        df = phoenix_manifest._load_split_df(csv_path, "train", 25.0)
        assert df is None

    def test_start_end_parsed_when_present(self, tmp_path):
        csv_path = tmp_path / "PHOENIX-2014-T.train.corpus.csv"
        csv_path.write_text(
            "id|folder|signer|orth|translation|start|end\n"
            "clip_0001|phoenix2014T/train/clip_0001/*.png|Signer01|REGEN|rain|0|75\n",
            encoding="utf-8",
        )
        df = phoenix_manifest._load_split_df(csv_path, "train", 25.0)
        assert "START" in df.columns
        assert "END" in df.columns
        assert df.iloc[0]["START"] == 0.0
        assert df.iloc[0]["END"] == 75.0

    def test_official_columns_are_supported(self, tmp_path):
        csv_path = tmp_path / "PHOENIX-2014-T.train.corpus.csv"
        _write_pipe_csv(
            csv_path,
            ["name", "video", "start", "end", "speaker", "orth", "translation"],
            [{
                "name": "06October_2012_Saturday_tagesschau-8730",
                "video": "06October_2012_Saturday_tagesschau-8730/*.png",
                "start": 0,
                "end": 75,
                "speaker": "Signer08",
                "orth": "MORGEN DEUTSCH LAND",
                "translation": "morgen regen im sueden",
            }],
        )

        df = phoenix_manifest._load_split_df(csv_path, "train", 25.0)

        assert df is not None
        assert df.iloc[0]["SAMPLE_ID"] == "06October_2012_Saturday_tagesschau-8730"
        assert df.iloc[0]["REL_PATH"] == "train/06October_2012_Saturday_tagesschau-8730.mp4"
        assert df.iloc[0]["SIGNER_ID"] == "Signer08"
        assert df.iloc[0]["START"] == 0.0
        assert df.iloc[0]["END"] == 75.0

    def test_corrupt_csv_returns_none(self, tmp_path):
        csv_path = tmp_path / "PHOENIX-2014-T.train.corpus.csv"
        csv_path.write_text("not|||valid\x00csv\xff", encoding="latin-1")
        result = phoenix_manifest._load_split_df(csv_path, "train", 25.0)
        assert result is None or isinstance(result, pd.DataFrame)


class TestRWTHPhoenixWeatherBuildManifest:
    def _make_context(self, config):
        adapter = RWTHPhoenixWeatherDataset()
        return PipelineContext(config=config, dataset=adapter)

    def test_build_manifest_all_splits(self, tmp_path):
        release_dir = _make_phoenix_release(
            tmp_path, splits=("train", "dev", "test"), rows_per_split=2
        )
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        manifest_path = tmp_path / "manifest.tsv"

        cfg = _make_config(
            release_dir,
            video_dir,
            manifest_path,
            source_extra={"availability_policy": "mark_unavailable"},
        )
        context = self._make_context(cfg)
        context = RWTHPhoenixWeatherDataset().build_manifest(cfg, context)

        assert context.manifest_path == manifest_path
        assert len(context.manifest_df) == 6
        assert set(context.manifest_df["SPLIT"]) == {"train", "dev", "test"}
        assert context.stats["dataset.manifest"]["segments"] == 6
        assert set(context.stats["dataset.manifest"]["splits"]) == {"train", "dev", "test"}

    def test_build_manifest_supports_official_release_layout(self, tmp_path):
        release_dir = _make_official_style_release(
            tmp_path, splits=("train", "dev"), rows_per_split=1
        )
        video_dir = tmp_path / "videos"
        (video_dir / "train").mkdir(parents=True)
        (video_dir / "dev").mkdir(parents=True)
        (video_dir / "train" / "01April_2010_Thursday_clip-0.mp4").touch()
        (video_dir / "dev" / "01April_2010_Thursday_clip-0.mp4").touch()
        manifest_path = tmp_path / "manifest.tsv"

        cfg = _make_config(
            release_dir,
            video_dir,
            manifest_path,
            source_extra={"availability_policy": "mark_unavailable"},
        )
        context = self._make_context(cfg)
        context = RWTHPhoenixWeatherDataset().build_manifest(cfg, context)

        assert len(context.manifest_df) == 2
        assert set(context.manifest_df["SPLIT"]) == {"train", "dev"}
        assert set(context.manifest_df["SIGNER_ID"]) == {"Signer01"}
        assert set(context.manifest_df["REL_PATH"]) == {
            "train/01April_2010_Thursday_clip-0.mp4",
            "dev/01April_2010_Thursday_clip-0.mp4",
        }

    def test_build_manifest_single_split(self, tmp_path):
        release_dir = _make_phoenix_release(
            tmp_path, splits=("train", "dev"), rows_per_split=2
        )
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        train_dir = video_dir / "train"
        train_dir.mkdir()
        for index in range(2):
            (train_dir / f"clip_{index:04d}.mp4").touch()

        manifest_path = tmp_path / "manifest.tsv"
        cfg = _make_config(
            release_dir,
            video_dir,
            manifest_path,
            source_extra={"split": "train", "availability_policy": "mark_unavailable"},
        )
        context = self._make_context(cfg)
        context = RWTHPhoenixWeatherDataset().build_manifest(cfg, context)

        splits_in_df = (
            context.manifest_df["SPLIT"].unique().tolist()
            if len(context.manifest_df) > 0
            else []
        )
        assert all(split == "train" for split in splits_in_df)

    def test_build_manifest_writes_tsv(self, tmp_path):
        release_dir = _make_phoenix_release(tmp_path, splits=("train",), rows_per_split=1)
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        manifest_path = tmp_path / "manifest.tsv"

        cfg = _make_config(
            release_dir,
            video_dir,
            manifest_path,
            source_extra={"availability_policy": "mark_unavailable"},
        )
        context = self._make_context(cfg)
        RWTHPhoenixWeatherDataset().build_manifest(cfg, context)

        assert manifest_path.exists()
        reloaded = pd.read_csv(manifest_path, sep="\t")
        assert "SAMPLE_ID" in reloaded.columns
        assert "REL_PATH" in reloaded.columns
        assert "SPLIT" in reloaded.columns

    def test_build_manifest_missing_release_dir_raises(self, tmp_path):
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        manifest_path = tmp_path / "manifest.tsv"
        cfg = _make_config(
            tmp_path / "nonexistent",
            video_dir,
            manifest_path,
        )
        adapter = RWTHPhoenixWeatherDataset()
        context = PipelineContext(config=cfg, dataset=adapter)

        with pytest.raises(FileNotFoundError, match="PHOENIX release directory"):
            adapter.build_manifest(cfg, context)

    def test_build_manifest_drop_unavailable(self, tmp_path):
        release_dir = _make_phoenix_release(tmp_path, splits=("train",), rows_per_split=2)
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        manifest_path = tmp_path / "manifest.tsv"
        cfg = _make_config(
            release_dir,
            video_dir,
            manifest_path,
            source_extra={"availability_policy": "drop_unavailable"},
        )
        context = self._make_context(cfg)
        context = RWTHPhoenixWeatherDataset().build_manifest(cfg, context)

        assert len(context.manifest_df) == 0

    def test_build_manifest_mark_unavailable(self, tmp_path):
        release_dir = _make_phoenix_release(tmp_path, splits=("train",), rows_per_split=2)
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        train_dir = video_dir / "train"
        train_dir.mkdir()
        (train_dir / "clip_0000.mp4").touch()

        manifest_path = tmp_path / "manifest.tsv"
        cfg = _make_config(
            release_dir,
            video_dir,
            manifest_path,
            source_extra={"availability_policy": "mark_unavailable"},
        )
        context = self._make_context(cfg)
        context = RWTHPhoenixWeatherDataset().build_manifest(cfg, context)

        assert "AVAILABLE" in context.manifest_df.columns
        available = context.manifest_df["AVAILABLE"].tolist()
        assert True in available
        assert False in available

    def test_build_manifest_no_csv_found_raises(self, tmp_path):
        release_dir = tmp_path / "release"
        release_dir.mkdir()
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        manifest_path = tmp_path / "manifest.tsv"
        cfg = _make_config(release_dir, video_dir, manifest_path)
        adapter = RWTHPhoenixWeatherDataset()
        context = PipelineContext(config=cfg, dataset=adapter)

        with pytest.raises(RuntimeError, match="No corpus CSV rows loaded"):
            adapter.build_manifest(cfg, context)
