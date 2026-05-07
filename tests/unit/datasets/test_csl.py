"""Tests for the CSL dataset package."""

from pathlib import Path
import pytest

from signdata.config.schema import Config
from signdata.datasets.csl import CSLDataset, CSLSourceConfig
from signdata.datasets.csl import manifest as csl_manifest
from signdata.datasets.csl import source as csl_source
from signdata.pipeline.context import PipelineContext
from signdata.registry import DATASET_REGISTRY
from signdata.utils.manifest import get_timing_columns, resolve_video_path


def _touch_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _write_frames(sample_dir: Path, count: int = 2) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, count + 1):
        (sample_dir / f"{index:06d}.jpg").touch()


def _write_corpus(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{sentence_id}\t{text}\n" for sentence_id, text in rows),
        encoding="utf-8",
    )


def _make_config(
    release_dir,
    manifest_path,
    *,
    paths=None,
    source=None,
):
    dataset_source = {"release_dir": str(release_dir)}
    if source:
        dataset_source.update(source)

    config_paths = {"manifest": str(manifest_path)}
    if paths:
        config_paths.update(paths)

    return Config(
        dataset={
            "name": "csl",
            "source": dataset_source,
        },
        paths=config_paths,
    )


class TestCSLRegistration:
    def test_registered(self):
        assert "csl" in DATASET_REGISTRY

    def test_instance_has_name(self):
        assert CSLDataset().name == "csl"


class TestCSLValidateConfig:
    def test_valid_config_passes_with_release_dir(self, tmp_path):
        cfg = Config(
            dataset={
                "name": "csl",
                "source": {"release_dir": str(tmp_path)},
            },
        )
        CSLDataset.validate_config(cfg)

    def test_valid_config_passes_with_paths_root(self, tmp_path):
        cfg = Config(
            dataset={"name": "csl"},
            paths={"root": str(tmp_path)},
        )
        CSLDataset.validate_config(cfg)

    def test_missing_release_dir_and_root_raises(self):
        cfg = Config(dataset={"name": "csl"})
        with pytest.raises(ValueError, match="release_dir|paths.root"):
            CSLDataset.validate_config(cfg)

    def test_invalid_variant_raises(self, tmp_path):
        cfg = Config(
            dataset={
                "name": "csl",
                "source": {
                    "release_dir": str(tmp_path),
                    "variant": "isolated_500",
                },
            },
        )
        with pytest.raises(ValueError, match="variant"):
            CSLDataset.validate_config(cfg)

    def test_invalid_protocol_raises(self, tmp_path):
        cfg = Config(
            dataset={
                "name": "csl",
                "source": {
                    "release_dir": str(tmp_path),
                    "protocol": "split_x",
                },
            },
        )
        with pytest.raises(ValueError, match="protocol"):
            CSLDataset.validate_config(cfg)

    def test_invalid_split_raises(self, tmp_path):
        cfg = Config(
            dataset={
                "name": "csl",
                "source": {
                    "release_dir": str(tmp_path),
                    "split": "dev",
                },
            },
        )
        with pytest.raises(ValueError, match="split"):
            CSLDataset.validate_config(cfg)

    def test_non_positive_video_fps_raises(self, tmp_path):
        cfg = Config(
            dataset={
                "name": "csl",
                "source": {
                    "release_dir": str(tmp_path),
                    "video_fps": 0,
                },
            },
        )
        with pytest.raises(ValueError, match="video_fps|positive"):
            CSLDataset.validate_config(cfg)


class TestCSLSourceConfig:
    def test_defaults(self, tmp_path):
        cfg = Config(
            dataset={
                "name": "csl",
                "source": {"release_dir": str(tmp_path)},
            },
        )
        source = CSLDataset().get_source_config(cfg)
        assert isinstance(source, CSLSourceConfig)
        assert source.release_dir == str(tmp_path)
        assert source.variant == "continuous_2015"
        assert source.protocol == "split_i"
        assert source.split == "all"
        assert source.prepare_mode == "materialize_missing"
        assert source.availability_policy == "drop_unavailable"
        assert source.video_fps == 30.0


class TestCSLDownload:
    def test_validate_mode_uses_rgb_video_dir(self, tmp_path):
        release_dir = tmp_path / "release"
        _write_corpus(release_dir / "corpus.txt", [("000000", "你好")])
        _touch_video(release_dir / "color" / "000000" / "sample_01_01.mp4")
        cfg = _make_config(
            release_dir,
            tmp_path / "manifest.tsv",
            paths={"root": str(release_dir), "videos": str(tmp_path / "videos")},
            source={"prepare_mode": "validate"},
        )

        adapter = CSLDataset()
        context = PipelineContext(config=cfg, dataset=adapter)
        context = adapter.download(cfg, context)

        assert context.stats["dataset.download"]["validated"] is True
        assert context.stats["dataset.download"]["runtime_video_dir"] == str(
            release_dir / "color"
        )
        assert context.videos_dir == release_dir / "color"

    def test_materialize_mode_converts_frame_dirs(self, tmp_path, monkeypatch):
        release_dir = tmp_path / "release"
        _write_corpus(release_dir / "corpus.txt", [("000000", "你好")])
        sample_dir = release_dir / "color" / "000000" / "sample_01_01"
        _write_frames(sample_dir)
        output_dir = tmp_path / "videos"
        calls = []

        def fake_materialize(input_dir, output_path, fps, pattern, overwrite):
            calls.append((input_dir, output_path, fps, pattern, overwrite))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()

        monkeypatch.setattr(
            csl_source,
            "materialize_frames_to_video",
            fake_materialize,
        )

        cfg = _make_config(
            release_dir,
            tmp_path / "manifest.tsv",
            paths={"root": str(release_dir), "videos": str(output_dir)},
        )

        adapter = CSLDataset()
        context = PipelineContext(config=cfg, dataset=adapter)
        context = adapter.download(cfg, context)

        stats = context.stats["dataset.download"]
        assert stats["validated"] is True
        assert stats["materialized"] == 1
        assert stats["runtime_video_dir"] == str(output_dir)
        assert context.videos_dir == output_dir
        assert calls == [
            (
                sample_dir,
                output_dir / "000000" / "sample_01_01.mp4",
                30.0,
                "*.jpg",
                False,
            ),
        ]

    def test_materialize_mode_uses_discovered_bmp_pattern(self, tmp_path, monkeypatch):
        release_dir = tmp_path / "release"
        _write_corpus(release_dir / "corpus.txt", [("000000", "你好")])
        sample_dir = release_dir / "color" / "000000" / "sample_01_01"
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "000001.bmp").touch()
        output_dir = tmp_path / "videos"
        captured = {}

        def fake_materialize(input_dir, output_path, fps, pattern, overwrite):
            captured["call"] = (input_dir, output_path, fps, pattern, overwrite)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()

        monkeypatch.setattr(
            csl_source,
            "materialize_frames_to_video",
            fake_materialize,
        )

        cfg = _make_config(
            release_dir,
            tmp_path / "manifest.tsv",
            paths={"root": str(release_dir), "videos": str(output_dir)},
        )

        adapter = CSLDataset()
        context = PipelineContext(config=cfg, dataset=adapter)
        context = adapter.download(cfg, context)

        assert context.stats["dataset.download"]["materialized"] == 1
        assert captured["call"] == (
            sample_dir,
            output_dir / "000000" / "sample_01_01.mp4",
            30.0,
            "*.bmp",
            False,
        )

    def test_materialize_mode_preserves_uppercase_frame_extension(
        self, tmp_path, monkeypatch
    ):
        release_dir = tmp_path / "release"
        _write_corpus(release_dir / "corpus.txt", [("000000", "你好")])
        sample_dir = release_dir / "color" / "000000" / "sample_01_01"
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "000001.JPG").touch()
        output_dir = tmp_path / "videos"
        captured = {}

        def fake_materialize(input_dir, output_path, fps, pattern, overwrite):
            captured["call"] = (input_dir, output_path, fps, pattern, overwrite)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()

        monkeypatch.setattr(
            csl_source,
            "materialize_frames_to_video",
            fake_materialize,
        )

        cfg = _make_config(
            release_dir,
            tmp_path / "manifest.tsv",
            paths={"root": str(release_dir), "videos": str(output_dir)},
        )

        adapter = CSLDataset()
        context = PipelineContext(config=cfg, dataset=adapter)
        context = adapter.download(cfg, context)

        assert context.stats["dataset.download"]["materialized"] == 1
        assert captured["call"] == (
            sample_dir,
            output_dir / "000000" / "sample_01_01.mp4",
            30.0,
            "*.JPG",
            False,
        )

    def test_materialize_mode_collects_mixed_case_frame_extensions(
        self, tmp_path, monkeypatch
    ):
        release_dir = tmp_path / "release"
        _write_corpus(release_dir / "corpus.txt", [("000000", "你好")])
        sample_dir = release_dir / "color" / "000000" / "sample_01_01"
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "000001.jpg").touch()
        (sample_dir / "000002.JPG").touch()
        output_dir = tmp_path / "videos"
        captured = {}

        def fake_materialize(input_dir, output_path, fps, pattern, overwrite):
            captured["call"] = (input_dir, output_path, fps, pattern, overwrite)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()

        monkeypatch.setattr(
            csl_source,
            "materialize_frames_to_video",
            fake_materialize,
        )

        cfg = _make_config(
            release_dir,
            tmp_path / "manifest.tsv",
            paths={"root": str(release_dir), "videos": str(output_dir)},
        )

        adapter = CSLDataset()
        context = PipelineContext(config=cfg, dataset=adapter)
        context = adapter.download(cfg, context)

        assert context.stats["dataset.download"]["materialized"] == 1
        assert captured["call"][:3] == (
            sample_dir,
            output_dir / "000000" / "sample_01_01.mp4",
            30.0,
        )
        assert set(captured["call"][3]) == {"*.jpg", "*.JPG"}
        assert captured["call"][4] is False

    def test_materialize_mode_raises_when_no_videos_are_produced(
        self, tmp_path, monkeypatch
    ):
        release_dir = tmp_path / "release"
        _write_corpus(release_dir / "corpus.txt", [("000000", "你好")])
        sample_dir = release_dir / "color" / "000000" / "sample_01_01"
        _write_frames(sample_dir)
        output_dir = tmp_path / "videos"

        def fake_materialize(*args, **kwargs):
            raise RuntimeError("ffmpeg failed")

        monkeypatch.setattr(
            csl_source,
            "materialize_frames_to_video",
            fake_materialize,
        )

        cfg = _make_config(
            release_dir,
            tmp_path / "manifest.tsv",
            paths={"root": str(release_dir), "videos": str(output_dir)},
        )

        adapter = CSLDataset()
        context = PipelineContext(config=cfg, dataset=adapter)

        with pytest.raises(RuntimeError, match="did not produce any usable videos"):
            adapter.download(cfg, context)


class TestCSLBuildManifest:
    def _make_context(self, config):
        adapter = CSLDataset()
        return PipelineContext(config=config, dataset=adapter)

    def test_build_manifest_emits_canonical_timing_columns(self, tmp_path, monkeypatch):
        release_dir = tmp_path / "release"
        _write_corpus(
            release_dir / "corpus.txt",
            [("000000", "你好"), ("000001", "谢谢")],
        )
        video_dir = release_dir / "color"
        _touch_video(video_dir / "000000" / "sample_01_01.mp4")
        _touch_video(video_dir / "000000" / "sample_41_02.mp4")
        _touch_video(video_dir / "000001" / "sample_02_01.mp4")

        monkeypatch.setattr(csl_manifest, "get_video_duration", lambda _: 2.5)
        monkeypatch.setattr(csl_manifest, "get_video_fps", lambda _: 29.97)

        cfg = _make_config(
            release_dir,
            tmp_path / "manifest.tsv",
            paths={"root": str(release_dir), "videos": str(tmp_path / "videos")},
            source={"prepare_mode": "validate"},
        )
        context = self._make_context(cfg)
        context = CSLDataset().build_manifest(cfg, context)

        df = context.manifest_df
        assert list(df["SAMPLE_ID"]) == [
            "000000_01_01",
            "000000_41_02",
            "000001_02_01",
        ]
        assert list(df["SPLIT"]) == ["train", "test", "train"]
        assert list(df["START"]) == [0.0, 0.0, 0.0]
        assert list(df["END"]) == [2.5, 2.5, 2.5]
        assert list(df["FPS"]) == [29.97, 29.97, 29.97]
        assert list(df["TEXT"]) == ["你好", "你好", "谢谢"]
        assert context.videos_dir == release_dir / "color"

        start_col, end_col = get_timing_columns(df)
        assert (start_col, end_col) == ("START", "END")
        first_video = resolve_video_path(df.iloc[0], context.videos_dir)
        assert first_video == video_dir / "000000" / "sample_01_01.mp4"

    def test_build_manifest_uses_zero_based_split_ii_boundary(self, tmp_path, monkeypatch):
        release_dir = tmp_path / "release"
        _write_corpus(release_dir / "corpus.txt", [("000094", "再见")])
        _touch_video(release_dir / "color" / "000094" / "sample_01_01.mp4")

        monkeypatch.setattr(csl_manifest, "get_video_duration", lambda _: 1.0)
        monkeypatch.setattr(csl_manifest, "get_video_fps", lambda _: 30.0)

        cfg = _make_config(
            release_dir,
            tmp_path / "manifest.tsv",
            paths={"root": str(release_dir), "videos": str(tmp_path / "videos")},
            source={"prepare_mode": "validate", "protocol": "split_ii"},
        )
        context = self._make_context(cfg)
        context = CSLDataset().build_manifest(cfg, context)

        df = context.manifest_df
        assert list(df["SPLIT"]) == ["test"]

    def test_build_manifest_applies_custom_split_spec(self, tmp_path, monkeypatch):
        release_dir = tmp_path / "release"
        _write_corpus(release_dir / "corpus.txt", [("000000", "你好")])
        _touch_video(release_dir / "color" / "000000" / "sample_01_01.mp4")
        split_spec = tmp_path / "split_spec.tsv"
        split_spec.write_text("000000_01_01\ttest\n", encoding="utf-8")

        monkeypatch.setattr(csl_manifest, "get_video_duration", lambda _: 1.0)
        monkeypatch.setattr(csl_manifest, "get_video_fps", lambda _: 30.0)

        cfg = _make_config(
            release_dir,
            tmp_path / "manifest.tsv",
            paths={"root": str(release_dir), "videos": str(tmp_path / "videos")},
            source={
                "prepare_mode": "validate",
                "split_spec_file": str(split_spec),
            },
        )
        context = self._make_context(cfg)
        context = CSLDataset().build_manifest(cfg, context)

        df = context.manifest_df
        assert list(df["SPLIT"]) == ["test"]

    def test_build_manifest_prefers_rgb_over_stale_materialized_videos(
        self, tmp_path, monkeypatch
    ):
        release_dir = tmp_path / "release"
        _write_corpus(release_dir / "corpus.txt", [("000000", "你好")])
        rgb_dir = release_dir / "color"
        _touch_video(rgb_dir / "000000" / "sample_01_01.mp4")

        stale_dir = tmp_path / "videos"
        _touch_video(stale_dir / "000000" / "stale_41_05.mp4")

        monkeypatch.setattr(csl_manifest, "get_video_duration", lambda _: 1.0)
        monkeypatch.setattr(csl_manifest, "get_video_fps", lambda _: 30.0)

        cfg = _make_config(
            release_dir,
            tmp_path / "manifest.tsv",
            paths={"root": str(release_dir), "videos": str(stale_dir)},
            source={"prepare_mode": "validate"},
        )
        context = self._make_context(cfg)
        context = CSLDataset().build_manifest(cfg, context)

        df = context.manifest_df
        assert context.videos_dir == rgb_dir
        assert list(df["REL_PATH"]) == ["000000/sample_01_01.mp4"]
        assert list(df["SAMPLE_ID"]) == ["000000_01_01"]
