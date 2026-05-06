"""Tests for PipelineContext and PipelineRunner (4-stage pipeline)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import signdata.datasets  # noqa: F401 – trigger registrations

from signdata.config.schema import Config
from signdata.datasets.lsa64 import LSA64Dataset
from signdata.datasets.youtube_asl import YouTubeASLDataset
from signdata.pipeline.context import PipelineContext
from signdata.pipeline.runner import PipelineRunner
from signdata.registry import DATASET_REGISTRY


# ── PipelineContext ─────────────────────────────────────────────────────────

class TestPipelineContext:
    def test_instantiation(self):
        cfg = Config(dataset={"name": "youtube_asl"})
        ds = YouTubeASLDataset()
        ctx = PipelineContext(config=cfg, dataset=ds)
        assert ctx.config is cfg
        assert ctx.dataset is ds

    def test_defaults_empty(self):
        cfg = Config(dataset={"name": "youtube_asl"})
        ctx = PipelineContext(
            config=cfg,
            dataset=YouTubeASLDataset(),
        )
        assert ctx.manifest_path is None
        assert ctx.manifest_df is None
        assert ctx.output_dir is None
        assert ctx.webdataset_dir is None
        assert ctx.videos_dir is None
        assert ctx.completed_stages == []
        assert ctx.stats == {}

    def test_resolve_paths(self, tmp_path):
        cfg = Config(
            dataset={"name": "youtube_asl"},
            run_name="exp1",
            paths={
                "root": str(tmp_path),
                "videos": str(tmp_path / "videos"),
                "manifest": str(tmp_path / "manifest.csv"),
                "output": str(tmp_path / "output"),
                "webdataset": str(tmp_path / "webdataset"),
            },
        )
        ctx = PipelineContext(config=cfg, dataset=YouTubeASLDataset())
        ctx.resolve_paths()

        assert ctx.output_dir == Path(str(tmp_path / "output")) / "exp1"
        assert ctx.webdataset_dir == Path(str(tmp_path / "webdataset")) / "exp1"
        assert ctx.videos_dir == tmp_path / "videos"
        assert ctx.manifest_path == tmp_path / "manifest.csv"

    def test_resolve_paths_uses_dataset_specific_video_dir(self, tmp_path):
        release_root = tmp_path / "lsa64"
        raw_dir = release_root / "raw"
        cut_dir = release_root / "cut"
        raw_dir.mkdir(parents=True)
        cut_dir.mkdir(parents=True)

        cfg = Config(
            dataset={
                "name": "lsa64",
                "source": {
                    "release_dir": str(release_root),
                    "variant": "raw",
                },
            },
            run_name="exp1",
            paths={
                "root": str(tmp_path),
                "videos": str(release_root),
                "manifest": str(tmp_path / "manifest.tsv"),
                "output": str(tmp_path / "output"),
                "webdataset": str(tmp_path / "webdataset"),
            },
        )
        ctx = PipelineContext(config=cfg, dataset=LSA64Dataset())
        ctx.resolve_paths()

        assert ctx.videos_dir == raw_dir

    def test_load_manifest(self, tmp_path):
        manifest_path = tmp_path / "manifest.csv"
        manifest_path.write_text("SAMPLE_ID\tVIDEO_ID\ns1\tv1\n")

        cfg = Config(dataset={"name": "youtube_asl"})
        ctx = PipelineContext(config=cfg, dataset=YouTubeASLDataset())
        ctx.load_manifest(str(manifest_path))

        assert ctx.manifest_path == manifest_path
        assert ctx.manifest_df is not None
        assert len(ctx.manifest_df) == 1

    def test_load_manifest_normalizes_legacy_columns(self, tmp_path):
        manifest_path = tmp_path / "manifest.csv"
        manifest_path.write_text(
            "VIDEO_NAME\tSENTENCE_NAME\tSTART_REALIGNED\tEND_REALIGNED\n"
            "v1\ts1\t0.0\t1.0\n"
        )

        cfg = Config(dataset={"name": "youtube_asl"})
        ctx = PipelineContext(config=cfg, dataset=YouTubeASLDataset())
        ctx.load_manifest(str(manifest_path))

        assert ctx.manifest_df is not None
        assert "VIDEO_ID" in ctx.manifest_df.columns
        assert "SAMPLE_ID" in ctx.manifest_df.columns
        assert "START" in ctx.manifest_df.columns
        assert "END" in ctx.manifest_df.columns


# ── PipelineRunner init ───────────────────────────────────────────────────

class TestPipelineRunnerInit:
    def test_init_stores_config(self):
        cfg = Config(
            dataset={"name": "youtube_asl"},
        )
        runner = PipelineRunner(cfg)
        assert runner.config is cfg
        assert isinstance(runner.dataset, YouTubeASLDataset)

    def test_init_force_all(self):
        cfg = Config(dataset={"name": "youtube_asl"})
        runner = PipelineRunner(cfg, force_all=True)
        assert runner.force_all is True

    def test_unknown_dataset_raises(self):
        """PipelineRunner requires dataset to be in DATASET_REGISTRY."""
        cfg = Config(dataset={"name": "nonexistent"})
        with pytest.raises(KeyError):
            PipelineRunner(cfg)


# ── PipelineRunner.run orchestration ──────────────────────────────────────

class TestPipelineRunnerOrchestration:
    def _make_config(self, **overrides):
        base = {
            "dataset": {"name": "youtube_asl", "download": True, "manifest": True},
            "processing": {"enabled": False},
            "post_processing": {"enabled": False},
            "output": {"enabled": False},
        }
        base.update(overrides)
        return Config(**base)

    @patch.object(YouTubeASLDataset, "build_manifest")
    @patch.object(YouTubeASLDataset, "download")
    def test_runs_dataset_stages(self, mock_download, mock_manifest):
        mock_download.side_effect = lambda cfg, ctx: ctx
        mock_manifest.side_effect = lambda cfg, ctx: ctx

        cfg = self._make_config()
        runner = PipelineRunner(cfg)
        context = runner.run()

        mock_download.assert_called_once()
        mock_manifest.assert_called_once()
        assert "dataset.download" in context.completed_stages
        assert "dataset.manifest" in context.completed_stages

    @patch.object(YouTubeASLDataset, "build_manifest")
    @patch.object(YouTubeASLDataset, "download")
    def test_skips_disabled_download(self, mock_download, mock_manifest):
        mock_manifest.side_effect = lambda cfg, ctx: ctx

        cfg = self._make_config(
            dataset={"name": "youtube_asl", "download": False, "manifest": True},
        )
        runner = PipelineRunner(cfg)
        context = runner.run()

        mock_download.assert_not_called()
        assert "dataset.download" not in context.completed_stages

    @patch.object(YouTubeASLDataset, "build_manifest")
    @patch.object(YouTubeASLDataset, "download")
    def test_skips_disabled_manifest(self, mock_download, mock_manifest):
        mock_download.side_effect = lambda cfg, ctx: ctx

        cfg = self._make_config(
            dataset={"name": "youtube_asl", "download": True, "manifest": False},
        )
        runner = PipelineRunner(cfg)
        context = runner.run()

        mock_manifest.assert_not_called()
        assert "dataset.manifest" not in context.completed_stages

    @patch.object(YouTubeASLDataset, "download")
    def test_processing_stage(self, mock_download):
        mock_download.side_effect = lambda cfg, ctx: ctx

        mock_processor = MagicMock()
        mock_processor.return_value.run.side_effect = lambda ctx: ctx

        with patch.dict(
            "signdata.pipeline.runner.PROCESSOR_REGISTRY",
            {"video2pose": mock_processor},
        ):
            cfg = Config(
                dataset={"name": "youtube_asl", "download": True, "manifest": False},
                processing={
                    "enabled": True,
                    "processor": "video2pose",
                    "detection": "null",
                    "pose": "mediapipe",
                    "pose_config": {},
                },
                post_processing={"enabled": False},
                output={"enabled": False},
            )
            runner = PipelineRunner(cfg)
            context = runner.run()

        mock_processor.assert_called_once()
        assert "processing.video2pose" in context.completed_stages

    @patch.object(YouTubeASLDataset, "download")
    def test_unknown_processor_raises(self, mock_download):
        mock_download.side_effect = lambda cfg, ctx: ctx

        cfg = Config(
            dataset={"name": "youtube_asl", "download": True, "manifest": False},
            processing={
                "enabled": True,
                "processor": "video2pose",
                "detection": "null",
                "pose": "mediapipe",
                "pose_config": {},
            },
            post_processing={"enabled": False},
            output={"enabled": False},
        )
        runner = PipelineRunner(cfg)

        with patch.dict("signdata.pipeline.runner.PROCESSOR_REGISTRY", {}, clear=True):
            with pytest.raises(ValueError, match="Unknown processor"):
                runner.run()

    @patch.object(YouTubeASLDataset, "download")
    def test_post_processing_stage(self, mock_download):
        mock_download.side_effect = lambda cfg, ctx: ctx

        mock_pp = MagicMock()
        mock_pp.return_value.run.side_effect = lambda ctx: ctx

        with patch.dict(
            "signdata.pipeline.runner.POST_PROCESSOR_REGISTRY",
            {"normalize": mock_pp},
        ):
            cfg = Config(
                dataset={"name": "youtube_asl", "download": True, "manifest": False},
                processing={"enabled": False},
                post_processing={"enabled": True, "recipes": ["normalize"]},
                output={"enabled": False},
            )
            runner = PipelineRunner(cfg)
            context = runner.run()

        mock_pp.assert_called_once()
        assert "post_processing.normalize" in context.completed_stages

    @patch.object(YouTubeASLDataset, "download")
    def test_output_stage(self, mock_download):
        mock_download.side_effect = lambda cfg, ctx: ctx

        mock_output = MagicMock()
        mock_output.return_value.run.side_effect = lambda ctx: ctx

        with patch.dict(
            "signdata.pipeline.runner.OUTPUT_REGISTRY",
            {"webdataset": mock_output},
        ):
            cfg = Config(
                dataset={"name": "youtube_asl", "download": True, "manifest": False},
                processing={"enabled": False},
                post_processing={"enabled": False},
                output={"enabled": True, "type": "webdataset"},
            )
            runner = PipelineRunner(cfg)
            context = runner.run()

        mock_output.assert_called_once()
        assert "output.webdataset" in context.completed_stages
