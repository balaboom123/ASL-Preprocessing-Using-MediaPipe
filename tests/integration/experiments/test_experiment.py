"""Tests for Phase 6: Experiment Layer.

Covers:
  - ExperimentConfig / JobEntry schema validation
  - load_experiment() path resolution and error handling
  - ExperimentRunner orchestration (mocked pipeline)
  - CLI argument parsing for the experiment subcommand
  - load_config dict_overrides integration
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from signdata.cli import parse_args
from signdata.config.experiment import (
    ExperimentConfig,
    JobEntry,
    load_experiment,
)
from signdata.pipeline.experiment import ExperimentRunner, JobResult


# ── Schema ────────────────────────────────────────────────────────────

class TestExperimentConfigSchema:
    def test_minimal_valid(self):
        cfg = ExperimentConfig(
            name="test",
            jobs=[JobEntry(config="jobs/test.yaml")],
        )
        assert cfg.name == "test"
        assert cfg.description == ""
        assert len(cfg.jobs) == 1

    def test_with_description(self):
        cfg = ExperimentConfig(
            name="test",
            description="A longer description",
            jobs=[JobEntry(config="a.yaml")],
        )
        assert cfg.description == "A longer description"

    def test_multiple_jobs(self):
        cfg = ExperimentConfig(
            name="multi",
            jobs=[
                JobEntry(config="a.yaml"),
                JobEntry(config="b.yaml", overrides={"x": 1}),
            ],
        )
        assert len(cfg.jobs) == 2
        assert cfg.jobs[1].overrides == {"x": 1}

    def test_jobs_cannot_be_empty(self):
        with pytest.raises(ValueError, match="jobs"):
            ExperimentConfig(name="empty", jobs=[])

    def test_job_defaults(self):
        job = JobEntry(config="test.yaml")
        assert job.overrides == {}

    def test_job_with_overrides(self):
        job = JobEntry(
            config="test.yaml",
            overrides={
                "processing.sample_rate": None,
                "processing.detection_config.device": "cuda:1",
            },
        )
        assert job.overrides["processing.sample_rate"] is None
        assert job.overrides["processing.detection_config.device"] == "cuda:1"

    def test_nested_overrides_rejected(self):
        with pytest.raises(ValueError, match="dot-separated keys"):
            JobEntry(
                config="test.yaml",
                overrides={"processing": {"sample_rate": 30.0}},
            )


# ── load_experiment ───────────────────────────────────────────────────

class TestLoadExperiment:
    def _write_experiment(self, tmp_path, data, subdir="experiments"):
        """Write experiment YAML in configs/experiments/ structure."""
        exp_dir = tmp_path / "configs" / subdir
        exp_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = exp_dir / "test_experiment.yaml"
        yaml_path.write_text(yaml.dump(data))
        return str(yaml_path)

    def test_repository_baseline_jobs_exist(self, project_root):
        path = project_root / "configs" / "experiments" / "baseline_youtube_asl.yaml"
        experiment = load_experiment(str(path))

        assert [Path(job.config).name for job in experiment.jobs] == [
            "mediapipe.yaml",
            "mmpose.yaml",
        ]
        assert all(Path(job.config).is_file() for job in experiment.jobs)

    def test_basic_load(self, tmp_path):
        # Create a job config so the path exists
        jobs_dir = tmp_path / "configs" / "jobs"
        jobs_dir.mkdir(parents=True)
        (jobs_dir / "job_a.yaml").touch()

        path = self._write_experiment(tmp_path, {
            "name": "Test Experiment",
            "jobs": [{"config": "jobs/job_a.yaml"}],
        })

        exp = load_experiment(path)
        assert exp.name == "Test Experiment"
        assert len(exp.jobs) == 1
        # Path resolved to absolute under configs/
        assert exp.jobs[0].config == str(jobs_dir / "job_a.yaml")

    def test_resolves_relative_to_configs_root(self, tmp_path):
        """Job paths resolve relative to configs/, not configs/experiments/."""
        jobs_dir = tmp_path / "configs" / "jobs"
        jobs_dir.mkdir(parents=True)
        (jobs_dir / "my_job.yaml").touch()

        path = self._write_experiment(tmp_path, {
            "name": "test",
            "jobs": [{"config": "jobs/my_job.yaml"}],
        })

        exp = load_experiment(path)
        expected = str(tmp_path / "configs" / "jobs" / "my_job.yaml")
        assert exp.jobs[0].config == expected

    def test_absolute_path_preserved(self, tmp_path):
        abs_path = "/absolute/path/to/job.yaml"
        path = self._write_experiment(tmp_path, {
            "name": "test",
            "jobs": [{"config": abs_path}],
        })

        exp = load_experiment(path)
        assert exp.jobs[0].config == abs_path

    def test_overrides_preserved(self, tmp_path):
        path = self._write_experiment(tmp_path, {
            "name": "test",
            "jobs": [{
                "config": "jobs/a.yaml",
                "overrides": {"processing.sample_rate": None},
            }],
        })

        exp = load_experiment(path)
        assert exp.jobs[0].overrides == {"processing.sample_rate": None}

    def test_missing_name_raises(self, tmp_path):
        path = self._write_experiment(tmp_path, {
            "jobs": [{"config": "a.yaml"}],
        })
        with pytest.raises(ValueError, match="name"):
            load_experiment(path)

    def test_missing_jobs_raises(self, tmp_path):
        path = self._write_experiment(tmp_path, {
            "name": "test",
        })
        with pytest.raises(ValueError, match="jobs"):
            load_experiment(path)

    def test_empty_jobs_raises(self, tmp_path):
        path = self._write_experiment(tmp_path, {
            "name": "test",
            "jobs": [],
        })
        with pytest.raises(ValueError, match="jobs"):
            load_experiment(path)

    def test_non_experiments_dir_resolves_relative_to_parent(self, tmp_path):
        """When experiment file is NOT in experiments/, resolve relative to its dir."""
        custom_dir = tmp_path / "my_configs"
        custom_dir.mkdir()
        (custom_dir / "job.yaml").touch()

        yaml_path = custom_dir / "exp.yaml"
        yaml_path.write_text(yaml.dump({
            "name": "test",
            "jobs": [{"config": "job.yaml"}],
        }))

        exp = load_experiment(str(yaml_path))
        assert exp.jobs[0].config == str(custom_dir / "job.yaml")

    def test_nested_subdir_resolves_to_configs_root(self, tmp_path):
        """configs/experiments/baselines/foo.yaml resolves jobs/ to configs/jobs/."""
        jobs_dir = tmp_path / "configs" / "jobs"
        jobs_dir.mkdir(parents=True)
        (jobs_dir / "job_a.yaml").touch()

        nested_dir = tmp_path / "configs" / "experiments" / "baselines"
        nested_dir.mkdir(parents=True)
        yaml_path = nested_dir / "foo.yaml"
        yaml_path.write_text(yaml.dump({
            "name": "nested test",
            "jobs": [{"config": "jobs/job_a.yaml"}],
        }))

        exp = load_experiment(str(yaml_path))
        assert exp.jobs[0].config == str(jobs_dir / "job_a.yaml")

    def test_deeply_nested_subdir_resolves_to_configs_root(self, tmp_path):
        """configs/experiments/a/b/c/foo.yaml still finds configs/ root."""
        jobs_dir = tmp_path / "configs" / "jobs"
        jobs_dir.mkdir(parents=True)
        (jobs_dir / "deep.yaml").touch()

        deep_dir = tmp_path / "configs" / "experiments" / "a" / "b" / "c"
        deep_dir.mkdir(parents=True)
        yaml_path = deep_dir / "foo.yaml"
        yaml_path.write_text(yaml.dump({
            "name": "deep test",
            "jobs": [{"config": "jobs/deep.yaml"}],
        }))

        exp = load_experiment(str(yaml_path))
        assert exp.jobs[0].config == str(jobs_dir / "deep.yaml")

    def test_multiple_jobs_resolved(self, tmp_path):
        jobs_dir = tmp_path / "configs" / "jobs"
        jobs_dir.mkdir(parents=True)
        (jobs_dir / "a.yaml").touch()
        (jobs_dir / "b.yaml").touch()

        path = self._write_experiment(tmp_path, {
            "name": "multi",
            "jobs": [
                {"config": "jobs/a.yaml"},
                {"config": "jobs/b.yaml", "overrides": {"x": 1}},
            ],
        })

        exp = load_experiment(path)
        assert len(exp.jobs) == 2
        assert exp.jobs[0].config.endswith("a.yaml")
        assert exp.jobs[1].config.endswith("b.yaml")
        assert exp.jobs[1].overrides == {"x": 1}


# ── ExperimentRunner ──────────────────────────────────────────────────

class TestExperimentRunner:
    def _make_experiment(self, jobs):
        return ExperimentConfig(name="test", jobs=jobs)

    @patch("signdata.pipeline.experiment.PipelineRunner")
    @patch("signdata.pipeline.experiment.load_config")
    def test_runs_all_jobs(self, mock_load, mock_runner_cls):
        mock_context = MagicMock()
        mock_context.stats = {"acquire": {"total": 10}}
        mock_runner_cls.return_value.run.return_value = mock_context
        mock_load.return_value = MagicMock()

        exp = self._make_experiment([
            JobEntry(config="/path/a.yaml"),
            JobEntry(config="/path/b.yaml"),
        ])

        runner = ExperimentRunner(exp)
        results = runner.run()

        assert len(results) == 2
        assert all(r.status == "success" for r in results)
        assert mock_load.call_count == 2
        assert mock_runner_cls.call_count == 2

    @patch("signdata.pipeline.experiment.PipelineRunner")
    @patch("signdata.pipeline.experiment.load_config")
    def test_continues_after_failure(self, mock_load, mock_runner_cls):
        """If job 1 fails, job 2 still runs."""
        mock_context = MagicMock()
        mock_context.stats = {}

        # First call fails, second succeeds
        mock_runner_instance = MagicMock()
        mock_runner_instance.run.side_effect = [
            RuntimeError("job 1 failed"),
            mock_context,
        ]
        mock_runner_cls.return_value = mock_runner_instance
        mock_load.return_value = MagicMock()

        exp = self._make_experiment([
            JobEntry(config="/path/a.yaml"),
            JobEntry(config="/path/b.yaml"),
        ])

        runner = ExperimentRunner(exp)
        results = runner.run()

        assert len(results) == 2
        assert results[0].status == "failed"
        assert "job 1 failed" in results[0].error
        assert results[1].status == "success"

    @patch("signdata.pipeline.experiment.PipelineRunner")
    @patch("signdata.pipeline.experiment.load_config")
    def test_force_all_propagated(self, mock_load, mock_runner_cls):
        mock_context = MagicMock()
        mock_context.stats = {}
        mock_runner_cls.return_value.run.return_value = mock_context
        mock_load.return_value = MagicMock()

        exp = self._make_experiment([
            JobEntry(config="/path/a.yaml"),
        ])

        runner = ExperimentRunner(exp, force_all=True)
        runner.run()

        mock_runner_cls.assert_called_once_with(
            mock_load.return_value, force_all=True,
        )

    @patch("signdata.pipeline.experiment.PipelineRunner")
    @patch("signdata.pipeline.experiment.load_config")
    def test_overrides_passed_to_load_config(self, mock_load, mock_runner_cls):
        mock_context = MagicMock()
        mock_context.stats = {}
        mock_runner_cls.return_value.run.return_value = mock_context
        mock_load.return_value = MagicMock()

        exp = self._make_experiment([
            JobEntry(
                config="/path/a.yaml",
                overrides={"processing.sample_rate": None},
            ),
        ])

        runner = ExperimentRunner(exp)
        runner.run()

        mock_load.assert_called_once_with(
            "/path/a.yaml",
            dict_overrides={"processing.sample_rate": None},
        )

    @patch("signdata.pipeline.experiment.PipelineRunner")
    @patch("signdata.pipeline.experiment.load_config")
    def test_no_overrides_passes_none(self, mock_load, mock_runner_cls):
        mock_context = MagicMock()
        mock_context.stats = {}
        mock_runner_cls.return_value.run.return_value = mock_context
        mock_load.return_value = MagicMock()

        exp = self._make_experiment([
            JobEntry(config="/path/a.yaml"),
        ])

        runner = ExperimentRunner(exp)
        runner.run()

        mock_load.assert_called_once_with(
            "/path/a.yaml", dict_overrides=None,
        )

    @patch("signdata.pipeline.experiment.PipelineRunner")
    @patch("signdata.pipeline.experiment.load_config")
    def test_stats_collected(self, mock_load, mock_runner_cls):
        mock_context = MagicMock()
        mock_context.stats = {"extract": {"total": 100}}
        mock_runner_cls.return_value.run.return_value = mock_context
        mock_load.return_value = MagicMock()

        exp = self._make_experiment([
            JobEntry(config="/path/a.yaml"),
        ])

        runner = ExperimentRunner(exp)
        results = runner.run()

        assert results[0].stats == {"extract": {"total": 100}}

    @patch("signdata.pipeline.experiment.PipelineRunner")
    @patch("signdata.pipeline.experiment.load_config")
    def test_all_jobs_fail(self, mock_load, mock_runner_cls):
        mock_runner_cls.return_value.run.side_effect = RuntimeError("boom")
        mock_load.return_value = MagicMock()

        exp = self._make_experiment([
            JobEntry(config="/path/a.yaml"),
            JobEntry(config="/path/b.yaml"),
        ])

        runner = ExperimentRunner(exp)
        results = runner.run()

        assert len(results) == 2
        assert all(r.status == "failed" for r in results)

    @patch("signdata.pipeline.experiment.PipelineRunner")
    @patch("signdata.pipeline.experiment.load_config")
    def test_config_load_failure_captured(self, mock_load, mock_runner_cls):
        """If load_config fails, the job is marked as failed."""
        mock_load.side_effect = ValueError("bad config")

        exp = self._make_experiment([
            JobEntry(config="/path/a.yaml"),
        ])

        runner = ExperimentRunner(exp)
        results = runner.run()

        assert results[0].status == "failed"
        assert "bad config" in results[0].error


# ── CLI ───────────────────────────────────────────────────────────────

class TestExperimentCLI:
    def test_experiment_subcommand(self):
        args = parse_args(["experiment", "exp.yaml"])
        assert args.command == "experiment"
        assert args.config == "exp.yaml"

    def test_experiment_force_all(self):
        args = parse_args(["experiment", "exp.yaml", "--force-all"])
        assert args.force_all is True

    def test_experiment_force_all_default(self):
        args = parse_args(["experiment", "exp.yaml"])
        assert args.force_all is False

    def test_experiment_config_path_preserved(self):
        args = parse_args(["experiment", "/abs/path/exp.yaml"])
        assert args.config == "/abs/path/exp.yaml"


# ── load_config dict_overrides ────────────────────────────────────────

class TestLoadConfigDictOverrides:
    def _write_job_config(self, tmp_path):
        """Write a minimal job config in the standard directory structure."""
        configs_dir = tmp_path / "configs" / "datasets"
        configs_dir.mkdir(parents=True)
        yaml_path = configs_dir / "test_job.yaml"
        yaml_path.write_text(yaml.dump({
            "dataset": {
                "name": "youtube_asl",
                "source": {"video_ids_file": "assets/ids.txt"},
            },
            "processing": {"enabled": False, "sample_rate": 24.0},
        }))
        # Create the assets file so validation passes
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir(exist_ok=True)
        (assets_dir / "ids.txt").write_text("vid1\nvid2\n")
        return str(yaml_path)

    def test_dict_overrides_applied(self, tmp_path):
        from signdata.config.loader import load_config

        path = self._write_job_config(tmp_path)
        config = load_config(path, dict_overrides={
            "processing.sample_rate": 30.0,
        })
        assert config.processing.sample_rate == 30.0

    def test_dict_overrides_null_value(self, tmp_path):
        from signdata.config.loader import load_config

        path = self._write_job_config(tmp_path)
        config = load_config(path, dict_overrides={
            "processing.sample_rate": None,
        })
        assert config.processing.sample_rate is None

    def test_dict_overrides_after_cli_overrides(self, tmp_path):
        """Dict overrides take precedence over CLI overrides."""
        from signdata.config.loader import load_config

        path = self._write_job_config(tmp_path)
        config = load_config(
            path,
            overrides=["processing.sample_rate=10"],
            dict_overrides={"processing.sample_rate": 20},
        )
        assert config.processing.sample_rate == 20

    def test_dict_overrides_none_is_noop(self, tmp_path):
        from signdata.config.loader import load_config

        path = self._write_job_config(tmp_path)
        config = load_config(path, dict_overrides=None)
        assert config.processing.sample_rate == 24.0

    def test_removed_sampling_override_rejected(self, tmp_path):
        from signdata.config.loader import load_config

        path = self._write_job_config(tmp_path)

        with pytest.raises(ValueError, match="use processing.sample_rate"):
            load_config(path, dict_overrides={"processing.frame_skip": 2})

    def test_dict_overrides_dataset_validated_after_override(self, tmp_path):
        """Overriding dataset via dict_overrides validates the overridden name."""
        from signdata.config.loader import load_config

        # Base YAML has dataset=youtube_asl (valid).  Dict override changes
        # it to a name not in the registry.  With correct ordering the
        # registry check catches the overridden name; without the fix it
        # would validate "youtube_asl" and silently accept the bad name.
        path = self._write_job_config(tmp_path)

        with pytest.raises(ValueError, match="Unknown dataset 'nonexistent'"):
            load_config(str(path), dict_overrides={"dataset": "nonexistent"})
