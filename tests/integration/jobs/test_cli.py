"""Tests for CLI argument parsing (cli.py)."""

import pytest

from signdata.cli import parse_args


class TestParseArgs:
    def test_run_config_only(self):
        args = parse_args(["run", "config.yaml"])
        assert args.command == "run"
        assert args.config == "config.yaml"
        assert args.override == []

    def test_run_with_overrides(self):
        args = parse_args(["run", "config.yaml", "--override", "a=1", "b=2"])
        assert args.config == "config.yaml"
        assert args.override == ["a=1", "b=2"]

    def test_missing_subcommand(self):
        with pytest.raises(SystemExit):
            parse_args([])

    def test_run_override_empty_list(self):
        args = parse_args(["run", "my.yaml", "--override"])
        assert args.override == []

    def test_run_config_path_preserved(self):
        args = parse_args(["run", "/path/to/configs/test.yaml"])
        assert args.config == "/path/to/configs/test.yaml"

    def test_run_name_flag(self):
        args = parse_args(["run", "job.yaml", "--run-name", "exp1"])
        assert args.run_name == "exp1"

    def test_list_presets_flag(self):
        args = parse_args(["run", "job.yaml", "--list-presets"])
        assert args.list_presets is True

    def test_force_all_flag(self):
        args = parse_args(["run", "job.yaml", "--force-all"])
        assert args.force_all is True

    def test_run_missing_config_is_none(self):
        """config is optional (for --list-presets); None when omitted."""
        args = parse_args(["run"])
        assert args.config is None

    def test_list_presets_without_config(self):
        args = parse_args(["run", "--list-presets"])
        assert args.list_presets is True
        assert args.config is None
