"""Validate shipped configs against the strict schemas."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from signdata.config.experiment import load_experiment
from signdata.config.loader import _load_raw_config
from signdata.config.schema import Config

CONFIGS = Path(__file__).resolve().parents[3] / "configs"


@pytest.mark.parametrize(
    "path",
    sorted((CONFIGS / "jobs").rglob("*.yaml"))
    + sorted((CONFIGS / "test").rglob("*.yaml")),
    ids=lambda path: str(path.relative_to(CONFIGS)),
)
def test_job_config_validates(path):
    Config(**_load_raw_config(str(path)))


@pytest.mark.parametrize(
    "path",
    sorted((CONFIGS / "experiments").rglob("*.yaml")),
    ids=lambda path: path.name,
)
def test_experiment_config_validates(path):
    load_experiment(str(path))


def test_unknown_config_keys_are_rejected():
    with pytest.raises(ValidationError, match="stop_at"):
        Config(dataset={"name": "youtube_asl"}, stop_at="extract")

    with pytest.raises(ValidationError, match="sample_rat"):
        Config(
            dataset={"name": "youtube_asl"},
            processing={"enabled": False, "sample_rat": 0.5},
        )
