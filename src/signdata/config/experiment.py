"""Experiment config schema and loader."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class JobEntry(BaseModel):
    """A single pipeline job within an experiment."""

    config: str  # path to job YAML (relative to configs/ or absolute)
    overrides: dict[str, Any] = Field(default_factory=dict)

    @field_validator("overrides")
    @classmethod
    def require_flat_overrides(
        cls, overrides: dict[str, Any]
    ) -> dict[str, Any]:
        if any(
            "." not in key and isinstance(value, dict)
            for key, value in overrides.items()
        ):
            raise ValueError("Experiment overrides must use dot-separated keys")
        return overrides


class ExperimentConfig(BaseModel):
    """Top-level experiment definition.

    Attributes:
        name: Human-readable experiment name (for logging).
        description: Optional longer description.
        jobs: Ordered list of pipeline jobs to execute.
    """

    name: str
    description: str = ""
    jobs: list[JobEntry] = Field(min_length=1)


def load_experiment(yaml_path: str) -> ExperimentConfig:
    """Load an experiment config from YAML.

    Job config paths are resolved relative to the nearest ``configs/``
    ancestor directory. If no ``configs/`` ancestor exists, paths
    resolve relative to the experiment file's directory.

    Args:
        yaml_path: Path to the experiment YAML file.

    Returns:
        Validated :class:`ExperimentConfig` with resolved job paths.
    """
    yaml_path = Path(yaml_path).resolve()
    experiment_dir = yaml_path.parent

    configs_root = next(
        (
            path
            for path in (experiment_dir, *experiment_dir.parents)
            if path.name == "configs"
        ),
        experiment_dir,
    )

    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}

    experiment = ExperimentConfig.model_validate(raw)

    # Resolve relative job config paths
    for job in experiment.jobs:
        if not Path(job.config).is_absolute():
            job.config = str(configs_root / job.config)

    return experiment
