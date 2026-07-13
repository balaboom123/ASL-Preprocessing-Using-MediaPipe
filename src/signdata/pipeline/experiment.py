"""Experiment runner — execute multiple pipeline jobs sequentially."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config.experiment import ExperimentConfig
from ..config.loader import load_config
from .runner import PipelineRunner

logger = logging.getLogger(__name__)


@dataclass
class JobResult:
    """Result of a single job within an experiment."""

    config: str
    status: str  # "success" or "failed"
    stats: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class ExperimentRunner:
    """Execute experiment jobs sequentially, continuing after failures."""

    def __init__(
        self,
        experiment: ExperimentConfig,
        force_all: bool = False,
    ):
        self.experiment = experiment
        self.force_all = force_all

    def run(self) -> list[JobResult]:
        """Run all jobs and return per-job results."""
        n_jobs = len(self.experiment.jobs)

        logger.info("=" * 70)
        logger.info("Experiment: %s", self.experiment.name)
        if self.experiment.description:
            logger.info("  %s", self.experiment.description)
        logger.info("Jobs: %d", n_jobs)
        logger.info("=" * 70)

        results: list[JobResult] = []

        for idx, job in enumerate(self.experiment.jobs, start=1):
            job_label = Path(job.config).name
            logger.info(
                "[%d/%d] Running: %s", idx, n_jobs, job_label,
            )

            dict_overrides = job.overrides or None

            if dict_overrides:
                logger.info(
                    "  Overrides: %s",
                    ", ".join(f"{key}={value}" for key, value in dict_overrides.items()),
                )

            try:
                config = load_config(job.config, dict_overrides=dict_overrides)
                context = PipelineRunner(
                    config, force_all=self.force_all,
                ).run()

                results.append(JobResult(
                    config=job.config,
                    status="success",
                    stats=context.stats,
                ))
                logger.info("[%d/%d] Completed: %s", idx, n_jobs, job_label)

            except Exception as exc:
                logger.error(
                    "[%d/%d] Failed: %s — %s", idx, n_jobs, job_label, exc,
                )
                results.append(JobResult(
                    config=job.config,
                    status="failed",
                    error=str(exc),
                ))

        # Summary
        failed = sum(1 for r in results if r.status == "failed")
        succeeded = n_jobs - failed

        logger.info("=" * 70)
        logger.info(
            "Experiment complete: %d/%d succeeded, %d failed",
            succeeded, n_jobs, failed,
        )

        if failed:
            logger.error(
                "Failed jobs: %s",
                ", ".join(
                    Path(result.config).name
                    for result in results
                    if result.status == "failed"
                ),
            )

        logger.info("=" * 70)

        return results
