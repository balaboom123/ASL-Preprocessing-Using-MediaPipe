"""Four-stage pipeline runner."""

import logging
from typing import Optional

from ..config.schema import Config
from ..registry import (
    DATASET_REGISTRY,
    PROCESSOR_REGISTRY,
    POST_PROCESSOR_REGISTRY,
    OUTPUT_REGISTRY,
)
from .context import PipelineContext

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Execute the standardized 4-stage pipeline.

    Stages: dataset → processing → post_processing → output
    Each stage has an ``enabled`` flag to turn on/off.
    """

    def __init__(
        self,
        config: Config,
        force_all: bool = False,
    ):
        self.config = config
        self.dataset = DATASET_REGISTRY[config.dataset.name]()
        self.force_all = force_all

    def run(self) -> PipelineContext:
        context = PipelineContext(
            config=self.config,
            dataset=self.dataset,
            force_all=self.force_all,
        )

        # Resolve run-scoped artifact paths
        context.resolve_paths()

        logger.info("=" * 70)
        logger.info(
            "Pipeline: dataset=%s  processor=%s  run_name=%s",
            self.config.dataset.name,
            self.config.processing.processor if self.config.processing.enabled else "(disabled)",
            self.config.run_name,
        )
        logger.info("=" * 70)

        # Stage 1: dataset (download + manifest)
        if self.config.dataset.download:
            logger.info("Running: dataset.download")
            context = self.dataset.download(self.config, context)
            context.completed_stages.append("dataset.download")

        if self.config.dataset.manifest:
            logger.info("Running: dataset.manifest")
            context = self.dataset.build_manifest(self.config, context)
            context.completed_stages.append("dataset.manifest")
        else:
            # Load existing manifest so downstream stages can iterate it
            if self.config.paths.manifest:
                context.load_manifest(self.config.paths.manifest)
                self.dataset.validate_loaded_manifest(self.config, context)

        # Filter out unavailable rows before downstream stages
        if context.manifest_df is not None:
            from ..utils.availability import filter_available
            context.manifest_df = filter_available(context.manifest_df)

        # Stage 2: processing
        if self.config.processing.enabled:
            processor_name = self.config.processing.processor
            logger.info("Running: processing (%s)", processor_name)
            if processor_name not in PROCESSOR_REGISTRY:
                raise ValueError(
                    f"Unknown processor '{processor_name}'. "
                    f"Available: {list(PROCESSOR_REGISTRY.keys())}"
                )
            processor = PROCESSOR_REGISTRY[processor_name](self.config)
            context = processor.run(context)
            context.completed_stages.append(f"processing.{processor_name}")

        # Stage 3: post_processing
        if self.config.post_processing.enabled:
            for recipe_name in self.config.post_processing.recipes:
                logger.info("Running: post_processing.%s", recipe_name)
                if recipe_name not in POST_PROCESSOR_REGISTRY:
                    raise ValueError(
                        f"Unknown post-processor '{recipe_name}'. "
                        f"Available: {list(POST_PROCESSOR_REGISTRY.keys())}"
                    )
                pp = POST_PROCESSOR_REGISTRY[recipe_name](self.config)
                context = pp.run(context)
                context.completed_stages.append(f"post_processing.{recipe_name}")

        # Stage 4: output
        if self.config.output.enabled:
            output_type = self.config.output.type
            logger.info("Running: output (%s)", output_type)
            if output_type not in OUTPUT_REGISTRY:
                raise ValueError(
                    f"Unknown output type '{output_type}'. "
                    f"Available: {list(OUTPUT_REGISTRY.keys())}"
                )
            output = OUTPUT_REGISTRY[output_type](self.config)
            context = output.run(context)
            context.completed_stages.append(f"output.{output_type}")

        logger.info("Pipeline complete.")
        return context
