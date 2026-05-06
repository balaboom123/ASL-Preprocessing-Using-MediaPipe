"""Base dataset adapter class.

Dataset adapters bridge external data sources into the signdata pipeline.
Each adapter is responsible for:
- Downloading raw data (or validating existence)
- Building a manifest from raw data in the canonical format

Adapters never do experiment processing (pose extraction, normalization, etc.).
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from ..config.schema import Config
    from ..pipeline.context import PipelineContext


class DatasetAdapter(ABC):
    """Abstract base class for dataset adapters.

    Subclasses must implement ``download`` and ``build_manifest``.
    Override ``validate_config`` for dataset-specific config validation.
    Override ``get_source_config`` to parse adapter-specific typed config.
    """

    name: str

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"signdata.dataset.{self.name}")

    @classmethod
    def validate_config(cls, config: "Config") -> None:
        """Validate config for this dataset. Override for custom checks."""
        pass

    @abstractmethod
    def download(self, config: "Config", context: "PipelineContext") -> "PipelineContext":
        """Download raw data (or validate existence).

        For web-mined datasets: download videos and transcripts.
        For local datasets: validate that required files exist.

        Returns the updated context.
        """
        ...

    @abstractmethod
    def build_manifest(self, config: "Config", context: "PipelineContext") -> "PipelineContext":
        """Build a manifest from raw data.

        Must produce a TSV manifest file and set ``context.manifest_path``
        and ``context.manifest_df``.

        Returns the updated context.
        """
        ...

    def get_source_config(self, config: "Config") -> BaseModel:
        """Parse adapter-specific config into a typed Pydantic model.

        Reads from ``config.dataset.source`` dict and returns a typed SourceConfig.
        Override in subclasses to return a dataset-specific model.
        """
        return BaseModel()

    def resolve_videos_dir(self, config: "Config") -> Path | None:
        """Resolve the directory processors should treat as the video root."""
        videos = getattr(config.paths, "videos", "")
        return Path(videos) if videos else None


# Backward-compat alias
BaseDataset = DatasetAdapter
