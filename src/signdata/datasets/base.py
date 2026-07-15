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
from typing import Any, Dict, TYPE_CHECKING

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
    def validate_raw_inputs(cls, raw: Dict[str, Any]) -> None:
        """Validate user-provided raw YAML before defaults are applied.

        Override when an adapter must reject configs that look valid only
        because the loader synthesizes defaults (e.g. ``paths.videos``).
        """
        return None

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

    def resolve_videos_dir(self, config: "Config") -> Path | None:
        """Resolve the directory processors should treat as the video root."""
        videos = config.paths.videos
        return Path(videos) if videos else None

    @staticmethod
    def _set_manifest(
        context: "PipelineContext",
        manifest_path: str | Path,
        manifest_df: Any,
    ) -> None:
        context.manifest_path = Path(manifest_path)
        context.manifest_df = manifest_df

    def validate_loaded_manifest(
        self,
        config: "Config",
        context: "PipelineContext",
    ) -> None:
        """Validate a reused manifest before downstream stages run."""
        return None
