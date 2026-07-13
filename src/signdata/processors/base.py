"""Base processor class for pipeline steps."""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..pipeline.context import PipelineContext

from ..config.schema import Config


class BaseProcessor(ABC):
    """Abstract base class for pipeline processing steps.

    Subclasses must set ``name`` and implement ``run()``.
    """

    name: str  # Must match registry key

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(f"signdata.{self.name}")

    @abstractmethod
    def run(self, context: "PipelineContext") -> "PipelineContext":
        """Execute this processing step. Return updated context."""
        ...
