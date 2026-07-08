"""Pipeline context for shared state between processing stages."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from ..config.schema import Config
from ..datasets.base import DatasetAdapter


@dataclass
class PipelineContext:
    """Shared state passed between pipeline stages.

    The runner calls ``resolve_paths()`` once at startup to scope
    output and webdataset directories under ``run_name``.
    """

    config: Config
    dataset: DatasetAdapter

    # Run-scoped artifact directories (set by resolve_paths)
    output_dir: Optional[Path] = None
    webdataset_dir: Optional[Path] = None

    # Static paths (shared across runs)
    videos_dir: Optional[Path] = None
    manifest_path: Optional[Path] = None
    manifest_df: Optional["pd.DataFrame"] = None

    # Flags
    force_all: bool = False

    # Tracking
    completed_stages: List[str] = field(default_factory=list)
    stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def resolve_paths(self) -> None:
        """Scope output/webdataset under run_name to isolate runs."""
        cfg = self.config
        self.output_dir = Path(cfg.paths.output) / cfg.run_name
        self.webdataset_dir = Path(cfg.paths.webdataset) / cfg.run_name
        self.videos_dir = self.dataset.resolve_videos_dir(cfg)
        self.manifest_path = Path(cfg.paths.manifest) if cfg.paths.manifest else None

    def load_manifest(self, manifest_path: str) -> None:
        """Load an existing manifest (used when dataset.manifest is false)."""
        from ..utils.manifest import read_manifest

        self.manifest_path = Path(manifest_path)
        self.manifest_df = read_manifest(manifest_path, normalize_columns=True)
