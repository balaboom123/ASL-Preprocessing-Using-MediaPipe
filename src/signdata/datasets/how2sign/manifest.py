"""How2Sign manifest loading."""

from pathlib import Path

from ...utils.manifest import read_manifest
from .source import How2SignSourceConfig


def build(source: How2SignSourceConfig):
    """Load the existing How2Sign manifest CSV."""
    manifest_path = source.manifest_csv

    if not manifest_path or not Path(manifest_path).exists():
        raise FileNotFoundError(
            f"How2Sign manifest not found: {manifest_path}\n"
            f"Provide a valid manifest path via paths.manifest in config."
        )

    return read_manifest(manifest_path)
