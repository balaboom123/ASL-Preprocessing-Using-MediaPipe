"""How2Sign source config and validation."""

import logging
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class How2SignSourceConfig(BaseModel):
    """Typed config for How2Sign adapter."""

    model_config = ConfigDict(extra="forbid")

    manifest_csv: str = ""


def get_source_config(config) -> How2SignSourceConfig:
    """Parse ``config.dataset.source`` dict into typed model."""
    source_dict = dict(config.dataset.source)
    source_dict["manifest_csv"] = source_dict.get("manifest_csv") or config.paths.manifest
    return How2SignSourceConfig(**source_dict)


def validate(
    config,
    log: logging.Logger,
) -> dict:
    """Validate that How2Sign video directory exists."""
    video_dir = config.paths.videos

    if not video_dir:
        raise ValueError(
            "paths.videos is required for How2Sign. "
            "Set it in your config YAML."
        )

    if not Path(video_dir).exists():
        raise FileNotFoundError(
            f"How2Sign video directory not found: {video_dir}\n"
            "How2Sign requires manual download. "
            "See https://how2sign.github.io/ for instructions."
        )

    log.info("How2Sign video directory validated: %s", video_dir)
    return {"validated": True}
