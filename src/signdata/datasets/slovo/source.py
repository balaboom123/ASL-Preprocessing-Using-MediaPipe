"""SLoVo source config, path resolution, and release validation."""

import logging
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .._ingestion.availability import AvailabilityPolicy

# Optional class map path used when class_map_mode="bundled".
_DEFAULT_CLASS_MAP = (
    Path(__file__).resolve().parents[4] / "assets" / "slovo_class_map.tsv"
)

# Columns that must be present in annotations.csv
REQUIRED_COLUMNS = {"attachment_id", "user_id", "text", "train"}

# Columns that may be present and map to canonical passthrough names
OPTIONAL_PASSTHROUGH = {
    "width": "SRC_WIDTH",
    "height": "SRC_HEIGHT",
    "length": "FRAME_COUNT",
    "begin": "FRAME_START",
    "end": "FRAME_END",
}


class SlovoSourceConfig(BaseModel):
    """Typed config for the SLoVo adapter."""

    model_config = ConfigDict(extra="forbid")

    release_dir: str = ""
    annotations_csv: str = ""
    split: Literal["all", "train", "test"] = "all"
    availability_policy: AvailabilityPolicy = "fail_fast"
    class_map_file: str = ""
    class_map_mode: Literal["derive", "bundled", "none"] = "derive"
    include_background: bool = True
    background_labels: list[str] = Field(
        default_factory=lambda: ["no_event"]
    )


def get_source_config(config) -> SlovoSourceConfig:
    source_dict = dict(config.dataset.source)
    if not source_dict.get("release_dir") and config.paths.videos:
        source_dict["release_dir"] = config.paths.videos
    return SlovoSourceConfig(**source_dict)


def resolve_release_dir(source: SlovoSourceConfig, config) -> str:
    return source.release_dir or (config.paths.videos or "")


def resolve_annotations_csv(source: SlovoSourceConfig, video_dir: str) -> str:
    return source.annotations_csv or str(Path(video_dir) / "annotations.csv")


def load_annotations(path: str | Path) -> pd.DataFrame:
    ann = pd.read_csv(path)
    missing_cols = REQUIRED_COLUMNS - set(ann.columns)
    if missing_cols:
        raise ValueError(
            "SLoVo annotations.csv is missing required columns: "
            f"{sorted(missing_cols)}. Available columns: {list(ann.columns)}"
        )
    return ann


def parse_train_col(val) -> bool:
    """Robustly parse the ``train`` column value to a boolean."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes")
    return False


def validate(source: SlovoSourceConfig, config, log: logging.Logger) -> dict:
    """Validate SLoVo release directory and annotations."""
    video_dir = resolve_release_dir(source, config)
    if not video_dir:
        raise ValueError(
            "SLoVo requires a release directory. "
            "Set dataset.source.release_dir or paths.videos in your config YAML."
        )
    if not Path(video_dir).exists():
        raise FileNotFoundError(
            f"SLoVo release directory not found: {video_dir}\n"
            "SLoVo requires manual download. "
            "See https://github.com/hukenovs/slovo for instructions."
        )

    annotations_csv = resolve_annotations_csv(source, video_dir)
    if not Path(annotations_csv).exists():
        raise FileNotFoundError(
            f"SLoVo annotations CSV not found: {annotations_csv}\n"
            "Expected annotations.csv inside the release directory, or "
            "provide an explicit path via dataset.source.annotations_csv."
        )

    ann = load_annotations(annotations_csv)
    row_count = len(ann)

    log.info(
        "SLoVo release directory validated: %s (%d annotation rows)",
        video_dir, row_count,
    )
    return {"validated": True, "rows_found": row_count}


def get_bundled_class_map_path(source: SlovoSourceConfig) -> Path:
    return Path(source.class_map_file) if source.class_map_file else _DEFAULT_CLASS_MAP
