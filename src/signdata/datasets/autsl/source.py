"""AUTSL source config, path resolution, and release validation."""

import logging
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .._ingestion.availability import AvailabilityPolicy

# Modality suffix appended to the base sample key in filenames.
MODALITY_SUFFIX: dict[str, str] = {
    "rgb": "_color",
    "depth": "_depth",
}

# Known filesystem aliases for split directory names.
SPLIT_ALIASES: dict[str, list[str]] = {
    "val": ["validation"],
}

LABEL_OVERRIDE_FIELDS = {
    "train": "train_labels_file",
    "val": "val_labels_file",
    "test": "test_labels_file",
}


class AUTSLSourceConfig(BaseModel):
    """Typed config for AUTSL adapter."""

    model_config = ConfigDict(extra="forbid")

    release_dir: str = ""
    split: Literal["train", "val", "test", "all"] = "train"
    modality: Literal["rgb", "depth"] = "rgb"
    availability_policy: AvailabilityPolicy = "fail_fast"
    allow_unlabeled: bool = False
    class_id_file: str = ""
    train_labels_file: str = ""
    val_labels_file: str = ""
    test_labels_file: str = ""


def get_source_config(config) -> AUTSLSourceConfig:
    source_dict = dict(config.dataset.source)
    if not source_dict.get("release_dir") and config.paths.videos:
        source_dict["release_dir"] = config.paths.videos
    return AUTSLSourceConfig(**source_dict)


def resolve_release_root(source: AUTSLSourceConfig, config) -> Path:
    raw = source.release_dir or (config.paths.videos or "")
    return Path(raw)


def get_selected_splits(source: AUTSLSourceConfig) -> tuple[str, ...]:
    return ("train", "val", "test") if source.split == "all" else (source.split,)


def parse_signer_id(sample_key: str) -> str:
    """Extract the numeric signer ID from a sample key such as 'signer0_sample1'."""
    match = re.match(r"signer(\d+)", sample_key)
    return match.group(1) if match else ""


def discover_split_dir(release_root: Path, split: str) -> Path:
    """Return the first existing split directory under *release_root*.

    Raises
    ------
    FileNotFoundError
        When neither the canonical name nor any alias exists.
    """
    candidates = [split] + SPLIT_ALIASES.get(split, [])
    for name in candidates:
        candidate = release_root / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"AUTSL split directory not found for split='{split}' under "
        f"{release_root}. Tried: {candidates}. "
        f"Ensure the dataset is extracted correctly."
    )


def discover_class_id_file(release_root: Path) -> Path | None:
    """Search *release_root* recursively for a class correspondence CSV."""
    for pattern in ("SignList*.csv", "classId*.csv", "class_id*.csv",
                    "*class*correspondence*.csv", "*sign_list*.csv"):
        for match in release_root.rglob(pattern):
            return match
    return None


def discover_labels_file(release_root: Path, split: str) -> Path | None:
    """Search *release_root* for the labels file for *split*."""
    for split_name in [split] + SPLIT_ALIASES.get(split, []):
        for path in (
            release_root / f"{split_name}_labels.csv",
            release_root / f"{split_name}" / f"{split_name}_labels.csv",
            release_root / "labels" / f"{split_name}_labels.csv",
        ):
            if path.exists():
                return path
    return None


def resolve_class_id_file(
    source: AUTSLSourceConfig,
    release_root: Path,
) -> Path | None:
    if source.class_id_file:
        return Path(source.class_id_file)
    return discover_class_id_file(release_root)


def resolve_labels_file(
    split: str,
    source: AUTSLSourceConfig,
    release_root: Path,
) -> Path | None:
    field_name = LABEL_OVERRIDE_FIELDS.get(split)
    explicit = getattr(source, field_name, "") if field_name else ""
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(
                f"AUTSL dataset.source.{field_name} not found: {p}"
            )
        return p
    return discover_labels_file(release_root, split)


def validate(
    source: AUTSLSourceConfig,
    config,
    log: logging.Logger,
) -> dict:
    """Validate AUTSL release directory and required files."""
    release_root = resolve_release_root(source, config)
    if not release_root.exists():
        raise FileNotFoundError(
            f"AUTSL release directory not found: {release_root}\n"
            f"Set dataset.source.release_dir (or paths.videos) to the "
            f"extracted challenge root and try again."
        )

    selected_splits = get_selected_splits(source)
    resolved_split_dirs: dict[str, str] = {}
    for split_name in selected_splits:
        split_dir = discover_split_dir(release_root, split_name)
        resolved_split_dirs[split_name] = split_dir.name

    class_id_path = resolve_class_id_file(source, release_root)
    if class_id_path is None or not class_id_path.exists():
        raise FileNotFoundError(
            f"AUTSL class correspondence file not found. "
            f"Specify dataset.source.class_id_file or ensure a file "
            f"matching 'SignList*.csv' / 'classId*.csv' exists under "
            f"{release_root}."
        )

    for field_name in LABEL_OVERRIDE_FIELDS.values():
        explicit = getattr(source, field_name)
        if explicit and not Path(explicit).exists():
            raise FileNotFoundError(
                f"AUTSL dataset.source.{field_name} not found: {explicit}"
            )

    log.info(
        "AUTSL release validated: root=%s, splits=%s, class_file=%s",
        release_root, sorted(resolved_split_dirs), class_id_path,
    )
    return {
        "validated": True,
        "release_root": str(release_root),
        "splits": resolved_split_dirs,
        "modality": source.modality,
        "class_id_file": str(class_id_path),
    }
