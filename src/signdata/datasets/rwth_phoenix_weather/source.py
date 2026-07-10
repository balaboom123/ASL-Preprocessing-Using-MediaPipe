"""RWTH-PHOENIX-Weather source config, path resolution, and preparation."""

import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from pydantic import BaseModel

from .._ingestion.availability import AvailabilityPolicy
from .._ingestion.media import materialize_frames_to_video

ALL_SPLITS = ("train", "dev", "test")
VALID_SPLITS = {*ALL_SPLITS, "all"}
VALID_PREPARE_MODES = {"validate", "materialize_missing", "rematerialize_all"}


class RWTHPhoenixWeatherSourceConfig(BaseModel):
    """Typed source config for the RWTH-PHOENIX-Weather adapter."""

    release_dir: str = ""
    variant: str = "phoenix_2014_t"
    split: str = "all"
    prepare_mode: str = "materialize_missing"
    availability_policy: AvailabilityPolicy = "drop_unavailable"
    video_fps: float = 25.0


def get_source_config(config) -> RWTHPhoenixWeatherSourceConfig:
    source_dict = dict(config.dataset.source)
    if not source_dict.get("release_dir") and getattr(config.paths, "videos", ""):
        source_dict["release_dir"] = config.paths.videos
    return RWTHPhoenixWeatherSourceConfig(**source_dict)


def validate_source_config(source: RWTHPhoenixWeatherSourceConfig) -> None:
    if source.split not in VALID_SPLITS:
        raise ValueError(
            f"Unsupported PHOENIX split '{source.split}'. "
            f"Expected one of {sorted(VALID_SPLITS)}."
        )
    if source.prepare_mode not in VALID_PREPARE_MODES:
        raise ValueError(
            f"Unsupported PHOENIX prepare_mode '{source.prepare_mode}'. "
            f"Expected one of {sorted(VALID_PREPARE_MODES)}."
        )
    if source.video_fps <= 0:
        raise ValueError("PHOENIX video_fps must be positive.")


def find_corpus_csvs(release_dir: Path, split: str) -> List[Path]:
    """Locate PHOENIX corpus CSV files for *split* under *release_dir*."""
    pattern = f"PHOENIX-2014-T.{split}.corpus.csv"
    return sorted(release_dir.rglob(pattern))


def derive_clip_id(name: str) -> str:
    """Normalise a PHOENIX folder/id field into a safe file stem."""
    return _normalise_frame_path(name).lstrip("/").replace("/", "_").replace(" ", "_")


def _normalise_frame_path(path_value: str) -> str:
    """Strip trailing frame-glob suffixes from PHOENIX folder paths."""
    path = Path(path_value.strip().rstrip("/"))
    return str(path.parent) if any(char in path.name for char in "*?[]") else str(path)


def _get_first_present_column(columns, *candidates: str) -> str | None:
    """Return the first matching column name from *candidates*."""
    return next((column for column in candidates if column in columns), None)


def get_identifier_column(columns) -> str | None:
    """Return the preferred PHOENIX clip identifier column."""
    return _get_first_present_column(columns, "id", "name", "video")


def get_frame_path_column(columns) -> str | None:
    """Return the preferred PHOENIX frame-path column."""
    return _get_first_present_column(columns, "folder", "video")


def get_signer_column(columns) -> str | None:
    """Return the preferred PHOENIX signer/speaker column."""
    return _get_first_present_column(columns, "signer", "speaker")


def _iter_release_anchors(csv_path: Path, release_dir: Path):
    """Yield resolver anchors that stay confined to the configured release root."""
    yield release_dir

    for parent in csv_path.parents:
        if parent == release_dir:
            break
        if parent.is_relative_to(release_dir):
            yield parent


def _resolve_frame_dir(
    csv_path: Path,
    release_dir: Path,
    split: str,
    path_value: str,
) -> Path:
    """Resolve a frame directory from either plan-style or official PHOENIX metadata."""
    normalized = _normalise_frame_path(path_value).lstrip("/")
    candidates: list[Path] = []

    for anchor in _iter_release_anchors(csv_path, release_dir):
        for base_dir in (
            anchor,
            anchor / split,
            anchor / "features" / "fullFrame-210x260px",
            anchor / "features" / "fullFrame-210x260px" / split,
        ):
            candidate = base_dir / normalized if normalized else base_dir
            if candidate not in candidates:
                candidates.append(candidate)
            if candidate.is_dir():
                return candidate

    return candidates[0] if candidates else release_dir / normalized


def prepare(
    source: RWTHPhoenixWeatherSourceConfig,
    config,
    log: logging.Logger,
) -> dict:
    """Validate the PHOENIX release directory and optionally materialise videos."""
    release_dir = Path(source.release_dir)
    validate_source_config(source)

    if not release_dir.exists():
        raise FileNotFoundError(
            f"PHOENIX release directory not found: {release_dir}\n"
            "RWTH-PHOENIX-Weather requires manual download from "
            "https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/"
        )

    log.info("PHOENIX release directory validated: %s", release_dir)

    splits = ALL_SPLITS if source.split == "all" else (source.split,)
    mode = source.prepare_mode

    if mode == "validate":
        return {"validated": True, "mode": mode}

    overwrite = mode == "rematerialize_all"
    video_dir = Path(config.paths.videos) if getattr(config.paths, "videos", "") else release_dir

    total_materialized = 0
    total_errors = 0
    total_validated = 0

    for split in splits:
        csv_paths = find_corpus_csvs(release_dir, split)
        if not csv_paths:
            log.warning(
                "No corpus CSV found for split '%s' under %s — skipping.",
                split, release_dir,
            )
            continue
        for csv_path in csv_paths:
            mat, err, val = _materialise_split(
                csv_path=csv_path,
                split=split,
                release_dir=release_dir,
                video_dir=video_dir,
                fps=source.video_fps,
                overwrite=overwrite,
                log=log,
            )
            total_materialized += mat
            total_errors += err
            total_validated += val

    log.info(
        "PHOENIX prepare complete: %d materialized, %d validated, %d errors.",
        total_materialized, total_validated, total_errors,
    )
    return {
        "mode": mode,
        "validated": total_validated,
        "materialized": total_materialized,
        "errors": total_errors,
    }


def _materialise_split(
    csv_path: Path,
    split: str,
    release_dir: Path,
    video_dir: Path,
    fps: float,
    overwrite: bool,
    log: logging.Logger,
) -> Tuple[int, int, int]:
    """Materialise videos for a single corpus CSV.

    Returns (materialized_count, error_count, validated_count).
    """
    try:
        df = pd.read_csv(csv_path, delimiter="|")
    except Exception as exc:
        log.error("Failed to read corpus CSV %s: %s", csv_path, exc)
        return 0, 1, 0

    df.columns = [column.strip().lower() for column in df.columns]

    id_col = get_identifier_column(df.columns)
    folder_col = get_frame_path_column(df.columns)

    if id_col is None:
        log.error("Corpus CSV %s has no 'id', 'name', or 'video' column — skipping.", csv_path)
        return 0, 1, 0

    materialized = 0
    errors = 0
    validated = 0

    for _, row in df.iterrows():
        raw_id = str(row[id_col])
        clip_id = derive_clip_id(raw_id)
        raw_folder = raw_id
        if folder_col and pd.notna(row[folder_col]):
            raw_folder = str(row[folder_col])
        frame_dir = _resolve_frame_dir(csv_path, release_dir, split, raw_folder)
        output_path = video_dir / split / f"{clip_id}.mp4"

        if output_path.exists() and not overwrite:
            validated += 1
            continue

        if not frame_dir.is_dir():
            log.debug(
                "Frame directory not found, skipping clip '%s': %s",
                clip_id, frame_dir,
            )
            errors += 1
            continue

        try:
            materialize_frames_to_video(frame_dir, output_path, fps=fps, overwrite=overwrite)
        except Exception as exc:
            log.warning("Failed to materialise clip '%s': %s", clip_id, exc)
            errors += 1
            continue

        if output_path.exists():
            materialized += 1
            continue

        log.warning(
            "Materialisation did not produce an output file for clip '%s': %s",
            clip_id, output_path,
        )
        errors += 1

    return materialized, errors, validated
