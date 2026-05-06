"""RWTH-PHOENIX-Weather manifest building."""

import logging
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .._ingestion.availability import apply_availability_policy_paths
from .source import (
    ALL_SPLITS,
    RWTHPhoenixWeatherSourceConfig,
    derive_clip_id,
    find_corpus_csvs,
    get_identifier_column,
    get_signer_column,
)

_log = logging.getLogger(__name__)


def build(
    config,
    source: RWTHPhoenixWeatherSourceConfig,
    log: logging.Logger,
) -> pd.DataFrame:
    """Build a canonical TSV manifest from PHOENIX corpus CSV files."""
    release_dir = Path(source.release_dir)
    video_dir = Path(config.paths.videos) if getattr(config.paths, "videos", "") else release_dir
    manifest_path = config.paths.manifest

    if not release_dir.exists():
        raise FileNotFoundError(
            f"PHOENIX release directory not found: {release_dir}"
        )

    splits = ALL_SPLITS if source.split == "all" else (source.split,)
    split_frames: List[pd.DataFrame] = []

    for split in splits:
        csv_paths = find_corpus_csvs(release_dir, split)
        if not csv_paths:
            log.warning(
                "No corpus CSV found for split '%s' under %s.",
                split, release_dir,
            )
            continue
        for csv_path in csv_paths:
            split_df = _load_split_df(csv_path, split, source.video_fps)
            if split_df is not None and len(split_df) > 0:
                split_frames.append(split_df)
                log.info(
                    "Loaded %d rows for split '%s' from %s",
                    len(split_df), split, csv_path,
                )

    if not split_frames:
        raise RuntimeError(
            f"No corpus CSV rows loaded for splits {splits} under {release_dir}. "
            "Check that release_dir points to the unpacked PHOENIX archive."
        )

    df = pd.concat(split_frames, ignore_index=True)

    df = apply_availability_policy_paths(
        df,
        base_dir=video_dir,
        policy=source.availability_policy,
        rel_path_col="REL_PATH",
    )

    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    df.to_csv(manifest_path, sep="\t", index=False)
    return df


def _load_split_df(
    csv_path: Path,
    split: str,
    video_fps: float,
) -> Optional[pd.DataFrame]:
    """Parse a single corpus CSV and return a canonical DataFrame."""
    try:
        raw = pd.read_csv(csv_path, delimiter="|")
    except Exception as exc:
        _log.error("Failed to read corpus CSV %s: %s", csv_path, exc)
        return None

    raw.columns = [c.strip().lower() for c in raw.columns]
    id_col = get_identifier_column(raw.columns)
    if id_col is None:
        _log.error("Corpus CSV %s has no 'id', 'name', or 'video' column — skipping.", csv_path)
        return None

    signer_col = get_signer_column(raw.columns)

    rows = []
    for _, row in raw.iterrows():
        clip_id = derive_clip_id(str(row[id_col]))

        entry = {
            "SAMPLE_ID": clip_id,
            "VIDEO_ID": clip_id,
            "REL_PATH": f"{split}/{clip_id}.mp4",
            "SPLIT": split,
            "FPS": video_fps,
        }

        if "orth" in raw.columns and pd.notna(row["orth"]):
            entry["GLOSS"] = str(row["orth"]).strip()
        if "translation" in raw.columns and pd.notna(row["translation"]):
            entry["TEXT"] = str(row["translation"]).strip()
        if signer_col and pd.notna(row[signer_col]):
            entry["SIGNER_ID"] = str(row[signer_col]).strip()

        try:
            start_val = float(row["start"]) if "start" in raw.columns else None
            end_val = float(row["end"]) if "end" in raw.columns else None
            if (
                start_val is not None
                and end_val is not None
                and start_val >= 0
                and end_val > start_val
            ):
                entry["START"] = start_val
                entry["END"] = end_val
        except (TypeError, ValueError):
            pass

        rows.append(entry)

    return pd.DataFrame(rows)
