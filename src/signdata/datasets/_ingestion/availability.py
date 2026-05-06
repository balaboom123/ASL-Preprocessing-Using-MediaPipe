"""Dataset-ingestion availability helpers.

These helpers are used exclusively by dataset adapters during the
download and build_manifest stages.  The pipeline-level ``filter_available``
function and the ``AvailabilityPolicy`` type remain in
``signdata.utils.availability``.

Policies:
    fail_fast          -- raise on any missing video/file
    drop_unavailable   -- remove rows with missing files from manifest
    mark_unavailable   -- keep all rows, add ``AVAILABLE`` bool column
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Set, Union

import pandas as pd

# Re-export AvailabilityPolicy from its canonical definition so adapters
# only need to import from this module.
from ...utils.availability import AvailabilityPolicy  # noqa: F401

logger = logging.getLogger(__name__)

_VIDEO_EXTENSIONS = ("mp4", "webm", "mkv", "avi", "mov")


def _iter_video_files(directory: Union[str, Path], recursive: bool) -> Iterable[Path]:
    base_dir = Path(directory)
    for ext in _VIDEO_EXTENSIONS:
        pattern = f"**/*.{ext}" if recursive else f"*.{ext}"
        yield from base_dir.glob(pattern)


def get_existing_video_ids(directory: str, recursive: bool = False) -> Set[str]:
    """Return set of stem IDs from video files with any common extension."""
    ids: Set[str] = set()
    for path in _iter_video_files(directory, recursive):
        ids.add(path.stem)
    return ids


def apply_availability_policy(
    df: pd.DataFrame,
    video_dir: str,
    policy: AvailabilityPolicy,
) -> pd.DataFrame:
    """Filter or annotate manifest rows based on video availability.

    Parameters
    ----------
    df : pd.DataFrame
        Manifest with at least a ``VIDEO_ID`` column.
    video_dir : str
        Directory where downloaded videos reside.
    policy : AvailabilityPolicy
        How to handle missing videos.

    Returns
    -------
    pd.DataFrame
        Modified manifest.

    Raises
    ------
    RuntimeError
        If *policy* is ``fail_fast`` and any VIDEO_IDs are missing.
    """
    available_ids = get_existing_video_ids(video_dir)
    is_available = df["VIDEO_ID"].isin(available_ids)
    missing_count = int((~is_available).sum())

    if missing_count == 0:
        if policy == "mark_unavailable":
            df = df.copy()
            df["AVAILABLE"] = True
        return df

    missing_ids = sorted(df.loc[~is_available, "VIDEO_ID"].unique())
    logger.warning(
        "%d rows reference %d unavailable VIDEO_IDs (policy=%s)",
        missing_count, len(missing_ids), policy,
    )

    if policy == "fail_fast":
        raise RuntimeError(
            f"{len(missing_ids)} video(s) not found in {video_dir}. "
            f"First 5: {missing_ids[:5]}. "
            f"Set availability_policy to 'drop_unavailable' or "
            f"'mark_unavailable' to continue without them."
        )

    if policy == "drop_unavailable":
        before = len(df)
        df = df[is_available].reset_index(drop=True)
        logger.info(
            "Dropped %d rows with unavailable videos (%d remaining).",
            before - len(df), len(df),
        )
    elif policy == "mark_unavailable":
        df = df.copy()
        df["AVAILABLE"] = is_available

    return df


def apply_availability_policy_paths(
    df: pd.DataFrame,
    base_dir: Union[str, Path],
    policy: AvailabilityPolicy,
    *,
    rel_path_col: str = "REL_PATH",
) -> pd.DataFrame:
    """Filter or annotate manifest rows using file-path existence checks.

    Unlike ``apply_availability_policy`` which checks ``VIDEO_ID`` stems
    in a flat directory, this function checks whether the file at
    ``base_dir / REL_PATH`` actually exists.  Falls back to ``VIDEO_ID``
    stem lookup when *rel_path_col* is absent.

    Parameters
    ----------
    df : pd.DataFrame
        Manifest with ``VIDEO_ID`` and optionally *rel_path_col*.
    base_dir : str or Path
        Root directory for resolving relative paths.
    policy : AvailabilityPolicy
        How to handle missing files.
    rel_path_col : str
        Column containing relative paths from *base_dir*.

    Returns
    -------
    pd.DataFrame
        Modified manifest.
    """
    base_dir = Path(base_dir)

    if rel_path_col not in df.columns:
        return apply_availability_policy(df, str(base_dir), policy)

    is_available = df[rel_path_col].apply(
        lambda p: (base_dir / str(p)).exists() if pd.notna(p) and str(p).strip() else False
    )
    missing_count = int((~is_available).sum())

    if missing_count == 0:
        if policy == "mark_unavailable":
            df = df.copy()
            df["AVAILABLE"] = True
        return df

    logger.warning(
        "%d rows reference unavailable files (policy=%s)",
        missing_count, policy,
    )

    if policy == "fail_fast":
        missing_paths = df.loc[~is_available, rel_path_col].head(5).tolist()
        raise RuntimeError(
            f"{missing_count} file(s) not found under {base_dir}. "
            f"First 5: {missing_paths}. "
            f"Set availability_policy to 'drop_unavailable' or "
            f"'mark_unavailable' to continue without them."
        )

    if policy == "drop_unavailable":
        before = len(df)
        df = df[is_available].reset_index(drop=True)
        logger.info(
            "Dropped %d rows with unavailable files (%d remaining).",
            before - len(df), len(df),
        )
    elif policy == "mark_unavailable":
        df = df.copy()
        df["AVAILABLE"] = is_available

    return df


def write_acquire_report(
    report_dir: str,
    stats: Dict,
    missing: List[Dict],
) -> None:
    """Write acquire report files.

    Parameters
    ----------
    report_dir : str
        Directory for report files (created if needed).
    stats : dict
        Summary stats (total, downloaded, errors, skipped).
    missing : list of dict
        Each entry has ``VIDEO_ID`` and ``REASON`` keys.
    """
    os.makedirs(report_dir, exist_ok=True)

    report_path = os.path.join(report_dir, "download_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    missing_path = os.path.join(report_dir, "missing_videos.csv")
    if missing:
        pd.DataFrame(missing).to_csv(missing_path, index=False)
    else:
        pd.DataFrame(columns=["VIDEO_ID", "REASON"]).to_csv(
            missing_path, index=False,
        )
