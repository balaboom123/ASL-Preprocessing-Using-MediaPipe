"""Dataset-ingestion availability helpers."""

import json
import logging
from pathlib import Path
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

AvailabilityPolicy = Literal["fail_fast", "drop_unavailable", "mark_unavailable"]
_VIDEO_EXTENSIONS = {".mp4", ".webm", ".mkv", ".avi", ".mov"}


def get_existing_video_ids(
    directory: str | Path, recursive: bool = False
) -> set[str]:
    """Return set of stem IDs from video files with any common extension."""
    directory = Path(directory)
    paths = directory.rglob("*") if recursive else directory.glob("*")
    return {
        path.stem
        for path in paths
        if path.is_file() and path.suffix.lower() in _VIDEO_EXTENSIONS
    }


def _apply_policy(
    df: pd.DataFrame,
    is_available: pd.Series,
    policy: AvailabilityPolicy,
    item_name: str,
) -> pd.DataFrame:
    if policy == "mark_unavailable":
        return df.assign(AVAILABLE=is_available)

    if policy != "drop_unavailable" or is_available.all():
        return df

    result = df[is_available].reset_index(drop=True)
    logger.info(
        "Dropped %d rows with unavailable %s (%d remaining).",
        len(df) - len(result), item_name, len(result),
    )
    return result


def apply_availability_policy(
    df: pd.DataFrame,
    video_dir: str | Path,
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

    if missing_count:
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

    return _apply_policy(df, is_available, policy, "videos")


def apply_availability_policy_paths(
    df: pd.DataFrame,
    base_dir: str | Path,
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
        return apply_availability_policy(df, base_dir, policy)

    is_available = df[rel_path_col].apply(
        lambda path: (
            (base_dir / str(path)).exists()
            if pd.notna(path) and str(path).strip()
            else False
        )
    )
    missing_count = int((~is_available).sum())

    if missing_count:
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

    return _apply_policy(df, is_available, policy, "files")


def write_acquire_report(
    report_dir: str | Path,
    stats: dict,
    missing: list[dict],
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
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "download_report.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    missing_rows = pd.DataFrame(missing)
    if missing_rows.empty:
        missing_rows = missing_rows.reindex(columns=["VIDEO_ID", "REASON"])
    missing_rows.to_csv(report_dir / "missing_videos.csv", index=False)
