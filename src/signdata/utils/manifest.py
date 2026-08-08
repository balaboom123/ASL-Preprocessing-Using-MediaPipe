"""Canonical manifest schema and shared manifest I/O utilities."""

from pathlib import Path

import pandas as pd

# Common video extensions produced by yt-dlp and ffmpeg
_VIDEO_EXTENSIONS = (".mp4", ".webm", ".mkv", ".avi", ".mov")

# ---------------------------------------------------------------------------
# Column alias mapping — old name → canonical name
# ---------------------------------------------------------------------------

_COLUMN_ALIASES = {
    # How2Sign / YouTube-ASL legacy names
    "SENTENCE_NAME": "SAMPLE_ID",
    "VIDEO_NAME": "VIDEO_ID",
    "START_REALIGNED": "START",
    "END_REALIGNED": "END",
    "SENTENCE": "TEXT",
    "CAPTION": "TEXT",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename legacy column names to their canonical equivalents.

    Only renames a column if the canonical name does *not* already exist in
    the DataFrame (avoids overwriting an explicit canonical column).

    When multiple aliases map to the same canonical name (e.g. both
    ``SENTENCE`` and ``CAPTION`` → ``TEXT``), only the first alias found
    (in ``_COLUMN_ALIASES`` iteration order) is renamed.  This prevents
    ``df.rename()`` from producing duplicate column names.
    """
    rename_map = {}
    claimed = set(df.columns)
    for old_name, canonical in _COLUMN_ALIASES.items():
        if old_name in claimed and canonical not in claimed:
            rename_map[old_name] = canonical
            claimed.add(canonical)
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def row_value(row: pd.Series, column: str) -> str:
    """Return a manifest row value as stripped text, or empty when missing."""
    value = row.get(column)
    return "" if pd.isna(value) else str(value).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_manifest(path: str | Path) -> pd.DataFrame:
    """Read a TSV manifest and normalize column names.

    Parameters
    ----------
    path : str or Path
        Path to the manifest TSV file.
    Returns
    -------
    pd.DataFrame
        The manifest data with normalized column names.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    return _normalize_columns(
        pd.read_csv(path, delimiter="\t", on_bad_lines="warn")
    )


def write_manifest(df: pd.DataFrame, path: str | Path) -> None:
    """Write a canonical TSV manifest, creating its parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def find_video_file(
    base_dir: str | Path,
    stem: str,
) -> Path:
    """Find a video file by stem, trying common video extensions.

    Tries ``.mp4`` first (most common), then other extensions.
    Falls back to ``{stem}.mp4`` if no file is found on disk, so that
    callers can rely on a deterministic return value.

    Parameters
    ----------
    base_dir : str or Path
        Directory containing video files.
    stem : str
        File stem (e.g. a VIDEO_ID or SAMPLE_ID).

    Returns
    -------
    Path
        Path to the first matching video file, or ``base_dir/{stem}.mp4``
        as a fallback.
    """
    base_dir = Path(base_dir)
    for ext in _VIDEO_EXTENSIONS:
        candidate = base_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return base_dir / f"{stem}.mp4"


def resolve_video_path(
    row: pd.Series,
    base_dir: str | Path,
) -> Path:
    """Resolve the physical video file path for a manifest row.

    Resolution order:
    1. If ``REL_PATH`` column is present and non-null → ``base_dir / REL_PATH``,
       falling back to the same stem under another video extension when that
       exact file is absent
    2. Otherwise if ``VIDEO_NAME`` is present and non-null → use that stem
    3. Otherwise → ``find_video_file(base_dir, VIDEO_ID)`` (extension-aware)

    Parameters
    ----------
    row : pd.Series
        A single manifest row.
    base_dir : str or Path
        The base directory for video files (e.g., ``config.paths.videos``
        or ``context.video_dir``).

    Returns
    -------
    Path
        Resolved absolute path to the video file.
    """
    base_dir = Path(base_dir)

    rel_path = row_value(row, "REL_PATH")
    if rel_path:
        candidate = base_dir / rel_path
        if candidate.exists():
            return candidate
        # video2compression re-encodes into .mp4 but passes other sources
        # through under their own container, so a manifest written against
        # videos/ names a different extension than the mirror holds. Retry on
        # the stem so compressed/ stays the drop-in replacement it claims to
        # be. find_video_file still returns a deterministic path on a miss.
        return find_video_file(candidate.parent, candidate.stem)

    video_name = row_value(row, "VIDEO_NAME")
    if video_name:
        return find_video_file(base_dir, video_name)

    return find_video_file(base_dir, row_value(row, "VIDEO_ID"))


def get_timing_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Return canonical timing columns when present.

    Returns
    -------
    tuple of (str, str)
        The start and end column names.

    Raises
    ------
    ValueError
        If no recognized timestamp columns are found.
    """
    if {"START", "END"}.issubset(df.columns):
        return "START", "END"

    raise ValueError(
        "No canonical timestamp columns found in manifest. Expected START and END."
    )
