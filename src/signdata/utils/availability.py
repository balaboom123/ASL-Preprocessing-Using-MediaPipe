"""Pipeline-level availability filtering."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def filter_available(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows marked unavailable (``AVAILABLE == False``).

    If the ``AVAILABLE`` column is not present, returns the DataFrame
    unchanged.  This is called by the runner after loading a manifest
    produced with ``mark_unavailable`` so that downstream processors
    only iterate over rows with actual video files on disk.
    """
    if "AVAILABLE" not in df.columns:
        return df
    filtered = df[df["AVAILABLE"]].reset_index(drop=True)
    n_dropped = len(df) - len(filtered)
    if n_dropped:
        logger.info(
            "Filtered %d unavailable rows from manifest (%d remaining).",
            n_dropped, len(filtered),
        )
    return filtered
