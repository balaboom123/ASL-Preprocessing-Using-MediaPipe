"""Class-map loading utilities."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_class_map(path: str | Path) -> pd.DataFrame:
    """Load a TSV class map and require CLASS_ID and GLOSS columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Class map file not found: {path}")

    df = pd.read_csv(
        path,
        delimiter="\t",
        keep_default_na=False,
        na_values=[""],
    )

    required = {"CLASS_ID", "GLOSS"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Class map {path} missing required columns: {sorted(missing)}. "
            f"Available: {list(df.columns)}"
        )

    df["CLASS_ID"] = pd.to_numeric(df["CLASS_ID"], errors="coerce")

    dup_ids = df["CLASS_ID"].duplicated(keep=False)
    if dup_ids.any():
        dups = sorted(df.loc[dup_ids, "CLASS_ID"].unique().tolist())
        logger.warning(
            "Class map has %d duplicate CLASS_ID values: %s",
            len(dups), dups[:10],
        )

    return df
