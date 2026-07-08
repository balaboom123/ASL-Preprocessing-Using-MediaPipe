"""Class-map loading and joining utilities.

Datasets that map numeric class IDs to gloss labels (AUTSL, LSA64, SLoVo,
MS-ASL) share the same load-and-join pattern.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


def load_class_map(
    path: Union[str, Path],
    *,
    id_column: str = "CLASS_ID",
    gloss_column: str = "GLOSS",
    delimiter: str = "\t",
    extra_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load and validate a class-map file.

    Parameters
    ----------
    path : str or Path
        Path to the class-map file (TSV by default).
    id_column : str
        Column name for the class ID.
    gloss_column : str
        Column name for the gloss label.
    delimiter : str
        Field delimiter.
    extra_columns : list[str], optional
        Additional columns to validate and retain.

    Returns
    -------
    pd.DataFrame
        DataFrame with at least ``id_column`` and ``gloss_column``.

    Raises
    ------
    FileNotFoundError
        If the class-map file does not exist.
    ValueError
        If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Class map file not found: {path}")

    df = pd.read_csv(
        path,
        delimiter=delimiter,
        keep_default_na=False,
        na_values=[""],
    )

    required = {id_column, gloss_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Class map {path} missing required columns: {sorted(missing)}. "
            f"Available: {list(df.columns)}"
        )

    if extra_columns:
        extra_missing = set(extra_columns) - set(df.columns)
        if extra_missing:
            logger.warning(
                "Class map %s missing optional columns: %s",
                path, sorted(extra_missing),
            )

    df[id_column] = pd.to_numeric(df[id_column], errors="coerce")

    dup_ids = df[id_column].duplicated(keep=False)
    if dup_ids.any():
        dups = sorted(df.loc[dup_ids, id_column].unique().tolist())
        logger.warning(
            "Class map has %d duplicate %s values: %s",
            len(dups), id_column, dups[:10],
        )

    return df


def join_class_map(
    df: pd.DataFrame,
    class_map: pd.DataFrame,
    *,
    on: str = "CLASS_ID",
    gloss_column: str = "GLOSS",
    text_column: Optional[str] = None,
    extra_columns: Optional[List[str]] = None,
    strict: bool = True,
) -> pd.DataFrame:
    """Join class labels onto a manifest DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Manifest DataFrame with a class-ID column.
    class_map : pd.DataFrame
        Class map loaded via ``load_class_map``.
    on : str
        Column name to join on (must exist in both DataFrames).
    gloss_column : str
        Name of the gloss column in *class_map*.
    text_column : str, optional
        If set, also copy this column from *class_map* into *df* as ``TEXT``.
    extra_columns : list[str], optional
        Additional columns to carry over from *class_map*.
    strict : bool
        If *True*, warn when manifest rows have class IDs not found in the map.

    Returns
    -------
    pd.DataFrame
        Manifest with ``GLOSS`` (and optionally ``TEXT``) columns added.
    """
    df = df.copy()
    lookup = class_map.set_index(on)

    df["GLOSS"] = df[on].map(lookup[gloss_column])

    if text_column and text_column in lookup.columns:
        df["TEXT"] = df[on].map(lookup[text_column])

    if extra_columns:
        for col in extra_columns:
            if col in lookup.columns:
                df[col] = df[on].map(lookup[col])

    if strict:
        unmapped = df["GLOSS"].isna().sum()
        if unmapped:
            logger.warning(
                "%d rows have %s values not found in class map.",
                unmapped, on,
            )

    return df
