"""OpenASL manifest building."""

import json
import logging
from pathlib import Path

import pandas as pd

from .._ingestion.availability import apply_availability_policy
from .._ingestion.text import normalize_text
from ...utils.manifest import write_manifest
from .source import OpenASLSourceConfig, read_manifest_tsv


def build(config, source: OpenASLSourceConfig, log: logging.Logger) -> pd.DataFrame:
    """Build canonical manifest from OpenASL TSV."""
    tsv = read_manifest_tsv(source)

    missing = {"vid", "yid", "start", "end"} - set(tsv.columns)
    if missing:
        raise ValueError(
            f"OpenASL TSV missing required columns: {sorted(missing)}. "
            f"Available: {list(tsv.columns)}"
        )

    text_col = source.text_column
    has_text = text_col in tsv.columns

    if not has_text:
        log.warning(
            "Text column '%s' not found in TSV. "
            "Manifest will have no TEXT column. "
            "Available columns: %s",
            text_col, list(tsv.columns),
        )

    df = pd.DataFrame({
        "SAMPLE_ID": tsv["vid"].astype(str),
        "VIDEO_ID": tsv["yid"].astype(str),
        "START": tsv["start"].astype(float),
        "END": tsv["end"].astype(float),
    })

    if has_text:
        text_opts = source.text_processing.model_dump()
        df["TEXT"] = (
            tsv[text_col]
            .fillna("")
            .astype(str)
            .apply(lambda text: normalize_text(text, **text_opts))
        )

    for src_col, canon_col in (("split", "SPLIT"), ("signer_id", "SIGNER_ID")):
        if src_col in tsv.columns:
            df[canon_col] = tsv[src_col].astype(str)

    if source.bbox_json and Path(source.bbox_json).exists():
        df = _merge_bboxes(df, source.bbox_json)
        log.info("Merged bounding boxes from %s", source.bbox_json)

    video_dir = config.paths.videos
    if video_dir and Path(video_dir).is_dir():
        df = apply_availability_policy(df, video_dir, source.availability_policy)

    write_manifest(df, config.paths.manifest)
    return df


def _merge_bboxes(df: pd.DataFrame, bbox_path: str) -> pd.DataFrame:
    bboxes = json.loads(Path(bbox_path).read_text(encoding="utf-8"))
    rows = []

    for vid in df["SAMPLE_ID"]:
        bbox = bboxes.get(str(vid))
        if isinstance(bbox, dict):
            bbox = bbox.get("bbox", bbox.get("box"))
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            rows.append([float(value) for value in bbox[:4]] + [True])
        else:
            rows.append([None, None, None, None, False])

    columns = ["BBOX_X1", "BBOX_Y1", "BBOX_X2", "BBOX_Y2", "PERSON_DETECTED"]
    return df.join(pd.DataFrame(rows, columns=columns, index=df.index))
