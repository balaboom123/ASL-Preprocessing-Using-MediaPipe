"""OpenASL manifest building."""

import json
import logging
from pathlib import Path

import pandas as pd

from .._ingestion.availability import apply_availability_policy
from .._ingestion.text import normalize_text
from ...utils.manifest import write_manifest
from .source import OpenASLSourceConfig


def build(config, source: OpenASLSourceConfig, log: logging.Logger) -> pd.DataFrame:
    """Build canonical manifest from OpenASL TSV."""
    manifest_path = config.paths.manifest

    tsv_path = source.manifest_tsv
    if not tsv_path or not Path(tsv_path).exists():
        raise FileNotFoundError(f"OpenASL manifest TSV not found: {tsv_path}")

    tsv = pd.read_csv(tsv_path, delimiter="\t")

    for required in ("vid", "yid", "start", "end"):
        if required not in tsv.columns:
            raise ValueError(
                f"OpenASL TSV missing required column '{required}'. "
                f"Available: {list(tsv.columns)}"
            )

    text_col = source.text_column
    text_opts = source.text_processing.model_dump()
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
        df["TEXT"] = (
            tsv[text_col]
            .fillna("")
            .astype(str)
            .apply(lambda t: normalize_text(t, **text_opts) if t else "")
        )

    optional_passthrough = {
        "split": "SPLIT",
        "signer_id": "SIGNER_ID",
    }
    for src_col, canon_col in optional_passthrough.items():
        if src_col in tsv.columns:
            df[canon_col] = tsv[src_col].astype(str)

    if source.bbox_json and Path(source.bbox_json).exists():
        df = _merge_bboxes(df, source.bbox_json)
        log.info("Merged bounding boxes from %s", source.bbox_json)

    video_dir = config.paths.videos
    if video_dir and Path(video_dir).is_dir():
        df = apply_availability_policy(df, video_dir, source.availability_policy)

    write_manifest(df, manifest_path)
    return df


def _merge_bboxes(df: pd.DataFrame, bbox_path: str) -> pd.DataFrame:
    df = df.copy()

    bboxes = json.loads(Path(bbox_path).read_text(encoding="utf-8"))

    x1, y1, x2, y2, detected = [], [], [], [], []

    for vid in df["SAMPLE_ID"]:
        bbox = bboxes.get(str(vid))
        if bbox is not None:
            if isinstance(bbox, dict):
                bbox = bbox.get("bbox", bbox.get("box"))
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                x1.append(float(bbox[0]))
                y1.append(float(bbox[1]))
                x2.append(float(bbox[2]))
                y2.append(float(bbox[3]))
                detected.append(True)
                continue

        x1.append(None)
        y1.append(None)
        x2.append(None)
        y2.append(None)
        detected.append(False)

    df["BBOX_X1"] = x1
    df["BBOX_Y1"] = y1
    df["BBOX_X2"] = x2
    df["BBOX_Y2"] = y2
    df["PERSON_DETECTED"] = detected

    return df
