"""SLoVo manifest building."""

import logging
from pathlib import Path

import pandas as pd

from .._ingestion.availability import apply_availability_policy_paths
from .._ingestion.classmap import load_class_map
from .._ingestion.media import get_video_duration, get_video_fps
from ...utils.manifest import write_manifest
from .source import (
    OPTIONAL_PASSTHROUGH,
    SlovoSourceConfig,
    get_bundled_class_map_path,
    load_annotations,
    parse_train_col,
    resolve_annotations_csv,
    resolve_release_dir,
)


def build(config, source: SlovoSourceConfig, log: logging.Logger) -> pd.DataFrame:
    """Build the canonical manifest from SLoVo annotations.csv."""
    video_dir = resolve_release_dir(source, config)
    annotations_csv = resolve_annotations_csv(source, video_dir)

    ann = load_annotations(annotations_csv)
    ann["_split"] = ann["train"].apply(parse_train_col).map(
        {True: "train", False: "test"}
    )

    if source.split != "all":
        before = len(ann)
        ann = ann[ann["_split"] == source.split].reset_index(drop=True)
        log.info(
            "Filtered to split='%s': %d → %d rows",
            source.split, before, len(ann),
        )

    if not source.include_background:
        before = len(ann)
        ann = ann[~ann["text"].isin(source.background_labels)].reset_index(drop=True)
        log.info(
            "Dropped background labels %s: %d → %d rows",
            source.background_labels, before, len(ann),
        )

    durations: list[float] = []
    fps_values: list[float] = []
    rel_paths: list[str] = []

    attachment_ids = ann["attachment_id"].astype(str)
    for attachment_id in attachment_ids:
        rel_path = f"{attachment_id}.mp4"
        video_path = str(Path(video_dir) / rel_path)
        durations.append(get_video_duration(video_path))
        fps_values.append(get_video_fps(video_path))
        rel_paths.append(rel_path)

    df = pd.DataFrame({
        "SAMPLE_ID": attachment_ids,
        "VIDEO_ID": attachment_ids,
        "REL_PATH": rel_paths,
        "SPLIT": ann["_split"],
        "START": 0.0,
        "END": durations,
        "GLOSS": ann["text"].astype(str),
        "TEXT": ann["text"].astype(str),
        "SIGNER_ID": ann["user_id"].astype(str),
        "FPS": fps_values,
    })

    for src_col, canon_col in OPTIONAL_PASSTHROUGH.items():
        if src_col in ann.columns:
            df[canon_col] = ann[src_col].values

    if source.class_map_mode != "none":
        df = _apply_class_map(df, source, ann, log)

    df = apply_availability_policy_paths(
        df,
        base_dir=video_dir,
        policy=source.availability_policy,
    )

    write_manifest(df, config.paths.manifest)
    return df


def _apply_class_map(
    df: pd.DataFrame,
    source: SlovoSourceConfig,
    ann: pd.DataFrame,
    log: logging.Logger,
) -> pd.DataFrame:
    """Join or derive CLASS_ID and update GLOSS/TEXT accordingly."""
    if source.class_map_mode == "bundled":
        class_map_path = get_bundled_class_map_path(source)
        if not class_map_path.exists():
            log.warning(
                "Class map file not found: %s. Skipping CLASS_ID assignment.",
                class_map_path,
            )
            return df

        class_map = load_class_map(class_map_path)
        gloss_to_id = class_map.set_index("GLOSS")["CLASS_ID"]
        df = df.copy()
        df["CLASS_ID"] = df["GLOSS"].map(gloss_to_id)
        unmatched = df["CLASS_ID"].isna().sum()
        if unmatched:
            log.warning(
                "%d rows could not be matched to a CLASS_ID "
                "from the bundled class map.",
                unmatched,
            )

    else:
        unique_labels = sorted(ann["text"].dropna().unique())
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        df = df.copy()
        df["CLASS_ID"] = df["GLOSS"].map(label_to_id)
        log.info("Derived CLASS_ID from %d unique labels.", len(unique_labels))

    return df
