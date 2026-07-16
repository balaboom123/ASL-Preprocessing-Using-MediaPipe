"""AUTSL manifest building."""

import logging
from pathlib import Path

import pandas as pd

from .._ingestion.availability import apply_availability_policy_paths
from .._ingestion.media import get_video_duration, get_video_fps
from ...utils.manifest import write_manifest
from .source import (
    AUTSLSourceConfig,
    MODALITY_SUFFIX,
    discover_split_dir,
    get_selected_splits,
    parse_signer_id,
    resolve_class_id_file,
    resolve_labels_file,
    resolve_release_root,
)


def build(config, source: AUTSLSourceConfig, log: logging.Logger) -> pd.DataFrame:
    """Build canonical manifest from AUTSL release directory."""
    release_root = resolve_release_root(source, config)
    selected_splits = get_selected_splits(source)

    class_id_path = resolve_class_id_file(source, release_root)
    if class_id_path is None or not class_id_path.exists():
        raise FileNotFoundError(
            f"AUTSL class correspondence file not found under "
            f"{release_root}. Set dataset.source.class_id_file explicitly."
        )
    class_map = pd.read_csv(
        class_id_path,
        header=None,
        names=["CLASS_ID", "GLOSS_TR", "GLOSS_EN"],
    )
    class_lookup: dict[int, tuple[str, str]] = {}
    for _, row in class_map.iterrows():
        try:
            cid = int(row["CLASS_ID"])
        except (ValueError, TypeError):
            continue
        gloss_tr = str(row["GLOSS_TR"]) if pd.notna(row["GLOSS_TR"]) else ""
        gloss_en = str(row["GLOSS_EN"]) if pd.notna(row["GLOSS_EN"]) else gloss_tr
        class_lookup[cid] = (gloss_tr, gloss_en)

    modality_suffix = MODALITY_SUFFIX[source.modality]
    split_dfs: list[pd.DataFrame] = []

    for split in selected_splits:
        split_df = _build_split_df(
            split=split,
            release_root=release_root,
            source=source,
            modality_suffix=modality_suffix,
            class_lookup=class_lookup,
            log=log,
        )
        if split_df is not None and not split_df.empty:
            split_dfs.append(split_df)

    if not split_dfs:
        raise RuntimeError(
            f"AUTSL build_manifest produced no rows for splits "
            f"{selected_splits}. Check split directories and label files."
        )

    df = pd.concat(split_dfs, ignore_index=True)

    df = apply_availability_policy_paths(
        df,
        base_dir=release_root,
        policy=source.availability_policy,
        rel_path_col="REL_PATH",
    )

    write_manifest(df, config.paths.manifest)
    return df


def _build_split_df(
    split: str,
    release_root: Path,
    source: AUTSLSourceConfig,
    modality_suffix: str,
    class_lookup: dict[int, tuple[str, str]],
    log: logging.Logger,
) -> pd.DataFrame | None:
    """Build a manifest DataFrame for a single split."""
    try:
        split_dir = discover_split_dir(release_root, split)
    except FileNotFoundError as exc:
        log.warning("Skipping split '%s': %s", split, exc)
        return None

    labels_path = resolve_labels_file(split, source, release_root)
    if labels_path is None:
        if source.allow_unlabeled:
            log.warning(
                "No labels file found for split '%s'; "
                "allow_unlabeled=True so continuing without labels.",
                split,
            )
            labels_df = None
        else:
            log.warning(
                "No labels file found for split '%s'. "
                "Set allow_unlabeled=True to include unlabeled samples.",
                split,
            )
            return None
    else:
        labels_df = pd.read_csv(
            labels_path,
            header=None,
            names=["sample_key", "label_id"],
        )
        log.info(
            "Loaded %d label rows for split '%s' from %s",
            len(labels_df), split, labels_path,
        )

    if labels_df is not None:
        rows_iter = (
            (str(row.sample_key), int(row.label_id))
            for row in labels_df.itertuples(index=False)
        )
    else:
        rows_iter = (
            (path.stem.removesuffix(modality_suffix), None)
            for path in sorted(split_dir.glob(f"*{modality_suffix}.mp4"))
        )

    records = []
    skipped = 0
    for sample_key, label_id in rows_iter:
        physical_stem = f"{sample_key}{modality_suffix}"
        rel_path = f"{split_dir.name}/{physical_stem}.mp4"
        video_path = release_root / rel_path
        video_exists = video_path.exists()

        if not video_exists:
            log.debug(
                "Video file not found, will be handled by availability policy: %s",
                video_path,
            )
            skipped += 1

        gloss_tr, gloss_en = (
            class_lookup.get(label_id, ("", "")) if label_id is not None else ("", "")
        )

        if video_exists:
            duration = get_video_duration(str(video_path))
            fps = get_video_fps(str(video_path)) or 30.0
        else:
            duration = 0.0
            fps = 30.0

        records.append({
            "SAMPLE_ID": f"{split}-{physical_stem}",
            "VIDEO_ID": physical_stem,
            "REL_PATH": rel_path,
            "START": 0.0,
            "END": duration,
            "CLASS_ID": label_id,
            "GLOSS": gloss_tr,
            "TEXT": gloss_en if gloss_en else gloss_tr,
            "SIGNER_ID": parse_signer_id(sample_key),
            "FPS": fps,
            "SPLIT": split,
        })

    if skipped:
        log.warning(
            "Split '%s': %d / %d video files not found on disk "
            "(availability policy will resolve these).",
            split, skipped, len(records),
        )

    return pd.DataFrame(records) if records else None
