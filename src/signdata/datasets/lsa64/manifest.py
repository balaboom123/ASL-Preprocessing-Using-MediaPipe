"""LSA64 manifest building."""

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from .._ingestion.availability import apply_availability_policy_paths
from .._ingestion.classmap import join_class_map
from .._ingestion.media import get_video_duration, get_video_fps
from .source import DEFAULT_FPS, LSA64SourceConfig, load_lsa64_class_map, resolve_video_dir


def build(
    config,
    source: LSA64SourceConfig,
    video_dir: Optional[Path] = None,
    log: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Discover .mp4 files, parse filenames, join class map, write TSV manifest."""
    if log is None:
        log = logging.getLogger(__name__)
    if video_dir is None:
        video_dir = resolve_video_dir(config, source)

    if not video_dir or not video_dir.exists():
        raise FileNotFoundError(
            f"LSA64 video directory not found: {video_dir!r}. "
            f"Run the download stage first or set release_dir / paths.videos."
        )

    mp4_files = sorted(video_dir.glob("*.mp4"))
    if not mp4_files:
        raise FileNotFoundError(f"No .mp4 files found in: {video_dir}")

    log.info(
        "Discovering LSA64 videos in %s: found %d .mp4 files",
        video_dir, len(mp4_files),
    )

    rows = []
    skipped = 0
    for mp4 in mp4_files:
        stem = mp4.stem
        parts = stem.split("_")
        if len(parts) != 3:
            log.warning("Skipping file with unexpected name format: %s", mp4.name)
            skipped += 1
            continue
        try:
            class_id = int(parts[0])
            signer_id = int(parts[1])
            repetition_id = int(parts[2])
        except ValueError:
            log.warning("Skipping file with non-integer ID components: %s", mp4.name)
            skipped += 1
            continue

        video_path = str(mp4)
        duration = get_video_duration(video_path)
        fps = get_video_fps(video_path) or DEFAULT_FPS

        rows.append({
            "SAMPLE_ID": f"{source.variant}-{stem}",
            "VIDEO_ID": stem,
            "REL_PATH": mp4.name,
            "SOURCE_VARIANT": source.variant,
            "CLASS_ID": class_id,
            "SIGNER_ID": signer_id,
            "REPETITION_ID": repetition_id,
            "START": 0.0,
            "END": duration,
            "FPS": fps,
        })

    if skipped:
        log.warning("Skipped %d file(s) with unexpected filename format.", skipped)

    if not rows:
        raise RuntimeError(
            f"No valid LSA64 filenames found in {video_dir}. "
            f"Expected format: {{CLASS_ID}}_{{SIGNER_ID}}_{{REPETITION_ID}}.mp4 "
            f"(e.g. 001_003_005.mp4)"
        )

    df = pd.DataFrame(rows)

    class_map = load_lsa64_class_map(source, log)
    if class_map is not None:
        df = join_class_map(
            df,
            class_map,
            on="CLASS_ID",
            gloss_column="GLOSS",
            extra_columns=["HANDEDNESS"] if "HANDEDNESS" in class_map.columns else None,
        )
        if "TEXT" not in df.columns:
            df["TEXT"] = df["GLOSS"]
    else:
        df["GLOSS"] = None
        df["TEXT"] = None
        df["HANDEDNESS"] = None

    df = _apply_split_strategy(df, source)

    if source.split != "all":
        before = len(df)
        df = df[df["SPLIT"] == source.split].reset_index(drop=True)
        log.info(
            "Filtered to split='%s': %d -> %d rows",
            source.split, before, len(df),
        )

    df = apply_availability_policy_paths(
        df,
        base_dir=video_dir,
        policy=source.availability_policy,
        rel_path_col="REL_PATH",
    )

    canonical_columns = [
        "SAMPLE_ID", "VIDEO_ID", "REL_PATH", "SPLIT",
        "SOURCE_VARIANT",
        "START", "END", "CLASS_ID", "GLOSS", "TEXT",
        "SIGNER_ID", "FPS", "REPETITION_ID", "HANDEDNESS",
    ]
    ordered = [c for c in canonical_columns if c in df.columns]
    extra = [c for c in df.columns if c not in ordered]
    df = df[ordered + extra]

    manifest_path = config.paths.manifest
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    df.to_csv(manifest_path, sep="\t", index=False)
    return df


def _apply_split_strategy(
    df: pd.DataFrame,
    source: LSA64SourceConfig,
) -> pd.DataFrame:
    """Assign SPLIT column according to the configured strategy."""
    if source.split_strategy == "none":
        df = df.copy()
        df["SPLIT"] = "all"
        return df

    if source.split_strategy == "community_signer_8_1_1":
        train_set = set(source.train_signers)
        val_set = set(source.val_signers)
        test_set = set(source.test_signers)

        def _assign(signer_id: int) -> str:
            if signer_id in train_set:
                return "train"
            if signer_id in val_set:
                return "val"
            if signer_id in test_set:
                return "test"
            return "unknown"

        df = df.copy()
        df["SPLIT"] = df["SIGNER_ID"].apply(_assign)

        unknown_count = (df["SPLIT"] == "unknown").sum()
        if unknown_count:
            logging.getLogger(__name__).warning(
                "%d rows have SIGNER_IDs not assigned to any split "
                "(train=%s, val=%s, test=%s).",
                unknown_count,
                sorted(train_set),
                sorted(val_set),
                sorted(test_set),
            )
        return df

    raise ValueError(
        f"Unknown split_strategy: {source.split_strategy!r}. "
        f"Valid options: 'none', 'community_signer_8_1_1'."
    )
