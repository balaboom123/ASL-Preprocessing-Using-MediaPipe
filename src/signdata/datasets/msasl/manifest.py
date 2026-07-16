"""MS-ASL manifest building."""

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .._ingestion.availability import apply_availability_policy_paths
from ...utils.manifest import write_manifest
from .source import (
    MSASLSourceConfig,
    extract_video_id,
    get_selected_splits,
    load_classes_json,
    load_split_json,
)

_VIDEO_EXTENSIONS = {".mp4", ".webm", ".mkv", ".avi", ".mov"}
_MANIFEST_COLUMNS = [
    "SAMPLE_ID",
    "VIDEO_ID",
    "START",
    "END",
    "CLASS_ID",
    "GLOSS",
    "TEXT",
    "SIGNER_ID",
    "FPS",
    "SOURCE_URL",
    "REL_PATH",
    "SPLIT",
    "BBOX_X1",
    "BBOX_Y1",
    "BBOX_X2",
    "BBOX_Y2",
    "PERSON_DETECTED",
]


def build(config, source: MSASLSourceConfig) -> pd.DataFrame:
    """Build canonical manifest from MS-ASL JSON annotation files."""
    ann_dir = Path(source.annotations_dir)

    if not ann_dir.exists():
        raise FileNotFoundError(f"MS-ASL annotations_dir not found: {ann_dir}")

    class_lookup = _load_class_lookup(ann_dir)
    rel_path_lookup = _index_video_rel_paths(config.paths.videos)
    selected_entries = [
        (split, entry)
        for split in get_selected_splits(source)
        for entry in load_split_json(ann_dir, split)
        if not source.subset or int(entry["label"]) < source.subset
    ]
    video_id_counts = Counter(
        extract_video_id(str(entry["url"])) for _, entry in selected_entries
    )
    allow_shared_video_paths = source.download_mode == "download_missing"
    rows = [
        _build_row(
            entry,
            split,
            class_lookup,
            rel_path_lookup,
            video_id_counts,
            allow_shared_video_paths=allow_shared_video_paths,
        )
        for split, entry in selected_entries
    ]

    df = pd.DataFrame.from_records(rows, columns=_MANIFEST_COLUMNS)

    video_dir = config.paths.videos
    if video_dir and Path(video_dir).is_dir():
        df = apply_availability_policy_paths(df, video_dir, source.availability_policy)

    write_manifest(df, config.paths.manifest)
    return df


def _load_class_lookup(ann_dir: Path) -> dict[int, str]:
    raw = load_classes_json(ann_dir)

    if isinstance(raw, dict):
        for key in ("classes", "data", "items"):
            if key in raw and isinstance(raw[key], list):
                raw = raw[key]
                break

    if not isinstance(raw, list):
        raise ValueError(
            "MS-ASL classes JSON must be a list or dict containing a list of classes."
        )

    lookup: dict[int, str] = {}
    for idx, item in enumerate(raw):
        if isinstance(item, str):
            lookup[idx] = item
            continue

        if not isinstance(item, dict):
            raise ValueError(
                "MS-ASL classes JSON entries must be strings or objects."
            )

        class_id = _coerce_int(
            item.get("label", item.get("class_id", item.get("id", idx)))
        )
        gloss = (
            item.get("clean_text")
            or item.get("gloss")
            or item.get("text")
            or item.get("name")
            or item.get("label_name")
        )
        if class_id is None or not gloss:
            raise ValueError(
                "MS-ASL classes JSON entries must define a class id and gloss text."
            )
        lookup[class_id] = str(gloss)

    return lookup


def _index_video_rel_paths(video_dir: str | None) -> dict[str, list[str]]:
    if not video_dir or not Path(video_dir).is_dir():
        return {}

    base_dir = Path(video_dir)
    files = sorted(
        (
            path
            for path in base_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in _VIDEO_EXTENSIONS
        ),
        key=lambda path: (len(path.relative_to(base_dir).parts), str(path.relative_to(base_dir))),
    )

    rel_paths: dict[str, list[str]] = {}
    for path in files:
        rel_paths.setdefault(path.stem, []).append(path.relative_to(base_dir).as_posix())
    return rel_paths


def _build_row(
    entry: dict[str, Any],
    split: str,
    class_lookup: dict[int, str],
    rel_path_lookup: dict[str, list[str]],
    video_id_counts: Counter[str],
    *,
    allow_shared_video_paths: bool,
) -> dict[str, Any]:
    video_id = extract_video_id(str(entry["url"]))
    start = float(entry["start_time"])
    end = float(entry["end_time"])
    class_id = int(entry["label"])
    gloss = class_lookup.get(class_id)
    if gloss is None:
        raise ValueError(f"MS-ASL class id {class_id} missing from MSASL_classes.json")

    sample_id = (
        f"{split}-{video_id}"
        f"-{int(start * 1000)}-{int(end * 1000)}"
        f"-{class_id}"
    )

    text = str(entry.get("text", "")).strip() or gloss
    row: dict[str, Any] = {
        "SAMPLE_ID": sample_id,
        "VIDEO_ID": video_id,
        "START": start,
        "END": end,
        "CLASS_ID": class_id,
        "GLOSS": gloss,
        "TEXT": text,
        "SIGNER_ID": str(entry.get("signer_id", "")),
        "FPS": entry.get("fps", 0),
        "SOURCE_URL": entry["url"],
        "REL_PATH": _resolve_rel_path(
            sample_id,
            video_id,
            rel_path_lookup,
            video_id_counts,
            allow_shared_video_paths=allow_shared_video_paths,
        ),
        "SPLIT": split,
    }

    box = entry.get("box")
    if isinstance(box, list) and len(box) == 4:
        row["BBOX_X1"] = float(box[1])
        row["BBOX_Y1"] = float(box[0])
        row["BBOX_X2"] = float(box[3])
        row["BBOX_Y2"] = float(box[2])
        row["PERSON_DETECTED"] = True

    return row


def _resolve_rel_path(
    sample_id: str,
    video_id: str,
    rel_path_lookup: dict[str, list[str]],
    video_id_counts: Counter[str],
    *,
    allow_shared_video_paths: bool,
) -> str:
    sample_matches = rel_path_lookup.get(sample_id, [])
    if sample_matches:
        return sample_matches[0]

    video_matches = rel_path_lookup.get(video_id, [])
    if video_id_counts[video_id] > 1 and not allow_shared_video_paths:
        return f"{sample_id}.mp4"

    return video_matches[0] if video_matches else f"{video_id}.mp4"


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None
