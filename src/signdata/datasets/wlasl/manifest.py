"""WLASL manifest building."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .._ingestion.availability import apply_availability_policy
from .._ingestion.media import get_video_duration
from ...utils.manifest import find_video_file, write_manifest
from .source import WLASLSourceConfig, iter_filtered_instances


def build(config, source: WLASLSourceConfig) -> pd.DataFrame:
    """Build canonical manifest from WLASL_v0.3.json."""
    manifest_path = config.paths.manifest
    video_dir = config.paths.videos

    metadata_json = source.metadata_json
    if not metadata_json or not Path(metadata_json).exists():
        raise FileNotFoundError(
            f"WLASL metadata JSON not found: {metadata_json}\n"
            f"Set dataset.source.metadata_json in your config YAML."
        )

    entries = json.loads(Path(metadata_json).read_text(encoding="utf-8"))

    records = _flatten_instances(entries, video_dir, source)

    if not records:
        raise RuntimeError(
            "WLASL build_manifest produced no rows. "
            "Check that metadata_json is a valid WLASL JSON file."
        )

    df = pd.DataFrame(records)

    if video_dir and Path(video_dir).is_dir():
        df = apply_availability_policy(df, video_dir, source.availability_policy)

    write_manifest(df, manifest_path)
    return df


def _flatten_instances(
    entries: list[dict],
    video_dir: str | None,
    source: WLASLSourceConfig,
) -> list[dict]:
    records = []
    use_preprocessed_timing = source.download_mode == "validate"

    for gloss_idx, inst_idx, entry, inst in iter_filtered_instances(entries, source):
        gloss = entry.get("gloss", "")
        video_id = inst.get("video_id", "")
        variation_id = _coerce_int(inst.get("variation_id"))
        sample_id = video_id if video_id else f"{gloss_idx}-{inst_idx}"

        fps = _coerce_float(inst.get("fps"), default=0.0)
        frame_start = _coerce_int(inst.get("frame_start"))
        frame_end = _coerce_int(inst.get("frame_end"))
        bbox = inst.get("bbox")
        has_bbox = isinstance(bbox, (list, tuple)) and len(bbox) >= 4

        start, end = _resolve_timing(
            video_id=video_id,
            fps=fps,
            frame_start=frame_start,
            frame_end=frame_end,
            video_dir=video_dir,
            use_preprocessed_timing=use_preprocessed_timing,
        )

        records.append({
            "SAMPLE_ID": sample_id,
            "VIDEO_ID": video_id,
            "REL_PATH": _resolve_rel_path(video_id, sample_id, video_dir),
            "SPLIT": str(inst.get("split", "")),
            "GLOSS": gloss,
            "CLASS_ID": gloss_idx,
            "SIGNER_ID": str(inst.get("signer_id", "")),
            "SOURCE_URL": str(inst.get("url", "")),
            "SOURCE": str(inst.get("source", "")),
            "FPS": fps,
            "VARIATION_ID": variation_id,
            "FRAME_START": frame_start,
            "FRAME_END": frame_end,
            "START": start,
            "END": end,
            "BBOX_X1": _coerce_float(bbox[0]) if has_bbox else None,
            "BBOX_Y1": _coerce_float(bbox[1]) if has_bbox else None,
            "BBOX_X2": _coerce_float(bbox[2]) if has_bbox else None,
            "BBOX_Y2": _coerce_float(bbox[3]) if has_bbox else None,
            "PERSON_DETECTED": has_bbox,
        })
    return records


def _resolve_rel_path(video_id: str, sample_id: str, video_dir: str | None) -> str:
    stem = video_id or sample_id
    if video_dir:
        return find_video_file(video_dir, stem).name
    return f"{stem}.mp4"


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _resolve_timing(
    video_id: str,
    fps: float,
    frame_start: Any,
    frame_end: Any,
    video_dir: str | None,
    use_preprocessed_timing: bool,
) -> tuple[float, float]:
    source_timing = _estimate_source_timing(frame_start, frame_end, fps)
    if not use_preprocessed_timing and source_timing is not None:
        return source_timing

    clip_duration = _get_clip_duration(video_id, video_dir)
    if clip_duration > 0:
        return 0.0, clip_duration

    if use_preprocessed_timing and source_timing is not None:
        return 0.0, (float(frame_end) - float(frame_start)) / fps

    return 0.0, 0.0


def _estimate_source_timing(
    frame_start: Any,
    frame_end: Any,
    fps: float,
) -> tuple[float, float] | None:
    if frame_start is None or frame_end is None or fps <= 0:
        return None

    try:
        start = float(frame_start)
        end = float(frame_end)
    except (TypeError, ValueError):
        return None

    if start < 0 or end <= start:
        return None

    return start / fps, end / fps


def _get_clip_duration(video_id: str, video_dir: str | None) -> float:
    if not video_id or not video_dir:
        return 0.0

    video_path = find_video_file(video_dir, video_id)
    if video_path.exists():
        return get_video_duration(str(video_path))

    return 0.0
