"""BOBSL manifest building."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .._ingestion.availability import apply_availability_policy_paths
from ...utils.manifest import write_manifest
from .source import (
    BOBSLSourceConfig,
    load_split_map,
    resolve_annotation_path,
    resolve_episode_split,
    resolve_metadata_file,
    resolve_release_dir,
    resolve_subtitles_root,
)

_VIDEO_EXTENSIONS = {".mp4", ".webm", ".mkv", ".avi", ".mov"}
_SUBTITLE_EXTENSIONS = {".vtt", ".csv", ".tsv"}
_EPISODE_KEYS = ("episode", "episode_id", "video_id", "video", "file", "fname", "name")
_START_KEYS = ("start", "start_time", "segment_start")
_END_KEYS = ("end", "end_time", "segment_end")
_GLOSS_KEYS = ("gloss", "annotation", "label", "sign")
_CLASS_ID_KEYS = ("class_id", "label_id", "gloss_id", "id")
_SIGNER_KEYS = ("signer_id", "signer", "speaker")
_SPLIT_KEYS = ("split", "subset")


def build(config, source: BOBSLSourceConfig, log: logging.Logger) -> pd.DataFrame:
    """Build a canonical manifest from the BOBSL release."""
    manifest_path = config.paths.manifest
    release_dir = resolve_release_dir(source)
    metadata_path = resolve_metadata_file(source, release_dir, log)
    split_map = load_split_map(metadata_path)
    video_lookup = _index_video_rel_paths(config.paths.videos)

    if source.view == "subtitle_slt":
        df = _build_subtitle_manifest(
            source=source,
            release_dir=release_dir,
            split_map=split_map,
            video_lookup=video_lookup,
            log=log,
        )
    elif source.view == "isolated_signs":
        df = _build_isolated_manifest(
            source=source,
            release_dir=release_dir,
            split_map=split_map,
            video_lookup=video_lookup,
            log=log,
        )
    else:
        raise ValueError(f"Unsupported BOBSL view: {source.view!r}")

    if df.empty:
        raise RuntimeError(
            f"BOBSL build_manifest produced no rows for view={source.view!r}. "
            "Check the configured split and release contents."
        )

    video_dir = config.paths.videos
    if video_dir and Path(video_dir).is_dir():
        df = apply_availability_policy_paths(
            df,
            base_dir=video_dir,
            policy=source.availability_policy,
            rel_path_col="REL_PATH",
        )

    write_manifest(df, manifest_path)
    return df


def _build_subtitle_manifest(
    *,
    source: BOBSLSourceConfig,
    release_dir: Path,
    split_map: Dict[str, str],
    video_lookup: Dict[str, List[str]],
    log: logging.Logger,
) -> pd.DataFrame:
    subtitles_root = resolve_subtitles_root(source, release_dir, log)
    subtitle_files = sorted(
        path
        for path in subtitles_root.rglob("*")
        if path.is_file() and path.suffix.lower() in _SUBTITLE_EXTENSIONS
    )
    if not subtitle_files:
        raise FileNotFoundError(
            f"No subtitle files found under {subtitles_root}."
        )

    rows: List[Dict[str, Any]] = []
    for subtitle_path in subtitle_files:
        episode = _episode_id_from_path(subtitle_path)
        split_label = resolve_episode_split(
            episode,
            split_map,
            fallback=_infer_split_from_path(subtitles_root, subtitle_path),
        )
        if not _split_selected(split_label, source.split):
            continue

        rel_path = _resolve_rel_path(episode, video_lookup)
        for index, segment in enumerate(_read_subtitle_segments(subtitle_path)):
            start = float(segment["start"])
            end = float(segment["end"])
            if end <= start:
                continue
            sample_id = _build_sample_id(
                episode=episode,
                split_label=split_label,
                start=start,
                end=end,
                index=index,
            )
            rows.append(
                {
                    "SAMPLE_ID": sample_id,
                    "VIDEO_ID": episode,
                    "REL_PATH": rel_path,
                    "SPLIT": split_label,
                    "START": start,
                    "END": end,
                    "TEXT": str(segment["text"]).strip(),
                }
            )

    return pd.DataFrame(rows)


def _build_isolated_manifest(
    *,
    source: BOBSLSourceConfig,
    release_dir: Path,
    split_map: Dict[str, str],
    video_lookup: Dict[str, List[str]],
    log: logging.Logger,
) -> pd.DataFrame:
    annotation_path = resolve_annotation_path(source, release_dir, log)
    records = _read_isolated_records(annotation_path)
    if not records:
        raise FileNotFoundError(
            f"No isolated-sign annotation records found at {annotation_path}."
        )

    rows: List[Dict[str, Any]] = []
    missing_class_rows: List[tuple[str, str]] = []

    for index, record in enumerate(records):
        episode = _coerce_episode(record)
        start = _coerce_float(_first_value(record, _START_KEYS))
        end = _coerce_float(_first_value(record, _END_KEYS))
        gloss = str(_first_value(record, _GLOSS_KEYS, default="")).strip()
        if not episode or start is None or end is None or end <= start or not gloss:
            continue

        split_label = resolve_episode_split(
            episode,
            split_map,
            fallback=_first_value(record, _SPLIT_KEYS, default=""),
        )
        if not _split_selected(split_label, source.split):
            continue

        class_id = _coerce_int(_first_value(record, _CLASS_ID_KEYS))
        if class_id is None:
            missing_class_rows.append((episode, gloss))
            class_id = -1

        rel_path = _resolve_rel_path(episode, video_lookup)
        sample_id = _build_sample_id(
            episode=episode,
            split_label=split_label,
            start=start,
            end=end,
            index=index,
            suffix=str(class_id) if class_id >= 0 else gloss,
        )
        row = {
            "SAMPLE_ID": sample_id,
            "VIDEO_ID": episode,
            "REL_PATH": rel_path,
            "SPLIT": split_label,
            "START": start,
            "END": end,
            "GLOSS": gloss,
            "TEXT": gloss,
            "CLASS_ID": class_id,
        }
        signer_id = _first_value(record, _SIGNER_KEYS, default="")
        if signer_id not in (None, ""):
            row["SIGNER_ID"] = str(signer_id)
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if missing_class_rows:
        explicit_gloss_to_class: Dict[str, int] = {}
        used_class_ids = {
            class_id
            for class_id in (
                _coerce_int(row.get("CLASS_ID")) for _, row in df.iterrows()
            )
            if class_id is not None and class_id >= 0
        }
        for _, row in df.iterrows():
            class_id = _coerce_int(row.get("CLASS_ID"))
            gloss = str(row.get("GLOSS", ""))
            if class_id is not None and class_id >= 0 and gloss:
                explicit_gloss_to_class.setdefault(gloss, class_id)

        generated_gloss_to_class: Dict[str, int] = {}
        next_class_id = (max(used_class_ids) + 1) if used_class_ids else 0
        resolved_ids: List[int] = []
        for _, row in df.iterrows():
            class_id = _coerce_int(row.get("CLASS_ID"))
            gloss = str(row.get("GLOSS", ""))
            if class_id is None or class_id < 0:
                if gloss in explicit_gloss_to_class:
                    resolved_ids.append(explicit_gloss_to_class[gloss])
                    continue

                if gloss not in generated_gloss_to_class:
                    while next_class_id in used_class_ids:
                        next_class_id += 1
                    generated_gloss_to_class[gloss] = next_class_id
                    used_class_ids.add(next_class_id)
                    next_class_id += 1
                resolved_ids.append(generated_gloss_to_class[gloss])
            else:
                resolved_ids.append(class_id)
        df["CLASS_ID"] = resolved_ids

    return df


def _read_subtitle_segments(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".vtt":
        return _read_vtt_segments(path)
    return _read_tabular_subtitle_segments(path)


def _read_vtt_segments(path: Path) -> List[Dict[str, Any]]:
    blocks: List[List[str]] = []
    current: List[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip("\ufeff")
        if line.strip():
            current.append(line)
            continue
        if current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)

    segments: List[Dict[str, Any]] = []
    for block in blocks:
        if not block:
            continue
        if block[0].strip().upper().startswith("WEBVTT"):
            continue

        timing_index = 0
        if "-->" not in block[0]:
            if len(block) < 2 or "-->" not in block[1]:
                continue
            timing_index = 1

        timing_line = block[timing_index]
        start_text, end_text = timing_line.split("-->", 1)
        start = _parse_timestamp(start_text.strip())
        end = _parse_timestamp(end_text.strip().split()[0])
        text_lines = block[timing_index + 1 :]
        text = " ".join(line.strip() for line in text_lines if line.strip())
        if text:
            segments.append({"start": start, "end": end, "text": text})

    return segments


def _read_tabular_subtitle_segments(path: Path) -> List[Dict[str, Any]]:
    separator = "\t" if path.suffix.lower() == ".tsv" else ","
    table = pd.read_csv(path, sep=separator)
    start_col = _select_column(
        table.columns,
        (
            "start",
            "start_time",
            "subtitle_start",
            "start sub (after alignement heuristic 1)",
        ),
    )
    end_col = _select_column(
        table.columns,
        (
            "end",
            "end_time",
            "subtitle_end",
            "end sub (after alignement heuristic 1)",
        ),
    )
    text_col = _select_column(
        table.columns,
        ("text", "subtitle", "english sentence", "sentence", "translation"),
    )
    if not start_col or not end_col or not text_col:
        raise ValueError(
            f"BOBSL subtitle table missing timing/text columns: {path}"
        )

    segments = []
    for _, row in table.iterrows():
        start = _coerce_float(row.get(start_col))
        end = _coerce_float(row.get(end_col))
        text = str(row.get(text_col, "")).strip()
        if start is None or end is None or end <= start or not text:
            continue
        segments.append({"start": start, "end": end, "text": text})
    return segments


def _read_isolated_records(path: Path) -> List[Dict[str, Any]]:
    if path.is_dir():
        records: List[Dict[str, Any]] = []
        files = sorted(
            file_path
            for file_path in path.rglob("*")
            if file_path.is_file() and file_path.suffix.lower() in {".csv", ".tsv", ".json"}
        )
        for file_path in files:
            records.extend(_read_isolated_records(file_path))
        return records

    if path.suffix.lower() in {".csv", ".tsv"}:
        separator = "\t" if path.suffix.lower() == ".tsv" else ","
        table = pd.read_csv(path, sep=separator)
        return table.to_dict(orient="records")

    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return _flatten_json_records(raw)

    return []


def _flatten_json_records(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return [entry for entry in raw if isinstance(entry, dict)]

    if isinstance(raw, dict):
        for key in ("items", "data", "entries", "annotations"):
            value = raw.get(key)
            if isinstance(value, list):
                return [entry for entry in value if isinstance(entry, dict)]

        flattened: List[Dict[str, Any]] = []
        for episode, value in raw.items():
            if isinstance(value, list):
                for entry in value:
                    if isinstance(entry, dict):
                        item = dict(entry)
                        item.setdefault("episode", episode)
                        flattened.append(item)
            elif isinstance(value, dict):
                item = dict(value)
                item.setdefault("episode", episode)
                flattened.append(item)
        return flattened

    return []


def _index_video_rel_paths(video_dir: str) -> Dict[str, List[str]]:
    base_dir = Path(video_dir)
    if not base_dir.is_dir():
        return {}

    files = sorted(
        (
            path
            for path in base_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in _VIDEO_EXTENSIONS
        ),
        key=lambda path: (
            len(path.relative_to(base_dir).parts),
            str(path.relative_to(base_dir)),
        ),
    )

    rel_paths: Dict[str, List[str]] = {}
    for path in files:
        rel_path = path.relative_to(base_dir).as_posix()
        rel_paths.setdefault(path.stem, []).append(rel_path)
        rel_paths.setdefault(path.stem.casefold(), []).append(rel_path)
    return rel_paths


def _resolve_rel_path(episode: str, video_lookup: Dict[str, List[str]]) -> str:
    matches = video_lookup.get(episode) or video_lookup.get(episode.casefold()) or []
    if matches:
        return matches[0]
    return f"{episode}.mp4"


def _split_selected(split_label: str, selected_split: str) -> bool:
    return selected_split == "all" or split_label == selected_split


def _build_sample_id(
    *,
    episode: str,
    split_label: str,
    start: float,
    end: float,
    index: int,
    suffix: str = "",
) -> str:
    start_ms = int(round(start * 1000))
    end_ms = int(round(end * 1000))
    parts = [split_label or "unknown", episode, str(start_ms), str(end_ms), str(index)]
    if suffix:
        parts.append(str(suffix).replace(" ", "_"))
    return "-".join(parts)


def _episode_id_from_path(path: Path) -> str:
    return path.stem


def _infer_split_from_path(root: Path, path: Path) -> str:
    relative_parts = [part.casefold() for part in path.relative_to(root).parts[:-1]]
    for part in relative_parts:
        if part in {"train", "val", "test", "challenge"}:
            return part
    return ""


def _parse_timestamp(value: str) -> float:
    timestamp = value.strip().replace(",", ".")
    parts = timestamp.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = "0"
        minutes, seconds = parts
    else:
        raise ValueError(f"Unsupported subtitle timestamp: {value!r}")
    return (int(hours) * 3600) + (int(minutes) * 60) + float(seconds)


def _select_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    normalized = {column.casefold(): column for column in columns}
    for candidate in candidates:
        match = normalized.get(candidate.casefold())
        if match is not None:
            return match
    return None


def _coerce_episode(record: Dict[str, Any]) -> str:
    value = _first_value(record, _EPISODE_KEYS)
    return Path(str(value or "")).stem


def _first_value(record: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return default


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None
