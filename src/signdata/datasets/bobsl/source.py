"""BOBSL source config, release discovery, and validation."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator

from .._ingestion.availability import AvailabilityPolicy, get_existing_video_ids

ViewName = Literal["subtitle_slt", "isolated_signs"]
SubtitleAlignment = Literal["manual", "original"]
_SUBTITLE_FILE_EXTENSIONS = {".vtt", ".csv", ".tsv"}

_SPLIT_ALIASES = {
    "all": "all",
    "train": "train",
    "val": "val",
    "validation": "val",
    "dev": "val",
    "test": "test",
    "public_test": "test",
    "challenge": "challenge",
    "challenge_test": "challenge",
}
_ALIGNMENT_ALIASES = {
    "manual": "manual",
    "signing": "manual",
    "signed": "manual",
    "original": "original",
    "audio": "original",
    "heuristic": "original",
}


class BOBSLSourceConfig(BaseModel):
    """Typed config for the BOBSL adapter."""

    model_config = ConfigDict(extra="forbid")

    release_dir: str = ""
    view: ViewName = "subtitle_slt"
    split: str = "all"
    subtitle_alignment: SubtitleAlignment = "manual"
    availability_policy: AvailabilityPolicy = "drop_unavailable"
    annotation_root: str = ""
    metadata_file: str = ""
    subtitles_root: str = ""

    @field_validator("split", mode="before")
    @classmethod
    def normalize_split(cls, value: Any) -> str:
        split = str(value or "all").strip().lower()
        if split not in _SPLIT_ALIASES:
            raise ValueError(
                "split must be one of: all, train, val, test, challenge"
            )
        return _SPLIT_ALIASES[split]

    @field_validator("subtitle_alignment", mode="before")
    @classmethod
    def normalize_alignment(cls, value: Any) -> str:
        alignment = str(value or "manual").strip().lower()
        if alignment not in _ALIGNMENT_ALIASES:
            raise ValueError("subtitle_alignment must be 'manual' or 'original'")
        return _ALIGNMENT_ALIASES[alignment]


def get_source_config(config) -> BOBSLSourceConfig:
    return BOBSLSourceConfig(**config.dataset.source)


def resolve_release_dir(source: BOBSLSourceConfig) -> Path:
    return Path(source.release_dir)


def validate_release(
    source: BOBSLSourceConfig,
    config,
    log: logging.Logger,
) -> dict:
    """Validate local BOBSL release files required for the selected view."""
    release_dir = resolve_release_dir(source)
    if not release_dir.exists():
        raise FileNotFoundError(
            f"BOBSL release directory not found: {release_dir}\n"
            "Download the BOBSL release first and set dataset.source.release_dir."
        )

    video_dir = config.paths.videos
    if not video_dir:
        raise ValueError(
            "paths.videos is required for BOBSL. Set it in your config YAML."
        )
    video_path = Path(video_dir)
    if not video_path.exists():
        raise FileNotFoundError(
            f"BOBSL video directory not found: {video_dir}\n"
            "Set paths.videos to the directory containing interpreter-cropped MP4 files."
        )

    resolve_metadata_file(source, release_dir, log)
    if source.view == "subtitle_slt":
        resolve_subtitles_root(source, release_dir, log)
    else:
        resolve_annotation_path(source, release_dir, log)

    existing = get_existing_video_ids(video_dir, recursive=True)
    log.info(
        "BOBSL release validated: %s (%d videos found under %s)",
        release_dir, len(existing), video_dir,
    )
    return {"validated": True, "videos_on_disk": len(existing)}


def resolve_metadata_file(
    source: BOBSLSourceConfig,
    release_dir: Path,
    log: Optional[logging.Logger] = None,
) -> Path:
    """Resolve the metadata JSON describing episode-to-split assignments."""
    explicit = source.metadata_file
    if explicit:
        metadata_path = Path(explicit)
        if not metadata_path.exists():
            raise FileNotFoundError(f"BOBSL metadata_file not found: {metadata_path}")
        return metadata_path

    resolved = _first_existing_path(
        [
            release_dir / "metadata" / "subset2episode.json",
            release_dir / "splits" / "subset2episode.json",
            release_dir / "subset2episode.json",
        ]
    )
    if resolved is not None:
        return resolved

    matches = sorted(release_dir.rglob("subset2episode.json"))
    if matches:
        return matches[0]

    fuzzy = sorted(release_dir.rglob("*subset*episode*.json"))
    if fuzzy:
        if log:
            log.warning("Using fuzzy-matched BOBSL metadata file: %s", fuzzy[0])
        return fuzzy[0]

    raise FileNotFoundError(
        f"Could not auto-discover BOBSL metadata under {release_dir}. "
        "Set dataset.source.metadata_file explicitly."
    )


def resolve_subtitles_root(
    source: BOBSLSourceConfig,
    release_dir: Path,
    log: Optional[logging.Logger] = None,
) -> Path:
    """Resolve the subtitle directory for the requested alignment view."""
    explicit = source.subtitles_root
    if explicit:
        subtitles_root = Path(explicit)
        if not subtitles_root.exists():
            raise FileNotFoundError(
                f"BOBSL subtitles_root not found: {subtitles_root}"
            )
        return subtitles_root

    candidates = []
    if source.subtitle_alignment == "manual":
        candidates.extend(
            [
                release_dir / "subtitles" / "manually-aligned",
                release_dir / "manually-aligned",
            ]
        )
    else:
        candidates.extend(
            [
                release_dir / "subtitles" / "audio-aligned-heuristic-correction",
                release_dir / "audio-aligned-heuristic-correction",
                release_dir / "subtitles" / "original",
            ]
        )

    resolved = _first_existing_path(candidates)
    if resolved is not None:
        return resolved

    pattern = (
        "manually-aligned"
        if source.subtitle_alignment == "manual"
        else "audio-aligned-heuristic-correction"
    )
    matches = sorted(path for path in release_dir.rglob(pattern) if path.is_dir())
    if matches:
        return matches[0]

    fuzzy = sorted(
        path
        for path in release_dir.rglob("*")
        if path.is_dir()
        and _contains_subtitle_files(path)
        and _matches_subtitle_alignment(
            source.subtitle_alignment,
            path.relative_to(release_dir).as_posix().casefold(),
        )
    )
    if fuzzy:
        if log:
            log.warning("Using fuzzy-matched BOBSL subtitles root: %s", fuzzy[0])
        return fuzzy[0]

    raise FileNotFoundError(
        f"Could not auto-discover BOBSL subtitles for alignment="
        f"{source.subtitle_alignment!r} under {release_dir}. "
        "Set dataset.source.subtitles_root explicitly."
    )


def resolve_annotation_path(
    source: BOBSLSourceConfig,
    release_dir: Path,
    log: Optional[logging.Logger] = None,
) -> Path:
    """Resolve the isolated-sign annotation path or directory."""
    explicit = source.annotation_root
    if explicit:
        annotation_path = Path(explicit)
        if not annotation_path.exists():
            raise FileNotFoundError(
                f"BOBSL annotation_root not found: {annotation_path}"
            )
        return annotation_path

    resolved = _first_existing_path(
        [
            release_dir / "annotations" / "isolated_signs.csv",
            release_dir / "annotations" / "isolated_signs.tsv",
            release_dir / "annotations" / "islr.csv",
            release_dir / "annotations" / "islr.tsv",
            release_dir / "annotations",
            release_dir / "islr",
            release_dir / "isolated_signs",
        ]
    )
    if resolved is not None:
        return resolved

    patterns = (
        "*isolated*sign*.csv",
        "*isolated*sign*.tsv",
        "*isolated*sign*.json",
        "*islr*.csv",
        "*islr*.tsv",
        "*islr*.json",
    )
    for pattern in patterns:
        matches = sorted(release_dir.rglob(pattern))
        if matches:
            return matches[0]

    fuzzy_dirs = sorted(
        path
        for path in release_dir.rglob("*")
        if path.is_dir() and ("annotation" in path.name.lower() or "islr" in path.name.lower())
    )
    if fuzzy_dirs:
        if log:
            log.warning("Using fuzzy-matched BOBSL annotation root: %s", fuzzy_dirs[0])
        return fuzzy_dirs[0]

    raise FileNotFoundError(
        f"Could not auto-discover BOBSL isolated-sign annotations under {release_dir}. "
        "Set dataset.source.annotation_root explicitly."
    )


def load_split_map(metadata_path: Path) -> Dict[str, str]:
    """Load episode-to-split mapping from BOBSL metadata JSON."""
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))

    split_map: Dict[str, str] = {}

    def add_entry(episode: Any, split: Any) -> None:
        if episode in (None, "") or split in (None, ""):
            return
        episode_key = _normalize_episode_key(episode)
        split_key = _normalize_split_name(split)
        if not episode_key or not split_key:
            return
        split_map[episode_key] = split_key

    if isinstance(raw, dict):
        list_container = _first_list_container(raw)
        if list_container is not None:
            _ingest_split_entries(list_container, add_entry)
            return split_map

        if any(_normalize_split_name(key) for key in raw):
            for split_name, episodes in raw.items():
                normalized_split = _normalize_split_name(split_name)
                if normalized_split and isinstance(episodes, list):
                    for episode in episodes:
                        add_entry(episode, normalized_split)
            if split_map:
                return split_map

        for episode, value in raw.items():
            if isinstance(value, str):
                add_entry(episode, value)
            elif isinstance(value, dict):
                add_entry(
                    value.get("episode", value.get("name", value.get("file", episode))),
                    value.get("split", value.get("subset")),
                )
        return split_map

    if isinstance(raw, list):
        _ingest_split_entries(raw, add_entry)

    return split_map


def resolve_episode_split(
    episode: Any,
    split_map: Dict[str, str],
    fallback: Any = "",
) -> str:
    """Resolve the normalized split label for an episode."""
    episode_key = _normalize_episode_key(episode)
    if episode_key in split_map:
        return split_map[episode_key]
    return _normalize_split_name(fallback) or ""


def _ingest_split_entries(entries: Iterable[Any], add_entry) -> None:
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        episode = (
            entry.get("episode")
            or entry.get("name")
            or entry.get("fname")
            or entry.get("file")
            or entry.get("video")
        )
        split = entry.get("split") or entry.get("subset")
        add_entry(episode, split)


def _first_list_container(raw: Dict[str, Any]) -> Optional[list]:
    for key in ("episodes", "items", "data", "entries"):
        value = raw.get(key)
        if isinstance(value, list):
            return value
    return None


def _normalize_split_name(value: Any) -> str:
    split = str(value or "").strip().lower()
    return _SPLIT_ALIASES.get(split, "")


def _normalize_episode_key(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return Path(text).stem.casefold()


def _first_existing_path(candidates: Iterable[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _contains_subtitle_files(directory: Path) -> bool:
    return any(
        path.is_file() and path.suffix.lower() in _SUBTITLE_FILE_EXTENSIONS
        for path in directory.rglob("*")
    )


def _matches_subtitle_alignment(alignment: str, relative_path: str) -> bool:
    if alignment == "manual":
        return any(token in relative_path for token in ("manual", "signed", "signing"))

    return any(token in relative_path for token in ("audio", "heuristic", "original"))
