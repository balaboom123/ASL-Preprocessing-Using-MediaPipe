"""MS-ASL source config, path resolution, and download."""

import json
import logging
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .._ingestion.availability import (
    AvailabilityPolicy,
    get_existing_video_ids,
    write_acquire_report,
)
from .._ingestion.youtube import download_youtube_videos

SPLITS = ("train", "val", "test")
SplitName = Literal["train", "val", "test", "all"]
DownloadMode = Literal["validate", "download_missing"]


class MSASLSourceConfig(BaseModel):
    """Typed config for MS-ASL adapter."""

    model_config = ConfigDict(extra="forbid")

    annotations_dir: str = ""
    split: SplitName = "all"
    subset: int = 1000
    availability_policy: AvailabilityPolicy = "drop_unavailable"
    download_mode: DownloadMode = "validate"
    download_format: str = "bestvideo[height>=480]+bestaudio/best"
    rate_limit: str = "5M"
    concurrent_fragments: int = 5


def get_source_config(config) -> MSASLSourceConfig:
    return MSASLSourceConfig(**config.dataset.source)


def extract_video_id(url: str) -> str:
    """Extract an 11-character YouTube video ID from a URL."""
    patterns = [
        r'(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'(?:embed/)([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return url.split("/")[-1][:11]


def load_split_json(ann_dir: Path, split: str) -> list[dict]:
    json_path = ann_dir / f"MSASL_{split}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"MS-ASL annotation file not found: {json_path}")
    return json.loads(json_path.read_text(encoding="utf-8"))


def load_classes_json(ann_dir: Path) -> Any:
    json_path = ann_dir / "MSASL_classes.json"
    if not json_path.exists():
        raise FileNotFoundError(f"MS-ASL classes JSON not found: {json_path}")
    return json.loads(json_path.read_text(encoding="utf-8"))


def get_selected_splits(source: MSASLSourceConfig) -> tuple[str, ...]:
    return SPLITS if source.split == "all" else (source.split,)


def validate(
    source: MSASLSourceConfig,
    config,
    log: logging.Logger,
) -> dict:
    """Validate annotation files and video directory."""
    ann_dir = Path(source.annotations_dir)
    if not ann_dir.exists():
        raise FileNotFoundError(f"MS-ASL annotations_dir not found: {ann_dir}")

    for split in SPLITS:
        json_path = ann_dir / f"MSASL_{split}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"MS-ASL annotation file not found: {json_path}")
    load_classes_json(ann_dir)

    video_dir = config.paths.videos
    if not video_dir:
        raise ValueError(
            "paths.videos is required for MS-ASL. Set it in your config YAML."
        )
    if not Path(video_dir).exists():
        raise FileNotFoundError(f"MS-ASL video directory not found: {video_dir}")

    log.info("MS-ASL annotations validated: %s", source.annotations_dir)
    return {"validated": True, "videos_on_disk": len(get_existing_video_ids(video_dir, recursive=True))}


def download_missing(
    source: MSASLSourceConfig,
    config,
    log: logging.Logger,
) -> dict:
    """Download videos not already present in paths.videos."""
    video_dir = config.paths.videos
    if not video_dir:
        raise ValueError(
            "paths.videos is required for MS-ASL. Set it in your config YAML."
        )

    Path(video_dir).mkdir(parents=True, exist_ok=True)

    ann_dir = Path(source.annotations_dir)
    load_classes_json(ann_dir)
    selected_splits = get_selected_splits(source)

    all_video_ids: set = set()
    for split in selected_splits:
        entries = load_split_json(ann_dir, split)
        for entry in entries:
            vid = extract_video_id(entry["url"])
            all_video_ids.add(vid)

    existing = all_video_ids & get_existing_video_ids(video_dir, recursive=True)
    to_download = sorted(all_video_ids - existing)

    if not to_download:
        log.info("All %d videos already downloaded.", len(all_video_ids))
        stats = {
            "total": len(all_video_ids),
            "downloaded": 0,
            "errors": 0,
            "skipped": len(all_video_ids),
        }
        report_dir = Path(config.paths.root) / "acquire_report"
        write_acquire_report(report_dir, stats, missing=[])
        return stats

    log.info("Downloading %d / %d videos...", len(to_download), len(all_video_ids))

    result = download_youtube_videos(
        to_download,
        video_dir,
        download_format=source.download_format,
        rate_limit=source.rate_limit,
        concurrent_fragments=source.concurrent_fragments,
        log=log,
    )
    stats = {
        "total": len(all_video_ids),
        "downloaded": result["downloaded"],
        "errors": result["errors"],
        "skipped": len(existing),
    }
    report_dir = Path(config.paths.root) / "acquire_report"
    write_acquire_report(report_dir, stats, result["missing"])

    if source.availability_policy == "fail_fast" and result["errors"] > 0:
        raise RuntimeError(
            f"{result['errors']} download(s) failed with "
            f"availability_policy='fail_fast'. "
            f"See {report_dir}/download_report.json for details."
        )
    return stats
