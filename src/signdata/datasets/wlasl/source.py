"""WLASL source config, path resolution, and download."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Literal

from pydantic import BaseModel, ConfigDict, Field

from .._ingestion.availability import (
    AvailabilityPolicy,
    get_existing_video_ids,
    write_acquire_report,
)
from .._ingestion.youtube import download_video_urls


class WLASLSourceConfig(BaseModel):
    """Typed config for WLASL adapter."""

    model_config = ConfigDict(extra="forbid")

    metadata_json: str = ""
    split: Literal["all", "train", "val", "test"] = "all"
    subset: int = Field(default=0, ge=0)
    availability_policy: AvailabilityPolicy = "drop_unavailable"
    download_mode: Literal["validate", "download_missing"] = "validate"
    download_format: str = "bestvideo[height>=480]+bestaudio/best"
    rate_limit: str = "5M"
    concurrent_fragments: int = 5


def iter_filtered_instances(entries: list[dict[str, Any]], source: WLASLSourceConfig):
    """Yield WLASL instances after applying subset/split filters."""
    for gloss_idx, entry in enumerate(entries):
        if source.subset > 0 and gloss_idx >= source.subset:
            continue

        for inst_idx, inst in enumerate(entry.get("instances", [])):
            split = str(inst.get("split", ""))
            if source.split != "all" and split != source.split:
                continue
            yield gloss_idx, inst_idx, entry, inst


def get_source_config(config) -> WLASLSourceConfig:
    return WLASLSourceConfig(**config.dataset.source)


def validate_release(
    source: WLASLSourceConfig,
    config,
    log: logging.Logger,
) -> dict:
    """Validate that the video directory exists."""
    video_dir = config.paths.videos
    if not video_dir:
        raise ValueError(
            "paths.videos is required for WLASL. Set it in your config YAML."
        )
    video_path = Path(video_dir)
    if not video_path.exists():
        raise FileNotFoundError(
            f"WLASL video directory not found: {video_dir}\n"
            f"Set paths.videos to the directory containing downloaded WLASL video files."
        )
    existing = get_existing_video_ids(video_dir)
    log.info(
        "WLASL video directory validated: %s (%d videos found)",
        video_dir, len(existing),
    )
    return {"validated": True, "videos_on_disk": len(existing)}


def download_missing(
    source: WLASLSourceConfig,
    config,
    log: logging.Logger,
) -> dict:
    """Download any WLASL videos missing from disk."""
    video_dir = config.paths.videos
    if not video_dir:
        raise ValueError(
            "paths.videos is required for WLASL. Set it in your config YAML."
        )

    metadata_json = source.metadata_json
    if not metadata_json or not Path(metadata_json).exists():
        raise FileNotFoundError(f"WLASL metadata JSON not found: {metadata_json}")

    Path(video_dir).mkdir(parents=True, exist_ok=True)

    entries = json.loads(Path(metadata_json).read_text(encoding="utf-8"))

    video_urls: Dict[str, str] = {}
    for _, _, _, inst in iter_filtered_instances(entries, source):
        url = inst.get("url", "")
        video_id = inst.get("video_id", "")
        if url and video_id:
            video_urls[video_id] = url

    all_ids = set(video_urls.keys())
    existing = get_existing_video_ids(video_dir)
    to_download_ids = sorted(all_ids - existing)

    if not to_download_ids:
        log.info("All %d videos already downloaded.", len(all_ids))
        stats = {"total": len(all_ids), "downloaded": 0, "errors": 0, "skipped": len(all_ids)}
        report_dir = Path(config.paths.root) / "acquire_report"
        write_acquire_report(report_dir, stats, missing=[])
        return stats

    log.info("Downloading %d / %d videos...", len(to_download_ids), len(all_ids))

    result = download_video_urls(
        {video_id: video_urls[video_id] for video_id in to_download_ids},
        video_dir,
        download_format=source.download_format,
        rate_limit=source.rate_limit,
        concurrent_fragments=source.concurrent_fragments,
        log=log,
    )
    missing = result["missing"]
    stats = {
        "total": len(all_ids),
        "downloaded": result["downloaded"],
        "errors": result["errors"],
        "skipped": len(existing),
    }
    report_dir = Path(config.paths.root) / "acquire_report"
    write_acquire_report(report_dir, stats, missing)

    if source.availability_policy == "fail_fast" and result["errors"] > 0:
        raise RuntimeError(
            f"{result['errors']} download(s) failed with "
            f"availability_policy='fail_fast'. "
            f"See {report_dir}/download_report.json for details."
        )
    return stats
