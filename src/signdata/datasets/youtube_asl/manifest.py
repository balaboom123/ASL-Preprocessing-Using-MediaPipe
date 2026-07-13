"""YouTube-ASL manifest building."""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from .._ingestion.availability import apply_availability_policy
from .._ingestion.text import normalize_text
from .source import YouTubeASLSourceConfig


def build(
    config,
    source: YouTubeASLSourceConfig,
    log: logging.Logger,
):
    """Build segmented manifest from transcript JSON files."""
    transcript_dir = Path(config.paths.transcripts)
    manifest_path = Path(config.paths.manifest)
    json_files = sorted(transcript_dir.glob("*.json"))

    if not json_files:
        log.warning("No transcript files found in %s", transcript_dir)
    else:
        log.info(
            "Processing %d transcript files from %s",
            len(json_files),
            transcript_dir,
        )

    text_opts = source.text_processing.model_dump()
    processed_count = 0
    all_segments = []

    for json_file in tqdm(json_files, desc="Building manifest"):
        video_id = json_file.stem
        try:
            transcript_data = json.loads(json_file.read_text(encoding="utf-8"))

            if not transcript_data:
                continue

            segments = _process_segments(
                transcript_data,
                video_id,
                source.max_text_length,
                source.min_duration,
                source.max_duration,
                text_opts,
            )

            if segments:
                all_segments.extend(segments)
                processed_count += 1

        except Exception as e:
            log.error("Error processing %s: %s", video_id, e)

    columns = ["VIDEO_ID", "SAMPLE_ID", "START", "END", "TEXT"]
    df = pd.DataFrame(all_segments, columns=columns)
    video_dir = config.paths.videos
    if not df.empty and video_dir and Path(video_dir).is_dir():
        df = apply_availability_policy(df, video_dir, source.availability_policy)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        manifest_path,
        sep="\t",
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_ALL,
    )

    return manifest_path, df, {
        "videos": processed_count,
        "segments": len(all_segments),
    }


def _process_segments(
    transcripts: List[Dict],
    video_id: str,
    max_text_length: int,
    min_duration: float,
    max_duration: float,
    text_options: Optional[Dict] = None,
) -> List[Dict]:
    processed = []
    idx = 0

    valid = [
        t for t in transcripts
        if "text" in t and "start" in t and "duration" in t
    ]

    text_kw = text_options or {}

    for entry in valid:
        text = normalize_text(entry["text"], **text_kw)
        dur = entry["duration"]

        if (
            text
            and len(text) <= max_text_length
            and min_duration <= dur <= max_duration
        ):
            processed.append({
                "VIDEO_ID": video_id,
                "SAMPLE_ID": f"{video_id}-{idx:03d}",
                "START": entry["start"],
                "END": entry["start"] + dur,
                "TEXT": text,
            })
            idx += 1

    return processed
