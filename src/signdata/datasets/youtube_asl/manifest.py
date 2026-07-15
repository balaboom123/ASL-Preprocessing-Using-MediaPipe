"""YouTube-ASL manifest building."""

import csv
import json
import logging
from pathlib import Path

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

            all_segments.extend(segments)

        except Exception as exc:
            log.error("Error processing %s: %s", video_id, exc)

    df = pd.DataFrame(
        all_segments,
        columns=["VIDEO_ID", "SAMPLE_ID", "START", "END", "TEXT"],
    )
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
        "videos": int(df["VIDEO_ID"].nunique()),
        "segments": len(df),
    }


def _process_segments(
    transcripts: list[dict],
    video_id: str,
    max_text_length: int,
    min_duration: float,
    max_duration: float,
    text_options: dict | None = None,
) -> list[dict]:
    processed = []
    text_kw = text_options or {}

    for entry in transcripts:
        if not {"text", "start", "duration"}.issubset(entry):
            continue

        text = normalize_text(entry["text"], **text_kw)
        dur = entry["duration"]

        if (
            text
            and len(text) <= max_text_length
            and min_duration <= dur <= max_duration
        ):
            processed.append({
                "VIDEO_ID": video_id,
                "SAMPLE_ID": f"{video_id}-{len(processed):03d}",
                "START": entry["start"],
                "END": entry["start"] + dur,
                "TEXT": text,
            })

    return processed
