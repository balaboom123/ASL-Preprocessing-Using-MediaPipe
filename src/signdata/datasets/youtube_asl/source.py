"""YouTube-ASL source config and acquisition."""

import json
import logging
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from .._ingestion.availability import (
    AvailabilityPolicy,
    get_existing_video_ids,
    write_acquire_report,
)
from .._ingestion.text import TextProcessingConfig
from .._ingestion.youtube import download_youtube_videos

DEFAULT_TRANSCRIPT_LANGUAGES = [
    "en",
    "ase",
    "en-US",
    "en-CA",
    "en-GB",
    "en-AU",
    "en-NZ",
    "en-IN",
    "en-ZA",
    "en-IE",
    "en-SG",
    "en-PH",
    "en-NG",
    "en-PK",
    "en-JM",
]

# yt-dlp's format sort ranks vcodec above size, so a bare `worstvideo` resolves
# to the worst-ranked *codec* (h264) — the largest stream at a given resolution.
# YouTube's av01/vp9 renditions are independent encodes of the same master, not
# transcodes, so preferring them is a free ~30-50% size win. Each added branch
# keeps the original ">=720p, lowest qualifying resolution" pick; only the codec
# changes. Falls through to the previous behaviour when neither is offered.
DEFAULT_DOWNLOAD_FORMAT = (
    "worstvideo[height>=720][fps>=24][vcodec^=av01]+worstaudio"
    "/worstvideo[height>=720][fps>=24][vcodec^=vp09]+worstaudio"
    "/worstvideo[height>=720][fps>=24][vcodec^=vp9]+worstaudio"
    "/worstvideo[height>=720][fps>=24]+worstaudio"
    "/bestvideo[height>=480][height<720][fps>=24][fps<=60]+worstaudio"
    "/bestvideo[height>=480][height<=1080][fps>=14]+worstaudio"
    "/best"
)


class YouTubeASLSourceConfig(BaseModel):
    """Typed config for YouTube-ASL adapter."""

    model_config = ConfigDict(extra="forbid")

    video_ids_file: str = ""
    languages: list[str] = Field(
        default_factory=lambda: DEFAULT_TRANSCRIPT_LANGUAGES.copy()
    )
    availability_policy: AvailabilityPolicy = "drop_unavailable"
    download_format: str = DEFAULT_DOWNLOAD_FORMAT
    rate_limit: str = "5M"
    concurrent_fragments: int = 5
    transcript_proxy_http: str | None = None
    transcript_proxy_https: str | None = None
    stop_on_transcript_block: bool = True
    max_text_length: int = 300
    min_duration: float = 0.2
    max_duration: float = 60.0
    text_processing: TextProcessingConfig = Field(
        default_factory=TextProcessingConfig
    )


def get_source_config(config) -> YouTubeASLSourceConfig:
    return YouTubeASLSourceConfig(**config.dataset.source)


def download(
    source: YouTubeASLSourceConfig,
    config,
    log: logging.Logger,
) -> dict:
    """Download YouTube videos and transcripts."""
    video_dir = config.paths.videos
    transcript_dir = config.paths.transcripts

    Path(transcript_dir).mkdir(parents=True, exist_ok=True)
    Path(video_dir).mkdir(parents=True, exist_ok=True)

    log.info("Starting transcript download...")
    transcript_stats = _download_transcripts(
        source.video_ids_file,
        transcript_dir,
        source,
        log,
    )

    log.info("Starting video download...")
    video_result = _download_videos(
        source.video_ids_file,
        video_dir,
        source,
        log,
    )
    video_stats = {
        key: video_result[key]
        for key in ("total", "downloaded", "errors")
    }

    report_dir = Path(config.paths.root) / "acquire_report"
    write_acquire_report(report_dir, video_stats, video_result["missing"])

    if source.availability_policy == "fail_fast" and video_stats["errors"] > 0:
        raise RuntimeError(
            f"{video_stats['errors']} download(s) failed with "
            f"availability_policy='fail_fast'. "
            f"See {report_dir}/download_report.json for details."
        )

    return {
        "transcripts": transcript_stats,
        "videos": video_stats,
    }


def _load_video_ids(file_path: str) -> set[str]:
    """Load video IDs from a text file."""
    return {
        line.strip()
        for line in Path(file_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _download_transcripts(
    video_id_file: str,
    transcript_dir: str,
    source: YouTubeASLSourceConfig,
    log: logging.Logger,
) -> dict:
    from youtube_transcript_api._errors import (
        IpBlocked,
        NoTranscriptFound,
        RequestBlocked,
        TranscriptsDisabled,
        VideoUnavailable,
    )

    existing_ids = {path.stem for path in Path(transcript_dir).glob("*.json")}
    all_ids = _load_video_ids(video_id_file)
    ids = sorted(all_ids - existing_ids)

    if not ids:
        log.info("All transcripts already downloaded.")
        return {
            "total": len(all_ids),
            "attempted": 0,
            "downloaded": 0,
            "errors": 0,
            "blocked": False,
        }

    sleep_time = 0.2
    error_count = 0
    downloaded = 0
    blocked = False
    transcript_client = _build_transcript_client(source)

    with tqdm(ids, desc="Downloading transcripts") as pbar:
        for video_id in pbar:
            sleep_time = min(sleep_time, 2)
            time.sleep(sleep_time)
            try:
                transcript = transcript_client.fetch(
                    video_id,
                    languages=source.languages,
                )
                transcript = _normalize_transcript_payload(transcript)
                (Path(transcript_dir) / f"{video_id}.json").write_text(
                    json.dumps(transcript), encoding="utf-8"
                )
                downloaded += 1
            except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
                log.warning("Transcript unavailable for %s: %s", video_id, e)
                error_count += 1
            except (RequestBlocked, IpBlocked) as e:
                error_count += 1
                blocked = True
                log.error("Transcript download blocked for %s: %s", video_id, e)
                if source.stop_on_transcript_block:
                    log.error(
                        "Stopping transcript download early after an IP block. "
                        "Set dataset.source.transcript_proxy_http / "
                        "dataset.source.transcript_proxy_https or use a rotating "
                        "residential proxy to continue."
                    )
                    pbar.set_postfix(errors=error_count, blocked=1)
                    break
            except Exception as e:
                sleep_time += 0.1
                log.error("Error downloading transcript for %s: %s", video_id, e)
                error_count += 1
            pbar.set_postfix(errors=error_count)

    return {
        "total": len(all_ids),
        "attempted": downloaded + error_count,
        "downloaded": downloaded,
        "errors": error_count,
        "blocked": blocked,
    }


def _build_transcript_client(source: YouTubeASLSourceConfig) -> Any:
    from youtube_transcript_api import YouTubeTranscriptApi

    proxy_config = None
    if source.transcript_proxy_http or source.transcript_proxy_https:
        from youtube_transcript_api.proxies import GenericProxyConfig

        proxy_config = GenericProxyConfig(
            http_url=source.transcript_proxy_http,
            https_url=source.transcript_proxy_https,
        )

    return YouTubeTranscriptApi(proxy_config=proxy_config)


def _normalize_transcript_payload(transcript: Any) -> list[dict]:
    if hasattr(transcript, "to_raw_data"):
        transcript = transcript.to_raw_data()

    if isinstance(transcript, list):
        return transcript

    raise TypeError(
        "Unexpected transcript payload type "
        f"{type(transcript).__name__}; expected a list or object with "
        "to_raw_data()."
    )


def _download_videos(
    video_id_file: str,
    video_dir: str,
    source: YouTubeASLSourceConfig,
    log: logging.Logger,
) -> dict:
    existing_ids = get_existing_video_ids(video_dir)
    all_ids = _load_video_ids(video_id_file)
    ids = sorted(all_ids - existing_ids)

    if not ids:
        log.info("All videos already downloaded.")
        return {
            "total": len(all_ids),
            "downloaded": 0,
            "errors": 0,
            "missing": [],
        }

    result = download_youtube_videos(
        ids,
        video_dir,
        download_format=source.download_format,
        rate_limit=source.rate_limit,
        concurrent_fragments=source.concurrent_fragments,
        log=log,
    )

    return {
        "total": len(all_ids),
        "downloaded": result["downloaded"],
        "errors": result["errors"],
        "missing": result["missing"],
    }
