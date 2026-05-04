"""Shared video download helpers using yt-dlp.

Provides reusable yt-dlp wrappers for dataset adapters that download from
YouTube IDs or explicit per-sample source URLs.
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple, TypedDict

from tqdm import tqdm

logger = logging.getLogger(__name__)


class DownloadResult(TypedDict):
    downloaded: int
    errors: int
    missing: List[Dict[str, str]]


def _build_yt_config(
    video_dir: str,
    output_stem: str,
    *,
    download_format: str,
    rate_limit: str,
    concurrent_fragments: int,
) -> Dict[str, object]:
    return {
        "format": download_format,
        "merge_output_format": "mp4",
        "postprocessors": [
            {"key": "FFmpegVideoConvertor", "preferedformat": "mp4"},
        ],
        "writesubtitles": False,
        "outtmpl": os.path.join(video_dir, f"{output_stem}.%(ext)s"),
        "nocheckcertificate": True,
        "geo-bypass": True,
        "limit_rate": rate_limit,
        "http-chunk-size": 10485760,
        "noplaylist": True,
        "no-metadata-json": True,
        "no-metadata": True,
        "concurrent-fragments": concurrent_fragments,
        "hls-prefer-ffmpeg": True,
        "sleep-interval": 0,
    }


def _download_targets(
    targets: List[Tuple[str, str]],
    video_dir: str,
    *,
    download_format: str,
    rate_limit: str = "5M",
    concurrent_fragments: int = 5,
    sleep_interval: float = 0.2,
    log: Optional[logging.Logger] = None,
) -> DownloadResult:
    """Download explicit ``(video_id, url)`` targets via yt-dlp."""
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import (
        DownloadError,
        ExtractorError,
        PostProcessingError,
        UnavailableVideoError,
    )

    _log = log or logger

    error_count = 0
    downloaded = 0
    missing: List[Dict[str, str]] = []

    with tqdm(targets, desc="Downloading videos", unit="video") as pbar:
        for video_id, url in pbar:
            time.sleep(sleep_interval)
            yt_config = _build_yt_config(
                video_dir,
                video_id,
                download_format=download_format,
                rate_limit=rate_limit,
                concurrent_fragments=concurrent_fragments,
            )
            try:
                with YoutubeDL(yt_config) as yt:
                    yt.extract_info(url, download=True)
                downloaded += 1
            except (
                DownloadError,
                ExtractorError,
                PostProcessingError,
                UnavailableVideoError,
            ) as e:
                _log.error("Error downloading %s from %s: %s", video_id, url, e)
                missing.append({
                    "VIDEO_ID": video_id,
                    "SOURCE_URL": url,
                    "REASON": str(e),
                })
                error_count += 1
            except Exception as e:
                _log.error("Unexpected error for %s from %s: %s", video_id, url, e)
                missing.append({
                    "VIDEO_ID": video_id,
                    "SOURCE_URL": url,
                    "REASON": str(e),
                })
                error_count += 1
            pbar.set_postfix(errors=error_count)

    return DownloadResult(
        downloaded=downloaded,
        errors=error_count,
        missing=missing,
    )


def download_video_urls(
    video_urls: Dict[str, str],
    video_dir: str,
    *,
    download_format: str,
    rate_limit: str = "5M",
    concurrent_fragments: int = 5,
    sleep_interval: float = 0.2,
    log: Optional[logging.Logger] = None,
) -> DownloadResult:
    """Download videos from explicit source URLs keyed by output video ID."""
    return _download_targets(
        list(video_urls.items()),
        video_dir,
        download_format=download_format,
        rate_limit=rate_limit,
        concurrent_fragments=concurrent_fragments,
        sleep_interval=sleep_interval,
        log=log,
    )


def download_youtube_videos(
    video_ids: List[str],
    video_dir: str,
    *,
    download_format: str,
    rate_limit: str = "5M",
    concurrent_fragments: int = 5,
    sleep_interval: float = 0.2,
    log: Optional[logging.Logger] = None,
) -> DownloadResult:
    """Download YouTube videos via yt-dlp.

    Parameters
    ----------
    video_ids : list[str]
        YouTube video IDs to download.
    video_dir : str
        Directory to save downloaded videos.
    download_format : str
        yt-dlp format selector string.
    rate_limit : str
        Download rate limit (e.g. ``"5M"``).
    concurrent_fragments : int
        Number of parallel download fragments.
    sleep_interval : float
        Seconds to wait between downloads.
    log : logging.Logger, optional
        Logger instance. Falls back to module-level logger.

    Returns
    -------
    DownloadResult
        Dict with ``downloaded``, ``errors``, and ``missing`` keys.
    """
    return _download_targets(
        [(video_id, f"https://www.youtube.com/watch?v={video_id}") for video_id in video_ids],
        video_dir,
        download_format=download_format,
        rate_limit=rate_limit,
        concurrent_fragments=concurrent_fragments,
        sleep_interval=sleep_interval,
        log=log,
    )
