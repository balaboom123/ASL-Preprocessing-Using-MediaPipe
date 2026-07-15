"""Video processing utilities for the pipeline."""

import logging
import subprocess

import cv2

logger = logging.getLogger(__name__)


def get_video_duration(video_path: str) -> float:
    """Return video duration in seconds, or 0.0 when unavailable."""
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            cap.release()
            if fps > 0 and frame_count > 0:
                return frame_count / fps
    except Exception:
        pass

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass

    logger.warning("Could not determine duration for %s", video_path)
    return 0.0


def resolve_effective_sample_fps(
    src_fps: float,
    sample_rate: float | None,
) -> float | None:
    """Resolve a user-facing sample rate to an effective FPS.

    Rules:
      - ``None`` => native FPS (no resampling)
      - ``0 < sample_rate < 1`` => keep that ratio of source frames
      - ``sample_rate >= 1`` => downsample to that absolute FPS
    """
    if sample_rate is None:
        return None

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive or null")

    if sample_rate < 1:
        if src_fps <= 0:
            return None
        return src_fps * sample_rate

    if src_fps > 0:
        return min(sample_rate, src_fps)

    return sample_rate
