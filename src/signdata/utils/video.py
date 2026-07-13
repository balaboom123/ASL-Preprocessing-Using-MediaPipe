"""Video processing utilities for the pipeline.

Duration/FPS probing for dataset ingestion has moved to
``signdata.datasets._ingestion.media``.
"""

import logging
import subprocess
from typing import List, Optional

import cv2
import numpy as np

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
    sample_rate: Optional[float],
) -> Optional[float]:
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

    if 0 < sample_rate < 1:
        if src_fps <= 0:
            return None
        return src_fps * sample_rate

    if src_fps > 0:
        return min(sample_rate, src_fps)

    return sample_rate


class FPSSampler:
    """Frame sampler using native, ratio, or absolute-FPS semantics."""

    def __init__(self, src_fps: float, sample_rate: Optional[float]):
        effective_fps = resolve_effective_sample_fps(src_fps, sample_rate)
        target_fps = src_fps if effective_fps is None else effective_fps
        self.r = 1.0 if src_fps <= 0 else target_fps / src_fps
        self.acc = 0.0

    def take(self) -> bool:
        """Returns True if current frame should be sampled."""
        self.acc += self.r
        if self.acc >= 1.0:
            self.acc -= 1.0
            return True
        return False

    def reset(self) -> None:
        self.acc = 0.0


def read_sampled_frames(
    video_path: str,
    start_sec: float,
    end_sec: float,
    sampler: FPSSampler,
    source_fps: Optional[float] = None,
) -> List[np.ndarray]:
    """Read sampled frames from a video segment."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = source_fps or cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        cap.release()
        return []

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    sampler.reset()
    frames = []
    current = start_frame
    while current <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if sampler.take():
            frames.append(frame)
        current += 1

    cap.release()
    return frames
