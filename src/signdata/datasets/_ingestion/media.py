"""Video media utilities for dataset ingestion.

Provides video duration/FPS probing and frame-sequence materialisation.
These are used exclusively during dataset download and manifest-building stages.
Pipeline-level video utilities (FPSSampler, resolve_effective_sample_fps) remain
in ``signdata.utils.video``.
"""

import logging
import subprocess
from pathlib import Path
from typing import Sequence, Union

import cv2

logger = logging.getLogger(__name__)


def get_video_fps(video_path: str) -> float:
    """Return video FPS (frames per second) as float."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        cap.release()
        return float(fps)
    except Exception:
        return 0.0


def get_video_duration(video_path: str) -> float:
    """Return video duration in seconds.

    Uses OpenCV frame-count/FPS first.  Falls back to ``ffprobe`` when
    OpenCV returns zero or invalid metadata (common with certain codecs).

    Returns 0.0 if duration cannot be determined.
    """
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


def materialize_frames_to_video(
    frame_dir: Union[str, Path],
    output_path: Union[str, Path],
    *,
    fps: float = 25.0,
    codec: str = "libx264",
    pattern: Union[str, Sequence[str]] = "*.png",
    overwrite: bool = False,
) -> Path:
    """Encode a directory of ordered frame images into a video file.

    Frames are sorted lexicographically to ensure deterministic ordering
    regardless of filesystem glob order.

    Parameters
    ----------
    frame_dir : str or Path
        Directory containing the frame images.
    output_path : str or Path
        Destination video file path.
    fps : float
        Frame rate for the output video.
    codec : str
        Video codec (default ``libx264``).
    pattern : str or sequence[str]
        Glob pattern or patterns for frame files (default ``*.png``).
    overwrite : bool
        If *False* (default), skip materialisation when *output_path* exists.

    Returns
    -------
    Path
        The output video path.

    Raises
    ------
    FileNotFoundError
        If *frame_dir* does not exist or contains no matching frames.
    RuntimeError
        If ffmpeg exits with a non-zero return code.
    """
    frame_dir = Path(frame_dir)
    output_path = Path(output_path)

    if not frame_dir.is_dir():
        raise FileNotFoundError(f"Frame directory not found: {frame_dir}")

    if output_path.exists() and not overwrite:
        logger.debug("Video already exists, skipping: %s", output_path)
        return output_path

    frames = _resolve_frame_paths(frame_dir, pattern)
    if not frames:
        raise FileNotFoundError(
            f"No frames matching {pattern!r} in {frame_dir}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    concat_content = "\n".join(
        f"file '{frame.resolve()}'\nduration {1.0 / fps:.6f}"
        for frame in frames
    )

    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-f", "concat",
        "-safe", "0",
        "-protocol_whitelist", "file,pipe",
        "-i", "pipe:0",
        "-r", str(fps),
        "-c:v", codec,
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]

    result = subprocess.run(
        cmd,
        input=concat_content,
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}) for {frame_dir}:\n"
            f"{result.stderr[-500:]}"
        )

    logger.debug(
        "Materialized %d frames -> %s (%.1f fps)",
        len(frames), output_path, fps,
    )
    return output_path


def _resolve_frame_paths(
    frame_dir: Path,
    pattern: Union[str, Sequence[str]],
) -> list[Path]:
    """Resolve frame files from one or more glob patterns."""
    patterns = [pattern] if isinstance(pattern, str) else list(pattern)
    matched: dict[Path, None] = {}

    for glob_pattern in patterns:
        for frame in sorted(frame_dir.glob(glob_pattern)):
            if frame.is_file():
                matched.setdefault(frame, None)

    return sorted(matched)
