"""Video media utilities for dataset ingestion."""

import logging
import subprocess
from pathlib import Path

import cv2

from ...utils.video import get_video_duration

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


def materialize_frames_to_video(
    frame_dir: str | Path,
    output_path: str | Path,
    *,
    fps: float,
    overwrite: bool,
    pattern: str | tuple[str, ...] = "*.png",
) -> Path:
    """Encode lexically ordered frame images into a video file."""
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
        "-c:v", "libx264",
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
    pattern: str | tuple[str, ...],
) -> list[Path]:
    """Resolve frame files from one or more glob patterns."""
    patterns = (pattern,) if isinstance(pattern, str) else pattern
    return sorted({
        frame
        for glob_pattern in patterns
        for frame in frame_dir.glob(glob_pattern)
        if frame.is_file()
    })
