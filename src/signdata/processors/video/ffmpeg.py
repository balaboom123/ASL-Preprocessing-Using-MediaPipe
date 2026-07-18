"""FFmpeg-based video processing utilities."""

import logging
import shutil
import subprocess
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ...config.schema import VideoProcessingConfig
from ...utils.video import resolve_effective_sample_fps

logger = logging.getLogger(__name__)
_opencv_fallback_reported = False


@dataclass(frozen=True)
class CropWindow:
    """A fixed crop origin active over a half-open frame interval."""

    start_frame: int
    end_frame: int
    x: int
    y: int


@dataclass(frozen=True)
class CropPlan:
    """Constant-size crop whose origin may change only between shots."""

    width: int
    height: int
    windows: tuple[CropWindow, ...]


def probe_video(video_path: str) -> tuple[int, int, float] | None:
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return None
        return (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        )
    finally:
        cap.release()


def probe_frame_count(video_path: str) -> int | None:
    """Read the container frame count without decoding the video."""
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return None
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return frame_count if frame_count > 0 else None
    finally:
        cap.release()


def probe_video_stream_size(video_path: str) -> int | None:
    """Return encoded bytes belonging to the first video stream."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None

    cmd = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "packet=size",
        "-of", "default=nw=1:nk=1",
        video_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    total = 0
    found = False
    for raw_line in result.stdout.splitlines():
        try:
            size = int(raw_line.strip())
        except (TypeError, ValueError):
            continue
        if size >= 0:
            total += size
            found = True
    return total if found else None


def ffmpeg_pipe_frames(
    video_path: str,
    start_sec: float,
    end_sec: float,
    sample_rate: float | None,
) -> list[np.ndarray]:
    """Decode frames from a video segment via ffmpeg pipe.

    Pass 1 of the two-pass video2crop pipeline. Returns BGR frames
    that can be fed to the detection backend.

    Args:
        video_path: Path to the video file.
        start_sec: Segment start time.
        end_sec: Segment end time.
        sample_rate: Shared sampling rate for both passes.

    Returns:
        List of BGR frames as numpy arrays.
    """
    if shutil.which("ffmpeg") is None:
        return _opencv_sample_frames(
            video_path,
            start_sec,
            end_sec,
            sample_rate,
        )

    try:
        frames: list[np.ndarray] = []
        for batch in iter_ffmpeg_frame_batches(
            video_path, start_sec, end_sec, sample_rate, batch_size=64,
        ):
            frames.extend(batch)
        return frames
    except FileNotFoundError:
        return _opencv_sample_frames(
            video_path,
            start_sec,
            end_sec,
            sample_rate,
        )
    except Exception as exc:
        logger.error("ffmpeg pipe error: %s", exc)
        return []


def _opencv_sample_frames(
    video_path: str,
    start_sec: float,
    end_sec: float,
    sample_rate: float | None,
) -> list[np.ndarray]:
    """Decode sampled frames with OpenCV when ffmpeg is unavailable."""
    global _opencv_fallback_reported
    if not _opencv_fallback_reported:
        logger.warning(
            "ffmpeg executable not found; using OpenCV for frame decoding"
        )
        _opencv_fallback_reported = True

    if end_sec <= start_sec:
        return []

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return []

        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if source_fps <= 0:
            return []

        effective_fps = resolve_effective_sample_fps(source_fps, sample_rate)
        target_fps = source_fps if effective_fps is None else effective_fps
        sample_ratio = target_fps / source_fps
        accumulator = 0.0

        start_frame = max(0, int(start_sec * source_fps))
        end_frame = max(start_frame, int(end_sec * source_fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames: list[np.ndarray] = []
        current_frame = start_frame
        while current_frame <= end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            accumulator += sample_ratio
            if accumulator >= 1.0:
                accumulator -= 1.0
                frames.append(frame)
            current_frame += 1
        return frames
    finally:
        cap.release()


def iter_ffmpeg_frame_batches(
    video_path: str,
    start_sec: float,
    end_sec: float,
    sample_rate: float | None,
    batch_size: int,
) -> Iterator[list[np.ndarray]]:
    """Stream decoded frames from ffmpeg in bounded-size batches.

    Unlike :func:`ffmpeg_pipe_frames`, this function does not buffer the full
    rawvideo stream or the full decoded frame list in memory.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    metadata = probe_video(video_path)
    if metadata is None:
        return
    width, height, source_fps = metadata

    if width <= 0 or height <= 0:
        return

    duration = end_sec - start_sec
    effective_fps = resolve_effective_sample_fps(source_fps, sample_rate)

    # -hide_banner + -nostats prevent ffmpeg from filling the stderr PIPE
    # buffer and deadlocking the streaming read loop below.
    cmd = [
        "ffmpeg", "-y",
        "-hide_banner", "-nostats",
        "-ss", str(start_sec),
        "-i", video_path,
        "-t", str(duration),
    ]

    if effective_fps is not None:
        cmd.extend(["-vf", f"fps={effective_fps}"])

    cmd.extend([
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-v", "error",
        "pipe:1",
    ])

    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.stdout is None:
            raise RuntimeError("ffmpeg stdout pipe was not created")

        frame_size = width * height * 3
        batch: list[np.ndarray] = []

        while True:
            raw = proc.stdout.read(frame_size)
            if not raw:
                break
            if len(raw) != frame_size:
                logger.warning(
                    "ffmpeg produced a partial frame for %s; dropping "
                    "trailing bytes",
                    video_path,
                )
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                height, width, 3,
            ).copy()
            batch.append(frame)

            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

        stderr = b""
        if proc.stderr is not None:
            stderr = proc.stderr.read()
        returncode = proc.wait()
        if returncode != 0:
            message = stderr.decode(errors="replace")[:200]
            raise RuntimeError(f"ffmpeg error: {message}")

    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()


def _piecewise_crop_expression(
    windows: Sequence[CropWindow],
    coordinate: str,
    frame_offset: int = 0,
) -> str:
    """Build an ffmpeg expression selecting x/y by decoded frame number."""
    if coordinate not in {"x", "y"}:
        raise ValueError("coordinate must be 'x' or 'y'")
    if not windows:
        raise ValueError("crop plan must contain at least one window")

    previous_value = getattr(windows[0], coordinate)
    terms = [str(previous_value)]
    frame_number = "n" if frame_offset == 0 else f"(n+{frame_offset})"
    for window in windows[1:]:
        value = getattr(window, coordinate)
        delta = value - previous_value
        if delta:
            # Commas belong to expression functions, not the filter chain.
            terms.append(
                f"({delta})*gte({frame_number}\\,{window.start_frame})"
            )
        previous_value = value
    return "+".join(terms)


def _encoder_args(video_config: VideoProcessingConfig) -> list[str]:
    return [
        "-c:v", video_config.codec,
        "-preset", video_config.preset,
        "-crf", str(video_config.crf),
        "-pix_fmt", "yuv420p",
        "-an",
        "-movflags", "+faststart",
    ]


def transcode_with_crop_plan(
    video_path: str,
    crop_plan: CropPlan,
    video_config: VideoProcessingConfig,
    output_path: str,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
    source_fps: float | None = None,
) -> bool:
    """Transcode at native timing with a constant-size, shot-aware crop."""
    if crop_plan.width <= 0 or crop_plan.height <= 0:
        raise ValueError("crop plan dimensions must be positive")
    if not crop_plan.windows:
        raise ValueError("crop plan must contain at least one window")

    expected_start = 0
    for window in crop_plan.windows:
        if window.start_frame != expected_start:
            raise ValueError("crop plan windows must be contiguous")
        if window.end_frame <= window.start_frame:
            raise ValueError("crop plan windows must be non-empty")
        expected_start = window.end_frame

    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if end_frame is not None and end_frame <= start_frame:
        raise ValueError("end_frame must be greater than start_frame")

    if end_frame is not None:
        if source_fps is None or source_fps <= 0:
            metadata = probe_video(video_path)
            source_fps = metadata[2] if metadata is not None else 0.0
        if source_fps <= 0:
            raise ValueError("source_fps is required for a partial transcode")

    x_expression = _piecewise_crop_expression(
        crop_plan.windows, "x", frame_offset=start_frame
    )
    y_expression = _piecewise_crop_expression(
        crop_plan.windows, "y", frame_offset=start_frame
    )
    filters = [
        (
            f"crop={crop_plan.width}:{crop_plan.height}:"
            f"{x_expression}:{y_expression}:exact=1"
        )
    ]
    if video_config.resize:
        filters.append(
            f"scale={video_config.resize[0]}:{video_config.resize[1]}"
        )

    cmd = ["ffmpeg", "-y", "-hide_banner", "-nostats"]
    if end_frame is not None:
        cmd.extend(["-ss", str(start_frame / source_fps)])
    cmd.extend(["-i", video_path])
    if end_frame is not None:
        cmd.extend(["-t", str((end_frame - start_frame) / source_fps)])
    cmd.extend([
        "-vf", ",".join(filters),
        *_encoder_args(video_config),
        "-fps_mode", "passthrough",
        "-v", "error",
        output_path,
    ])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=video_config.encoding_timeout_seconds,
        )
        if proc.returncode != 0:
            logger.error(
                "ffmpeg compression error: %s",
                proc.stderr.decode()[:200],
            )
            return False
        return True
    except Exception as exc:
        logger.error("ffmpeg compression error: %s", exc)
        return False


def clip_and_crop(
    video_path: str,
    start_sec: float,
    end_sec: float,
    bbox: tuple[float, float, float, float],
    sample_rate: float | None,
    video_config: VideoProcessingConfig,
    output_path: str,
) -> bool:
    """Pass 2: clip + crop a video segment using ffmpeg.

    Uses the same timing parameters as pass 1 (ffmpeg_pipe_frames)
    plus a crop filter derived from the detection bbox.

    Args:
        video_path: Source video path.
        start_sec: Segment start time.
        end_sec: Segment end time.
        bbox: (x1, y1, x2, y2) crop region in pixels.
        sample_rate: Same sampling rate used in pass 1.
        video_config: VideoProcessingConfig with codec, padding, resize.
        output_path: Output file path.

    Returns:
        True if successful.
    """
    from ..detection.validation import apply_bbox_padding

    duration = end_sec - start_sec

    metadata = probe_video(video_path)
    if metadata is None:
        return False
    frame_w, frame_h, source_fps = metadata

    # Apply padding
    x1, y1, x2, y2 = apply_bbox_padding(
        bbox, video_config.padding, frame_w, frame_h
    )
    crop_w = x2 - x1
    crop_h = y2 - y1

    if crop_w <= 0 or crop_h <= 0:
        return False

    effective_fps = resolve_effective_sample_fps(source_fps, sample_rate)

    # Build ffmpeg command
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-i", video_path,
        "-t", str(duration),
    ]

    vf_filters = []
    if effective_fps is not None:
        vf_filters.append(f"fps={effective_fps}")
    vf_filters.append(f"crop={crop_w}:{crop_h}:{x1}:{y1}")

    if video_config.resize:
        vf_filters.append(
            f"scale={video_config.resize[0]}:{video_config.resize[1]}"
        )

    cmd.extend(["-vf", ",".join(vf_filters)])

    cmd.extend(
        [
            "-c:v", video_config.codec,
            "-preset", "medium",
            "-crf", "15",
            "-an",
        ]
    )
    cmd.extend(["-v", "error"])
    cmd.append(output_path)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=120)
        if proc.returncode != 0:
            logger.error("ffmpeg crop error: %s", proc.stderr.decode()[:200])
            return False
        return True
    except Exception as exc:
        logger.error("ffmpeg crop error: %s", exc)
        return False
