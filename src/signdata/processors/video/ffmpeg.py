"""FFmpeg-based video processing utilities."""

import json
import logging
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ...config.schema import NUMERIC_PRESET_SPECS, VideoProcessingConfig
from ...utils.video import resolve_effective_sample_fps

logger = logging.getLogger(__name__)
_opencv_fallback_reported = False

# ffmpeg prints this when it parsed an encoder option that no stream consumed —
# e.g. `-crf` on NVENC, which has no such private option and quietly encodes at
# its default rate instead. Silent quality loss, so we treat it as fatal.
_UNUSED_OPTION_MARKER = "has not been used for any stream"


class EncoderOptionError(RuntimeError):
    """ffmpeg parsed an encoder option and then silently discarded it.

    Always a configuration bug rather than a bad input file, so callers should
    let this abort the run instead of counting it as a per-video error.
    """


@dataclass(frozen=True)
class VideoInfo:
    """Everything the compression path needs, from one ffprobe call."""

    codec: str
    width: int
    height: int
    duration: float
    bitrate_bps: int
    pix_fmt: str = ""


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


def probe_media(video_path: str) -> VideoInfo | None:
    """Read codec, geometry, duration and bitrate in a single ffprobe call.

    Returns None when ffprobe is unavailable or the file is unreadable, so
    callers must treat "unknown" as a reason to skip rather than to guess.
    """
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None

    cmd = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,width,height,pix_fmt,bit_rate:format=duration,bit_rate",
        "-of", "json",
        video_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    try:
        payload = json.loads(result.stdout)
        stream = (payload.get("streams") or [{}])[0]
    except (ValueError, IndexError):
        return None
    container = payload.get("format") or {}

    duration = float(container.get("duration") or 0.0)
    # VP9/AV1 in WebM routinely omit the per-stream bit_rate, so fall back to
    # the container rate and finally to file bytes over duration.
    raw_bitrate = stream.get("bit_rate") or container.get("bit_rate")
    if raw_bitrate:
        bitrate_bps = int(raw_bitrate)
    elif duration > 0:
        bitrate_bps = int(Path(video_path).stat().st_size * 8 / duration)
    else:
        bitrate_bps = 0

    return VideoInfo(
        codec=str(stream.get("codec_name") or ""),
        width=int(stream.get("width") or 0),
        height=int(stream.get("height") or 0),
        duration=duration,
        bitrate_bps=bitrate_bps,
        pix_fmt=str(stream.get("pix_fmt") or ""),
    )


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
        # Start already "due" and test before accumulating, so the window's
        # first frame is always kept. FFmpeg's `fps=` filter emits frame 0, and
        # a fallback that dropped it would shift every landmark sequence by one
        # source frame depending on whether ffmpeg happened to be installed.
        accumulator = 1.0

        start_frame = max(0, int(start_sec * source_fps))
        end_frame = max(start_frame, int(end_sec * source_fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames: list[np.ndarray] = []
        current_frame = start_frame
        # Half-open window, matching `-ss start -t (end - start)`.
        while current_frame < end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            if accumulator >= 1.0:
                accumulator -= 1.0
                frames.append(frame)
            accumulator += sample_ratio
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


# ffprobe pix_fmt suffixes that mark >8-bit samples. nv12/nv16 (8-bit)
# deliberately do not match: their digits are a chroma-layout tag, not a depth.
_HIGH_DEPTH_SUFFIXES = ("10le", "10be", "12le", "12be", "16le", "16be")


def _output_pix_fmt(source_pix_fmt: str, codec: str) -> str:
    """Preserve the source bit depth instead of crushing 10-bit to 8-bit.

    A hardcoded yuv420p silently truncates 10-bit sources — the av01/vp9
    renditions the download step now prefers are often 10-bit — and VMAF is
    luma-only, so the calibration sweep never sees the loss. Chroma is still
    normalised to 4:2:0 (near-universal for delivery and required by every
    NVENC target); only bit depth carries signal the pose/crop stages can use.

    ponytail: h264 is 8-bit only, and >10-bit output caps at p010le/yuv420p10le
    because that is the deepest NVENC (and a stock libx265 build) will take.
    """
    if not source_pix_fmt.endswith(_HIGH_DEPTH_SUFFIXES):
        return "yuv420p"
    if codec in ("h264_nvenc", "libx264"):
        return "yuv420p"
    return "p010le" if codec.endswith("_nvenc") else "yuv420p10le"


def _encoder_args(
    video_config: VideoProcessingConfig,
    max_bitrate_bps: int | None,
    source_pix_fmt: str = "",
) -> list[str]:
    """Constant-quality encoder args for the configured codec family.

    NVENC has no `crf` private option. Passing one makes ffmpeg warn and then
    encode at its own default rate, which is how a "compression" pass ends up
    producing larger files than it consumed. Constant quality on NVENC is
    `-cq`, and it only takes effect together with `-rc vbr -b:v 0`.

    The speed knob is likewise not spelled the same everywhere: libaom-av1 and
    libvpx-vp9 have no `-preset`, only `-cpu-used`.
    """
    numeric_spec = NUMERIC_PRESET_SPECS.get(video_config.codec)
    speed_flag = numeric_spec[0] if numeric_spec else "-preset"
    args = ["-c:v", video_config.codec, speed_flag, video_config.preset]
    if video_config.codec.endswith("_nvenc") and video_config.nvenc_gpu is not None:
        args += ["-gpu", str(video_config.nvenc_gpu)]

    if video_config.codec.endswith("_nvenc"):
        args += [
            "-rc", "vbr",
            "-cq", str(video_config.crf),
            "-b:v", "0",
            # Adaptive quantisation spends bits on the hands and face instead
            # of flat background, which is exactly where the signal lives.
            "-spatial-aq", "1",
            "-aq-strength", str(video_config.aq_strength),
            # ponytail: temporal AQ needs Turing or newer. On older cards the
            # unused-option guard below fires; drop this line if so.
            "-temporal-aq", "1",
            "-rc-lookahead", "32",
        ]
    else:
        args += ["-crf", str(video_config.crf)]
        if speed_flag == "-cpu-used":
            # libvpx-vp9 and libaom-av1 refuse to open the encoder when
            # -maxrate/-bufsize arrive without an explicit -b:v ("Rate control
            # parameters set without a bitrate"), and -b:v 0 does not satisfy
            # them. Pointing the target at the ceiling turns the VBV cap into
            # constrained quality, which is what max_bitrate_ratio asks for;
            # 0 keeps pure constant quality when there is no cap.
            args += ["-b:v", str(max_bitrate_bps or 0)]

    if max_bitrate_bps:
        # A VBV ceiling, not a quality target: loose enough that -cq/-crf still
        # drives the encode, tight enough to clamp a pathological blow-up.
        args += [
            "-maxrate", str(max_bitrate_bps),
            "-bufsize", str(max_bitrate_bps * 4),
        ]

    # Keep the source bit depth (10-bit stays 10-bit) and drop audio: no
    # downstream stage reads it, and copying the usual YouTube Opus track into
    # mp4 fails the mux outright, which would error otherwise-fine sources.
    args += [
        "-pix_fmt", _output_pix_fmt(source_pix_fmt, video_config.codec),
        "-an",
        "-movflags", "+faststart",
    ]
    return args


def _run_ffmpeg(cmd: list[str], timeout: int) -> bool:
    """Run ffmpeg, raising when it silently discarded an encoder option."""
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except Exception as exc:
        logger.error("ffmpeg error: %s", exc)
        return False

    stderr = proc.stderr.decode(errors="replace")
    if _UNUSED_OPTION_MARKER in stderr:
        ignored = "\n".join(
            line for line in stderr.splitlines()
            if _UNUSED_OPTION_MARKER in line
        )
        raise EncoderOptionError(
            "ffmpeg ignored an encoder option, so this encode did not use the "
            f"configured quality settings:\n{ignored}"
        )
    if proc.returncode != 0:
        logger.error("ffmpeg failed: %s", stderr[-2000:])
        return False
    return True


def transcode(
    video_path: str,
    video_config: VideoProcessingConfig,
    output_path: str,
    *,
    max_bitrate_bps: int | None = None,
    source_pix_fmt: str = "",
) -> bool:
    """Re-encode a whole video, changing only the codec and quality level.

    Deliberately no crop, no scale and no frame-rate change: pose estimation,
    video2crop and video2parts all re-derive their own regions from the stored
    file, so the geometry and cadence they see must stay byte-for-byte the
    same shape as the source.
    """
    hwaccel_args = ["-hwaccel", "auto"]
    if video_config.codec.endswith("_nvenc") and video_config.nvenc_gpu is not None:
        hwaccel_args += ["-hwaccel_device", str(video_config.nvenc_gpu)]

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-nostats",
        # Falls back to software decode when the GPU cannot handle the source
        # codec, so an AV1 source on an older card still works.
        *hwaccel_args,
        "-i", video_path,
        *_encoder_args(video_config, max_bitrate_bps, source_pix_fmt),
        "-fps_mode", "passthrough",
        # Not `-v error`: the unused-option warning we check for is a warning.
        "-v", "warning",
        output_path,
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    return _run_ffmpeg(cmd, video_config.encoding_timeout_seconds)


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

    # Crops keep their own near-lossless quality target, but the preset and
    # rate-control flags still have to match the codec family: NVENC rejects
    # software preset names and has no `crf` option, so a hardcoded
    # `-preset medium -crf 15` silently encodes at NVENC's default rate.
    cmd.extend(["-c:v", video_config.codec, "-preset", video_config.preset])
    if video_config.codec.endswith("_nvenc"):
        cmd.extend(["-rc", "vbr", "-cq", "15", "-b:v", "0"])
    else:
        cmd.extend(["-crf", "15"])
    cmd.append("-an")
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
