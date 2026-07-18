"""video2compression processor: video → signer-safe compressed video (.mp4).

The processor keeps the source frame cadence, detects inexpensive scene cuts,
tracks person boxes within each continuous shot, and preserves every persistent
person.  Crop origins may change only at scene boundaries.
"""

import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .base import BaseProcessor
from .detection import apply_bbox_padding, create_detector, union_bbox_tuples
from .detection.base import Detection
from .video.ffmpeg import (
    CropPlan,
    CropWindow,
    iter_ffmpeg_frame_batches,
    probe_frame_count,
    probe_video,
    probe_video_stream_size,
    transcode_with_crop_plan,
)
from ..config.schema import VideoProcessingConfig
from ..registry import register_processor
from ..utils.manifest import resolve_video_path, row_value
from ..utils.video import get_video_duration


BBox = tuple[float, float, float, float]


@dataclass
class _Track:
    """Cheap bbox-only track; no appearance model or landmark inference."""

    last_bbox: BBox
    last_frame: int
    hits: int
    union_bbox: BBox


@dataclass(frozen=True)
class _ShotRegion:
    start_frame: int
    end_frame: int
    bbox: tuple[int, int, int, int]


def _get_output_relpath(row) -> Path:
    """Choose a stable video-level output path from a manifest row."""
    rel_path = row_value(row, "REL_PATH")
    if rel_path:
        return Path(rel_path).with_suffix(".mp4")

    stem = row_value(row, "VIDEO_NAME") or row_value(row, "VIDEO_ID")
    if stem:
        return Path(f"{stem}.mp4")

    raise ValueError(
        "video2compression requires one of REL_PATH, VIDEO_NAME, or VIDEO_ID "
        "to derive a video-level output name."
    )


def _build_video_tasks(df, video_dir: str) -> list[tuple[str, Path]]:
    """Deduplicate segment-level manifests into one task per source video."""
    tasks: list[tuple[str, Path]] = []
    seen_video_paths = set()

    for _, row in df.iterrows():
        video_path = str(resolve_video_path(row, video_dir))
        if video_path in seen_video_paths:
            continue
        seen_video_paths.add(video_path)
        tasks.append((video_path, _get_output_relpath(row)))

    return tasks


def _bbox_iou(first: BBox, second: BBox) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(
        0.0, first[3] - first[1]
    )
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _bbox_center_distance_ratio(first: BBox, second: BBox) -> float:
    first_center = ((first[0] + first[2]) / 2, (first[1] + first[3]) / 2)
    second_center = ((second[0] + second[2]) / 2, (second[1] + second[3]) / 2)
    distance = math.hypot(
        first_center[0] - second_center[0],
        first_center[1] - second_center[1],
    )
    scale = max(
        math.hypot(first[2] - first[0], first[3] - first[1]),
        math.hypot(second[2] - second[0], second[3] - second[1]),
        1.0,
    )
    return distance / scale


def _match_score(
    first: BBox,
    second: BBox,
    config: VideoProcessingConfig,
) -> float | None:
    iou = _bbox_iou(first, second)
    distance_ratio = _bbox_center_distance_ratio(first, second)
    if (
        iou < config.track_iou_threshold
        and distance_ratio > config.track_center_distance_ratio
    ):
        return None
    distance_score = max(
        0.0,
        1.0 - distance_ratio / config.track_center_distance_ratio,
    )
    return iou + distance_score


class _ShotAccumulator:
    """Associate all people within one shot using bbox geometry only."""

    def __init__(
        self,
        start_frame: int,
        frame_width: int,
        frame_height: int,
        config: VideoProcessingConfig,
        source_fps: float | None = None,
    ):
        self.start_frame = start_frame
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.config = config
        self.source_fps = source_fps
        self.frame_count = 0
        self.tracks: list[_Track] = []

    def add(self, frame_index: int, detections: list[Detection]) -> None:
        self.frame_count += 1
        active_tracks = [
            index
            for index, track in enumerate(self.tracks)
            if frame_index - track.last_frame <= self.config.max_track_gap + 1
        ]

        candidates: list[tuple[float, int, int]] = []
        for track_index in active_tracks:
            track = self.tracks[track_index]
            for detection_index, detection in enumerate(detections):
                score = _match_score(
                    track.last_bbox,
                    detection.bbox,
                    self.config,
                )
                if score is not None:
                    candidates.append((score, track_index, detection_index))

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for _, track_index, detection_index in sorted(
            candidates,
            reverse=True,
        ):
            if (
                track_index in matched_tracks
                or detection_index in matched_detections
            ):
                continue
            detection = detections[detection_index]
            track = self.tracks[track_index]
            track.last_bbox = detection.bbox
            track.last_frame = frame_index
            track.hits += 1
            track.union_bbox = union_bbox_tuples(
                [track.union_bbox, detection.bbox]
            )
            matched_tracks.add(track_index)
            matched_detections.add(detection_index)

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            self.tracks.append(
                _Track(
                    last_bbox=detection.bbox,
                    last_frame=frame_index,
                    hits=1,
                    union_bbox=detection.bbox,
                )
            )

    def finish(self, end_frame: int) -> _ShotRegion:
        if end_frame <= self.start_frame:
            raise ValueError("shot must contain at least one frame")

        required_hits = self.config.min_track_hits
        if self.source_fps is not None and self.source_fps > 0:
            required_hits = max(
                required_hits,
                math.ceil(self.source_fps * self.config.min_track_seconds),
            )
        required_hits = min(required_hits, self.frame_count)
        persistent_boxes = [
            track.union_bbox
            for track in self.tracks
            if track.hits >= required_hits
        ]

        if persistent_boxes:
            bbox = union_bbox_tuples(persistent_boxes)
            padded = apply_bbox_padding(
                bbox,
                self.config.padding,
                self.frame_width,
                self.frame_height,
            )
            if padded[2] > padded[0] and padded[3] > padded[1]:
                return _ShotRegion(self.start_frame, end_frame, padded)

        # No reliable track means no reliable signer selection. Preserve all.
        return _ShotRegion(
            self.start_frame,
            end_frame,
            (0, 0, self.frame_width, self.frame_height),
        )


def _frame_signature(frame: np.ndarray) -> np.ndarray:
    """Return a tiny colour thumbnail for cheap scene-cut detection."""
    return cv2.resize(frame, (32, 32), interpolation=cv2.INTER_AREA)


def _scene_change_score(previous: np.ndarray, current: np.ndarray) -> float:
    """Normalized mean absolute thumbnail difference in [0, 1]."""
    difference = cv2.absdiff(previous, current)
    return float(np.mean(difference, dtype=np.float64) / 255.0)


def _even_dimension(value: int, limit: int) -> int:
    value = max(2, min(int(value), limit))
    if value % 2:
        if value < limit:
            value += 1
        else:
            value -= 1
    return max(2, value)


def _crop_origin(
    bbox: tuple[int, int, int, int],
    crop_width: int,
    crop_height: int,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    x = round(center_x - crop_width / 2)
    y = round(center_y - crop_height / 2)
    x = max(0, min(x, frame_width - crop_width))
    y = max(0, min(y, frame_height - crop_height))
    return x, y


def _full_frame_crop_plan(
    regions: list[_ShotRegion],
    frame_width: int,
    frame_height: int,
) -> CropPlan:
    """Return a valid plan that preserves the complete source frame."""
    width = _even_dimension(frame_width, frame_width)
    height = _even_dimension(frame_height, frame_height)
    return CropPlan(
        width=width,
        height=height,
        windows=tuple(
            CropWindow(region.start_frame, region.end_frame, 0, 0)
            for region in regions
        ),
    )


def _build_crop_plan(
    regions: list[_ShotRegion],
    frame_width: int,
    frame_height: int,
    config: VideoProcessingConfig,
) -> CropPlan:
    if not regions:
        raise ValueError("at least one shot region is required")

    crop_width = _even_dimension(
        max(region.bbox[2] - region.bbox[0] for region in regions),
        frame_width,
    )
    crop_height = _even_dimension(
        max(region.bbox[3] - region.bbox[1] for region in regions),
        frame_height,
    )

    crop_area_ratio = (crop_width * crop_height) / (
        frame_width * frame_height
    )
    if crop_area_ratio >= config.max_crop_area_ratio:
        return _full_frame_crop_plan(regions, frame_width, frame_height)

    origins = [
        _crop_origin(
            region.bbox,
            crop_width,
            crop_height,
            frame_width,
            frame_height,
        )
        for region in regions
    ]
    for previous, current in zip(origins, origins[1:]):
        shift_ratio = max(
            abs(current[0] - previous[0]) / max(frame_width, 1),
            abs(current[1] - previous[1]) / max(frame_height, 1),
        )
        if shift_ratio > config.max_crop_shift_ratio:
            return _full_frame_crop_plan(regions, frame_width, frame_height)

    windows: list[CropWindow] = []
    for region, (x, y) in zip(regions, origins):
        if windows and windows[-1].x == x and windows[-1].y == y:
            previous = windows[-1]
            windows[-1] = CropWindow(
                start_frame=previous.start_frame,
                end_frame=region.end_frame,
                x=x,
                y=y,
            )
        else:
            windows.append(
                CropWindow(
                    start_frame=region.start_frame,
                    end_frame=region.end_frame,
                    x=x,
                    y=y,
                )
            )

    return CropPlan(
        width=crop_width,
        height=crop_height,
        windows=tuple(windows),
    )

def _metadata_path(output_path: Path) -> Path:
    return output_path.with_suffix(f"{output_path.suffix}.crop.json")


def _write_crop_metadata(
    output_path: Path,
    source_path: str,
    source_width: int,
    source_height: int,
    source_fps: float,
    crop_plan: CropPlan,
    config: VideoProcessingConfig,
    source_video_bytes: int | None = None,
    output_video_bytes: int | None = None,
    estimated_output_video_bytes: int | None = None,
) -> None:
    output_width, output_height = (
        tuple(config.resize)
        if config.resize
        else (crop_plan.width, crop_plan.height)
    )
    payload = {
        "format": "signdata.compression.v1",
        "source_path": source_path,
        "source_width": source_width,
        "source_height": source_height,
        "source_fps": source_fps,
        "output_width": output_width,
        "output_height": output_height,
        "fps_changed": False,
        "codec": config.codec,
        "crf": config.crf,
        "preset": config.preset,
        "crop_window_seconds": config.crop_window_seconds,
        "max_crop_area_ratio": config.max_crop_area_ratio,
        "max_crop_shift_ratio": config.max_crop_shift_ratio,
        "min_video_reduction_ratio": config.min_video_reduction_ratio,
        "source_video_bytes": source_video_bytes,
        "output_video_bytes": output_video_bytes,
        "estimated_output_video_bytes": estimated_output_video_bytes,
        "windows": [
            {
                "start_frame": window.start_frame,
                "end_frame": window.end_frame,
                "x": window.x,
                "y": window.y,
                "width": crop_plan.width,
                "height": crop_plan.height,
            }
            for window in crop_plan.windows
        ],
    }
    metadata_path = _metadata_path(output_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _temporary_output_path(output_path: Path) -> Path:
    return output_path.with_name(
        f".{output_path.stem}.compression-tmp{output_path.suffix}"
    )


def _temporary_pilot_path(output_path: Path, index: int) -> Path:
    return output_path.with_name(
        f".{output_path.stem}.compression-pilot-{index}{output_path.suffix}"
    )


def _video_stream_bytes(video_path: Path) -> int:
    """Use encoded video bytes, falling back to file bytes for test doubles."""
    stream_size = probe_video_stream_size(str(video_path))
    if stream_size is not None and stream_size > 0:
        return stream_size
    return video_path.stat().st_size


def _pilot_segments(
    frame_count: int,
    source_fps: float,
    segment_seconds: float,
) -> list[tuple[int, int]]:
    """Return beginning, middle, and end pilot intervals."""
    if frame_count <= 0 or source_fps <= 0:
        return []

    duration = frame_count / source_fps
    if duration < 30.0:
        return [(0, frame_count)]

    segment_frames = max(1, round(segment_seconds * source_fps))
    segment_frames = min(segment_frames, frame_count)
    starts = (
        0,
        max(0, (frame_count - segment_frames) // 2),
        max(0, frame_count - segment_frames),
    )
    segments: list[tuple[int, int]] = []
    for start_frame in starts:
        segment = (start_frame, min(frame_count, start_frame + segment_frames))
        if segment[1] > segment[0] and segment not in segments:
            segments.append(segment)
    return segments


def _estimate_pilot_output_bytes(
    source_path: Path,
    output_path: Path,
    crop_plan: CropPlan,
    video_config: VideoProcessingConfig,
    frame_count: int,
    source_fps: float,
) -> int | None:
    """Encode short samples and extrapolate their video-stream bitrate."""
    segments = _pilot_segments(
        frame_count, source_fps, video_config.pilot_segment_seconds
    )
    if not segments:
        return None

    pilot_bytes = 0
    pilot_frames = 0
    try:
        for index, (start_frame, end_frame) in enumerate(segments):
            pilot_path = _temporary_pilot_path(output_path, index)
            pilot_path.unlink(missing_ok=True)
            try:
                if start_frame == 0 and end_frame == frame_count:
                    ok = transcode_with_crop_plan(
                        str(source_path),
                        crop_plan,
                        video_config,
                        str(pilot_path),
                    )
                else:
                    ok = transcode_with_crop_plan(
                        str(source_path),
                        crop_plan,
                        video_config,
                        str(pilot_path),
                        start_frame=start_frame,
                        end_frame=end_frame,
                        source_fps=source_fps,
                    )
                if not ok or not pilot_path.exists():
                    return None
                pilot_bytes += _video_stream_bytes(pilot_path)
                pilot_frames += end_frame - start_frame
            finally:
                pilot_path.unlink(missing_ok=True)
    except (OSError, ValueError):
        return None

    if pilot_frames <= 0:
        return None
    return round(pilot_bytes * frame_count / pilot_frames)


@register_processor("video2compression")
class Video2CompressionProcessor(BaseProcessor):
    """Video → native-FPS, shot-aware signer-safe compressed MP4."""

    name = "video2compression"

    def run(self, context):
        cfg = self.config.processing
        video_config = cfg.video_config
        df = context.manifest_df
        if df is None:
            self.logger.warning("No manifest loaded, nothing to process.")
            context.stats["processing"] = {"total": 0}
            return context

        video_dir = str(context.videos_dir) if context.videos_dir else ""
        source_rows = len(df)
        tasks = _build_video_tasks(df, video_dir)
        total = len(tasks)

        if source_rows != total:
            self.logger.info(
                "video2compression deduplicated %d manifest rows to %d "
                "unique videos",
                source_rows,
                total,
            )

        if total == 0:
            context.stats["processing"] = {
                "total": 0,
                "source_rows": source_rows,
                "processed": 0,
                "skipped": 0,
                "not_smaller": 0,
                "errors": 0,
            }
            return context

        output_dir = context.output_dir / "compressed"
        output_dir.mkdir(parents=True, exist_ok=True)
        detector = create_detector(cfg.detection, cfg.detection_config)
        processed = skipped = not_smaller = errors = 0

        try:
            for video_path, output_relpath in tasks:
                output_path = output_dir / output_relpath
                output_label = str(output_relpath)

                if not context.force_all and output_path.exists():
                    skipped += 1
                    continue

                temporary_path = _temporary_output_path(output_path)
                temporary_path.unlink(missing_ok=True)

                try:
                    source_path = Path(video_path)
                    if not source_path.exists():
                        self.logger.warning("Video not found: %s", video_path)
                        errors += 1
                        continue

                    duration = get_video_duration(video_path)
                    metadata = probe_video(video_path)
                    if duration <= 0 or metadata is None:
                        self.logger.warning(
                            "Cannot read video metadata: %s",
                            video_path,
                        )
                        errors += 1
                        continue
                    frame_width, frame_height, source_fps = metadata
                    source_video_bytes = _video_stream_bytes(source_path)
                    if frame_width <= 1 or frame_height <= 1:
                        self.logger.warning(
                            "Invalid frame size: %s",
                            video_path,
                        )
                        errors += 1
                        continue

                    batch_size = getattr(
                        cfg.detection_config,
                        "batch_size",
                        16,
                    )
                    frame_index = 0
                    previous_signature: np.ndarray | None = None
                    regions: list[_ShotRegion] = []
                    window_frame_span = max(
                        1,
                        round(
                            source_fps * video_config.crop_window_seconds
                        ),
                    )
                    next_window_frame = window_frame_span
                    shot = _ShotAccumulator(
                        start_frame=0,
                        frame_width=frame_width,
                        frame_height=frame_height,
                        config=video_config,
                        source_fps=source_fps,
                    )

                    # Detection uses every source frame; output cadence is
                    # untouched. Crop state resets at scene changes and every
                    # fixed crop window.
                    for frames in iter_ffmpeg_frame_batches(
                        video_path,
                        0.0,
                        duration,
                        None,
                        batch_size=batch_size,
                    ):
                        detections = detector.detect_batch(frames)
                        if len(detections) != len(frames):
                            raise RuntimeError(
                                "detector returned a different number of "
                                "frame results"
                            )

                        for frame, frame_detections in zip(frames, detections):
                            signature = _frame_signature(frame)
                            is_scene_cut = (
                                previous_signature is not None
                                and _scene_change_score(
                                    previous_signature,
                                    signature,
                                )
                                >= video_config.scene_cut_threshold
                            )
                            if (
                                is_scene_cut
                                or frame_index >= next_window_frame
                            ):
                                regions.append(shot.finish(frame_index))
                                shot = _ShotAccumulator(
                                    start_frame=frame_index,
                                    frame_width=frame_width,
                                    frame_height=frame_height,
                                    config=video_config,
                                    source_fps=source_fps,
                                )
                                next_window_frame = (
                                    frame_index + window_frame_span
                                )

                            shot.add(frame_index, frame_detections)
                            previous_signature = signature
                            frame_index += 1

                        del detections
                        del frames

                    if frame_index == 0:
                        errors += 1
                        continue

                    regions.append(shot.finish(frame_index))
                    crop_plan = _build_crop_plan(
                        regions,
                        frame_width,
                        frame_height,
                        video_config,
                    )
                    estimated_output_video_bytes = (
                        _estimate_pilot_output_bytes(
                            source_path,
                            output_path,
                            crop_plan,
                            video_config,
                            frame_index,
                            source_fps,
                        )
                    )
                    if estimated_output_video_bytes is None:
                        self.logger.warning(
                            "Pilot encoding failed for %s", output_label
                        )
                        errors += 1
                        continue

                    maximum_output_bytes = source_video_bytes * (
                        1.0 - video_config.min_video_reduction_ratio
                    )
                    if estimated_output_video_bytes > maximum_output_bytes:
                        skipped += 1
                        not_smaller += 1
                        self.logger.info(
                            "Keeping source after pilot: %s source_video=%d "
                            "estimated_video=%d",
                            output_label,
                            source_video_bytes,
                            estimated_output_video_bytes,
                        )
                        continue

                    ok = transcode_with_crop_plan(
                        video_path,
                        crop_plan,
                        video_config,
                        str(temporary_path),
                    )
                    if not ok or not temporary_path.exists():
                        errors += 1
                        continue

                    encoded_metadata = probe_video(str(temporary_path))
                    if encoded_metadata is None:
                        self.logger.warning(
                            "Encoded output is not readable: %s",
                            output_label,
                        )
                        errors += 1
                        continue
                    output_width, output_height, output_fps = encoded_metadata
                    expected_dimensions = (
                        tuple(video_config.resize)
                        if video_config.resize
                        else (crop_plan.width, crop_plan.height)
                    )
                    if (output_width, output_height) != expected_dimensions:
                        self.logger.warning(
                            "Encoded dimensions changed unexpectedly for %s: "
                            "expected=%s output=%s",
                            output_label,
                            expected_dimensions,
                            (output_width, output_height),
                        )
                        errors += 1
                        continue

                    output_frame_count = probe_frame_count(
                        str(temporary_path)
                    )
                    if output_frame_count != frame_index:
                        self.logger.warning(
                            "Encoded frame count changed for %s: "
                            "source=%d output=%s",
                            output_label,
                            frame_index,
                            output_frame_count,
                        )
                        errors += 1
                        continue

                    fps_tolerance = max(0.05, source_fps * 0.002)
                    if (
                        source_fps > 0
                        and abs(output_fps - source_fps) > fps_tolerance
                    ):
                        self.logger.warning(
                            "Encoded FPS changed for %s: source=%.3f "
                            "output=%.3f",
                            output_label,
                            source_fps,
                            output_fps,
                        )
                        errors += 1
                        continue

                    output_video_bytes = _video_stream_bytes(
                        temporary_path
                    )
                    if output_video_bytes > maximum_output_bytes:
                        temporary_path.unlink(missing_ok=True)
                        skipped += 1
                        not_smaller += 1
                        self.logger.info(
                            "Keeping source because compressed video does not "
                            "meet the reduction gate: %s source_video=%d "
                            "output_video=%d minimum_reduction=%.3f",
                            output_label,
                            source_video_bytes,
                            output_video_bytes,
                            video_config.min_video_reduction_ratio,
                        )
                        continue

                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary_path.replace(output_path)
                    _write_crop_metadata(
                        output_path,
                        video_path,
                        frame_width,
                        frame_height,
                        source_fps,
                        crop_plan,
                        video_config,
                        source_video_bytes=source_video_bytes,
                        output_video_bytes=output_video_bytes,
                        estimated_output_video_bytes=estimated_output_video_bytes,
                    )
                    processed += 1

                except Exception as exc:
                    self.logger.error(
                        "Error processing %s: %s",
                        output_label,
                        exc,
                    )
                    errors += 1
                finally:
                    temporary_path.unlink(missing_ok=True)

        finally:
            detector.close()
            gc.collect()

        context.stats["processing"] = {
            "total": total,
            "source_rows": source_rows,
            "processed": processed,
            "skipped": skipped,
            "not_smaller": not_smaller,
            "errors": errors,
        }
        self.logger.info(
            "video2compression: processed=%d skipped=%d not_smaller=%d "
            "errors=%d total=%d source_rows=%d",
            processed,
            skipped,
            not_smaller,
            errors,
            total,
            source_rows,
        )
        return context
