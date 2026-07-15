"""video2compression processor: video → cropped & compressed video (.mp4).

Compresses videos by detecting all persons across the entire video and
cropping to the union bounding box that covers every detection in every
frame.  Unlike video2crop, this processor:

- Processes the whole video (no START/END time segmentation)
- Never skips multi-person videos
- Unions ALL detections (not just the largest per frame)
"""

import gc
from pathlib import Path

from .base import BaseProcessor
from .detection import create_detector, union_bbox_tuples
from .detection.base import Detection
from .video.ffmpeg import clip_and_crop, iter_ffmpeg_frame_batches
from ..registry import register_processor
from ..utils.manifest import resolve_video_path, row_value
from ..utils.video import get_video_duration


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


def _update_union_bbox(
    current: tuple[float, float, float, float] | None,
    detections: list[list[Detection]],
) -> tuple[float, float, float, float] | None:
    """Fold detection bboxes into an existing union bbox."""
    bboxes = [det.bbox for frame_dets in detections for det in frame_dets]
    if current is not None:
        bboxes.append(current)
    return union_bbox_tuples(bboxes) if bboxes else None


@register_processor("video2compression")
class Video2CompressionProcessor(BaseProcessor):
    """High-level processor: video → cropped & compressed video (.mp4).

    Detects all persons across the entire video and crops to the union
    bounding box that includes every person in every frame.

    Orchestrates:
    - video/ffmpeg_pipe for frame decoding (pass 1)
    - detection/ backends for person detection
    - video/clip_and_crop for final output (pass 2, same ffmpeg params + crop)
    """

    name = "video2compression"

    def run(self, context):
        cfg = self.config.processing
        output_dir = context.output_dir / "compressed"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load manifest
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
                "video2compression deduplicated %d manifest rows to %d unique videos",
                source_rows, total,
            )

        if total == 0:
            context.stats["processing"] = {
                "total": 0,
                "source_rows": source_rows,
                "processed": 0,
                "skipped": 0,
                "errors": 0,
            }
            return context

        # Create building blocks only after inputs are known
        detector = create_detector(cfg.detection, cfg.detection_config)
        processed = skipped = errors = 0

        try:
            for video_path, output_relpath in tasks:
                output_path = output_dir / output_relpath
                output_label = str(output_relpath)

                # Skip existing (unless force_all)
                if not context.force_all and output_path.exists():
                    skipped += 1
                    continue

                try:
                    if not Path(video_path).exists():
                        self.logger.warning("Video not found: %s", video_path)
                        errors += 1
                        continue

                    duration = get_video_duration(video_path)
                    if duration <= 0:
                        self.logger.warning("Cannot read duration: %s", video_path)
                        errors += 1
                        continue

                    batch_size = getattr(cfg.detection_config, "batch_size", 16)
                    bbox: tuple[float, float, float, float] | None = None
                    decoded_frames = 0

                    # Pass 1: stream frames for detection (whole video)
                    for frames in iter_ffmpeg_frame_batches(
                        video_path, 0.0, duration, None,
                        batch_size=batch_size,
                    ):
                        decoded_frames += len(frames)
                        detections = detector.detect_batch(frames)
                        bbox = _update_union_bbox(bbox, detections)
                        del detections
                        del frames

                    if decoded_frames == 0:
                        errors += 1
                        continue

                    if bbox is None:
                        self.logger.debug("No detections, skipping: %s", output_label)
                        skipped += 1
                        continue

                    # Pass 2: crop whole video with union bbox
                    ok = clip_and_crop(
                        video_path, 0.0, duration,
                        bbox, None, cfg.video_config,
                        str(output_path),
                    )
                    if ok:
                        processed += 1
                    else:
                        errors += 1

                except Exception as exc:
                    self.logger.error("Error processing %s: %s", output_label, exc)
                    errors += 1

        finally:
            detector.close()
            gc.collect()

        context.stats["processing"] = {
            "total": total,
            "source_rows": source_rows,
            "processed": processed,
            "skipped": skipped,
            "errors": errors,
        }
        self.logger.info(
            "video2compression: processed=%d skipped=%d errors=%d total=%d source_rows=%d",
            processed, skipped, errors, total, source_rows,
        )
        return context
