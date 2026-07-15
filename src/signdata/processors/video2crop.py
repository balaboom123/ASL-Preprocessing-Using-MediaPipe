"""video2crop processor: video → cropped video (.mp4)."""

import gc

from .base import BaseProcessor
from .detection import create_detector, single_person_check, union_bboxes
from .video.ffmpeg import clip_and_crop, ffmpeg_pipe_frames
from ..registry import register_processor
from ..utils.manifest import get_timing_columns, resolve_video_path


@register_processor("video2crop")
class Video2CropProcessor(BaseProcessor):
    """High-level processor: video → cropped video (.mp4).

    Uses ffmpeg as the single frame source for both detection and output,
    ensuring frame-level consistency (no OpenCV/ffmpeg mismatch).

    Orchestrates:
    - video/ffmpeg_pipe for frame decoding (pass 1)
    - detection/ backends for person detection
    - video/clip_and_crop for final output (pass 2, same ffmpeg params + crop)
    """

    name = "video2crop"

    def run(self, context):
        cfg = self.config.processing
        output_dir = context.output_dir / "raw"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create building blocks
        detector = create_detector(cfg.detection, cfg.detection_config)
        # Load manifest
        df = context.manifest_df
        if df is None:
            self.logger.warning("No manifest loaded, nothing to process.")
            context.stats["processing"] = {"total": 0}
            return context

        start_col, end_col = get_timing_columns(df)
        video_dir = str(context.videos_dir) if context.videos_dir else ""

        processed = skipped = errors = 0
        total = len(df)

        try:
            for _, row in df.iterrows():
                sample_id = row["SAMPLE_ID"]
                output_path = output_dir / f"{sample_id}.mp4"

                # Skip existing (unless force_all)
                if not context.force_all and output_path.exists():
                    skipped += 1
                    continue

                try:
                    video_path = resolve_video_path(row, video_dir)
                    if not video_path.exists():
                        self.logger.warning("Video not found: %s", video_path)
                        errors += 1
                        continue
                    video_path = str(video_path)

                    start_sec = float(row[start_col])
                    end_sec = float(row[end_col])

                    # Pass 1: decode frames for detection
                    frames = ffmpeg_pipe_frames(
                        video_path, start_sec, end_sec, cfg.sample_rate,
                    )

                    if not frames:
                        errors += 1
                        continue

                    # Detect persons
                    detections = detector.detect_batch(frames)

                    # Validate single person
                    if not single_person_check(detections):
                        self.logger.debug("Multi-person detected, skipping: %s", sample_id)
                        skipped += 1
                        continue

                    # Compute union bbox across all frames
                    bbox = union_bboxes(detections)
                    if bbox is None:
                        self.logger.debug("No detections, skipping: %s", sample_id)
                        skipped += 1
                        continue

                    # Pass 2: clip + crop with same params
                    ok = clip_and_crop(
                        video_path, start_sec, end_sec,
                        bbox, cfg.sample_rate, cfg.video_config,
                        str(output_path),
                    )
                    if ok:
                        processed += 1
                    else:
                        errors += 1

                except Exception as e:
                    self.logger.error("Error processing %s: %s", sample_id, e)
                    errors += 1

        finally:
            detector.close()
            gc.collect()

        context.stats["processing"] = {
            "total": total,
            "processed": processed,
            "skipped": skipped,
            "errors": errors,
        }
        self.logger.info(
            "video2crop: processed=%d skipped=%d errors=%d total=%d",
            processed, skipped, errors, total,
        )
        return context
