"""video2pose processor: video → pose landmarks (.npy)."""

import gc
import logging

import cv2
import numpy as np

from .base import BaseProcessor
from .detection import create_detector, single_person_check
from .pose import create_estimator
from ..registry import register_processor
from ..utils.manifest import get_timing_columns, resolve_video_path
from ..utils.video import FPSSampler, read_sampled_frames

logger = logging.getLogger(__name__)


def _extract_bboxes(detections, frames):
    """Convert upstream detections to per-frame bbox arrays for the pose estimator.

    Each returned element is an (N, 4) array suitable for ``inference_topdown``,
    or a full-frame fallback when a frame has zero detections.
    """
    bboxes = []
    for i, frame_dets in enumerate(detections):
        if frame_dets:
            d = frame_dets[0]  # single person (verified by single_person_check)
            bboxes.append(np.array([list(d.bbox)], dtype=np.float32))
        else:
            h, w = frames[i].shape[:2]
            bboxes.append(np.array([[0, 0, w, h]], dtype=np.float32))
    return bboxes


@register_processor("video2pose")
class Video2PoseProcessor(BaseProcessor):
    """High-level processor: video → pose landmarks (.npy).

    Orchestrates:
    - sampler for frame selection (native / ratio / absolute FPS)
    - detection/ backends for person detection
    - pose/ backends for pose estimation
    """

    name = "video2pose"

    def run(self, context):
        cfg = self.config.processing
        output_dir = context.output_dir / "raw"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create building blocks
        detector = create_detector(cfg.detection, cfg.detection_config)
        estimator = create_estimator(cfg.pose, cfg.pose_config)

        batch_size = cfg.pose_config.batch_size

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
                output_path = output_dir / f"{sample_id}.npy"

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

                    # Get source FPS for sampler
                    cap = cv2.VideoCapture(video_path)
                    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                    cap.release()

                    if src_fps <= 0:
                        errors += 1
                        continue

                    # Read sampled frames
                    sampler = FPSSampler(src_fps, cfg.sample_rate)
                    frames = read_sampled_frames(
                        video_path, start_sec, end_sec, sampler, src_fps,
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

                    # Extract per-frame bboxes from upstream detection
                    frame_bboxes = _extract_bboxes(detections, frames)

                    # Extract pose landmarks in batches
                    sequences = []
                    num_landmarks = estimator.num_landmarks

                    for i in range(0, len(frames), batch_size):
                        batch_frames = frames[i:i + batch_size]
                        batch_bboxes = frame_bboxes[i:i + batch_size]
                        batch_results = estimator.process_batch(
                            batch_frames, bboxes=batch_bboxes,
                        )
                        for landmarks in batch_results:
                            if landmarks is None:
                                landmarks = np.zeros((num_landmarks, 4), dtype=np.float32)
                            sequences.append(landmarks)

                    # Save
                    if sequences:
                        arr = np.array(sequences)
                        if arr.size > 0 and np.any(arr):
                            np.save(output_path, arr)
                            processed += 1

                except Exception as e:
                    self.logger.error("Error processing %s: %s", sample_id, e)
                    errors += 1

        finally:
            detector.close()
            estimator.close()
            gc.collect()

        context.stats["processing"] = {
            "total": total,
            "processed": processed,
            "skipped": skipped,
            "errors": errors,
        }
        self.logger.info(
            "video2pose: processed=%d skipped=%d errors=%d total=%d",
            processed, skipped, errors, total,
        )
        return context
