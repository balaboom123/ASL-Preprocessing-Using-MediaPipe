"""video2parts processor: video -> SignMusketeers-style part streams."""

import gc
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .base import BaseProcessor
from .pose import create_estimator
from .sampler import create_sampler
from ..registry import register_processor
from ..utils.manifest import get_timing_columns, resolve_video_path
from ..utils.video import resolve_effective_sample_fps


BODY_POSE_INDICES = [0, 11, 12, 13, 14, 15, 16]
LEFT_HAND_HINT = [17, 19, 21]
RIGHT_HAND_HINT = [18, 20, 22]


def _has_points(points: np.ndarray) -> bool:
    return bool(
        points.size
        and np.isfinite(points[:, :2]).all()
        and np.any(points[:, :2])
    )


def _clip_square(cx: float, cy: float, side: float, width: int, height: int):
    side = int(round(max(1.0, min(side, width, height))))
    x1 = int(round(cx - side / 2))
    y1 = int(round(cy - side / 2))
    x1 = max(0, min(x1, width - side))
    y1 = max(0, min(y1, height - side))
    return (x1, y1, x1 + side, y1 + side)


def _bbox_from_landmarks(
    landmarks: np.ndarray,
    frame_shape: Tuple[int, int, int],
    scale: float,
):
    if not _has_points(landmarks):
        return None

    height, width = frame_shape[:2]
    xy = landmarks[:, :2].astype(np.float32)
    xs = xy[:, 0] * width
    ys = xy[:, 1] * height
    cx = float((xs.min() + xs.max()) / 2)
    cy = float((ys.min() + ys.max()) / 2)
    side = float(max(xs.max() - xs.min(), ys.max() - ys.min()) * scale)
    return _clip_square(cx, cy, side, width, height)


def _bbox_from_pose_points(
    pose: np.ndarray,
    indices: List[int],
    frame_shape: Tuple[int, int, int],
    scale: float,
):
    return _bbox_from_landmarks(pose[indices], frame_shape, scale)


def _bbox_from_center(
    center_xy: np.ndarray,
    side: float,
    frame_shape: Tuple[int, int, int],
):
    height, width = frame_shape[:2]
    return _clip_square(
        float(center_xy[0] * width),
        float(center_xy[1] * height),
        side,
        width,
        height,
    )


def _crop_resize(frame: np.ndarray, bbox, crop_size: int) -> np.ndarray:
    if bbox is None:
        return np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    return cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)


def _split_mediapipe_landmarks(landmarks: np.ndarray):
    if landmarks is None or landmarks.shape[0] < 33 + 21 + 21:
        raise ValueError("Expected MediaPipe holistic landmarks")

    face_count = landmarks.shape[0] - 33 - 21 - 21
    if face_count <= 0:
        raise ValueError("Expected MediaPipe face landmarks")

    pose = landmarks[:33]
    face = landmarks[33:33 + face_count]
    left = landmarks[33 + face_count:33 + face_count + 21]
    right = landmarks[33 + face_count + 21:33 + face_count + 42]
    return pose, face, left, right


def _point_valid(points: np.ndarray, index: int) -> bool:
    point = points[index]
    return bool(np.isfinite(point[:2]).all() and np.any(point[:2]))


def _normalize_upper_body_pose(
    pose: np.ndarray,
    missing_value: float,
    previous: Optional[np.ndarray] = None,
):
    if not all(_point_valid(pose, i) for i in BODY_POSE_INDICES):
        if previous is not None:
            return previous.copy(), False
        return np.full(14, missing_value, dtype=np.float32), False

    left_shoulder = pose[11, :2]
    right_shoulder = pose[12, :2]
    head_unit = float(np.linalg.norm(left_shoulder - right_shoulder) / 2.0)
    if head_unit <= 1e-6:
        if previous is not None:
            return previous.copy(), False
        return np.full(14, missing_value, dtype=np.float32), False

    nose = pose[0, :2]
    eyes = [pose[i, :2] for i in (2, 5) if _point_valid(pose, i)]
    eye_y = float(np.mean([p[1] for p in eyes])) if eyes else float(nose[1])

    width = 6.0 * head_unit
    height = 7.0 * head_unit
    left = float(nose[0] - width / 2.0)
    top = eye_y - head_unit

    points = pose[BODY_POSE_INDICES, :2]
    normalized = np.column_stack([
        (points[:, 0] - left) / width,
        (points[:, 1] - top) / height,
    ])
    return normalized.astype(np.float32).reshape(-1), True


def _read_sampled_frames_with_index(
    video_path: str,
    start_sec: float,
    end_sec: float,
    sampler,
    source_fps: float,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], [], []

    start_frame = int(start_sec * source_fps)
    end_frame = int(end_sec * source_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    sampler.reset()
    frames, indices, times = [], [], []
    current = start_frame

    while current <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if sampler.take():
            frames.append(frame)
            indices.append(current)
            times.append(current / source_fps)
        current += 1

    cap.release()
    return frames, indices, times


def _write_video(path: Path, frames: List[np.ndarray], fps: float, codec: str) -> bool:
    if not frames:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        return False
    for frame in frames:
        writer.write(frame)
    writer.release()
    return True


def _row_str(row, column: str) -> str:
    if column not in row.index:
        return ""
    value = row.get(column)
    if value is None:
        return ""
    value = str(value).strip()
    return "" if value.lower() == "nan" else value


@register_processor("video2parts")
class Video2PartsProcessor(BaseProcessor):
    """Build face, hand, and upper-body pose streams from a video segment."""

    name = "video2parts"

    def run(self, context):
        cfg = self.config.processing
        parts_cfg = cfg.parts_config
        output_dir = context.output_dir / "raw"
        output_dir.mkdir(parents=True, exist_ok=True)

        df = context.manifest_df
        if df is None:
            self.logger.warning("No manifest loaded, nothing to process.")
            context.stats["processing"] = {"total": 0}
            return context

        estimator = create_estimator(cfg.pose, cfg.pose_config)
        batch_size = getattr(cfg.pose_config, "batch_size", 16)
        video_dir = str(context.videos_dir) if context.videos_dir else ""
        start_col, end_col = get_timing_columns(df)

        processed = skipped = errors = 0

        try:
            for _, row in df.iterrows():
                sample_id = str(row["SAMPLE_ID"])
                sample_dir = output_dir / sample_id
                expected = [
                    sample_dir / "face.mp4",
                    sample_dir / "left_hand.mp4",
                    sample_dir / "right_hand.mp4",
                    sample_dir / "pose.npz",
                    sample_dir / "meta.json",
                ]

                if not getattr(context, "force_all", False) and all(p.exists() for p in expected):
                    skipped += 1
                    continue

                try:
                    video_path = str(resolve_video_path(row, video_dir))
                    if not os.path.exists(video_path):
                        self.logger.warning("Video not found: %s", video_path)
                        errors += 1
                        continue

                    cap = cv2.VideoCapture(video_path)
                    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                    cap.release()
                    if src_fps <= 0:
                        errors += 1
                        continue

                    start_sec = float(row[start_col])
                    end_sec = float(row[end_col])
                    sampler = create_sampler(cfg.sample_rate, src_fps)
                    frames, frame_indices, frame_times = _read_sampled_frames_with_index(
                        video_path, start_sec, end_sec, sampler, src_fps,
                    )
                    if not frames:
                        errors += 1
                        continue

                    landmarks = []
                    for i in range(0, len(frames), batch_size):
                        landmarks.extend(estimator.process_batch(
                            frames[i:i + batch_size],
                            fallback_on_error=True,
                        ))

                    face_frames, left_frames, right_frames = [], [], []
                    body_pose, bboxes, valid = [], [], []
                    last_left = last_right = last_body = None

                    for frame, lm in zip(frames, landmarks):
                        if lm is None:
                            lm = np.zeros((estimator.num_landmarks, 4), dtype=np.float32)
                        pose, face, left, right = _split_mediapipe_landmarks(lm)

                        face_valid = _has_points(face)
                        left_valid = _has_points(left)
                        right_valid = _has_points(right)

                        face_bbox = _bbox_from_landmarks(
                            face, frame.shape, parts_cfg.bbox_scale,
                        )
                        if face_bbox is None:
                            face_bbox = _bbox_from_pose_points(
                                pose, list(range(11)), frame.shape, parts_cfg.bbox_scale,
                            )
                        face_side = (
                            face_bbox[2] - face_bbox[0]
                            if face_bbox is not None
                            else min(frame.shape[:2]) * 0.25
                        )

                        left_bbox = _bbox_from_landmarks(
                            left, frame.shape, parts_cfg.bbox_scale,
                        )
                        if left_bbox is None and _has_points(pose[LEFT_HAND_HINT]):
                            left_bbox = _bbox_from_center(
                                pose[LEFT_HAND_HINT, :2].mean(axis=0),
                                face_side,
                                frame.shape,
                            )
                        if left_bbox is None:
                            left_bbox = last_left
                        elif left_valid:
                            last_left = left_bbox

                        right_bbox = _bbox_from_landmarks(
                            right, frame.shape, parts_cfg.bbox_scale,
                        )
                        if right_bbox is None and _has_points(pose[RIGHT_HAND_HINT]):
                            right_bbox = _bbox_from_center(
                                pose[RIGHT_HAND_HINT, :2].mean(axis=0),
                                face_side,
                                frame.shape,
                            )
                        if right_bbox is None:
                            right_bbox = last_right
                        elif right_valid:
                            last_right = right_bbox

                        body, body_valid = _normalize_upper_body_pose(
                            pose, parts_cfg.missing_value, last_body,
                        )
                        if body_valid:
                            last_body = body

                        face_frames.append(_crop_resize(frame, face_bbox, parts_cfg.crop_size))
                        left_frames.append(_crop_resize(frame, left_bbox, parts_cfg.crop_size))
                        right_frames.append(_crop_resize(frame, right_bbox, parts_cfg.crop_size))
                        body_pose.append(body)
                        bboxes.append([
                            face_bbox or (parts_cfg.missing_value,) * 4,
                            left_bbox or (parts_cfg.missing_value,) * 4,
                            right_bbox or (parts_cfg.missing_value,) * 4,
                        ])
                        valid.append([face_valid, left_valid, right_valid, body_valid])

                    effective_fps = resolve_effective_sample_fps(src_fps, cfg.sample_rate)
                    output_fps = float(effective_fps or src_fps)

                    ok = all([
                        _write_video(sample_dir / "face.mp4", face_frames, output_fps, parts_cfg.codec),
                        _write_video(sample_dir / "left_hand.mp4", left_frames, output_fps, parts_cfg.codec),
                        _write_video(sample_dir / "right_hand.mp4", right_frames, output_fps, parts_cfg.codec),
                    ])
                    if not ok:
                        errors += 1
                        continue

                    bbox_arr = np.asarray(bboxes, dtype=np.float32)
                    np.savez_compressed(
                        sample_dir / "pose.npz",
                        body_pose=np.asarray(body_pose, dtype=np.float32),
                        frame_time_s=np.asarray(frame_times, dtype=np.float32),
                        frame_index=np.asarray(frame_indices, dtype=np.int64),
                        face_bbox_xyxy=bbox_arr[:, 0],
                        left_bbox_xyxy=bbox_arr[:, 1],
                        right_bbox_xyxy=bbox_arr[:, 2],
                        valid=np.asarray(valid, dtype=bool),
                    )

                    source = self.config.dataset.source
                    meta = {
                        "format": "signdata.parts.v1",
                        "sample_id": sample_id,
                        "video_id": str(row["VIDEO_ID"]),
                        "source_dataset": self.config.dataset.name,
                        "sign_language": source.get("sign_language", ""),
                        "spoken_language": source.get("spoken_language", ""),
                        "country": source.get("country", ""),
                        "start": start_sec,
                        "end": end_sec,
                        "source_fps": src_fps,
                        "sample_rate": output_fps,
                        "frame_count": len(frames),
                        "crop_size": parts_cfg.crop_size,
                        "bbox_scale": parts_cfg.bbox_scale,
                        "pose_backend": "mediapipe_holistic",
                        "pose_landmark_layout": f"mediapipe_{estimator.num_landmarks}",
                        "body_pose_layout": "signmusketeers_upper_body_14",
                        "license": source.get("license", ""),
                        "source_url": _row_str(row, "URL"),
                    }
                    (sample_dir / "meta.json").write_text(
                        json.dumps(meta, ensure_ascii=True, sort_keys=True),
                        encoding="utf-8",
                    )
                    processed += 1

                except Exception as e:
                    self.logger.error("Error processing %s: %s", sample_id, e)
                    errors += 1

        finally:
            estimator.close()
            gc.collect()

        context.stats["processing"] = {
            "total": len(df),
            "processed": processed,
            "skipped": skipped,
            "errors": errors,
        }
        self.logger.info(
            "video2parts: processed=%d skipped=%d errors=%d total=%d",
            processed, skipped, errors, len(df),
        )
        return context
