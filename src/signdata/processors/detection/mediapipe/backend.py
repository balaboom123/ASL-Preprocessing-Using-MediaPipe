"""MediaPipe detection-only backend."""

from typing import List

import numpy as np

from ..base import Detection, PersonDetector


class MediaPipeDetector(PersonDetector):
    """Lightweight MediaPipe-based person detection.

    Uses MediaPipe's pose detection (not the full holistic pipeline)
    to locate a person in the frame.
    """

    def __init__(self, config):
        """
        Args:
            config: MediaPipeDetectionConfig with min_detection_confidence.
        """
        import mediapipe as mp
        self.detector = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=0,  # lightweight for detection only
            min_detection_confidence=config.min_detection_confidence,
        )

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        import cv2
        all_detections: List[List[Detection]] = []

        for frame in frames:
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.detector.process(rgb)

            frame_detections = []
            if result.pose_landmarks:
                points = [
                    (lm.x * w, lm.y * h)
                    for lm in result.pose_landmarks.landmark
                    if lm.visibility > 0.3
                ]
                if points:
                    xs, ys = zip(*points)
                    frame_detections.append(
                        Detection(
                            bbox=(min(xs), min(ys), max(xs), max(ys)),
                            confidence=1.0,
                        )
                    )
            all_detections.append(frame_detections)

        return all_detections

    def close(self) -> None:
        self.detector.close()


__all__ = ["MediaPipeDetector"]
