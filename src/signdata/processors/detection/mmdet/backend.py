"""MMDet person detection backend."""

import logging

import numpy as np

from .._cuda_utils import (
    clear_cuda_cache,
    describe_device,
    format_cuda_oom_message,
    format_effective_batch_size_message,
    is_cuda_oom_error,
    validate_cuda_device,
)
from ..base import Detection, PersonDetector

logger = logging.getLogger(__name__)


class MMDetDetector(PersonDetector):
    """MMDetection-based person detector.

    Uses the same RTMDet models that MMPose ships for top-down pose estimation.
    """

    def __init__(self, config):
        """
        Args:
            config: MMDetDetectionConfig with det_model_config,
                    det_model_checkpoint, device.
        """
        from mmdet.apis import init_detector
        from mmpose.utils import adapt_mmdet_pipeline

        validate_cuda_device(config.device)
        self.config = config
        self.device_label = describe_device(config.device)
        self._effective_batch_size = int(config.batch_size)
        self._reported_effective_batch_size = self._effective_batch_size
        self.detector = init_detector(
            config.det_model_config,
            config.det_model_checkpoint,
            device=config.device,
        )
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
        self._use_sequential = False
        logger.info(
            "MMDet detector loaded: %s on %s (configured_batch_size=%d)",
            config.det_model_config,
            self.device_label,
            self.config.batch_size,
        )

    def _report_effective_batch_size_if_needed(self) -> None:
        current_batch_size = self._effective_batch_size
        previous_batch_size = self._reported_effective_batch_size
        if current_batch_size >= previous_batch_size:
            return

        self._reported_effective_batch_size = current_batch_size
        logger.warning(
            format_effective_batch_size_message(
                backend="MMDet",
                device=self.config.device,
                previous_batch_size=previous_batch_size,
                new_batch_size=current_batch_size,
            )
        )

    def _raise_terminal_oom(self, attempted_batch_size: int, exc: BaseException) -> None:
        learned_batch_size = None
        if self._effective_batch_size < self.config.batch_size:
            learned_batch_size = self._effective_batch_size

        raise RuntimeError(
            format_cuda_oom_message(
                backend="MMDet",
                device=self.config.device,
                configured_batch_size=self.config.batch_size,
                attempted_batch_size=attempted_batch_size,
                model=self.config.det_model_config,
                learned_batch_size=learned_batch_size,
            )
        ) from exc

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        if not frames:
            return []

        from mmdet.apis import inference_detector

        all_detections: list[list[Detection]] = []
        start = 0

        while start < len(frames):
            batch_size = self._effective_batch_size
            chunk = frames[start : start + batch_size]
            chunk_len = len(chunk)
            batch_results = self._infer_chunk_adaptive(inference_detector, chunk)
            self._report_effective_batch_size_if_needed()

            for result in batch_results:
                frame_dets = []
                pred = result.pred_instances
                if pred is not None and len(pred) > 0:
                    bboxes = pred.bboxes.cpu().numpy()
                    scores = pred.scores.cpu().numpy()
                    labels = pred.labels.cpu().numpy()

                    for j in range(len(labels)):
                        if int(labels[j]) != 0:  # person class
                            continue
                        x1, y1, x2, y2 = bboxes[j]
                        frame_dets.append(Detection(
                            bbox=(float(x1), float(y1), float(x2), float(y2)),
                            confidence=float(scores[j]),
                        ))

                all_detections.append(frame_dets)

            del batch_results
            start += chunk_len

        return all_detections

    def _infer_chunk_adaptive(
        self,
        inference_detector,
        chunk: list[np.ndarray],
    ) -> list:
        try:
            return self._infer_chunk(inference_detector, chunk)
        except Exception as exc:
            if not is_cuda_oom_error(exc):
                raise
            if len(chunk) == 1:
                self._raise_terminal_oom(1, exc)

            clear_cuda_cache(self.config.device)
            self._effective_batch_size = min(
                self._effective_batch_size,
                len(chunk) // 2,
            )
            mid = len(chunk) // 2
            logger.debug(
                "MMDet inference OOM for batch size %d; retrying as %d + %d",
                len(chunk), mid, len(chunk) - mid,
            )
            return self._infer_chunk_adaptive(
                inference_detector,
                chunk[:mid],
            ) + self._infer_chunk_adaptive(
                inference_detector,
                chunk[mid:],
            )

    def _infer_chunk(self, inference_detector, chunk: list[np.ndarray]):
        if self._use_sequential:
            return [inference_detector(self.detector, f) for f in chunk]

        try:
            batch_results = inference_detector(self.detector, chunk)
            if not isinstance(batch_results, (list, tuple)):
                batch_results = [batch_results]
            return list(batch_results)
        except Exception as exc:
            if is_cuda_oom_error(exc):
                clear_cuda_cache(self.config.device)
                raise
            logger.warning(
                "MMDet batched inference failed; falling back to "
                "sequential mode for all subsequent frames."
            )
            self._use_sequential = True
            return [inference_detector(self.detector, f) for f in chunk]

    def close(self) -> None:
        del self.detector
        clear_cuda_cache(self.config.device)


__all__ = ["MMDetDetector"]
