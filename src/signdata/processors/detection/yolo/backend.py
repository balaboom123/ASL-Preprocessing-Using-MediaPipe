"""YOLO person detection backend."""

import logging

import numpy as np

from .._cuda_utils import (
    clear_cuda_cache,
    describe_device,
    format_cuda_oom_message,
    format_effective_batch_size_message,
    is_cuda_device,
    is_cuda_oom_error,
    parse_cuda_device_index,
    validate_cuda_device,
)
from ..base import Detection, PersonDetector
from .resolver import VALID_ALIASES, resolve_yolo_model

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO, __version__ as ULTRALYTICS_VERSION
except ImportError:
    YOLO = None  # type: ignore[assignment,misc]
    ULTRALYTICS_VERSION = "unknown"


PERSON_CLASS_ID = 0


class YOLODetector(PersonDetector):
    """YOLO-based person detector using ultralytics."""

    def __init__(self, config):
        """
        Args:
            config: YOLODetectionConfig with model, device,
                    confidence_threshold, min_bbox_area.
        """
        if YOLO is None:
            raise ImportError(
                "ultralytics is required for YOLO detection. "
                "Install with: pip install ultralytics"
            )

        validate_cuda_device(config.device)

        self.config = config
        self._predict_device = str(config.device)
        if is_cuda_device(self._predict_device):
            predict_index = parse_cuda_device_index(self._predict_device) or 0
            self._expected_runtime_device = f"cuda:{predict_index}"
        else:
            self._expected_runtime_device = self._predict_device
        self.device_label = describe_device(config.device)
        self.use_fp16 = is_cuda_device(config.device)
        self._effective_batch_size = int(config.batch_size)
        self._reported_effective_batch_size = self._effective_batch_size
        self._runtime_device_verified = False

        resolved_model = resolve_yolo_model(
            config.model,
            allow_download=config.allow_download,
            weights_dir=config.weights_dir,
        )
        # Key alias_mode off the resolver's output, not the raw config, so a
        # cwd-local file named like an alias does not trip the download-error
        # branch below.
        alias_mode = resolved_model in VALID_ALIASES

        try:
            self.model = YOLO(resolved_model)
        except FileNotFoundError as e:
            if alias_mode:
                raise RuntimeError(
                    f"YOLO model {config.model!r} is a valid alias but could not "
                    f"be loaded. This usually means the download failed or the "
                    f"Ultralytics cache is missing/unwritable. Check your internet "
                    f"connection and cache directory permissions."
                ) from e
            raise FileNotFoundError(
                f"YOLO weights not found: {config.model!r}. Provide an existing "
                "local weights path or use a model alias supported by "
                f"ultralytics {ULTRALYTICS_VERSION}."
            ) from e
        except Exception as e:
            msg = str(e).lower()
            if "download" in msg or "connection" in msg or "url" in msg:
                raise RuntimeError(
                    f"Failed to download YOLO model {config.model!r}. "
                    "Check your internet connection, or provide a local "
                    "weights path instead."
                ) from e
            raise

        logger.info(
            "YOLO detector loaded: %s (resolved: %s) on %s "
            "(fp16=%s, configured_batch_size=%d)",
            config.model,
            resolved_model,
            self.device_label,
            self.use_fp16,
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
                backend="YOLO",
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
                backend="YOLO",
                device=self.config.device,
                configured_batch_size=self.config.batch_size,
                attempted_batch_size=attempted_batch_size,
                model=self.config.model,
                learned_batch_size=learned_batch_size,
            )
        ) from exc

    def _reset_predictor(self) -> None:
        # Ultralytics reuses the existing predictor as long as the device stays
        # the same, so a previous fp16 AutoBackend would keep running even if
        # we pass half=False on a later call. Null it out to force a rebuild.
        if hasattr(self.model, "predictor"):
            self.model.predictor = None
        self._runtime_device_verified = False

    @staticmethod
    def _canonical_device_text(device) -> str:
        text = str(device)
        if text == "cuda":
            return "cuda:0"
        return text

    def _verify_runtime_device(self) -> None:
        if self._runtime_device_verified:
            return

        predictor = getattr(self.model, "predictor", None)
        if predictor is None:
            return

        runtime_device = getattr(predictor, "device", None)
        if runtime_device is None:
            logger.debug("YOLO predictor device is unavailable for verification.")
            return

        expected_device = self._expected_runtime_device
        actual_device = self._canonical_device_text(runtime_device)
        backend_device = getattr(getattr(predictor, "model", None), "device", None)
        backend_suffix = ""
        if backend_device is not None:
            backend_suffix = f"; backend={describe_device(str(backend_device))}"

        if actual_device != expected_device:
            raise RuntimeError(
                "YOLO runtime device mismatch: configured "
                f"{describe_device(expected_device)} but Ultralytics predictor "
                f"initialized on {describe_device(actual_device)}{backend_suffix}."
            )

        logger.info(
            "YOLO runtime device verified: configured=%s predictor=%s%s",
            describe_device(expected_device),
            describe_device(actual_device),
            backend_suffix,
        )
        self._runtime_device_verified = True

    def _predict_chunk(self, chunk: list[np.ndarray], *, half: bool):
        results = self.model.predict(
            source=chunk,
            device=self._predict_device,
            verbose=False,
            half=half,
        )
        self._verify_runtime_device()
        return results

    def _predict_with_precision_fallback(self, chunk: list[np.ndarray]):
        if self.use_fp16:
            try:
                return self._predict_chunk(chunk, half=True)
            except Exception as fp16_exc:
                if is_cuda_oom_error(fp16_exc):
                    clear_cuda_cache(self.config.device)
                    raise

                logger.warning(
                    "FP16 inference failed (%s); resetting the Ultralytics "
                    "predictor and retrying with half=False for the rest of "
                    "this run.",
                    fp16_exc,
                )
                self._reset_predictor()
                try:
                    results = self._predict_chunk(chunk, half=False)
                except Exception as fp32_exc:
                    if is_cuda_oom_error(fp32_exc):
                        self.use_fp16 = False
                        clear_cuda_cache(self.config.device)
                    raise RuntimeError(
                        "YOLO inference failed with fp16, and the fp32 "
                        "retry after predictor reset also failed. "
                        f"fp16 error: {fp16_exc}; fp32 retry error: {fp32_exc}"
                    ) from fp32_exc
                self.use_fp16 = False
                return results

        return self._predict_chunk(chunk, half=False)

    def _predict_chunk_adaptive(
        self,
        chunk: list[np.ndarray],
    ) -> list:
        try:
            return list(self._predict_with_precision_fallback(chunk))
        except Exception as exc:
            if not is_cuda_oom_error(exc):
                raise
            if len(chunk) == 1:
                # Last-ditch: if a single frame OOMs while FP16 is still on,
                # drop to FP32 once before giving up. FP32 uses more memory
                # but avoids FP16 allocator fragmentation, which is a common
                # cause of single-frame OOMs after many batches.
                if self.use_fp16:
                    logger.warning(
                        "YOLO OOM at batch size 1 with FP16; disabling FP16 "
                        "and retrying in FP32."
                    )
                    self.use_fp16 = False
                    self._reset_predictor()
                    clear_cuda_cache(self.config.device)
                    try:
                        return list(self._predict_chunk(chunk, half=False))
                    except Exception as retry_exc:
                        if is_cuda_oom_error(retry_exc):
                            clear_cuda_cache(self.config.device)
                            self._raise_terminal_oom(1, retry_exc)
                        raise
                self._raise_terminal_oom(1, exc)

            clear_cuda_cache(self.config.device)
            self._effective_batch_size = min(
                self._effective_batch_size,
                len(chunk) // 2,
            )
            mid = len(chunk) // 2
            logger.debug(
                "YOLO inference OOM for batch size %d; retrying as %d + %d",
                len(chunk), mid, len(chunk) - mid,
            )
            return (
                self._predict_chunk_adaptive(chunk[:mid])
                + self._predict_chunk_adaptive(chunk[mid:])
            )

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        if not frames:
            return []

        all_detections: list[list[Detection]] = []
        start = 0

        while start < len(frames):
            batch_size = self._effective_batch_size
            chunk = frames[start : start + batch_size]
            chunk_len = len(chunk)
            results = self._predict_chunk_adaptive(chunk)
            self._report_effective_batch_size_if_needed()

            for i, result in enumerate(results):
                h, w = chunk[i].shape[:2]
                frame_area = float(w * h)
                frame_dets = []

                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    all_detections.append(frame_dets)
                    continue

                class_ids = boxes.cls.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                xyxy = boxes.xyxy.cpu().numpy()

                for j in range(len(class_ids)):
                    if int(class_ids[j]) != PERSON_CLASS_ID:
                        continue
                    conf = float(confidences[j])
                    if conf < self.config.confidence_threshold:
                        continue

                    x1, y1, x2, y2 = xyxy[j]
                    bbox_area = (x2 - x1) * (y2 - y1)
                    if frame_area > 0 and (bbox_area / frame_area) < self.config.min_bbox_area:
                        continue

                    frame_dets.append(Detection(
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        confidence=conf,
                    ))

                all_detections.append(frame_dets)

            del results
            start += chunk_len

        return all_detections

    def close(self) -> None:
        del self.model
        clear_cuda_cache(self.config.device)


__all__ = ["YOLO", "YOLODetector"]
