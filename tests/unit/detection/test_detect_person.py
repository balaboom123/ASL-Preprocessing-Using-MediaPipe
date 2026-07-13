"""Tests for detection utilities and config schema defaults.

Covers:
  - Pure logic helpers (union_bbox_tuples, bbox padding/clamp)
  - Schema defaults for detection/video config classes
  - Pipeline processor registration (video2pose / video2crop)
"""

import logging
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

import pytest
import torch


# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from signdata.config.schema import (
    Config,
    MMDetDetectionConfig,
    PathsConfig,
    ProcessingConfig,
    VideoProcessingConfig,
    YOLODetectionConfig,
)
from signdata.processors.detection.mmdet.backend import MMDetDetector
from signdata.processors.detection.validation import union_bbox_tuples
from signdata.processors.detection.yolo.backend import YOLODetector
import signdata.processors  # noqa: F401 – trigger registrations
from signdata.registry import PROCESSOR_REGISTRY


# ===========================================================================
# 1. Schema defaults (config classes)
# ===========================================================================

class TestSchemaDefaults:
    def test_yolo_detection_defaults(self):
        cfg = YOLODetectionConfig()
        assert cfg.model == "yolo11m.pt"
        assert cfg.confidence_threshold == 0.5
        assert cfg.device == "cpu"
        assert cfg.min_bbox_area == 0.05

    def test_mmdet_detection_requires_paths(self):
        with pytest.raises(Exception):
            MMDetDetectionConfig()  # missing required fields

    def test_mmdet_detection_with_paths(self):
        cfg = MMDetDetectionConfig(
            det_model_config="model.py",
            det_model_checkpoint="model.pth",
        )
        assert cfg.device == "cuda:0"

    def test_video_processing_defaults(self):
        cfg = VideoProcessingConfig()
        assert cfg.codec == "libx264"
        assert cfg.padding == 0.0
        assert cfg.resize is None

    def test_paths_config_has_output_and_webdataset(self):
        p = PathsConfig()
        assert hasattr(p, "output")
        assert hasattr(p, "webdataset")
        assert p.output == ""
        assert p.webdataset == ""

    def test_config_processing_defaults(self):
        cfg = Config(dataset={"name": "youtube_asl"})
        assert isinstance(cfg.processing, ProcessingConfig)
        assert cfg.processing.enabled is False


# ===========================================================================
# 2. union_bbox_tuples helper
# ===========================================================================

class TestUnionBboxTuples:
    def test_single_bbox(self):
        result = union_bbox_tuples([(10, 20, 100, 200)])
        assert result == (10, 20, 100, 200)

    def test_union_of_two_overlapping(self):
        b1 = (10.0, 20.0, 100.0, 200.0)
        b2 = (50.0,  5.0, 150.0, 180.0)
        x1, y1, x2, y2 = union_bbox_tuples([b1, b2])
        assert x1 == 10.0   # min x1
        assert y1 == 5.0    # min y1
        assert x2 == 150.0  # max x2
        assert y2 == 200.0  # max y2

    def test_union_of_non_overlapping(self):
        b1 = (0.0,   0.0,  50.0,  50.0)
        b2 = (60.0, 60.0, 120.0, 120.0)
        x1, y1, x2, y2 = union_bbox_tuples([b1, b2])
        assert x1 == 0.0
        assert y1 == 0.0
        assert x2 == 120.0
        assert y2 == 120.0

    def test_union_identical_bboxes(self):
        b = (5.0, 10.0, 80.0, 90.0)
        assert union_bbox_tuples([b, b, b]) == b

    def test_union_five_frames(self):
        bboxes = [(i * 5.0, i * 3.0, i * 5.0 + 100.0, i * 3.0 + 200.0) for i in range(5)]
        x1, y1, x2, y2 = union_bbox_tuples(bboxes)
        assert x1 == 0.0
        assert y1 == 0.0
        assert x2 == pytest.approx(4 * 5.0 + 100.0)
        assert y2 == pytest.approx(4 * 3.0 + 200.0)


# ===========================================================================
# 3. Pipeline registration
# ===========================================================================

class TestPipelineRegistration:
    def test_video2pose_registered(self):
        assert "video2pose" in PROCESSOR_REGISTRY

    def test_video2crop_registered(self):
        assert "video2crop" in PROCESSOR_REGISTRY

    def test_video2compression_registered(self):
        assert "video2compression" in PROCESSOR_REGISTRY


# ===========================================================================
# 4. YOLO init errors
# ===========================================================================

class TestYOLODetectorInit:
    def test_missing_weights_raises_clear_error(self, tmp_path):
        # Use an existing local-path so the resolver passes through to YOLO();
        # unrecognized alias strings are now rejected earlier by the resolver.
        weights = tmp_path / "custom-weights.pt"
        weights.touch()
        cfg = YOLODetectionConfig(model=str(weights), device="cpu")

        with patch(
            "signdata.processors.detection.yolo.backend.YOLO",
            side_effect=FileNotFoundError("missing"),
        ):
            with pytest.raises(FileNotFoundError, match="YOLO weights not found"):
                YOLODetector(cfg)

    def test_cuda_unavailable_raises_clear_error(self):
        cfg = YOLODetectionConfig(model="yolov8n.pt", device="cuda:0")

        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="CUDA is unavailable"):
                YOLODetector(cfg)


# ===========================================================================
# 5. FP16 inference policy
# ===========================================================================

class _EmptyResult:
    boxes = None


class _StatefulFakeYOLO:
    """Minimal fake that mimics Ultralytics predictor reuse semantics."""

    def __init__(
        self,
        *,
        fp16_failures: int = 0,
        fp32_failures: int = 0,
        fp16_error: str = "fp16 not supported on this device",
        fp32_error: str = "fp32 fallback failed",
        runtime_device=None,
    ):
        self.predictor = None
        self.calls = []
        self.predictor_builds = 0
        self._fp16_failures = fp16_failures
        self._fp32_failures = fp32_failures
        self._fp16_error = fp16_error
        self._fp32_error = fp32_error
        self._runtime_device = runtime_device

    def predict(self, *, source, device, verbose, half):
        # Ultralytics reuses the predictor when device is unchanged, so the
        # effective precision comes from the existing predictor, not the new
        # half= kwarg, unless caller resets predictor first.
        predictor_device = self._runtime_device or device
        if self.predictor is None:
            self.predictor_builds += 1
            self.predictor = SimpleNamespace(fp16=half, device=predictor_device)
        else:
            self.predictor.device = predictor_device

        effective_half = self.predictor.fp16
        self.calls.append({
            "requested_half": half,
            "effective_half": effective_half,
            "device": device,
            "predictor_builds": self.predictor_builds,
        })

        if effective_half and self._fp16_failures > 0:
            self._fp16_failures -= 1
            raise RuntimeError(self._fp16_error)
        if not effective_half and self._fp32_failures > 0:
            self._fp32_failures -= 1
            raise RuntimeError(self._fp32_error)

        return [_EmptyResult() for _ in source]


class _BatchLimitFakeYOLO:
    """Fake model that raises CUDA OOM when a predict call is too large."""

    def __init__(self, max_batch_size: int):
        self.predictor = None
        self.max_batch_size = max_batch_size
        self.calls = []

    def predict(self, *, source, device, verbose, half):
        self.predictor = SimpleNamespace(device=device)
        self.calls.append({
            "batch_size": len(source),
            "device": device,
            "half": half,
        })
        if len(source) > self.max_batch_size:
            raise RuntimeError("CUDA out of memory")
        return [_EmptyResult() for _ in source]


class _EmptyPredInstances:
    def __len__(self):
        return 0


class _BatchLimitFakeMMDetApis:
    """Fake MMDet APIs that raise CUDA OOM when a batch is too large."""

    def __init__(self, max_batch_size: int):
        self.max_batch_size = max_batch_size
        self.calls = []
        self.detector = SimpleNamespace(cfg=SimpleNamespace())

    def init_detector(self, config, checkpoint, device):
        self.init_args = {
            "config": config,
            "checkpoint": checkpoint,
            "device": device,
        }
        return self.detector

    def inference_detector(self, detector, source):
        assert detector is self.detector
        batch_size = len(source) if isinstance(source, list) else 1
        self.calls.append(batch_size)
        if batch_size > self.max_batch_size:
            raise RuntimeError("CUDA out of memory")

        results = [
            SimpleNamespace(pred_instances=_EmptyPredInstances())
            for _ in range(batch_size)
        ]
        if isinstance(source, list):
            return results
        return results[0]


class TestYOLODetectorFP16Policy:
    """Verify CUDA→FP16 policy and real predictor-reset fallback behavior."""

    _FRAME = np.zeros((100, 100, 3), dtype=np.uint8)

    def _make_detector(self, device: str, model=None, *, cuda_available: bool = True):
        from contextlib import ExitStack

        cfg = YOLODetectionConfig(model="yolov8n.pt", device=device)
        fake_model = model or _StatefulFakeYOLO()
        with ExitStack() as stack:
            stack.enter_context(
                patch("signdata.processors.detection.yolo.backend.YOLO", return_value=fake_model)
            )
            if device.startswith("cuda"):
                _, _, suffix = device.partition(":")
                requested_index = int(suffix) if suffix else 0
                stack.enter_context(
                    patch("torch.cuda.is_available", return_value=cuda_available)
                )
                stack.enter_context(
                    patch(
                        "torch.cuda.device_count",
                        return_value=(requested_index + 1) if cuda_available else 0,
                    )
                )
                stack.enter_context(
                    patch(
                        "torch.cuda.get_device_name",
                        side_effect=lambda index: f"Fake GPU {index}",
                    )
                )
            detector = YOLODetector(cfg)
        return detector, fake_model

    def test_cpu_sets_use_fp16_false(self):
        detector, _ = self._make_detector("cpu")
        assert detector.use_fp16 is False

    def test_cuda_sets_use_fp16_true(self):
        detector, _ = self._make_detector("cuda:0")
        assert detector.use_fp16 is True

    def test_cpu_calls_predict_with_half_false(self):
        detector, fake_model = self._make_detector("cpu")

        detector.detect_batch([self._FRAME])

        assert len(fake_model.calls) == 1
        assert fake_model.calls[0]["requested_half"] is False
        assert fake_model.calls[0]["effective_half"] is False

    def test_cuda_calls_predict_with_half_true(self):
        detector, fake_model = self._make_detector("cuda:0")

        detector.detect_batch([self._FRAME])

        assert len(fake_model.calls) == 1
        assert fake_model.calls[0]["requested_half"] is True
        assert fake_model.calls[0]["effective_half"] is True

    def test_cuda_device_is_logged_and_passed_through(self, caplog):
        fake_model = _StatefulFakeYOLO()
        cfg = YOLODetectionConfig(model="yolov8n.pt", device="cuda:1")

        with caplog.at_level(logging.INFO):
            with patch("torch.cuda.is_available", return_value=True):
                with patch("torch.cuda.device_count", return_value=2):
                    with patch("torch.cuda.get_device_name", return_value="Fake GPU 1"):
                        with patch(
                            "signdata.processors.detection.yolo.backend.YOLO",
                            return_value=fake_model,
                        ):
                            detector = YOLODetector(cfg)
                            detector.detect_batch([self._FRAME])

        assert fake_model.calls[0]["device"] == torch.device("cuda:1")
        assert "cuda:1 (GPU 1: Fake GPU 1)" in caplog.text
        assert "YOLO runtime device verified" in caplog.text

    def test_plain_cuda_device_verifies_as_cuda_zero(self):
        detector, fake_model = self._make_detector("cuda")

        detector.detect_batch([self._FRAME])

        assert fake_model.calls[0]["device"] == torch.device("cuda")

    def test_cuda_runtime_device_mismatch_raises_clear_error(self):
        fake_model = _StatefulFakeYOLO(runtime_device=torch.device("cuda:0"))
        cfg = YOLODetectionConfig(model="yolov8n.pt", device="cuda:1")

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=2):
                with patch("torch.cuda.get_device_name", side_effect=lambda index: f"Fake GPU {index}"):
                    with patch(
                        "signdata.processors.detection.yolo.backend.YOLO",
                        return_value=fake_model,
                    ):
                        detector = YOLODetector(cfg)

        with pytest.raises(RuntimeError, match="YOLO runtime device mismatch"):
            detector.detect_batch([self._FRAME])

    def test_fp16_failure_resets_predictor_before_fp32_retry(self):
        fake_model = _StatefulFakeYOLO(
            fp16_failures=2,
            fp16_error="CUDA kernel for float16 not supported",
        )
        detector, fake_model = self._make_detector("cuda:0", model=fake_model)
        assert detector.use_fp16 is True

        detector.detect_batch([self._FRAME])

        assert detector.use_fp16 is False
        assert len(fake_model.calls) == 2
        assert fake_model.calls[0]["requested_half"] is True
        assert fake_model.calls[0]["effective_half"] is True
        assert fake_model.calls[1]["requested_half"] is False
        assert fake_model.calls[1]["effective_half"] is False
        assert fake_model.predictor_builds == 2

    def test_fp16_fallback_persists_for_subsequent_batches(self):
        fake_model = _StatefulFakeYOLO(
            fp16_failures=2,
            fp16_error="CUDA kernel for float16 not supported",
        )
        cfg = YOLODetectionConfig(model="yolov8n.pt", device="cuda:0", batch_size=1)
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=1):
                with patch("torch.cuda.get_device_name", return_value="Fake GPU 0"):
                    with patch(
                        "signdata.processors.detection.yolo.backend.YOLO",
                        return_value=fake_model,
                    ):
                        detector = YOLODetector(cfg)

        detector.detect_batch([self._FRAME, self._FRAME])

        assert len(fake_model.calls) == 3
        assert fake_model.calls[0]["effective_half"] is True
        assert fake_model.calls[1]["effective_half"] is False
        assert fake_model.calls[2]["requested_half"] is False
        assert fake_model.calls[2]["effective_half"] is False
        assert fake_model.predictor_builds == 2
        assert fake_model.calls[1]["predictor_builds"] == fake_model.calls[2]["predictor_builds"]

    def test_fp32_retry_failure_raises_combined_error(self):
        fake_model = _StatefulFakeYOLO(
            fp16_failures=2,
            fp32_failures=1,
            fp16_error="no kernel image is available for execution on the device",
            fp32_error="invalid device function",
        )
        detector, _ = self._make_detector("cuda:0", model=fake_model)

        with pytest.raises(
            RuntimeError,
            match="fp16, and the fp32 retry after predictor reset also failed",
        ):
            detector.detect_batch([self._FRAME])

    def test_cuda_oom_retries_with_smaller_batches(self):
        fake_model = _BatchLimitFakeYOLO(max_batch_size=2)
        cfg = YOLODetectionConfig(
            model="yolov8n.pt",
            device="cuda:0",
            batch_size=4,
        )
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=1):
                with patch("torch.cuda.get_device_name", return_value="Fake GPU 0"):
                    with patch(
                        "signdata.processors.detection.yolo.backend.YOLO",
                        return_value=fake_model,
                    ):
                        detector = YOLODetector(cfg)

        with patch("torch.cuda.empty_cache") as empty_cache:
            detections = detector.detect_batch([self._FRAME] * 4)

        assert len(detections) == 4
        assert [call["batch_size"] for call in fake_model.calls] == [4, 2, 2]
        assert empty_cache.called

    def test_cuda_oom_learns_smaller_effective_batch_size(self, caplog):
        fake_model = _BatchLimitFakeYOLO(max_batch_size=2)
        cfg = YOLODetectionConfig(
            model="yolov8n.pt",
            device="cuda:0",
            batch_size=4,
        )
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=1):
                with patch("torch.cuda.get_device_name", return_value="Fake GPU 0"):
                    with patch(
                        "signdata.processors.detection.yolo.backend.YOLO",
                        return_value=fake_model,
                    ):
                        detector = YOLODetector(cfg)

        with patch("torch.cuda.empty_cache") as empty_cache:
            with caplog.at_level(logging.WARNING):
                detections = detector.detect_batch([self._FRAME] * 8)

        assert len(detections) == 8
        assert [call["batch_size"] for call in fake_model.calls] == [4, 2, 2, 2, 2]
        assert detector._effective_batch_size == 2
        assert empty_cache.called
        assert "set processing.detection_config.batch_size to 2 or lower" in caplog.text

    def test_deep_cuda_oom_reports_one_warning_for_each_new_lower_bound(self, caplog):
        fake_model = _BatchLimitFakeYOLO(max_batch_size=8)
        cfg = YOLODetectionConfig(
            model="yolov8n.pt",
            device="cuda:0",
            batch_size=64,
        )
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=1):
                with patch("torch.cuda.get_device_name", return_value="Fake GPU 0"):
                    with patch(
                        "signdata.processors.detection.yolo.backend.YOLO",
                        return_value=fake_model,
                    ):
                        detector = YOLODetector(cfg)

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=1):
                with patch("torch.cuda.get_device_name", return_value="Fake GPU 0"):
                    with patch("torch.cuda.empty_cache"):
                        with caplog.at_level(logging.WARNING):
                            detector.detect_batch([self._FRAME] * 64)
                            detector.detect_batch([self._FRAME] * 16)

        lowered_messages = [
            record.message for record in caplog.records
            if "lowered the effective batch size" in record.message
        ]
        retry_messages = [
            record.message for record in caplog.records
            if "retrying as" in record.message
        ]
        assert detector._effective_batch_size == 8
        assert lowered_messages == [
            "YOLO lowered the effective batch size for this run from 64 to 8 "
            "on cuda:0 (GPU 0: Fake GPU 0) after CUDA out of memory. To reduce "
            "VRAM usage and avoid repeated retries, set "
            "processing.detection_config.batch_size to 8 or lower."
        ]
        assert retry_messages == []

    def test_terminal_cuda_oom_raises_actionable_message(self):
        fake_model = _BatchLimitFakeYOLO(max_batch_size=0)
        cfg = YOLODetectionConfig(
            model="yolov8n.pt",
            device="cuda:0",
            batch_size=1,
        )
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=1):
                with patch("torch.cuda.get_device_name", return_value="Fake GPU 0"):
                    with patch(
                        "signdata.processors.detection.yolo.backend.YOLO",
                        return_value=fake_model,
                    ):
                        detector = YOLODetector(cfg)

        with patch("torch.cuda.empty_cache"):
            with pytest.raises(
                RuntimeError,
                match="processing\\.detection_config\\.batch_size=1",
            ):
                detector.detect_batch([self._FRAME])


class TestMMDetDetectorOOMPolicy:
    _FRAME = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_cuda_oom_retries_with_smaller_batches(self):
        fake_apis = _BatchLimitFakeMMDetApis(max_batch_size=2)
        fake_mmdet = ModuleType("mmdet")
        fake_mmdet.__path__ = []
        fake_mmdet_apis = ModuleType("mmdet.apis")
        fake_mmdet_apis.init_detector = fake_apis.init_detector
        fake_mmdet_apis.inference_detector = fake_apis.inference_detector
        fake_mmdet.apis = fake_mmdet_apis

        fake_mmpose = ModuleType("mmpose")
        fake_mmpose.__path__ = []
        fake_mmpose_utils = ModuleType("mmpose.utils")
        fake_mmpose_utils.adapt_mmdet_pipeline = lambda cfg: cfg
        fake_mmpose.utils = fake_mmpose_utils

        cfg = MMDetDetectionConfig(
            det_model_config="model.py",
            det_model_checkpoint="model.pth",
            device="cuda:1",
            batch_size=4,
        )

        with patch.dict(
            sys.modules,
            {
                "mmdet": fake_mmdet,
                "mmdet.apis": fake_mmdet_apis,
                "mmpose": fake_mmpose,
                "mmpose.utils": fake_mmpose_utils,
            },
        ):
            with patch("torch.cuda.is_available", return_value=True):
                with patch("torch.cuda.device_count", return_value=2):
                    with patch("torch.cuda.get_device_name", return_value="Fake GPU 1"):
                        detector = MMDetDetector(cfg)
                        with patch("torch.cuda.empty_cache") as empty_cache:
                            detections = detector.detect_batch([self._FRAME] * 8)

        assert len(detections) == 8
        assert fake_apis.init_args["device"] == "cuda:1"
        assert fake_apis.calls == [4, 2, 2, 2, 2]
        assert detector._effective_batch_size == 2
        assert empty_cache.called

    def test_deep_cuda_oom_reports_one_warning_for_each_new_lower_bound(self, caplog):
        fake_apis = _BatchLimitFakeMMDetApis(max_batch_size=8)
        fake_mmdet = ModuleType("mmdet")
        fake_mmdet.__path__ = []
        fake_mmdet_apis = ModuleType("mmdet.apis")
        fake_mmdet_apis.init_detector = fake_apis.init_detector
        fake_mmdet_apis.inference_detector = fake_apis.inference_detector
        fake_mmdet.apis = fake_mmdet_apis

        fake_mmpose = ModuleType("mmpose")
        fake_mmpose.__path__ = []
        fake_mmpose_utils = ModuleType("mmpose.utils")
        fake_mmpose_utils.adapt_mmdet_pipeline = lambda cfg: cfg
        fake_mmpose.utils = fake_mmpose_utils

        cfg = MMDetDetectionConfig(
            det_model_config="model.py",
            det_model_checkpoint="model.pth",
            device="cuda:1",
            batch_size=64,
        )

        with patch.dict(
            sys.modules,
            {
                "mmdet": fake_mmdet,
                "mmdet.apis": fake_mmdet_apis,
                "mmpose": fake_mmpose,
                "mmpose.utils": fake_mmpose_utils,
            },
        ):
            with patch("torch.cuda.is_available", return_value=True):
                with patch("torch.cuda.device_count", return_value=2):
                    with patch("torch.cuda.get_device_name", return_value="Fake GPU 1"):
                        detector = MMDetDetector(cfg)
                        with patch("torch.cuda.empty_cache"):
                            with caplog.at_level(logging.WARNING):
                                detector.detect_batch([self._FRAME] * 64)
                                detector.detect_batch([self._FRAME] * 16)

        lowered_messages = [
            record.message for record in caplog.records
            if "lowered the effective batch size" in record.message
        ]
        retry_messages = [
            record.message for record in caplog.records
            if "retrying as" in record.message
        ]
        assert detector._effective_batch_size == 8
        assert lowered_messages == [
            "MMDet lowered the effective batch size for this run from 64 to 8 "
            "on cuda:1 (GPU 1: Fake GPU 1) after CUDA out of memory. To reduce "
            "VRAM usage and avoid repeated retries, set "
            "processing.detection_config.batch_size to 8 or lower."
        ]
        assert retry_messages == []
