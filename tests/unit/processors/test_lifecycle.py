"""Processor lifecycle checks."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from signdata.config.schema import Config
from signdata.datasets.how2sign import How2SignDataset
from signdata.pipeline.context import PipelineContext
from signdata.processors.detection import Detection, union_bboxes
from signdata.processors.video2compression import Video2CompressionProcessor
from signdata.processors.video2crop import Video2CropProcessor
from signdata.processors.video2parts import Video2PartsProcessor
from signdata.processors.video2pose import Video2PoseProcessor


def _config(processor_name):
    return Config(
        dataset={"name": "how2sign"},
        processing={
            "enabled": True,
            "processor": processor_name,
            "detection": "null",
            "pose": "mediapipe",
            "pose_config": {},
        },
    )


def test_union_bboxes_uses_largest_detection_per_frame():
    detections = [
        [
            Detection((0, 0, 2, 2), 0.9),
            Detection((0, 0, 1, 1), 1.0),
        ],
        [Detection((1, -1, 3, 1), 0.8)],
    ]

    assert union_bboxes(detections) == (0, -1, 3, 2)


@pytest.mark.parametrize(
    ("processor_class", "processor_name", "factory"),
    [
        # Compression has no detection backend; its expensive call is ffmpeg.
        (Video2CompressionProcessor, "video2compression", "transcode"),
        (Video2CropProcessor, "video2crop", "create_detector"),
        (Video2PartsProcessor, "video2parts", "create_estimator"),
        (Video2PoseProcessor, "video2pose", "create_detector"),
        (Video2PoseProcessor, "video2pose", "create_estimator"),
    ],
)
def test_no_manifest_creates_no_backend_or_output(
    tmp_path, processor_class, processor_name, factory
):
    config = _config(processor_name)
    output_dir = tmp_path / "output"
    context = PipelineContext(
        config=config,
        dataset=How2SignDataset(),
        output_dir=output_dir,
    )

    with patch(f"{processor_class.__module__}.{factory}") as factory_mock:
        result = processor_class(config).run(context)

    factory_mock.assert_not_called()
    assert result.stats["processing"] == {"total": 0}
    assert not output_dir.exists()


def test_pose_closes_detector_when_estimator_creation_fails(tmp_path):
    config = _config("video2pose")
    context = PipelineContext(
        config=config,
        dataset=How2SignDataset(),
        manifest_df=pd.DataFrame({"START": [0.0], "END": [1.0]}),
        output_dir=tmp_path / "output",
    )
    detector = MagicMock()

    with patch(
        "signdata.processors.video2pose.create_detector", return_value=detector
    ), patch(
        "signdata.processors.video2pose.create_estimator",
        side_effect=RuntimeError("failed"),
    ), pytest.raises(RuntimeError, match="failed"):
        Video2PoseProcessor(config).run(context)

    detector.close.assert_called_once_with()
