"""Tests for WebDataset packaging of video2parts bundles."""

import tarfile

import pandas as pd
import webdataset as wds

from signdata.config.schema import Config
from signdata.datasets.how2sign import How2SignDataset
from signdata.output.webdataset import WebDatasetOutput
from signdata.pipeline.context import PipelineContext


def test_webdataset_packages_video2parts_bundle(tmp_path):
    sample_id = "sample_001"
    raw_dir = tmp_path / "output" / "run" / "raw" / sample_id
    raw_dir.mkdir(parents=True)
    for name in ("face.mp4", "left_hand.mp4", "right_hand.mp4", "pose.npz"):
        (raw_dir / name).write_bytes(b"x")
    (raw_dir / "meta.json").write_text('{"format":"signdata.parts.v1"}')

    cfg = Config(
        dataset={"name": "how2sign"},
        processing={
            "enabled": True,
            "processor": "video2parts",
            "detection": "null",
            "pose": "mediapipe",
            "pose_config": {"model_complexity": 1},
        },
        output={"enabled": True, "type": "webdataset"},
        post_processing={"enabled": False},
    )
    ctx = PipelineContext(
        config=cfg,
        dataset=How2SignDataset(),
        manifest_df=pd.DataFrame({
            "VIDEO_ID": ["video_001"],
            "SAMPLE_ID": [sample_id],
            "START": [0.0],
            "END": [1.0],
            "TEXT": ["hello"],
        }),
        output_dir=tmp_path / "output" / "run",
        webdataset_dir=tmp_path / "wds" / "run",
    )

    WebDatasetOutput(cfg).run(ctx)

    shard_path = tmp_path / "wds" / "run" / "shard-000000.tar"

    with tarfile.open(shard_path) as tar:
        names = set(tar.getnames())

    assert names == {
        f"{sample_id}.face_mp4",
        f"{sample_id}.left_hand_mp4",
        f"{sample_id}.right_hand_mp4",
        f"{sample_id}.pose_npz",
        f"{sample_id}.json",
        f"{sample_id}.txt",
    }
    rows = list(wds.WebDataset(str(shard_path), shardshuffle=False))
    assert len(rows) == 1
    assert set(rows[0]) >= {
        "__key__",
        "face_mp4",
        "left_hand_mp4",
        "right_hand_mp4",
        "pose_npz",
        "json",
        "txt",
    }
    assert rows[0]["__key__"] == sample_id
    assert ctx.stats["output.webdataset"]["written"] == 1
