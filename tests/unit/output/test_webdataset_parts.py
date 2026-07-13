"""Tests for WebDataset packaging of video2parts bundles."""

import tarfile

import pandas as pd
import pytest

from signdata.config.schema import Config
from signdata.datasets.how2sign import How2SignDataset
from signdata.output.webdataset import WebDatasetOutput
from signdata.pipeline.context import PipelineContext


def test_webdataset_rejects_unsupported_processor(tmp_path):
    cfg = Config(
        dataset={"name": "how2sign"},
        processing={"enabled": False, "processor": "video2compression"},
    )
    ctx = PipelineContext(
        config=cfg,
        dataset=How2SignDataset(),
        webdataset_dir=tmp_path / "wds",
    )

    with pytest.raises(ValueError, match="video2compression"):
        WebDatasetOutput(cfg).run(ctx)


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
        output={"enabled": True},
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
        members = tar.getmembers()
        names = {member.name for member in members}

    assert names == {
        f"{sample_id}.face_mp4",
        f"{sample_id}.left_hand_mp4",
        f"{sample_id}.right_hand_mp4",
        f"{sample_id}.pose_npz",
        f"{sample_id}.json",
        f"{sample_id}.txt",
    }
    assert {member.mtime for member in members} == {0}
    assert ctx.stats["output.webdataset"]["written"] == 1


def test_webdataset_prefers_normalized_pose_bytes(tmp_path):
    sample_id = "sample_001"
    output_dir = tmp_path / "output" / "run"
    (output_dir / "raw").mkdir(parents=True)
    (output_dir / "normalized").mkdir()
    (output_dir / "raw" / f"{sample_id}.npy").write_bytes(b"raw")
    (output_dir / "normalized" / f"{sample_id}.npy").write_bytes(b"normalized")

    cfg = Config(
        dataset={"name": "how2sign"},
        processing={"enabled": False, "processor": "video2pose"},
        post_processing={"enabled": False},
        output={"enabled": True},
    )
    ctx = PipelineContext(
        config=cfg,
        dataset=How2SignDataset(),
        manifest_df=pd.DataFrame({
            "VIDEO_ID": ["video_001"],
            "SAMPLE_ID": [sample_id],
            "START": [0.0],
            "END": [1.0],
        }),
        output_dir=output_dir,
        webdataset_dir=tmp_path / "wds" / "run",
    )

    WebDatasetOutput(cfg).run(ctx)

    with tarfile.open(ctx.webdataset_dir / "shard-000000.tar") as tar:
        assert tar.extractfile(f"{sample_id}.npy").read() == b"normalized"
