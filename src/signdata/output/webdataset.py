"""WebDataset output: package outputs into tar shards."""

import io
import json
import os
import tarfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseOutput
from ..registry import register_output
from ..utils.manifest import read_manifest, get_timing_columns


class _ShardWriter:
    """Minimal shard writer using Python's tarfile module.

    Produces tar files that are fully compatible with webdataset readers.
    """

    def __init__(
        self,
        output_dir: str,
        max_count: int = 10_000,
        max_size: Optional[int] = None,
    ):
        self.output_dir = Path(output_dir)
        self.max_count = max_count
        self.max_size = max_size

        self._shard_idx = 0
        self._count = 0
        self._size = 0
        self._tar: Optional[tarfile.TarFile] = None
        self._open_shard()

    def _shard_path(self) -> Path:
        return self.output_dir / f"shard-{self._shard_idx:06d}.tar"

    def _open_shard(self):
        if self._tar is not None:
            self._tar.close()
        self._tar = tarfile.open(str(self._shard_path()), "w")
        self._count = 0
        self._size = 0

    def _next_shard(self):
        self._tar.close()
        self._shard_idx += 1
        self._open_shard()

    def _add_bytes(self, name: str, data: bytes):
        buf = io.BytesIO(data)
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        info.mtime = int(time.time())
        self._tar.addfile(info, buf)
        self._size += len(data)

    def write(self, sample: dict):
        key = sample["__key__"]
        if self._count >= self.max_count:
            self._next_shard()
        elif self.max_size and self._size >= self.max_size:
            self._next_shard()

        for ext, value in sample.items():
            if ext == "__key__":
                continue
            if isinstance(value, str):
                value = value.encode("utf-8")
            self._add_bytes(f"{key}.{ext}", value)
        self._count += 1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self._tar is not None:
            self._tar.close()
            self._tar = None

    @property
    def shard_count(self) -> int:
        return self._shard_idx + 1


@register_output("webdataset")
class WebDatasetOutput(BaseOutput):
    """Package pipeline outputs into WebDataset tar shards."""

    name = "webdataset"

    def run(self, context):
        cfg = self.config
        output_dir = str(context.webdataset_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Read manifest
        df = context.manifest_df
        if df is None and context.manifest_path:
            df = read_manifest(str(context.manifest_path), normalize_columns=True)

        if df is None or df.empty:
            self.logger.warning("No manifest data, nothing to package.")
            context.stats["output.webdataset"] = {"written": 0}
            return context

        start_col, end_col = get_timing_columns(df)
        sentence_col = "TEXT" if "TEXT" in df.columns else None

        output_config = cfg.output.config
        max_count = output_config.get("max_shard_count", 10000)
        max_size = output_config.get("max_shard_size")

        # Determine processor mode for content packaging
        processor = cfg.processing.processor

        # Source directories
        raw_dir = context.output_dir / "raw" if context.output_dir else None
        norm_dir = context.output_dir / "normalized" if context.output_dir else None

        written = skipped = 0

        with _ShardWriter(output_dir, max_count=max_count, max_size=max_size) as sink:
            for _, row in df.iterrows():
                sample_id = row.SAMPLE_ID
                video_id = row.VIDEO_ID

                caption = ""
                if sentence_col and pd.notna(row.get(sentence_col)):
                    caption = str(row[sentence_col])

                meta = {
                    "video_id": str(video_id),
                    "sample_id": str(sample_id),
                    "start": float(row[start_col]),
                    "end": float(row[end_col]),
                    "processor": processor,
                }

                sample = {"__key__": str(sample_id)}

                if processor == "video2pose":
                    # Prefer normalized, fallback to raw
                    npy_path = None
                    if norm_dir and (norm_dir / f"{sample_id}.npy").exists():
                        npy_path = str(norm_dir / f"{sample_id}.npy")
                    elif raw_dir and (raw_dir / f"{sample_id}.npy").exists():
                        npy_path = str(raw_dir / f"{sample_id}.npy")

                    if not npy_path:
                        skipped += 1
                        continue

                    arr = np.load(npy_path)
                    buf = io.BytesIO()
                    np.save(buf, arr)
                    sample["npy"] = buf.getvalue()

                elif processor == "video2crop":
                    clip_path = None
                    if raw_dir:
                        clip_path = str(raw_dir / f"{sample_id}.mp4")
                    if not clip_path or not os.path.exists(clip_path):
                        skipped += 1
                        continue
                    with open(clip_path, "rb") as f:
                        sample["mp4"] = f.read()

                elif processor == "video2parts":
                    if not raw_dir:
                        skipped += 1
                        continue
                    sample_dir = raw_dir / str(sample_id)
                    paths = {
                        "face_mp4": sample_dir / "face.mp4",
                        "left_hand_mp4": sample_dir / "left_hand.mp4",
                        "right_hand_mp4": sample_dir / "right_hand.mp4",
                        "pose_npz": sample_dir / "pose.npz",
                        "json": sample_dir / "meta.json",
                    }
                    if not all(path.exists() for path in paths.values()):
                        skipped += 1
                        continue
                    for ext, path in paths.items():
                        with open(path, "rb") as f:
                            sample[ext] = f.read()

                sample["txt"] = caption
                if "json" not in sample:
                    sample["json"] = json.dumps(meta).encode("utf-8")

                sink.write(sample)
                written += 1

        context.stats["output.webdataset"] = {
            "written": written,
            "skipped": skipped,
            "shards": sink.shard_count,
            "output_dir": output_dir,
        }
        self.logger.info(
            "WebDataset: wrote %d samples in %d shard(s), skipped %d → %s",
            written, sink.shard_count, skipped, output_dir,
        )
        return context
