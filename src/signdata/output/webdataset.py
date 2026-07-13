"""WebDataset output: package outputs into tar shards."""

import io
import json
import logging
import tarfile
from pathlib import Path
from typing import Optional

from ..utils.manifest import get_timing_columns, read_manifest, row_value


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
        self._tar = tarfile.open(self._shard_path(), "w")
        self._count = 0
        self._size = 0

    def _next_shard(self):
        self._shard_idx += 1
        self._open_shard()

    def _add_bytes(self, name: str, data: bytes):
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        info.mtime = 0
        self._tar.addfile(info, io.BytesIO(data))
        self._size += len(data)

    def write(self, sample: dict):
        key = sample["__key__"]
        if self._count >= self.max_count or (
            self.max_size and self._size >= self.max_size
        ):
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


class WebDatasetOutput:
    """Package pipeline outputs into WebDataset tar shards."""

    logger = logging.getLogger("signdata.output.webdataset")

    def __init__(self, config):
        self.config = config

    def run(self, context):
        cfg = self.config
        output_dir = context.webdataset_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        processor = cfg.processing.processor
        if processor not in {"video2pose", "video2crop", "video2parts"}:
            raise ValueError(
                f"WebDataset output does not support processor {processor!r}"
            )

        # Read manifest
        df = context.manifest_df
        if df is None and context.manifest_path:
            df = read_manifest(str(context.manifest_path))

        if df is None or df.empty:
            self.logger.warning("No manifest data, nothing to package.")
            context.stats["output.webdataset"] = {"written": 0}
            return context

        start_col, end_col = get_timing_columns(df)
        output_config = cfg.output.config
        max_count = output_config.get("max_shard_count", 10000)
        max_size = output_config.get("max_shard_size")

        raw_dir = context.output_dir / "raw"
        norm_dir = context.output_dir / "normalized"

        written = skipped = 0

        with _ShardWriter(output_dir, max_count=max_count, max_size=max_size) as sink:
            for _, row in df.iterrows():
                sample_id = row.SAMPLE_ID
                video_id = row.VIDEO_ID

                caption = row_value(row, "TEXT")

                meta = {
                    "video_id": str(video_id),
                    "sample_id": str(sample_id),
                    "start": float(row[start_col]),
                    "end": float(row[end_col]),
                    "processor": processor,
                }

                sample = {"__key__": str(sample_id)}

                if processor == "video2pose":
                    npy_path = next(
                        (
                            path
                            for path in (
                                norm_dir / f"{sample_id}.npy",
                                raw_dir / f"{sample_id}.npy",
                            )
                            if path.exists()
                        ),
                        None,
                    )

                    if not npy_path:
                        skipped += 1
                        continue

                    sample["npy"] = npy_path.read_bytes()

                elif processor == "video2crop":
                    clip_path = raw_dir / f"{sample_id}.mp4"
                    if not clip_path.exists():
                        skipped += 1
                        continue
                    sample["mp4"] = clip_path.read_bytes()

                elif processor == "video2parts":
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
                        sample[ext] = path.read_bytes()

                sample["txt"] = caption
                if "json" not in sample:
                    sample["json"] = json.dumps(meta)

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
