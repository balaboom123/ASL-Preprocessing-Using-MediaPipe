"""video2compression processor: re-encode videos smaller, pixel-for-pixel.

Compression here is purely a codec change. Geometry, frame cadence and colour
are left exactly as the source has them, because every downstream consumer —
video2pose, video2crop, video2parts — re-derives its own regions from the
stored file. Cropping or rescaling at this stage would bake one consumer's
assumptions into the archive and silently degrade the others.

The output directory is a *complete* mirror of the source: videos that shrank
are written re-encoded as .mp4 (with a .compression.json sidecar), and videos
that did not — already an efficient codec, no worthwhile saving, or a
read/encode error — are hardlinked through untouched, keeping their own
container extension because the bytes are unchanged. A mirror of .webm sources
is therefore a mix of .mp4 and .webm; `resolve_video_path` retries on the stem
so a manifest written against `videos/` still resolves here. So `compressed/`
is a drop-in replacement for `videos/`: verify it, then repoint paths.videos at
it or delete the originals yourself. This processor never deletes anything.
"""

import json
import os
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .base import BaseProcessor
from .video.ffmpeg import EncoderOptionError, VideoInfo, probe_media, transcode
from ..config.schema import VideoProcessingConfig
from ..registry import register_processor
from ..utils.manifest import resolve_video_path, row_value


# ffprobe codec_name, plus the encoder names and container tags that map to it.
_CODEC_FAMILIES = {
    "av1": ("av1_nvenc", "libsvtav1", "libaom-av1", "av01"),
    "hevc": ("hevc_nvenc", "libx265", "h265"),
    "h264": ("h264_nvenc", "libx264", "avc1"),
    "vp9": ("libvpx-vp9", "vp09"),
}


def _codec_family(name: str) -> str:
    """Normalise an ffprobe codec name or ffmpeg encoder name to a family."""
    for family, aliases in _CODEC_FAMILIES.items():
        if name == family or name in aliases:
            return family
    return name


def _get_output_relpath(row) -> Path:
    """Choose a stable video-level output path from a manifest row."""
    rel_path = row_value(row, "REL_PATH")
    if rel_path:
        return Path(rel_path).with_suffix(".mp4")

    stem = row_value(row, "VIDEO_NAME") or row_value(row, "VIDEO_ID")
    if stem:
        return Path(f"{stem}.mp4")

    raise ValueError(
        "video2compression requires one of REL_PATH, VIDEO_NAME, or VIDEO_ID "
        "to derive a video-level output name."
    )


def _build_video_tasks(df, video_dir: str) -> list[tuple[str, Path]]:
    """Deduplicate segment-level manifests into one task per source video."""
    tasks: list[tuple[str, Path]] = []
    seen_video_paths = set()

    for _, row in df.iterrows():
        video_path = str(resolve_video_path(row, video_dir))
        if video_path in seen_video_paths:
            continue
        seen_video_paths.add(video_path)
        tasks.append((video_path, _get_output_relpath(row)))

    return tasks


def _temporary_output_path(output_path: Path) -> Path:
    return output_path.with_name(
        f".{output_path.stem}.compression-tmp{output_path.suffix}"
    )


def _metadata_path(output_path: Path) -> Path:
    return output_path.with_suffix(f"{output_path.suffix}.compression.json")


def _passthrough_path(encoded_path: Path, video_path: str) -> Path:
    """Mirror path for a byte-for-byte passthrough of ``video_path``.

    A passthrough is hardlinked, not remuxed, so it has to keep the source's
    own container extension. Writing Matroska or WebM bytes into a ``.mp4``
    name produces files that strict readers reject and that misreport the
    archive's contents to anyone auditing it.
    """
    suffix = Path(video_path).suffix
    if not suffix:
        return encoded_path
    return encoded_path.with_suffix(suffix)


def _remove_output(path: Path) -> None:
    """Drop a mirror entry and its sidecar."""
    path.unlink(missing_ok=True)
    _metadata_path(path).unlink(missing_ok=True)


def _write_metadata(
    output_path: Path,
    source_path: str,
    source: VideoInfo,
    output: VideoInfo,
    config: VideoProcessingConfig,
    source_bytes: int,
    output_bytes: int,
) -> None:
    """Record what was re-encoded so a later audit can verify the archive."""
    payload = {
        "format": "signdata.compression.v2",
        "source_path": source_path,
        "source_codec": source.codec,
        "source_pix_fmt": source.pix_fmt,
        "source_bytes": source_bytes,
        "source_bitrate_bps": source.bitrate_bps,
        "output_codec": output.codec,
        "output_pix_fmt": output.pix_fmt,
        "output_bytes": output_bytes,
        "output_bitrate_bps": output.bitrate_bps,
        # Recorded once because compression must not change them.
        "width": output.width,
        "height": output.height,
        "duration": output.duration,
        "encoder": config.codec,
        "crf": config.crf,
        "preset": config.preset,
        "aq_strength": config.aq_strength,
        "nvenc_gpu": config.nvenc_gpu,
    }
    metadata_path = _metadata_path(output_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


@register_processor("video2compression")
class Video2CompressionProcessor(BaseProcessor):
    """Video → smaller video, with identical geometry and frame cadence."""

    name = "video2compression"

    def run(self, context):
        cfg = self.config.processing
        video_config = cfg.video_config
        df = context.manifest_df
        if df is None:
            self.logger.warning("No manifest loaded, nothing to process.")
            context.stats["processing"] = {"total": 0}
            return context

        if video_config.resize:
            raise ValueError(
                "video2compression must not rescale. Hand ROIs are re-cropped "
                "from the stored frame at full resolution by mediapipe and "
                "video2parts, so lost pixels here cannot be recovered "
                "downstream. Unset processing.video_config.resize."
            )

        video_dir = str(context.videos_dir) if context.videos_dir else ""
        source_rows = len(df)
        tasks = _build_video_tasks(df, video_dir)
        total = len(tasks)

        if source_rows != total:
            self.logger.info(
                "video2compression deduplicated %d manifest rows to %d "
                "unique videos",
                source_rows,
                total,
            )

        stats = {
            "total": total,
            "source_rows": source_rows,
            "processed": 0,
            "skipped": 0,
            "already_efficient": 0,
            "not_smaller": 0,
            "errors": 0,
        }
        if total == 0:
            context.stats["processing"] = stats
            return context

        output_dir = context.output_dir / "compressed"
        output_dir.mkdir(parents=True, exist_ok=True)
        target_family = _codec_family(video_config.codec)

        def compress(task: tuple[str, Path]) -> str:
            video_path, output_relpath = task
            encoded_path = output_dir / output_relpath
            passthrough_path = _passthrough_path(encoded_path, video_path)
            output_label = str(output_relpath)

            # Either name is a finished mirror entry for this video, so a
            # resumed run must recognise both or it re-does every passthrough.
            if not context.force_all and (
                encoded_path.exists() or passthrough_path.exists()
            ):
                return "skipped"

            temporary_path = _temporary_output_path(encoded_path)
            temporary_path.unlink(missing_ok=True)
            try:
                status = self._compress_one(
                    video_path,
                    encoded_path,
                    temporary_path,
                    output_label,
                    video_config,
                    target_family,
                )
            except EncoderOptionError:
                raise
            except Exception as exc:
                self.logger.error("Error processing %s: %s", output_label, exc)
                status = "errors"
            finally:
                temporary_path.unlink(missing_ok=True)

            # The two names collide whenever the source is already .mp4; only
            # a differing suffix can leave a stale entry from an earlier run,
            # and dropping it keeps one mirror entry per source video.
            if status == "processed":
                if passthrough_path != encoded_path:
                    _remove_output(passthrough_path)
            else:
                # Keep the mirror complete: whatever we did not re-encode is
                # passed through as the untouched original, so videos/ can be
                # swapped out or deleted wholesale after review.
                self._backfill_original(
                    video_path, passthrough_path, output_label
                )
                if passthrough_path != encoded_path:
                    _remove_output(encoded_path)
            return status

        # ffmpeg does the real work in a subprocess, so threads are enough to
        # keep several encodes in flight; the GIL is not the bottleneck here.
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
            outcomes = Counter(pool.map(compress, tasks))

        stats.update(
            processed=outcomes["processed"],
            already_efficient=outcomes["already_efficient"],
            not_smaller=outcomes["not_smaller"],
            errors=outcomes["errors"],
            # Anything kept as-is left the source file in place.
            skipped=(
                outcomes["skipped"]
                + outcomes["already_efficient"]
                + outcomes["not_smaller"]
            ),
        )
        context.stats["processing"] = stats
        self.logger.info(
            "video2compression: processed=%d skipped=%d already_efficient=%d "
            "not_smaller=%d errors=%d total=%d source_rows=%d",
            stats["processed"],
            stats["skipped"],
            stats["already_efficient"],
            stats["not_smaller"],
            stats["errors"],
            total,
            source_rows,
        )
        self.logger.info(
            "Mirror ready at %s (%d re-encoded, the rest passed through as "
            "originals). Verify it, then repoint paths.videos or delete the "
            "originals; %d video(s) errored — each is passed through as the "
            "original unless its source file was missing, in which case it is "
            "absent from the mirror.",
            output_dir,
            stats["processed"],
            stats["errors"],
        )
        return context

    def _backfill_original(
        self, video_path: str, output_path: Path, output_label: str
    ) -> None:
        """Pass an un-shrunk source through to the mirror, untouched.

        Hardlink where the filesystem allows it (no extra bytes, and it
        survives deleting the original); fall back to a copy across
        filesystems. A logged failure here means the mirror is missing this
        file, so the originals must not be deleted until it is resolved.
        """
        source_path = Path(video_path)
        if not source_path.exists():
            return
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.unlink(missing_ok=True)
            # A sidecar left over from an earlier run would now describe a
            # re-encode that no longer exists at this path.
            _metadata_path(output_path).unlink(missing_ok=True)
            try:
                os.link(source_path, output_path)
            except OSError:
                shutil.copy2(source_path, output_path)
        except OSError as exc:
            self.logger.error(
                "Mirror incomplete, could not pass through %s: %s",
                output_label,
                exc,
            )

    def _compress_one(
        self,
        video_path: str,
        output_path: Path,
        temporary_path: Path,
        output_label: str,
        video_config: VideoProcessingConfig,
        target_family: str,
    ) -> str:
        source_path = Path(video_path)
        if not source_path.exists():
            self.logger.warning("Video not found: %s", video_path)
            return "errors"

        source = probe_media(video_path)
        if source is None or source.duration <= 0 or source.width <= 1:
            self.logger.warning(
                "Cannot read video metadata (is ffprobe installed?): %s",
                video_path,
            )
            return "errors"

        # ponytail: re-encoding a stream that is already in the target family
        # only adds generational loss. Assumes such sources are not bloated,
        # which holds for the av01/vp9 renditions the download step prefers;
        # add a bitrate ceiling here if that stops being true.
        if _codec_family(source.codec) == target_family:
            self.logger.info(
                "Keeping source, already %s: %s", source.codec, output_label
            )
            return "already_efficient"

        source_bytes = source_path.stat().st_size
        max_bitrate_bps = None
        if video_config.max_bitrate_ratio and source.bitrate_bps > 0:
            max_bitrate_bps = round(
                source.bitrate_bps * video_config.max_bitrate_ratio
            )

        ok = transcode(
            video_path,
            video_config,
            str(temporary_path),
            max_bitrate_bps=max_bitrate_bps,
            source_pix_fmt=source.pix_fmt,
        )
        if not ok or not temporary_path.exists():
            return "errors"

        output = probe_media(str(temporary_path))
        if output is None:
            self.logger.warning(
                "Encoded output is not readable: %s", output_label
            )
            return "errors"

        # The whole contract of this processor: same pixels, fewer bytes.
        if (output.width, output.height) != (source.width, source.height):
            self.logger.warning(
                "Encoded dimensions changed for %s: source=%s output=%s",
                output_label,
                (source.width, source.height),
                (output.width, output.height),
            )
            return "errors"

        # Duration rather than frame count: YouTube sources are frequently
        # VFR, so a container frame count is an estimate and rejects good
        # encodes. passthrough preserves timestamps, so a good encode lands
        # well inside 0.2s; a looser floor would let a truncated tail slip
        # through on the short (~2s) isolated-sign clips in WLASL/MSASL.
        if abs(output.duration - source.duration) > max(
            0.2, source.duration * 0.01
        ):
            self.logger.warning(
                "Encoded duration changed for %s: source=%.3f output=%.3f",
                output_label,
                source.duration,
                output.duration,
            )
            return "errors"

        output_bytes = temporary_path.stat().st_size
        maximum_output_bytes = source_bytes * (
            1.0 - video_config.min_video_reduction_ratio
        )
        if output_bytes > maximum_output_bytes:
            self.logger.info(
                "Keeping source, compression did not meet the reduction gate: "
                "%s source=%d output=%d minimum_reduction=%.3f",
                output_label,
                source_bytes,
                output_bytes,
                video_config.min_video_reduction_ratio,
            )
            return "not_smaller"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.replace(output_path)
        _write_metadata(
            output_path,
            video_path,
            source,
            output,
            video_config,
            source_bytes,
            output_bytes,
        )
        return "processed"
