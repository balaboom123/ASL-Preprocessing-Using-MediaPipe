#!/usr/bin/env python3
"""Benchmark NVENC codecs/settings and write reproducible size records.

The benchmark deliberately re-encodes every source video directly, including
H.264 sources with h264_nvenc.  This is different from the production
video2compression mirror, which normally passes through an already-matching
codec family. Use ``--gpu`` to select the physical NVIDIA GPU used by
both the optional hardware decoder and NVENC encoder. Use ``--only`` to run a
profile subset.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from signdata.config.schema import VideoProcessingConfig
from signdata.processors.video.ffmpeg import (
    _encoder_args,
    probe_media,
    transcode,
)

LOGGER = logging.getLogger("nvenc_benchmark")
DEFAULT_SOURCE_ROOT = Path("/home/gorden/dataset/YouTube-ASL/test")
DEFAULT_OUTPUT_ROOT = Path(
    "/home/gorden/dataset/YouTube-ASL/test_output/nvenc_benchmark_gpu0"
)
GPU_INDEX = 0  # Overridden from --gpu in main().

# Unique profiles from the cross-codec comparison. AQ12 checks whether stronger
# adaptive quantization helps moving hands; H.264 remains a compatibility
# reference. Candidate names intentionally omit the GPU so a suite can run on
# any selected device while the reports record the actual physical GPU index.
CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "name": "av1_cq26_p6_r080_aq08",
        "label": "AV1 CQ26 / p6 / 0.80 / AQ8",
        "codec": "av1_nvenc",
        "crf": 26,
        "preset": "p6",
        "aq_strength": 8,
        "max_bitrate_ratio": 0.80,
        "target_reduction_percent": 25.0,
    },
    {
        "name": "av1_cq28_p6_r075_aq08",
        "label": "AV1 CQ28 / p6 / 0.75 / AQ8",
        "codec": "av1_nvenc",
        "crf": 28,
        "preset": "p6",
        "aq_strength": 8,
        "max_bitrate_ratio": 0.75,
        "target_reduction_percent": 30.0,
    },
    {
        "name": "av1_cq28_p7_r075_aq08",
        "label": "AV1 CQ28 / p7 / 0.75 / AQ8",
        "codec": "av1_nvenc",
        "crf": 28,
        "preset": "p7",
        "aq_strength": 8,
        "max_bitrate_ratio": 0.75,
        "target_reduction_percent": 30.0,
    },
    {
        "name": "av1_cq30_p7_r070_aq08",
        "label": "AV1 CQ30 / p7 / 0.70 / AQ8",
        "codec": "av1_nvenc",
        "crf": 30,
        "preset": "p7",
        "aq_strength": 8,
        "max_bitrate_ratio": 0.70,
        "target_reduction_percent": 35.0,
    },
    {
        "name": "av1_cq30_p7_r070_aq12",
        "label": "AV1 CQ30 / p7 / 0.70 / AQ12",
        "codec": "av1_nvenc",
        "crf": 30,
        "preset": "p7",
        "aq_strength": 12,
        "max_bitrate_ratio": 0.70,
        "target_reduction_percent": 35.0,
    },
    {
        "name": "av1_cq32_p7_r060_aq08",
        "label": "AV1 CQ32 / p7 / 0.60 / AQ8",
        "codec": "av1_nvenc",
        "crf": 32,
        "preset": "p7",
        "aq_strength": 8,
        "max_bitrate_ratio": 0.60,
        "target_reduction_percent": 50.0,
    },
    {
        "name": "av1_cq34_p7_r052_aq08",
        "label": "AV1 CQ34 / p7 / 0.52 / AQ8",
        "codec": "av1_nvenc",
        "crf": 34,
        "preset": "p7",
        "aq_strength": 8,
        "max_bitrate_ratio": 0.52,
        "target_reduction_percent": 55.0,
    },
    {
        "name": "av1_cq36_p7_r045_aq08",
        "label": "AV1 CQ36 / p7 / 0.45 / AQ8",
        "codec": "av1_nvenc",
        "crf": 36,
        "preset": "p7",
        "aq_strength": 8,
        "max_bitrate_ratio": 0.45,
        "target_reduction_percent": 60.0,
    },
    {
        "name": "hevc_cq26_p6_r080_aq08",
        "label": "HEVC CQ26 / p6 / 0.80 / AQ8",
        "codec": "hevc_nvenc",
        "crf": 26,
        "preset": "p6",
        "aq_strength": 8,
        "max_bitrate_ratio": 0.80,
        "target_reduction_percent": 25.0,
    },
    {
        "name": "hevc_cq28_p7_r075_aq08",
        "label": "HEVC CQ28 / p7 / 0.75 / AQ8",
        "codec": "hevc_nvenc",
        "crf": 28,
        "preset": "p7",
        "aq_strength": 8,
        "max_bitrate_ratio": 0.75,
        "target_reduction_percent": 30.0,
    },
    {
        "name": "hevc_cq30_p7_r070_aq08",
        "label": "HEVC CQ30 / p7 / 0.70 / AQ8",
        "codec": "hevc_nvenc",
        "crf": 30,
        "preset": "p7",
        "aq_strength": 8,
        "max_bitrate_ratio": 0.70,
        "target_reduction_percent": 35.0,
    },
    {
        "name": "hevc_cq30_p7_r070_aq12",
        "label": "HEVC CQ30 / p7 / 0.70 / AQ12",
        "codec": "hevc_nvenc",
        "crf": 30,
        "preset": "p7",
        "aq_strength": 12,
        "max_bitrate_ratio": 0.70,
        "target_reduction_percent": 35.0,
    },
    {
        "name": "hevc_cq32_p7_r060_aq08",
        "label": "HEVC CQ32 / p7 / 0.60 / AQ8",
        "codec": "hevc_nvenc",
        "crf": 32,
        "preset": "p7",
        "aq_strength": 8,
        "max_bitrate_ratio": 0.60,
        "target_reduction_percent": 50.0,
    },
    {
        "name": "hevc_cq34_p7_r052_aq08",
        "label": "HEVC CQ34 / p7 / 0.52 / AQ8",
        "codec": "hevc_nvenc",
        "crf": 34,
        "preset": "p7",
        "aq_strength": 8,
        "max_bitrate_ratio": 0.52,
        "target_reduction_percent": 55.0,
    },
    {
        "name": "hevc_cq36_p7_r045_aq08",
        "label": "HEVC CQ36 / p7 / 0.45 / AQ8",
        "codec": "hevc_nvenc",
        "crf": 36,
        "preset": "p7",
        "aq_strength": 8,
        "max_bitrate_ratio": 0.45,
        "target_reduction_percent": 60.0,
    },
    {
        "name": "h264_cq26_p7_r080_aq08",
        "label": "H.264 CQ26 / p7 / 0.80 / AQ8",
        "codec": "h264_nvenc",
        "crf": 26,
        "preset": "p7",
        "aq_strength": 8,
        "max_bitrate_ratio": 0.80,
        "target_reduction_percent": 25.0,
    },
    {
        "name": "h264_cq30_p7_r070_aq08",
        "label": "H.264 CQ30 / p7 / 0.70 / AQ8",
        "codec": "h264_nvenc",
        "crf": 30,
        "preset": "p7",
        "aq_strength": 8,
        "max_bitrate_ratio": 0.70,
        "target_reduction_percent": 35.0,
    },
)


def _config(candidate: dict[str, Any]) -> VideoProcessingConfig:
    return VideoProcessingConfig(
        codec=candidate["codec"],
        crf=candidate["crf"],
        preset=candidate["preset"],
        aq_strength=candidate["aq_strength"],
        max_bitrate_ratio=candidate["max_bitrate_ratio"],
        nvenc_gpu=GPU_INDEX,
    )


def _command(
    source_path: Path,
    output_path: Path,
    config: VideoProcessingConfig,
    max_bitrate_bps: int | None,
    source_pix_fmt: str,
) -> list[str]:
    return [
        "ffmpeg", "-y", "-hide_banner", "-nostats",
        "-hwaccel", "auto", "-hwaccel_device", str(GPU_INDEX),
        "-i", str(source_path),
        *_encoder_args(config, max_bitrate_bps, source_pix_fmt),
        "-fps_mode", "passthrough", "-v", "warning", str(output_path),
    ]


def _encode_one(
    candidate: dict[str, Any],
    source_path: Path,
    output_root: Path,
    force_all: bool,
) -> dict[str, Any]:
    name = candidate["name"]
    output_path = output_root / name / "compressed" / source_path.name
    sidecar_path = output_path.with_suffix(".mp4.compression.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source = probe_media(str(source_path))
    result: dict[str, Any] = {
        "candidate": name,
        "video": source_path.name,
        "source_path": str(source_path),
        "output_path": str(output_path),
        "status": "error",
    }
    if source is None or source.duration <= 0 or source.width <= 1:
        result["error"] = "source probe failed"
        return result

    max_bitrate_bps = (
        round(source.bitrate_bps * candidate["max_bitrate_ratio"])
        if candidate["max_bitrate_ratio"] and source.bitrate_bps > 0
        else None
    )
    config = _config(candidate)
    result.update({
        "source_bytes": source_path.stat().st_size,
        "source_codec": source.codec,
        "source_pix_fmt": source.pix_fmt,
        "source_bitrate_bps": source.bitrate_bps,
        "width": source.width,
        "height": source.height,
        "duration": source.duration,
        "max_bitrate_bps": max_bitrate_bps,
        "ffmpeg_command": _command(
            source_path, output_path, config, max_bitrate_bps, source.pix_fmt
        ),
    })

    started = time.monotonic()
    if not (output_path.exists() and sidecar_path.exists()) or force_all:
        temporary_path = output_path.with_name(
            f".{output_path.stem}.benchmark-tmp{output_path.suffix}"
        )
        temporary_path.unlink(missing_ok=True)
        try:
            ok = transcode(
                str(source_path), config, str(temporary_path),
                max_bitrate_bps=max_bitrate_bps,
                source_pix_fmt=source.pix_fmt,
            )
            if not ok or not temporary_path.exists():
                result["error"] = "ffmpeg encode failed"
                return result
            temporary_path.replace(output_path)
            result["status"] = "encoded"
        except Exception as exc:
            result["error"] = str(exc)
            return result
        finally:
            temporary_path.unlink(missing_ok=True)
    else:
        result["status"] = "reused"

    output = probe_media(str(output_path))
    if output is None:
        result["error"] = "output probe failed"
        return result

    output_bytes = output_path.stat().st_size
    result.update({
        "output_bytes": output_bytes,
        "output_codec": output.codec,
        "output_pix_fmt": output.pix_fmt,
        "output_bitrate_bps": output.bitrate_bps,
        "output_duration": output.duration,
        "output_width": output.width,
        "output_height": output.height,
        "reduction_percent": 100.0 * (1.0 - output_bytes / result["source_bytes"]),
        "wall_seconds": time.monotonic() - started,
    })
    sidecar_path.write_text(
        json.dumps({
            "format": "signdata.nvenc_benchmark.v1",
            "candidate": candidate,
            "nvenc_gpu": GPU_INDEX,
            "hwaccel_device": GPU_INDEX,
            "source": result,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _aggregate(candidate: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [item for item in results if "output_bytes" in item]
    source_bytes = sum(item["source_bytes"] for item in valid)
    output_bytes = sum(item["output_bytes"] for item in valid)
    return {
        "name": candidate["name"],
        "label": candidate["label"],
        "codec": candidate["codec"],
        "crf": candidate["crf"],
        "preset": candidate["preset"],
        "aq_strength": candidate["aq_strength"],
        "max_bitrate_ratio": candidate["max_bitrate_ratio"],
        "nvenc_gpu": GPU_INDEX,
        "target_reduction_percent": candidate["target_reduction_percent"],
        "videos_total": len(results),
        "videos_successful": len(valid),
        "source_bytes": source_bytes,
        "output_bytes": output_bytes,
        "reduction_percent": (
            100.0 * (1.0 - output_bytes / source_bytes) if source_bytes else None
        ),
        "files": results,
    }


def _conditions(
    output_root: Path,
    source_root: Path,
    selected: list[dict[str, Any]],
) -> dict[str, Any]:
    conditions: dict[str, Any] = {
        "original": {
            "label": "Original",
            "video_dir": str(source_root),
            "target_reduction_percent": 0.0,
        },
        "small": {
            "label": "Previous Small (25%)",
            "video_dir": "/home/gorden/dataset/YouTube-ASL/test_output/compressed-small-25/compressed",
            "target_reduction_percent": 25.0,
        },
        "base": {
            "label": "Previous Base (50%)",
            "video_dir": "/home/gorden/dataset/YouTube-ASL/test_output/compressed-base-50/compressed",
            "target_reduction_percent": 50.0,
        },
        "large": {
            "label": "Previous Large (60%)",
            "video_dir": "/home/gorden/dataset/YouTube-ASL/test_output/compressed-large-60/compressed",
            "target_reduction_percent": 60.0,
        },
    }
    for candidate in selected:
        conditions[candidate["name"]] = {
            "label": candidate["label"],
            "video_dir": str(output_root / candidate["name"] / "compressed"),
            "target_reduction_percent": candidate["target_reduction_percent"],
            "parameters": {
                **candidate,
                "nvenc_gpu": GPU_INDEX,
                "hwaccel_device": GPU_INDEX,
                "rate_control": "vbr + cq + b:v=0",
                "audio": "dropped (-an)",
                "resize": None,
                "fps_mode": "passthrough",
            },
        }
    return {"conditions": conditions}


def _write_reports(
    output_root: Path,
    selected: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    conditions: dict[str, Any],
    source_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "signdata.nvenc_benchmark.v1",
        "source_root": str(source_root),
        "gpu": {
            "index": GPU_INDEX,
            "selection": f"physical GPU {GPU_INDEX}",
        },
        "candidates": aggregates,
        "conditions_file": str(output_root / "mediapipe_conditions.json"),
    }
    (output_root / "compression_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_root / "compression_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "name", "label", "codec", "crf", "preset", "aq_strength",
            "max_bitrate_ratio", "nvenc_gpu", "target_reduction_percent",
            "videos_successful", "videos_total", "source_bytes", "output_bytes",
            "reduction_percent",
        ])
        writer.writeheader()
        for item in aggregates:
            writer.writerow({key: item.get(key) for key in writer.fieldnames})
    with (output_root / "compression_files.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "candidate", "video", "status", "source_bytes", "output_bytes",
            "reduction_percent", "source_codec", "output_codec", "source_bitrate_bps",
            "output_bitrate_bps", "wall_seconds", "error",
        ])
        writer.writeheader()
        for item in aggregates:
            for result in item["files"]:
                writer.writerow({key: result.get(key) for key in writer.fieldnames})
    (output_root / "mediapipe_conditions.json").write_text(
        json.dumps(conditions, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# NVENC compression benchmark",
        "",
        f"All candidates were encoded with FFmpeg on physical **GPU {GPU_INDEX}**. Hardware decode and NVENC use the same selected device; no resize, crop, or frame-rate conversion was applied; audio was dropped.",
        "",
        f"- Source root: `{source_root}`",
        f"- Candidate count: **{len(selected)}**",
        "- Full per-video records: `compression_files.csv` and `compression_results.json`.",
        "- MediaPipe condition file: `mediapipe_conditions.json`.",
        "",
        "| Candidate | Codec | CQ | Preset | AQ | Maxrate ratio | Output bytes | Reduction |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for item in aggregates:
        reduction = "—" if item["reduction_percent"] is None else f"{item['reduction_percent']:.2f}%"
        lines.append(
            f"| {item['label']} | `{item['codec']}` | {item['crf']} | {item['preset']} | "
            f"{item['aq_strength']} | {item['max_bitrate_ratio']:.2f} | "
            f"{item['output_bytes']:,} | {reduction} |"
        )
    lines += [
        "",
        "Run MediaPipe with `scripts/evaluate_mediapipe_compression.py --conditions-json mediapipe_conditions.json --video-root /home/gorden/dataset/YouTube-ASL/test --output-root /home/gorden/dataset/YouTube-ASL/test_output/mediapipe_eval --source-manifest /home/gorden/dataset/YouTube-ASL/youtube_asl.csv --sample-rate 0.5`.",
    ]
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument(
        "--gpu", type=int, default=GPU_INDEX,
        help="physical NVIDIA GPU index",
    )
    parser.add_argument("--force-all", action="store_true")
    parser.add_argument("--only", help="comma-separated candidate names")
    return parser.parse_args()


def main() -> None:
    global GPU_INDEX
    args = _parse_args()
    if args.gpu < 0:
        raise SystemExit("--gpu must be non-negative")
    GPU_INDEX = args.gpu
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    candidates = list(CANDIDATES)
    if args.only:
        wanted = {name.strip() for name in args.only.split(",") if name.strip()}
        candidates = [candidate for candidate in candidates if candidate["name"] in wanted]
        missing = wanted.difference(candidate["name"] for candidate in candidates)
        if missing:
            raise SystemExit(f"Unknown candidate(s): {sorted(missing)}")
    source_paths = sorted(source_root.glob("*.mp4"))
    if not source_paths:
        raise SystemExit(f"No .mp4 files found in {source_root}")

    aggregates: list[dict[str, Any]] = []
    for candidate in candidates:
        LOGGER.info("Starting %s (%s)", candidate["name"], candidate["label"])
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            results = list(pool.map(
                lambda path: _encode_one(candidate, path, output_root, args.force_all),
                source_paths,
            ))
        aggregate = _aggregate(candidate, results)
        aggregate["wall_seconds"] = time.monotonic() - started
        aggregates.append(aggregate)
        LOGGER.info(
            "Finished %s: %d/%d files, %.2f%% reduction in %.1f minutes",
            candidate["name"], aggregate["videos_successful"], aggregate["videos_total"],
            aggregate["reduction_percent"] or 0.0, aggregate["wall_seconds"] / 60.0,
        )
        _write_reports(
            output_root, candidates, aggregates,
            _conditions(output_root, source_root, candidates), source_root,
        )

    LOGGER.info("Wrote benchmark reports to %s", output_root)


if __name__ == "__main__":
    main()
