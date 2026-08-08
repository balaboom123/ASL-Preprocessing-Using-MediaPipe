#!/usr/bin/env python3
"""Evaluate compression influence through MediaPipe landmark stability.

This is intentionally a pilot evaluator for the three YouTube-ASL videos in
``/home/gorden/dataset/YouTube-ASL/test``.  It runs the repository's normal
``video2pose`` processor separately for each video directory, then compares
the resulting raw 553-keypoint MediaPipe sequences segment-by-segment.

The evaluator is resumable: by default the pipeline skips raw ``.npy`` files
that already exist, so re-running it completes an interrupted condition.
Pass ``--force-all`` when a clean re-extraction is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("mediapipe_compression_eval")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs/jobs/youtube_asl/mediapipe.yaml"
DEFAULT_SOURCE_MANIFEST = Path("/home/gorden/dataset/YouTube-ASL/youtube_asl.csv")
DEFAULT_OUTPUT_ROOT = Path("/home/gorden/dataset/YouTube-ASL/test_output/mediapipe_eval")

VIDEO_ROOT = Path("/home/gorden/dataset/YouTube-ASL/test")
CONDITIONS: dict[str, dict[str, Any]] = {
    "original": {
        "label": "Original",
        "video_dir": VIDEO_ROOT,
        "target_reduction_percent": 0.0,
    },
    "small": {
        "label": "Small (25%)",
        "video_dir": Path(
            "/home/gorden/dataset/YouTube-ASL/test_output/"
            "compressed-small-25/compressed"
        ),
        "target_reduction_percent": 25.0,
    },
    "base": {
        "label": "Base (50%)",
        "video_dir": Path(
            "/home/gorden/dataset/YouTube-ASL/test_output/"
            "compressed-base-50/compressed"
        ),
        "target_reduction_percent": 50.0,
    },
    "large": {
        "label": "Large (60%)",
        "video_dir": Path(
            "/home/gorden/dataset/YouTube-ASL/test_output/"
            "compressed-large-60/compressed"
        ),
        "target_reduction_percent": 60.0,
    },
}

VIDEO_IDS = tuple(
    path.stem for path in sorted(VIDEO_ROOT.glob("*.mp4"))
)

# MediaPipe Holistic refined output layout: 33 pose + 478 face + 21 left
# hand + 21 right hand = 553 keypoints.  Each keypoint is [x, y, z, vis].
COMPONENT_SLICES: dict[str, slice] = {
    "pose": slice(0, 33),
    "face": slice(33, 511),
    "left_hand": slice(511, 532),
    "right_hand": slice(532, 553),
}


def _canonical_manifest(source_path: Path, output_path: Path) -> pd.DataFrame:
    """Filter the source manifest to the three videos and write a TSV."""
    df = pd.read_csv(source_path, sep="\t")
    aliases = {
        "VIDEO_NAME": "VIDEO_ID",
        "SENTENCE_NAME": "SAMPLE_ID",
        "START_REALIGNED": "START",
        "END_REALIGNED": "END",
        "SENTENCE": "TEXT",
    }
    for old, new in aliases.items():
        if new not in df.columns and old in df.columns:
            df = df.rename(columns={old: new})

    required = {"VIDEO_ID", "SAMPLE_ID", "START", "END"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Manifest {source_path} is missing required columns: "
            f"{sorted(missing)}"
        )

    df = df[df["VIDEO_ID"].isin(VIDEO_IDS)].copy()
    df["START"] = pd.to_numeric(df["START"], errors="coerce")
    df["END"] = pd.to_numeric(df["END"], errors="coerce")
    df = df[df["START"].notna() & df["END"].notna()]
    df = df[df["END"] > df["START"]]
    df["VIDEO_ID"] = df["VIDEO_ID"].astype(str)
    df["SAMPLE_ID"] = df["SAMPLE_ID"].astype(str)
    df = df.sort_values(["VIDEO_ID", "START", "END", "SAMPLE_ID"])
    df = df.drop_duplicates(subset=["SAMPLE_ID"], keep="first")

    columns = ["VIDEO_ID", "SAMPLE_ID", "START", "END"]
    if "TEXT" in df.columns:
        columns.append("TEXT")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[columns].to_csv(output_path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    return df[columns].reset_index(drop=True)


def _run_condition(
    name: str,
    condition: dict[str, Any],
    manifest_path: Path,
    output_root: Path,
    sample_rate: float | None,
    force_all: bool,
) -> dict[str, Any]:
    """Run the existing Signdata MediaPipe pipeline for one condition."""
    # Import here so this helper can also be used for an analysis-only pass
    # without importing every optional dataset/processor backend at startup.
    import signdata.datasets  # noqa: F401
    import signdata.processors  # noqa: F401
    from signdata.config.loader import load_config
    from signdata.pipeline.runner import PipelineRunner

    run_output = output_root / "landmarks"
    overrides = [
        "dataset.download=false",
        "dataset.manifest=false",
        f"paths.videos={condition['video_dir']}",
        f"paths.manifest={manifest_path}",
        f"paths.output={run_output}",
        f"run_name={name}",
        f"processing.sample_rate={'null' if sample_rate is None else sample_rate}",
        # Raw landmarks are the evaluation source.  Disable normalization and
        # WebDataset packaging so the run does not create unrelated artifacts.
        "post_processing.enabled=false",
        "output.enabled=false",
    ]
    config = load_config(str(DEFAULT_CONFIG), overrides=overrides)

    LOGGER.info("Starting MediaPipe condition: %s (%s)", name, condition["video_dir"])
    started = time.monotonic()
    context = PipelineRunner(config, force_all=force_all).run()
    elapsed = time.monotonic() - started
    stats = {
        "condition": name,
        "label": condition["label"],
        "video_dir": str(condition["video_dir"]),
        "sample_rate": sample_rate,
        "force_all": force_all,
        "wall_seconds": elapsed,
        "pipeline_stats": context.stats,
    }
    stats_dir = output_root / "run_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    (stats_dir / f"{name}.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8"
    )
    LOGGER.info("Finished %s in %.1f minutes", name, elapsed / 60.0)
    return stats


def _empty_component() -> dict[str, float | int]:
    return {
        "total_frames": 0,
        "detected_frames": 0,
        "jitter_sum": 0.0,
        "jitter_count": 0,
        "error_sum": 0.0,
        "error_count": 0,
        "reference_detected_frames": 0,
        "joint_detected_frames": 0,
        "reference_landmarks": 0,
        "joint_landmarks": 0,
    }


def _new_accumulator() -> dict[str, Any]:
    return {
        "segments_manifest": 0,
        "segments_with_output": 0,
        "segments_missing_output": 0,
        "frames_extracted": 0,
        "resampled_segments": 0,
        "hand_both_detected_frames": 0,
        "components": {
            name: _empty_component()
            for name in (*COMPONENT_SLICES, "hands")
        },
    }


def _valid_landmarks(arr: np.ndarray) -> np.ndarray:
    """Return a (T, K) mask; MediaPipe missing landmarks are all zero."""
    return np.any(arr[..., :3] != 0.0, axis=-1)


def _component_points(arr: np.ndarray, name: str) -> np.ndarray:
    if name == "hands":
        return np.concatenate(
            [arr[:, COMPONENT_SLICES["left_hand"], :],
             arr[:, COMPONENT_SLICES["right_hand"], :]],
            axis=1,
        )
    return arr[:, COMPONENT_SLICES[name], :]


def _add_own_metrics(acc: dict[str, Any], arr: np.ndarray) -> None:
    """Add detection and temporal-jitter statistics for one output clip."""
    arr = np.asarray(arr)
    if arr.ndim != 3 or arr.shape[1] != 553 or arr.shape[2] != 4:
        raise ValueError(f"Expected (T, 553, 4), got {arr.shape}")
    frame_count = int(arr.shape[0])
    acc["frames_extracted"] += frame_count
    components = acc["components"]

    left_valid = _valid_landmarks(_component_points(arr, "left_hand"))
    right_valid = _valid_landmarks(_component_points(arr, "right_hand"))
    acc["hand_both_detected_frames"] += int(
        np.logical_and(left_valid.any(axis=1), right_valid.any(axis=1)).sum()
    )

    for name in components:
        points = _component_points(arr, name)
        valid = _valid_landmarks(points)
        item = components[name]
        item["total_frames"] += frame_count
        item["detected_frames"] += int(valid.any(axis=1).sum())

        if frame_count < 3:
            continue
        second_diff = points[2:, :, :3] - 2.0 * points[1:-1, :, :3]
        second_diff += points[:-2, :, :3]
        triple_valid = valid[2:] & valid[1:-1] & valid[:-2]
        distances = np.linalg.norm(second_diff, axis=-1)
        item["jitter_sum"] += float(distances[triple_valid].sum())
        item["jitter_count"] += int(triple_valid.sum())


def _align_to_reference(arr: np.ndarray, reference_frames: int) -> np.ndarray:
    """Nearest-time align a clip when codec timing yields a frame mismatch."""
    if arr.shape[0] == reference_frames:
        return arr
    if arr.shape[0] == 0:
        return arr
    positions = np.linspace(0, arr.shape[0] - 1, reference_frames)
    indices = np.rint(positions).astype(np.int64)
    return arr[indices]


def _add_pair_metrics(
    acc: dict[str, Any],
    reference: np.ndarray,
    candidate: np.ndarray,
) -> None:
    """Add candidate-vs-original landmark distances and capture rates."""
    if reference.shape[0] != candidate.shape[0]:
        acc["resampled_segments"] += 1
        candidate = _align_to_reference(candidate, reference.shape[0])
    if reference.shape[0] == 0 or candidate.shape[0] == 0:
        return

    for name in acc["components"]:
        ref_points = _component_points(reference, name)
        cur_points = _component_points(candidate, name)
        ref_valid = _valid_landmarks(ref_points)
        cur_valid = _valid_landmarks(cur_points)
        paired = ref_valid & cur_valid
        item = acc["components"][name]
        item["reference_detected_frames"] += int(ref_valid.any(axis=1).sum())
        item["joint_detected_frames"] += int(
            (ref_valid.any(axis=1) & cur_valid.any(axis=1)).sum()
        )
        item["reference_landmarks"] += int(ref_valid.sum())
        item["joint_landmarks"] += int(paired.sum())

        distance = np.linalg.norm(
            ref_points[..., :3] - cur_points[..., :3], axis=-1
        )
        item["error_sum"] += float(distance[paired].sum())
        item["error_count"] += int(paired.sum())


def _merge_accumulators(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in (
        "segments_manifest",
        "segments_with_output",
        "segments_missing_output",
        "frames_extracted",
        "resampled_segments",
        "hand_both_detected_frames",
    ):
        target[key] += source[key]
    for name in target["components"]:
        for key in target["components"][name]:
            target["components"][name][key] += source["components"][name][key]


def _finalize_accumulator(acc: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: value
        for key, value in acc.items()
        if key != "components"
    }
    components: dict[str, Any] = {}
    for name, item in acc["components"].items():
        total_frames = int(item["total_frames"])
        detected_frames = int(item["detected_frames"])
        components[name] = {
            "detection_rate": (
                detected_frames / total_frames if total_frames else None
            ),
            "detected_frames": detected_frames,
            "total_frames": total_frames,
            "temporal_jitter": (
                item["jitter_sum"] / item["jitter_count"]
                if item["jitter_count"] else None
            ),
            "jitter_samples": int(item["jitter_count"]),
            "landmark_error_3d": (
                item["error_sum"] / item["error_count"]
                if item["error_count"] else None
            ),
            "paired_landmarks": int(item["error_count"]),
            "reference_capture_rate": (
                item["joint_detected_frames"] / item["reference_detected_frames"]
                if item["reference_detected_frames"] else None
            ),
            "landmark_capture_rate": (
                item["joint_landmarks"] / item["reference_landmarks"]
                if item["reference_landmarks"] else None
            ),
        }
    result["components"] = components
    result["hand_both_detection_rate"] = (
        acc["hand_both_detected_frames"] / acc["frames_extracted"]
        if acc["frames_extracted"] else None
    )
    return result


def _analyze_condition(
    name: str,
    manifest: pd.DataFrame,
    landmark_root: Path,
    reference_root: Path | None,
) -> dict[str, Any]:
    """Aggregate one condition, optionally paired to original landmarks."""
    condition_acc = _new_accumulator()
    by_video: dict[str, dict[str, Any]] = {}
    raw_dir = landmark_root / name / "raw"
    ref_dir = reference_root / "raw" if reference_root else None

    for video_id, video_df in manifest.groupby("VIDEO_ID", sort=True):
        video_acc = _new_accumulator()
        video_acc["segments_manifest"] = len(video_df)
        condition_acc["segments_manifest"] += len(video_df)
        for row in video_df.itertuples(index=False):
            sample_id = str(row.SAMPLE_ID)
            output_path = raw_dir / f"{sample_id}.npy"
            if not output_path.exists():
                video_acc["segments_missing_output"] += 1
                condition_acc["segments_missing_output"] += 1
                continue
            arr = np.load(output_path, allow_pickle=False).astype(np.float32)
            video_acc["segments_with_output"] += 1
            condition_acc["segments_with_output"] += 1
            _add_own_metrics(video_acc, arr)
            _add_own_metrics(condition_acc, arr)

            if ref_dir is not None:
                reference_path = ref_dir / f"{sample_id}.npy"
                if reference_path.exists():
                    reference = np.load(reference_path, allow_pickle=False).astype(
                        np.float32
                    )
                    _add_pair_metrics(video_acc, reference, arr)
                    _add_pair_metrics(condition_acc, reference, arr)

        by_video[video_id] = _finalize_accumulator(video_acc)

    result = _finalize_accumulator(condition_acc)
    result["by_video"] = by_video
    return result


def _file_size_table() -> dict[str, Any]:
    """Collect bytes and actual reduction for every video condition."""
    source_bytes = {
        video_id: (VIDEO_ROOT / f"{video_id}.mp4").stat().st_size
        for video_id in VIDEO_IDS
    }
    result: dict[str, Any] = {}
    total_source = sum(source_bytes.values())
    for name, condition in CONDITIONS.items():
        files: dict[str, Any] = {}
        total_output = 0
        for video_id, original_size in source_bytes.items():
            path = condition["video_dir"] / f"{video_id}.mp4"
            size = path.stat().st_size if path.exists() else None
            if size is not None:
                total_output += size
            files[video_id] = {
                "bytes": size,
                "source_bytes": original_size,
                "reduction_percent": (
                    100.0 * (1.0 - size / original_size)
                    if size is not None else None
                ),
            }
        result[name] = {
            "files": files,
            "total_source_bytes": total_source,
            "total_output_bytes": total_output,
            "actual_reduction_percent": (
                100.0 * (1.0 - total_output / total_source)
                if total_source else None
            ),
        }
    return result


def _percent(value: float | None, digits: int = 2) -> str:
    return "—" if value is None or not math.isfinite(value) else f"{100.0 * value:.{digits}f}%"


def _number(value: float | None, digits: int = 6) -> str:
    return "—" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def _write_reports(
    output_root: Path,
    manifest: pd.DataFrame,
    condition_results: dict[str, Any],
    sizes: dict[str, Any],
    run_stats: dict[str, Any],
    sample_rate: float | None,
) -> None:
    """Write machine-readable results plus the concise final analysis."""
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": {
            "description": (
                "MediaPipe Holistic raw-landmark stability proxy; this does "
                "not measure translation semantics."
            ),
            "config": str(DEFAULT_CONFIG),
            "sample_rate": sample_rate,
            "manifest": str(output_root / "test_manifest.tsv"),
            "segments": int(len(manifest)),
            "videos": list(VIDEO_IDS),
            "landmark_layout": {
                "pose": "33 keypoints [0:33]",
                "face": "478 keypoints [33:511]",
                "left_hand": "21 keypoints [511:532]",
                "right_hand": "21 keypoints [532:553]",
            },
            "distance": "mean Euclidean 3D [x,y,z] distance when both landmarks are detected",
            "jitter": "mean Euclidean norm of p[t+1] - 2p[t] + p[t-1] over detected landmark triples",
        },
        "file_sizes": sizes,
        "run_stats": run_stats,
        "conditions": condition_results,
    }
    (output_root / "mediapipe_compression_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    csv_path = output_root / "mediapipe_compression_summary.csv"
    fields = [
        "condition",
        "label",
        "target_reduction_percent",
        "actual_reduction_percent",
        "segments_manifest",
        "segments_with_output",
        "segments_missing_output",
        "frames_extracted",
        "pose_detection_rate",
        "face_detection_rate",
        "left_hand_detection_rate",
        "right_hand_detection_rate",
        "either_hand_detection_rate",
        "both_hand_detection_rate",
        "hands_landmark_error_3d_vs_original",
        "hands_landmark_capture_rate_vs_original",
        "hands_temporal_jitter",
        "wall_seconds",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, condition in CONDITIONS.items():
            result = condition_results[name]
            components = result["components"]
            writer.writerow({
                "condition": name,
                "label": condition["label"],
                "target_reduction_percent": condition["target_reduction_percent"],
                "actual_reduction_percent": sizes[name]["actual_reduction_percent"],
                "segments_manifest": result["segments_manifest"],
                "segments_with_output": result["segments_with_output"],
                "segments_missing_output": result["segments_missing_output"],
                "frames_extracted": result["frames_extracted"],
                "pose_detection_rate": components["pose"]["detection_rate"],
                "face_detection_rate": components["face"]["detection_rate"],
                "left_hand_detection_rate": components["left_hand"]["detection_rate"],
                "right_hand_detection_rate": components["right_hand"]["detection_rate"],
                "either_hand_detection_rate": components["hands"]["detection_rate"],
                "both_hand_detection_rate": result["hand_both_detection_rate"],
                "hands_landmark_error_3d_vs_original": components["hands"]["landmark_error_3d"],
                "hands_landmark_capture_rate_vs_original": components["hands"]["landmark_capture_rate"],
                "hands_temporal_jitter": components["hands"]["temporal_jitter"],
                "wall_seconds": run_stats.get(name, {}).get("wall_seconds"),
            })

    lines = [
        "# MediaPipe compression-influence analysis",
        "",
        "## Conclusion",
        "",
        "MediaPipe was used as a visual/landmark stability proxy because no sign-language recognizer or translation model is available. It cannot establish whether the English meaning of a sign is preserved. The strongest acceptable compression is the one that saves storage while keeping hand detection and hand landmark capture close to the original, with little increase in temporal jitter.",
        "",
        "## Experiment",
        "",
        f"- Videos: `{', '.join(VIDEO_IDS)}`; matched manifest segments: **{len(manifest)}**.",
        f"- Same repository job/config: `{DEFAULT_CONFIG}`.",
        f"- Same MediaPipe Holistic settings: model complexity 1, refined face landmarks, detection/tracking confidence 0.5.",
        f"- Same sampling: `{sample_rate}` (0.5 keeps half the native frame rate).",
        "- Only the video directory changed. Each condition was extracted independently; compressed conditions did not reuse original landmark files.",
        "- Raw 553-keypoint arrays were compared per segment. Detection is the fraction of sampled frames with at least one valid keypoint in the component. Landmark error is mean paired 3D Euclidean distance in MediaPipe coordinates; missing landmarks are represented separately by capture/detection rates. Jitter is the second temporal difference norm over valid triples.",
        "",
        "## File-size result",
        "",
        "| Condition | Target reduction | Actual reduction | Output bytes |",
        "|---|---:|---:|---:|",
    ]
    for name, condition in CONDITIONS.items():
        size = sizes[name]
        lines.append(
            f"| {condition['label']} | {condition['target_reduction_percent']:.0f}% | "
            f"{size['actual_reduction_percent']:.2f}% | "
            f"{size['total_output_bytes']:,} |"
        )

    lines += [
        "",
        "## Overall MediaPipe result",
        "",
        "Detection rates are percentages of sampled frames. Hand error/capture are relative to the original landmarks; original error is zero by definition.",
        "",
        "| Condition | Pose det. | Face det. | Left hand det. | Right hand det. | Either hand det. | Both hands det. | Hand error 3D | Hand capture | Hand jitter |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, condition in CONDITIONS.items():
        result = condition_results[name]
        comp = result["components"]
        lines.append(
            f"| {condition['label']} | {_percent(comp['pose']['detection_rate'])} | "
            f"{_percent(comp['face']['detection_rate'])} | "
            f"{_percent(comp['left_hand']['detection_rate'])} | "
            f"{_percent(comp['right_hand']['detection_rate'])} | "
            f"{_percent(comp['hands']['detection_rate'])} | "
            f"{_percent(result['hand_both_detection_rate'])} | "
            f"{_number(comp['hands']['landmark_error_3d'])} | "
            f"{_percent(comp['hands']['landmark_capture_rate'])} | "
            f"{_number(comp['hands']['temporal_jitter'])} |"
        )

    lines += [
        "",
        "## Per-video result",
        "",
        "| Video | Condition | Actual reduction | Either hand det. | Hand error 3D | Hand capture | Hand jitter | Missing segments |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for video_id in VIDEO_IDS:
        for name, condition in CONDITIONS.items():
            result = condition_results[name]["by_video"][video_id]
            comp = result["components"]
            file_size = sizes[name]["files"][video_id]["reduction_percent"]
            lines.append(
                f"| `{video_id}` | {condition['label']} | "
                f"{file_size:.2f}% | {_percent(comp['hands']['detection_rate'])} | "
                f"{_number(comp['hands']['landmark_error_3d'])} | "
                f"{_percent(comp['hands']['landmark_capture_rate'])} | "
                f"{_number(comp['hands']['temporal_jitter'])} | "
                f"{result['segments_missing_output']} |"
            )

    lines += [
        "",
        "## Interpretation rule",
        "",
        "Choose the strongest compression whose either-hand detection/capture remains close to the original and whose hand error/jitter does not show a practically meaningful increase. Treat these three videos as a pilot. A translation or recognition model, human signer judgment, or a hand-focused perceptual metric is still required to make a semantic claim.",
        "",
        "## Artifacts",
        "",
        "- `test_manifest.tsv`: the exact 616-segment manifest used for every run.",
        "- `landmarks/<condition>/raw/*.npy`: independent raw MediaPipe outputs.",
        "- `run_stats/<condition>.json`: pipeline counts and wall time.",
        "- `mediapipe_compression_summary.csv`: one row per condition.",
        "- `mediapipe_compression_results.json`: complete machine-readable result.",
    ]
    (output_root / "mediapipe_compression_analysis.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sample-rate", type=float, default=0.5)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--video-root", type=Path, default=VIDEO_ROOT)
    parser.add_argument("--conditions-json", type=Path)
    parser.add_argument("--conditions", help="comma-separated condition names")
    parser.add_argument("--skip-runs", action="store_true")
    parser.add_argument("--force-all", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    global DEFAULT_CONFIG, VIDEO_ROOT, VIDEO_IDS, CONDITIONS
    DEFAULT_CONFIG = args.config.resolve()
    VIDEO_ROOT = args.video_root.resolve()
    VIDEO_IDS = tuple(path.stem for path in sorted(VIDEO_ROOT.glob("*.mp4")))
    if args.conditions_json:
        definitions = json.loads(args.conditions_json.read_text(encoding="utf-8"))
        raw_conditions = definitions.get("conditions", definitions)
        CONDITIONS = {}
        for name, condition in raw_conditions.items():
            normalized = dict(condition)
            normalized["video_dir"] = Path(normalized["video_dir"]).resolve()
            normalized.setdefault("label", name)
            normalized.setdefault("target_reduction_percent", 0.0)
            CONDITIONS[name] = normalized
    if args.conditions:
        selected = [name.strip() for name in args.conditions.split(",") if name.strip()]
        missing = set(selected).difference(CONDITIONS)
        if missing:
            raise SystemExit(f"Unknown condition(s): {sorted(missing)}")
        CONDITIONS = {name: CONDITIONS[name] for name in selected}

    manifest_path = args.output_root / "test_manifest.tsv"
    manifest = _canonical_manifest(args.source_manifest, manifest_path)
    LOGGER.info(
        "Prepared %d matched segments for %d videos: %s",
        len(manifest),
        manifest["VIDEO_ID"].nunique(),
        ", ".join(VIDEO_IDS),
    )

    run_stats: dict[str, Any] = {}
    if not args.skip_runs:
        for name, condition in CONDITIONS.items():
            try:
                run_stats[name] = _run_condition(
                    name,
                    condition,
                    manifest_path,
                    args.output_root,
                    args.sample_rate,
                    args.force_all,
                )
            except Exception:
                LOGGER.exception("Condition %s failed; continuing to the next condition", name)
                run_stats[name] = {
                    "condition": name,
                    "label": condition["label"],
                    "error": "pipeline failed; see log",
                }

    # If this is an analysis-only/resume pass, load any prior run metadata.
    stats_dir = args.output_root / "run_stats"
    for name in CONDITIONS:
        if name not in run_stats:
            path = stats_dir / f"{name}.json"
            if path.exists():
                run_stats[name] = json.loads(path.read_text(encoding="utf-8"))
            else:
                run_stats[name] = {}

    condition_results: dict[str, Any] = {}
    landmark_root = args.output_root / "landmarks"
    reference_root = landmark_root / "original"
    for name in CONDITIONS:
        condition_results[name] = _analyze_condition(
            name,
            manifest,
            landmark_root,
            None if name == "original" else reference_root,
        )

    sizes = _file_size_table()
    _write_reports(
        args.output_root,
        manifest,
        condition_results,
        sizes,
        run_stats,
        args.sample_rate,
    )
    LOGGER.info(
        "Reports written to %s; see mediapipe_compression_analysis.md",
        args.output_root,
    )


if __name__ == "__main__":
    main()
