"""Normalize post-processor: visibility masking + landmark normalization."""

import logging
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..processors.pose import resolve_keypoint_indices


def _load_clip(path: str) -> np.ndarray:
    """Load raw landmark clip from .npy file. Returns shape (T, K, 4)."""
    arr = np.load(path).astype(np.float32)

    if arr.ndim == 2:
        t_dim, feature_dim = arr.shape
        if feature_dim % 4 != 0:
            raise ValueError(f"Invalid feature count {feature_dim} in {path}")
        keypoint_count = feature_dim // 4
        if keypoint_count not in [85, 133, 543, 553]:
            raise ValueError(f"Unsupported shape {arr.shape} in {path}")
        arr = arr.reshape(t_dim, keypoint_count, 4)
    elif arr.ndim == 3:
        _, _, channels = arr.shape
        if channels != 4:
            raise ValueError(f"Expected 4 channels, got {channels} in {path}")
    else:
        raise ValueError(f"Unexpected ndim={arr.ndim} for {path}")

    return arr


def _apply_keypoint_reduction(
    clip_xyzv: np.ndarray,
    keypoint_indices: list[int],
) -> np.ndarray:
    """Reduce keypoints by selecting specific indices."""
    if clip_xyzv.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {clip_xyzv.shape}")
    if not keypoint_indices:
        raise ValueError("keypoint_indices is empty - cannot reduce to zero keypoints")
    _, keypoint_count, _ = clip_xyzv.shape
    if max(keypoint_indices) >= keypoint_count:
        raise ValueError(
            f"Index {max(keypoint_indices)} out of range for {keypoint_count} keypoints. "
            f"Check that keypoint_preset matches the extractor output."
        )
    return clip_xyzv[:, keypoint_indices, :]


def _apply_visibility_mask(
    clip_xyzv: np.ndarray,
    mask_empty_frames: bool,
    mask_low_confidence: bool,
    visibility_threshold: float,
    missing_value: float,
) -> np.ndarray:
    """Apply visibility masking. Returns (T, K, 3) with sentinels."""
    xyz = clip_xyzv[..., :3].copy()
    vis = clip_xyzv[..., 3]

    if mask_empty_frames:
        frame_all_zero = np.all(clip_xyzv == 0.0, axis=(1, 2))
        xyz[frame_all_zero] = missing_value

    if mask_low_confidence:
        low_vis = vis < visibility_threshold
        all_zero = np.all(clip_xyzv == 0.0, axis=-1)
        missing = np.logical_or(low_vis, all_zero)
        xyz[missing] = missing_value

    return xyz


def _normalize_clip_xyz(
    xyz_masked: np.ndarray,
    mode: str,
    missing_value: float,
) -> np.ndarray:
    """Normalize landmarks using whole-clip scaling."""
    if xyz_masked.ndim != 3 or xyz_masked.shape[-1] != 3:
        raise ValueError(f"Expected shape (T, K, 3), got {xyz_masked.shape}")

    out = xyz_masked.copy()
    invalid = xyz_masked == missing_value
    valid_points_mask = ~np.any(invalid, axis=-1)

    if not np.any(valid_points_mask):
        return out

    valid_coords = xyz_masked[valid_points_mask]
    x_vals, y_vals, z_vals = valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]

    if mode == "isotropic_3d":
        coord_min = valid_coords.min(axis=0)
        coord_max = valid_coords.max(axis=0)
        coord_range = coord_max - coord_min
        max_range = float(np.max(coord_range))

        if np.isclose(max_range, 0.0):
            scaled_valid = np.zeros_like(valid_coords, dtype=np.float32)
        else:
            scaled_valid = ((valid_coords - coord_min) / max_range).astype(np.float32)
    elif mode == "xy_isotropic_z_minmax":
        x_min, x_max = float(x_vals.min()), float(x_vals.max())
        y_min, y_max = float(y_vals.min()), float(y_vals.max())
        max_xy = max(x_max - x_min, y_max - y_min)

        if np.isclose(max_xy, 0.0):
            x_scaled = np.zeros_like(x_vals, dtype=np.float32)
            y_scaled = np.zeros_like(y_vals, dtype=np.float32)
        else:
            x_scaled = (x_vals - x_min) / max_xy
            y_scaled = (y_vals - y_min) / max_xy

        z_min, z_max = float(z_vals.min()), float(z_vals.max())
        z_range = z_max - z_min
        z_scaled = (
            np.zeros_like(z_vals, dtype=np.float32)
            if np.isclose(z_range, 0.0)
            else (z_vals - z_min) / z_range
        )
        scaled_valid = np.stack([x_scaled, y_scaled, z_scaled], axis=-1).astype(
            np.float32
        )
    else:
        raise ValueError(f"Unknown mode: '{mode}'")

    out[valid_points_mask] = scaled_valid
    return out


def _process_single_file(args) -> tuple[str, bool, str]:
    """Process a single landmark file through the normalization pipeline."""
    input_path, output_path, normalize_config = args
    input_path = Path(input_path)
    output_path = Path(output_path)
    filename = input_path.name

    try:
        if normalize_config.get("skip_existing") and output_path.exists():
            return filename, True, "skipped (exists)"

        clip_raw = _load_clip(input_path)

        # Keypoint reduction
        if normalize_config["select_keypoints"]:
            indices = resolve_keypoint_indices(
                normalize_config.get("keypoint_preset"),
                normalize_config.get("keypoint_indices"),
            )
            if indices is not None:
                clip_raw = _apply_keypoint_reduction(clip_raw, indices)

        # Visibility masking
        xyz_masked = _apply_visibility_mask(
            clip_raw,
            normalize_config["mask_empty_frames"],
            normalize_config["mask_low_confidence"],
            normalize_config["visibility_threshold"],
            normalize_config["missing_value"],
        )

        # Normalization
        xyz_norm = _normalize_clip_xyz(
            xyz_masked, normalize_config["mode"], normalize_config["missing_value"]
        )

        # Optionally drop z
        if normalize_config["remove_z"]:
            xyz_norm = xyz_norm[..., :2]

        t_dim, keypoint_count, channels_out = xyz_norm.shape
        flattened = xyz_norm.reshape(t_dim, keypoint_count * channels_out).astype(
            np.float32
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, flattened)

        return filename, True, ""
    except Exception as exc:
        return filename, False, str(exc)


class NormalizePostProcessor:
    """Normalize pose landmarks as a post-processing step.

    Reads from context.output_dir/raw, writes to context.output_dir/normalized.
    """

    logger = logging.getLogger("signdata.post.normalize")

    def __init__(self, config):
        self.config = config

    def run(self, context):
        cfg = self.config
        norm_cfg = cfg.post_processing.normalize

        if norm_cfg is None:
            self.logger.warning("No normalize config provided, skipping.")
            context.stats["post_processing.normalize"] = {"total": 0}
            return context

        input_dir = context.output_dir / "raw"
        output_dir = context.output_dir / "normalized"
        output_dir.mkdir(parents=True, exist_ok=True)
        npy_files = sorted(input_dir.rglob("*.npy"))

        if not npy_files:
            self.logger.warning("No .npy files found in %s", input_dir)
            context.stats["post_processing.normalize"] = {"total": 0}
            return context

        normalize_config = norm_cfg.model_dump()
        normalize_config["skip_existing"] = not context.force_all

        tasks = [
            (inp, output_dir / inp.relative_to(input_dir), normalize_config)
            for inp in npy_files
        ]

        self.logger.info("Normalizing %d files from %s", len(tasks), input_dir)

        success = skip = errors = 0

        executor = (
            ProcessPoolExecutor(max_workers=cfg.processing.max_workers)
            if cfg.processing.max_workers > 1
            else nullcontext()
        )
        with executor as pool:
            results = (
                pool.map(_process_single_file, tasks)
                if pool is not None
                else map(_process_single_file, tasks)
            )
            for fname, ok, msg in tqdm(
                results, total=len(tasks), desc="Normalizing"
            ):
                if ok:
                    if msg == "skipped (exists)":
                        skip += 1
                    else:
                        success += 1
                else:
                    errors += 1
                    self.logger.error("Failed: %s - %s", fname, msg)

        context.stats["post_processing.normalize"] = {
            "total": len(tasks),
            "success": success,
            "skipped": skip,
            "errors": errors,
        }
        return context
