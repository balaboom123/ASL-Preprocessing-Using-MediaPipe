"""LSA64 source config, path resolution, and release validation."""

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, get_args

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .._ingestion.availability import AvailabilityPolicy
from .._ingestion.classmap import load_class_map

_BUNDLED_CLASS_MAP = (
    Path(__file__).parents[4] / "assets" / "lsa64_class_map.tsv"
)

DEFAULT_FPS = 60.0
LSA64Variant = Literal["raw", "cut"]
_KNOWN_VARIANTS = frozenset(get_args(LSA64Variant))


class LSA64SourceConfig(BaseModel):
    """Typed config for LSA64 adapter."""

    model_config = ConfigDict(extra="forbid")

    release_dir: str = ""
    variant: LSA64Variant = "cut"
    split: Literal["all", "train", "val", "test"] = "all"
    split_strategy: Literal["none", "community_signer_8_1_1"] = "none"
    train_signers: list[int] = Field(
        default_factory=lambda: list(range(1, 9))
    )
    val_signers: list[int] = Field(default_factory=lambda: [9])
    test_signers: list[int] = Field(default_factory=lambda: [10])
    class_map_file: str = ""
    availability_policy: AvailabilityPolicy = "fail_fast"
    allow_missing_class_map: bool = False


def get_source_config(config) -> LSA64SourceConfig:
    source_dict = dict(config.dataset.source)
    if not source_dict.get("release_dir") and config.paths.videos:
        source_dict["release_dir"] = config.paths.videos
    return LSA64SourceConfig(**source_dict)


def resolve_release_dir(config, source: LSA64SourceConfig) -> Path | None:
    raw = source.release_dir or (config.paths.videos or "")
    return Path(raw) if str(raw).strip() else None


def infer_variant_from_path(path: Path) -> str | None:
    name = path.name.lower()
    return name if name in _KNOWN_VARIANTS else None


def validate_variant_path_consistency(config, source: LSA64SourceConfig) -> None:
    release_dir = resolve_release_dir(config, source)
    if release_dir is None:
        return

    explicit_variant = infer_variant_from_path(release_dir)
    if explicit_variant and explicit_variant != source.variant:
        raise ValueError(
            f"lsa64 variant={source.variant!r} conflicts with explicit "
            f"directory {release_dir!s} (looks like variant {explicit_variant!r}). "
            "Either point release_dir/paths.videos at the release root or "
            "set dataset.source.variant to match the explicit directory."
        )


def resolve_video_dir(config, source: LSA64SourceConfig) -> Path | None:
    release_dir = resolve_release_dir(config, source)
    if release_dir is None or infer_variant_from_path(release_dir):
        return release_dir

    variant_dir = release_dir / source.variant
    return variant_dir if variant_dir.is_dir() else release_dir


def _pick_unique_variant(values: Iterable[str], source_label: str) -> str | None:
    cleaned = {v for v in (str(x).strip().lower() for x in values) if v}
    if not cleaned:
        return None
    if len(cleaned) > 1:
        raise ValueError(
            f"LSA64 manifest has mixed {source_label} values: {sorted(cleaned)}"
        )
    return next(iter(cleaned))


def infer_manifest_variant(df: pd.DataFrame) -> str | None:
    """Infer which LSA64 variant an existing manifest was built from."""
    if "SOURCE_VARIANT" in df.columns:
        raw_values = df["SOURCE_VARIANT"].dropna().astype(str).str.strip().str.lower()
        raw_values = raw_values[raw_values != ""]
        unknown = set(raw_values.unique()) - _KNOWN_VARIANTS
        if unknown:
            raise ValueError(
                f"LSA64 manifest has unsupported SOURCE_VARIANT values: {sorted(unknown)}"
            )
        variant = _pick_unique_variant(raw_values.unique(), "SOURCE_VARIANT")
        if variant:
            return variant

    if "SAMPLE_ID" in df.columns:
        prefixes = (
            df["SAMPLE_ID"].dropna().astype(str)
            .str.split("-", n=1).str[0].str.strip().str.lower()
        )
        candidates = prefixes[prefixes.isin(_KNOWN_VARIANTS)].unique()
        return _pick_unique_variant(candidates, "SAMPLE_ID variant prefix")

    return None


def validate_loaded_manifest_variant(
    df: pd.DataFrame,
    manifest_path: Path | None,
    source: LSA64SourceConfig,
) -> None:
    """Reject reused manifests that were built for a different variant."""
    manifest_variant = infer_manifest_variant(df)
    if manifest_variant is None:
        raise ValueError(
            f"Cannot verify the source variant for existing LSA64 manifest "
            f"{manifest_path or '<unknown>'}. Expected SOURCE_VARIANT or "
            f"SAMPLE_ID values prefixed with 'raw-' or 'cut-'. Regenerate "
            f"the manifest with dataset.manifest=true."
        )
    if manifest_variant != source.variant:
        raise ValueError(
            f"Existing LSA64 manifest {manifest_path or '<unknown>'} was built "
            f"for variant={manifest_variant!r}, but current config requests "
            f"variant={source.variant!r}. Regenerate the manifest or use a "
            f"manifest built for the matching variant."
        )


def validate_release(
    source: LSA64SourceConfig,
    video_dir: Path | None,
    log: logging.Logger,
) -> dict:
    """Validate LSA64 release directory. Returns stats dict."""
    if video_dir is None:
        raise FileNotFoundError(
            "LSA64 requires a local release directory. "
            "Set dataset.source.release_dir or paths.videos in your config YAML.\n"
            "Download LSA64 from https://facundoq.github.io/datasets/lsa64/"
        )
    if not video_dir.exists():
        raise FileNotFoundError(
            f"LSA64 release directory not found: {video_dir}\n"
            f"LSA64 requires manual download. "
            f"See https://facundoq.github.io/datasets/lsa64/ for instructions."
        )
    mp4_files = list(video_dir.glob("*.mp4"))
    if not mp4_files:
        raise FileNotFoundError(
            f"No .mp4 files found in LSA64 directory: {video_dir}\n"
            f"Ensure the release has been extracted and the correct "
            f"variant directory is specified."
        )
    log.info(
        "LSA64 release directory validated: %s (%d .mp4 files)",
        video_dir, len(mp4_files),
    )
    release_root = video_dir.parent if infer_variant_from_path(video_dir) else video_dir
    return {
        "validated": True,
        "release_dir": str(release_root),
        "video_dir": str(video_dir),
        "variant": source.variant,
        "mp4_count": len(mp4_files),
    }


def load_lsa64_class_map(
    source: LSA64SourceConfig,
    log: logging.Logger,
) -> pd.DataFrame | None:
    """Load class map from source config or bundled asset.

    Returns None when the class map is not found and
    ``allow_missing_class_map`` is True.
    """
    candidates = []
    if source.class_map_file:
        candidates.append(Path(source.class_map_file))
    candidates.append(_BUNDLED_CLASS_MAP)

    for path in candidates:
        if path.exists():
            log.info("Loading LSA64 class map from %s", path)
            return load_class_map(path)

    if source.allow_missing_class_map:
        log.warning(
            "LSA64 class map not found (searched: %s). "
            "Proceeding without GLOSS/HANDEDNESS labels.",
            [str(c) for c in candidates],
        )
        return None

    raise FileNotFoundError(
        f"LSA64 class map not found. Searched:\n"
        + "\n".join(f"  {c}" for c in candidates)
        + "\nProvide a class_map_file in your config or place "
        "lsa64_class_map.tsv in the assets/ directory.\n"
        "Set allow_missing_class_map: true to skip class labels."
    )
