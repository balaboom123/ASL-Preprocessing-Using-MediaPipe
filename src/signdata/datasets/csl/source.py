"""CSL source config, path resolution, and release preparation."""

import logging
from pathlib import Path
from typing import Iterable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .._ingestion.availability import AvailabilityPolicy
from .._ingestion.media import materialize_frames_to_video

# Split-I: signer-independent boundary
SPLIT_I_TRAIN_SIGNERS = set(range(1, 41))
SPLIT_I_TEST_SIGNERS = set(range(41, 51))

# Split-II: unseen-sentence boundary
SPLIT_II_TRAIN_SENTENCES = set(range(1, 95))
SPLIT_II_TEST_SENTENCES = set(range(95, 101))

DEFAULT_FPS = 30.0
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
FRAME_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
REPETITIONS_PER_SIGNER = 5


class CSLSourceConfig(BaseModel):
    """Typed config for the continuous CSL adapter."""

    model_config = ConfigDict(extra="forbid")

    release_dir: str = ""
    protocol: Literal["split_i", "split_ii"] = "split_i"
    split: Literal["all", "train", "test"] = "all"
    split_spec_file: str = ""
    availability_policy: AvailabilityPolicy = "drop_unavailable"
    rgb_subdir: str = "color"
    corpus_file: str = ""
    prepare_mode: Literal[
        "validate", "materialize_missing", "rematerialize_all"
    ] = "materialize_missing"
    video_fps: float = Field(default=DEFAULT_FPS, gt=0)


def get_source_config(config) -> CSLSourceConfig:
    source_dict = dict(config.dataset.source)
    if not source_dict.get("release_dir") and config.paths.root:
        source_dict["release_dir"] = config.paths.root
    return CSLSourceConfig(**source_dict)


def resolve_release_dir(source: CSLSourceConfig, config) -> Path:
    raw = source.release_dir or (config.paths.root or "")
    return Path(raw)


def validate_source_config(source: CSLSourceConfig) -> None:
    """Validate adapter-level CSL source options."""
    if source.split_spec_file and not Path(source.split_spec_file).exists():
        raise ValueError(
            f"CSL split_spec_file not found: {source.split_spec_file}"
        )


def resolve_corpus_file(
    source: CSLSourceConfig,
    release_dir: Path,
    log: logging.Logger,
) -> Optional[Path]:
    if source.corpus_file:
        configured = Path(source.corpus_file)
        if configured.exists():
            return configured
        log.warning("Configured corpus_file not found: %s", source.corpus_file)

    candidates = [
        release_dir / "corpus.txt",
        release_dir / "corpus.tsv",
        release_dir / "sentences.txt",
        release_dir / "sentences.tsv",
        release_dir / "label.txt",
        release_dir / "label.tsv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_rgb_dir(release_dir: Path, source: CSLSourceConfig) -> Path:
    if source.rgb_subdir:
        candidate = release_dir / source.rgb_subdir
        if candidate.exists():
            return candidate
    return release_dir


def resolve_materialized_video_dir(config, release_dir: Path) -> Path:
    raw = config.paths.videos or str(release_dir / "videos")
    return Path(raw)


def resolve_runtime_video_dir(source: CSLSourceConfig, config) -> Path:
    release_dir = resolve_release_dir(source, config)
    rgb_dir = resolve_rgb_dir(release_dir, source)
    materialized_dir = resolve_materialized_video_dir(config, release_dir)

    if rgb_dir.exists() and has_video_files(rgb_dir):
        return rgb_dir
    if materialized_dir.exists() and has_video_files(materialized_dir):
        return materialized_dir
    return materialized_dir


def has_video_files(root: Path) -> bool:
    if not root.exists():
        return False
    return any(
        path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        for path in root.rglob("*")
    )


def has_frame_layout(root: Path) -> bool:
    return any(True for _ in iter_sample_frame_dirs(root))


def iter_sample_frame_dirs(root: Path) -> Iterable[tuple[Path, Path]]:
    if not root.exists():
        return

    sentence_dirs = sorted(
        path for path in root.iterdir()
        if path.is_dir() and _looks_like_sentence_dir(path.name)
    )
    for sentence_dir in sentence_dirs:
        sample_dirs = sorted(
            path for path in sentence_dir.iterdir()
            if path.is_dir() and _contains_frame_files(path)
        )
        for sample_dir in sample_dirs:
            yield sentence_dir, sample_dir


def prepare(source: CSLSourceConfig, config, log: logging.Logger) -> dict:
    """Validate or prepare a local CSL release for pipeline consumption."""
    release_dir = resolve_release_dir(source, config)

    if not str(release_dir).strip():
        raise FileNotFoundError(
            "CSL requires a local release directory. "
            "Set dataset.source.release_dir or paths.root in your config YAML.\n"
            "Download CSL from https://ustc-slr.github.io/datasets/2015_csl/"
        )
    if not release_dir.exists():
        raise FileNotFoundError(
            f"CSL release directory not found: {release_dir}\n"
            "CSL requires manual download. "
            "See https://ustc-slr.github.io/datasets/2015_csl/"
        )

    corpus_path = resolve_corpus_file(source, release_dir, log)
    if corpus_path is None:
        raise FileNotFoundError(
            f"CSL corpus file not found under {release_dir}.\n"
            "Tried: corpus.txt, corpus.tsv, sentences.txt, sentences.tsv, "
            "label.txt, label.tsv.\n"
            "Set dataset.source.corpus_file explicitly in your config YAML."
        )

    rgb_dir = resolve_rgb_dir(release_dir, source)
    if not rgb_dir.exists():
        raise FileNotFoundError(
            f"CSL RGB directory not found: {rgb_dir}\n"
            "Expected the continuous release to contain a color directory, or "
            "override dataset.source.rgb_subdir."
        )

    materialized_dir = resolve_materialized_video_dir(config, release_dir)
    if has_video_files(rgb_dir):
        log.info("CSL RGB directory already contains video files: %s", rgb_dir)
        return _build_prepare_stats(
            source=source,
            release_dir=release_dir,
            corpus_path=corpus_path,
            runtime_video_dir=rgb_dir,
            validated=True,
        )

    materialized_has_videos = has_video_files(materialized_dir)
    if source.prepare_mode == "validate":
        if materialized_has_videos:
            return _build_prepare_stats(
                source=source,
                release_dir=release_dir,
                corpus_path=corpus_path,
                runtime_video_dir=materialized_dir,
                validated=True,
            )
        if has_frame_layout(rgb_dir):
            raise FileNotFoundError(
                "CSL validate mode found frame directories but no video files. "
                "Set dataset.source.prepare_mode=materialize_missing to convert "
                "frame folders into .mp4 clips for the pipeline."
            )
        raise FileNotFoundError(
            "No CSL video files found in either the RGB release directory or "
            "paths.videos."
        )

    if not has_frame_layout(rgb_dir):
        if materialized_has_videos:
            log.info(
                "Using previously materialized CSL videos in %s",
                materialized_dir,
            )
            return _build_prepare_stats(
                source=source,
                release_dir=release_dir,
                corpus_path=corpus_path,
                runtime_video_dir=materialized_dir,
                validated=True,
            )
        raise FileNotFoundError(
            f"No CSL video files or frame directories found under {rgb_dir}.\n"
            "Expected either RGB video clips or per-sample frame folders."
        )

    overwrite = source.prepare_mode == "rematerialize_all"
    materialized, validated, errors = _materialize_frame_tree(
        source=source,
        rgb_dir=rgb_dir,
        output_dir=materialized_dir,
        overwrite=overwrite,
        log=log,
    )
    if materialized == 0 and validated == 0:
        raise RuntimeError(
            "CSL frame-folder preparation did not produce any usable videos. "
            f"Checked RGB source {rgb_dir} and output directory {materialized_dir}. "
            "Check ffmpeg availability, video_fps, and frame naming."
        )

    return _build_prepare_stats(
        source=source,
        release_dir=release_dir,
        corpus_path=corpus_path,
        runtime_video_dir=materialized_dir,
        validated=validated > 0 or materialized > 0,
        materialized=materialized,
        errors=errors,
    )


def _build_prepare_stats(
    *,
    source: CSLSourceConfig,
    release_dir: Path,
    corpus_path: Path,
    runtime_video_dir: Path,
    validated: bool,
    materialized: int = 0,
    errors: int = 0,
) -> dict:
    return {
        "validated": validated,
        "materialized": materialized,
        "errors": errors,
        "release_dir": str(release_dir),
        "corpus_file": str(corpus_path),
        "runtime_video_dir": str(runtime_video_dir),
        "protocol": source.protocol,
        "mode": source.prepare_mode,
    }


def _materialize_frame_tree(
    *,
    source: CSLSourceConfig,
    rgb_dir: Path,
    output_dir: Path,
    overwrite: bool,
    log: logging.Logger,
) -> tuple[int, int, int]:
    materialized = 0
    validated = 0
    errors = 0
    saw_samples = False

    for sentence_dir, sample_dir in iter_sample_frame_dirs(rgb_dir):
        saw_samples = True
        output_path = output_dir / sentence_dir.name / f"{sample_dir.name}.mp4"
        if output_path.exists() and not overwrite:
            validated += 1
            continue

        try:
            frame_pattern = _resolve_frame_pattern(sample_dir)
            materialize_frames_to_video(
                sample_dir,
                output_path,
                fps=source.video_fps,
                pattern=frame_pattern,
                overwrite=overwrite,
            )
        except Exception as exc:
            log.warning("Failed to materialize CSL sample %s: %s", sample_dir, exc)
            errors += 1
            continue

        if output_path.exists():
            materialized += 1
        else:
            errors += 1

    if not saw_samples:
        raise FileNotFoundError(
            f"No CSL frame directories found under {rgb_dir}. "
            "Expected a layout such as color/000000/<sample_dir>/000001.jpg."
        )

    return materialized, validated, errors


def _looks_like_sentence_dir(name: str) -> bool:
    stripped = name.strip()
    return stripped.isdigit() and len(stripped) <= 6


def _contains_frame_files(path: Path) -> bool:
    return any(
        child.is_file() and child.suffix.lower() in FRAME_EXTENSIONS
        for child in path.iterdir()
    )


def _resolve_frame_pattern(sample_dir: Path) -> str | tuple[str, ...]:
    patterns = sorted(
        {
            f"*{child.suffix}"
            for child in sample_dir.iterdir()
            if child.is_file() and child.suffix.lower() in FRAME_EXTENSIONS
        },
        key=lambda pattern: (pattern.lower(), pattern),
    )
    if patterns:
        return patterns[0] if len(patterns) == 1 else tuple(patterns)
    raise FileNotFoundError(
        f"No supported CSL frame files found in sample directory: {sample_dir}"
    )
