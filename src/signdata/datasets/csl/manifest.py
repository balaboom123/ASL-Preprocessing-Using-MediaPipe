"""CSL manifest building."""

import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from .._ingestion.availability import apply_availability_policy_paths
from .._ingestion.media import get_video_duration, get_video_fps
from ...utils.manifest import write_manifest
from .source import (
    CSLSourceConfig,
    REPETITIONS_PER_SIGNER,
    SPLIT_I_TEST_SIGNERS,
    SPLIT_I_TRAIN_SIGNERS,
    SPLIT_II_TEST_SENTENCES,
    SPLIT_II_TRAIN_SENTENCES,
    VIDEO_EXTENSIONS,
    resolve_corpus_file,
    resolve_release_dir,
    resolve_runtime_video_dir,
)


def build(config, source: CSLSourceConfig, log: logging.Logger) -> pd.DataFrame:
    """Build a canonical manifest from the continuous CSL release."""
    release_dir = resolve_release_dir(source, config)
    video_dir = resolve_runtime_video_dir(source, config)

    if not release_dir.exists():
        raise FileNotFoundError(
            f"CSL release directory not found: {release_dir!r}. "
            "Run the download stage first or set release_dir / paths.root."
        )
    if not video_dir.exists():
        raise FileNotFoundError(
            f"CSL runtime video directory not found: {video_dir!r}. "
            "Run the dataset.download stage first to validate or materialize videos."
        )

    corpus_path = resolve_corpus_file(source, release_dir, log)
    if corpus_path is None:
        raise FileNotFoundError(
            f"CSL corpus file not found under {release_dir}. "
            "Set dataset.source.corpus_file explicitly."
        )

    corpus = _parse_corpus(corpus_path, log)
    if corpus.empty:
        raise RuntimeError(
            f"CSL corpus file produced no rows: {corpus_path}. Check the file format."
        )

    text_lookup = dict(zip(corpus["sentence_id"], corpus["text"]))
    max_sentence_id = int(corpus["sentence_id"].max())
    sentence_base = 0 if max_sentence_id <= 99 else 1

    custom_splits: dict[str, str] | None = None
    if source.split_spec_file and Path(source.split_spec_file).exists():
        custom_splits = _load_split_spec(source.split_spec_file)
        log.info(
            "Loaded custom split spec from %s (%d entries)",
            source.split_spec_file,
            len(custom_splits),
        )

    sample_groups = _discover_video_groups(video_dir, log)
    if not sample_groups:
        raise FileNotFoundError(
            f"No CSL video files found under {video_dir}. "
            "Expected video clips after validation/materialization."
        )

    rows: list[dict] = []
    for sentence_id in sorted(sample_groups):
        sample_paths = sorted(
            sample_groups[sentence_id],
            key=lambda path: path.relative_to(video_dir).as_posix().casefold(),
        )
        sentence_text = text_lookup.get(sentence_id, "")
        if sentence_id not in text_lookup:
            log.warning("No corpus text found for CSL sentence_id=%s", sentence_id)

        variation_counts: Counter[int] = Counter()
        for ordinal, sample_path in enumerate(sample_paths):
            rel_path = sample_path.relative_to(video_dir).as_posix()
            signer_id, explicit_variation_id = _infer_signer_and_variation(
                sample_path,
                video_dir=video_dir,
                sentence_id=sentence_id,
                ordinal=ordinal,
            )
            if explicit_variation_id is not None:
                variation_id = explicit_variation_id
                variation_counts[signer_id] = max(
                    variation_counts[signer_id],
                    variation_id,
                )
            else:
                variation_counts[signer_id] += 1
                variation_id = variation_counts[signer_id]
            sample_id = f"{signer_id:03d}_{sentence_id:03d}_{variation_id:02d}"
            split_label = _resolve_split_label(
                custom_splits=custom_splits,
                sample_id=sample_id,
                rel_path=rel_path,
                sample_path=sample_path,
                signer_id=signer_id,
                sentence_id=sentence_id,
                protocol=source.protocol,
                sentence_base=sentence_base,
            )

            rows.append({
                "SAMPLE_ID": sample_id,
                "VIDEO_ID": sample_id,
                "REL_PATH": rel_path,
                "SPLIT": split_label,
                "START": 0.0,
                "END": get_video_duration(str(sample_path)),
                "TEXT": sentence_text,
                "SIGNER_ID": str(signer_id),
                "LANGUAGE": "zh",
                "FPS": get_video_fps(str(sample_path)) or source.video_fps,
                "VARIATION_ID": variation_id,
            })

    df = pd.DataFrame(rows)

    if source.split != "all":
        before = len(df)
        df = df[df["SPLIT"] == source.split].reset_index(drop=True)
        log.info("Filtered to split='%s': %d -> %d rows", source.split, before, len(df))

    df = apply_availability_policy_paths(
        df,
        base_dir=video_dir,
        policy=source.availability_policy,
        rel_path_col="REL_PATH",
    )

    write_manifest(df, config.paths.manifest)
    return df


def _parse_corpus(corpus_path: Path, log: logging.Logger) -> pd.DataFrame:
    rows = []
    skipped = 0
    for line in corpus_path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t") if "\t" in line else line.split()
        parts = [part.strip() for part in parts if part.strip()]
        if len(parts) < 2:
            skipped += 1
            continue

        try:
            sentence_id = int(parts[0])
        except ValueError:
            skipped += 1
            continue

        text_parts = parts[2:] if len(parts) >= 3 and parts[1].isdigit() else parts[1:]
        rows.append({"sentence_id": sentence_id, "text": " ".join(text_parts)})

    if skipped:
        log.warning("Skipped %d malformed CSL corpus lines in %s", skipped, corpus_path)

    return pd.DataFrame(rows)


def _discover_video_groups(video_dir: Path, log: logging.Logger) -> dict[int, list[Path]]:
    grouped: dict[int, list[Path]] = defaultdict(list)
    for path in sorted(video_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        sentence_id = _infer_sentence_id(path, video_dir)
        if sentence_id is None:
            log.debug("Skipping CSL video with unparseable sentence id: %s", path)
            continue
        grouped[sentence_id].append(path)
    return grouped


def _infer_sentence_id(path: Path, video_dir: Path) -> int | None:
    relative = path.relative_to(video_dir)
    for part in relative.parts[:-1]:
        if part.isdigit():
            return int(part)

    match = re.search(r"(\d{1,6})", path.stem)
    if match:
        return int(match.group(1))
    return None


def _infer_signer_and_variation(
    path: Path,
    *,
    video_dir: Path,
    sentence_id: int,
    ordinal: int,
) -> tuple[int, int | None]:
    numbers = _extract_sample_numbers(
        path,
        video_dir=video_dir,
        sentence_id=sentence_id,
    )
    if len(numbers) >= 2:
        return max(numbers[-2], 1), max(numbers[-1], 1)
    if len(numbers) == 1:
        return max(numbers[0], 1), None

    signer_id = ordinal // REPETITIONS_PER_SIGNER + 1
    variation_id = ordinal % REPETITIONS_PER_SIGNER + 1
    return signer_id, variation_id


def _extract_sample_numbers(
    path: Path,
    *,
    video_dir: Path,
    sentence_id: int,
) -> list[int]:
    relative = path.relative_to(video_dir)
    tokens: list[int] = []
    skipped_sentence_dir = False

    for part in relative.parts[:-1]:
        if (
            not skipped_sentence_dir
            and part.isdigit()
            and int(part) == sentence_id
        ):
            skipped_sentence_dir = True
            continue
        tokens.extend(_positive_numbers(part))

    tokens.extend(_positive_numbers(path.stem))
    return tokens


def _positive_numbers(value: str) -> list[int]:
    return [int(token) for token in re.findall(r"\d+", value)]


def _resolve_split_label(
    *,
    custom_splits: dict[str, str] | None,
    sample_id: str,
    rel_path: str,
    sample_path: Path,
    signer_id: int,
    sentence_id: int,
    protocol: str,
    sentence_base: int,
) -> str:
    if custom_splits is not None:
        for key in (sample_id, sample_path.stem, rel_path):
            if key in custom_splits:
                return custom_splits[key]
        return "unknown"

    return _assign_split(
        signer_id=signer_id,
        sentence_id=sentence_id,
        protocol=protocol,
        sentence_base=sentence_base,
    )


def _assign_split(
    *,
    signer_id: int,
    sentence_id: int,
    protocol: str,
    sentence_base: int,
) -> str:
    if protocol == "split_i":
        if signer_id in SPLIT_I_TRAIN_SIGNERS:
            return "train"
        if signer_id in SPLIT_I_TEST_SIGNERS:
            return "test"
        return "unknown"

    if protocol == "split_ii":
        normalized_sentence_id = sentence_id + 1 if sentence_base == 0 else sentence_id
        if normalized_sentence_id in SPLIT_II_TRAIN_SENTENCES:
            return "train"
        if normalized_sentence_id in SPLIT_II_TEST_SENTENCES:
            return "test"
        return "unknown"

    raise ValueError(
        f"Unknown CSL split protocol: {protocol!r}. "
        "Valid options: 'split_i', 'split_ii'."
    )


def _load_split_spec(spec_file: str) -> dict[str, str]:
    df = pd.read_csv(spec_file, sep="\t", header=None, names=["sample_id", "split"])
    return dict(zip(df["sample_id"].astype(str), df["split"].astype(str)))
