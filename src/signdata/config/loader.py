"""YAML config loading."""

import copy
import os
import warnings
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, List, Optional

import yaml

from .schema import Config
from ..registry import DATASET_REGISTRY


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base. Override values take precedence."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _normalize_dataset_shorthand(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Expand dataset shorthand so nested merges preserve dataset.name."""
    dataset = raw.get("dataset")
    if isinstance(dataset, str):
        raw["dataset"] = {"name": dataset}
    return raw


def _load_yaml_mapping(yaml_path: str) -> Dict[str, Any]:
    """Load a YAML file and require a mapping at the top level."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        return {}

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {yaml_path}")

    return _normalize_dataset_shorthand(raw)


def _resolve_base_config_paths(base_ref: Any, config_path: Path) -> List[Path]:
    """Resolve one or more base-config paths relative to the current config."""
    if isinstance(base_ref, str):
        refs = [base_ref]
    elif isinstance(base_ref, list) and all(isinstance(item, str) for item in base_ref):
        refs = base_ref
    else:
        raise ValueError(
            f"'base' must be a string or list[str] in config: {config_path}"
        )

    resolved: List[Path] = []
    for ref in refs:
        resolved.append(_resolve_path(ref, config_path.parent).resolve())
    return resolved


def _load_raw_config(
    yaml_path: str,
    stack: Optional[List[Path]] = None,
) -> Dict[str, Any]:
    """Load a YAML config with optional recursive base-config merging."""
    config_path = _coerce_path(yaml_path)
    if not config_path.is_absolute():
        config_path = Path(os.path.abspath(str(config_path)))
    config_path = config_path.resolve()
    stack = stack or []

    if config_path in stack:
        chain = " -> ".join(str(path) for path in stack + [config_path])
        raise ValueError(f"Circular config base reference detected: {chain}")

    raw = _load_yaml_mapping(str(config_path))
    base_ref = raw.pop("base", None)
    if base_ref is None:
        return raw

    merged: Dict[str, Any] = {}
    for base_path in _resolve_base_config_paths(base_ref, config_path):
        merged = deep_merge(
            merged,
            _load_raw_config(str(base_path), stack + [config_path]),
        )

    return deep_merge(merged, raw)


def _coerce_path(path_str: str) -> Path:
    """Convert config path strings into paths on the current platform."""
    path = Path(path_str)
    if path.is_absolute():
        return path

    windows_path = PureWindowsPath(path_str)
    if windows_path.is_absolute():
        if windows_path.drive.endswith(":"):
            drive = windows_path.drive[:-1].lower()
            return Path("/mnt", drive, *windows_path.parts[1:])
        return Path(str(windows_path).replace("\\", "/"))

    if "\\" in path_str:
        return Path(*windows_path.parts)

    return path


def _resolve_path(path_str: str, root: Path) -> Path:
    """Resolve a config path against a root after normalizing separators."""
    path = _coerce_path(path_str)
    return path if path.is_absolute() else root / path


PACKAGE_DIR_ALIASES = ("signdata", "sltpipe", "sign_prep")
RESOURCE_CONFIG_ROOTS = {
    "pose_model_config": ("resources", "pose_models"),
    "det_model_config": ("resources", "detection_models"),
}
RESOURCE_CHECKPOINT_ATTRS = {"pose_model_checkpoint", "det_model_checkpoint"}


def _alternate_package_dirs(path: Path) -> List[Path]:
    """Return alternate package-dir candidates for migrated model assets."""
    parts = list(path.parts)
    for i in range(len(parts) - 1):
        if parts[i] != "src":
            continue
        current = parts[i + 1]
        if current not in PACKAGE_DIR_ALIASES:
            continue

        alternates = []
        for candidate in PACKAGE_DIR_ALIASES:
            if candidate == current:
                continue
            alt_parts = parts.copy()
            alt_parts[i + 1] = candidate
            alternates.append(Path(*alt_parts))
        return alternates
    return []


def _alternate_resource_model_configs(
    path: Path, project_root: Path, attr_name: str
) -> List[Path]:
    """Return resource-path candidates for legacy model config locations."""
    if attr_name not in RESOURCE_CONFIG_ROOTS:
        return []

    parts = list(path.parts)
    is_legacy_config_path = any(
        parts[i] == "src"
        and parts[i + 1] in PACKAGE_DIR_ALIASES
        and parts[i + 2] == "models"
        and parts[i + 3] == "configs"
        for i in range(len(parts) - 4)
    )
    if not is_legacy_config_path:
        return []

    resource_root = project_root.joinpath(*RESOURCE_CONFIG_ROOTS[attr_name])
    if not resource_root.exists():
        return []

    return sorted(
        candidate
        for candidate in resource_root.rglob(path.name)
        if candidate.is_file()
    )


def _alternate_legacy_model_checkpoints(
    path: Path, project_root: Path, attr_name: str
) -> List[Path]:
    """Return legacy checkpoint candidates for resource-path checkpoint refs."""
    if attr_name not in RESOURCE_CHECKPOINT_ATTRS:
        return []

    parts = list(path.parts)
    is_resource_checkpoint_path = any(
        parts[i] == "resources"
        and parts[i + 1] in {"pose_models", "detection_models"}
        and parts[i + 3] == "checkpoints"
        for i in range(len(parts) - 4)
    )
    if not is_resource_checkpoint_path:
        return []

    return [
        project_root / "src" / package_dir / "models" / "checkpoints" / path.name
        for package_dir in PACKAGE_DIR_ALIASES
    ]


def _resolve_model_path(path_str: str, project_root: Path, attr_name: str) -> str:
    """Resolve model paths relative to project root with migration fallbacks."""
    resolved = _resolve_path(path_str, project_root)
    alternate_paths = _alternate_package_dirs(resolved)
    alternate_paths.extend(
        _alternate_resource_model_configs(resolved, project_root, attr_name)
    )
    alternate_paths.extend(
        _alternate_legacy_model_checkpoints(resolved, project_root, attr_name)
    )

    if not resolved.exists():
        for alternate in alternate_paths:
            if alternate.exists():
                return str(alternate)

    return str(resolved)


def _find_project_root(config_dir: Path) -> Path:
    """Resolve the project root for configs at any nesting depth."""
    cursor = config_dir
    while cursor != cursor.parent:
        if cursor.name == "configs":
            return cursor.parent
        cursor = cursor.parent
    return config_dir


def resolve_paths(config: Config, project_root: Path) -> Config:
    """Resolve relative paths to absolute paths based on project root."""
    paths = config.paths
    dataset_name = config.dataset.name

    if not paths.root:
        paths.root = str(project_root / "dataset" / dataset_name)

    root = _resolve_path(paths.root, project_root)
    paths.root = str(root)

    run_name = config.run_name

    if not paths.videos:
        paths.videos = str(root / "videos")
    else:
        paths.videos = str(_resolve_path(paths.videos, project_root))

    if not paths.transcripts:
        paths.transcripts = str(root / "transcripts")
    else:
        paths.transcripts = str(_resolve_path(paths.transcripts, project_root))

    if not paths.manifest:
        paths.manifest = str(root / "manifest.csv")
    else:
        paths.manifest = str(_resolve_path(paths.manifest, project_root))

    if not paths.output:
        paths.output = str(root / "output")
    else:
        paths.output = str(_resolve_path(paths.output, project_root))

    if not paths.webdataset:
        paths.webdataset = str(root / "webdataset")
    else:
        paths.webdataset = str(_resolve_path(paths.webdataset, project_root))

    # Resolve source paths relative to project root
    source = config.dataset.source
    for source_key in (
        "video_ids_file", "manifest_tsv", "manifest_csv",
        "release_dir", "class_map_file", "annotations_csv",
        "annotations_dir", "metadata_json", "annotation_json", "bbox_json",
        "corpus_file", "split_spec_file",
    ):
        val = source.get(source_key, "")
        if val:
            config.dataset.source[source_key] = str(_resolve_path(val, project_root))

    # Resolve model paths in detection_config and pose_config
    proc = config.processing
    if proc.detection_config:
        for attr in ("det_model_config", "det_model_checkpoint"):
            val = getattr(proc.detection_config, attr, None)
            if val:
                setattr(
                    proc.detection_config, attr,
                    _resolve_model_path(val, project_root, attr),
                )

    if proc.pose_config:
        for attr in ("pose_model_config", "pose_model_checkpoint"):
            val = getattr(proc.pose_config, attr, None)
            if val:
                setattr(
                    proc.pose_config, attr,
                    _resolve_model_path(val, project_root, attr),
                )

    return config


def load_config(
    yaml_path: str,
    overrides: Optional[List[str]] = None,
    dict_overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """Load config from YAML file.

    Merge order (later overrides earlier):
    1. Pydantic defaults (hardcoded in schema)
    2. Base YAML(s), if declared via top-level ``base:``
    3. YAML config file
    4. CLI overrides (key=value pairs)
    5. Dict overrides (from experiment layer, already typed)
    """
    config_path = _coerce_path(yaml_path)
    if not config_path.is_absolute():
        config_path = Path(os.path.abspath(str(config_path)))
    config_path = config_path.resolve()
    config_dir = config_path.parent

    project_root = _find_project_root(config_dir)
    raw = _load_raw_config(str(config_path))

    # Apply CLI overrides
    if overrides:
        for override in overrides:
            if "=" not in override:
                raise ValueError(f"Override must be key=value, got: {override}")
            key, value = override.split("=", 1)
            key, parsed_value = _normalize_legacy_sampling_override(
                key, _parse_value(value),
            )
            _set_nested(raw, key, parsed_value)

    # Apply dict overrides (from experiment layer — values already typed)
    if dict_overrides:
        for key, value in dict_overrides.items():
            key, value = _normalize_legacy_sampling_override(key, value)
            _set_nested(raw, key, value)

    raw = _normalize_dataset_shorthand(raw)

    # Validate required fields
    dataset_raw = raw.get("dataset")
    if not dataset_raw:
        raise ValueError("Config must specify 'dataset' field")

    dataset_name = (
        dataset_raw if isinstance(dataset_raw, str)
        else dataset_raw.get("name") if isinstance(dataset_raw, dict)
        else None
    )
    if not dataset_name:
        raise ValueError("Config must specify 'dataset.name' field")

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    config = Config(**raw)
    config = resolve_paths(config, project_root)

    # Run dataset-specific validation against the final config
    dataset_cls = DATASET_REGISTRY[dataset_name]
    dataset_cls.validate_config(config)

    return config


def _set_nested(d: Dict, key: str, value: Any) -> None:
    """Set a nested dictionary value using dot-separated key."""
    parts = key.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


def _normalize_legacy_sampling_override(key: str, value: Any) -> tuple[str, Any]:
    """Translate legacy sampling override keys to sample_rate with warnings."""
    if key == "processing.frame_skip":
        frame_skip = int(value)
        if frame_skip <= 0:
            raise ValueError("processing.frame_skip must be a positive integer")
        sample_rate = None if frame_skip == 1 else (1.0 / frame_skip)
        warnings.warn(
            "processing.frame_skip is deprecated and was mapped to "
            f"processing.sample_rate={sample_rate!r}.",
            FutureWarning,
            stacklevel=2,
        )
        return "processing.sample_rate", sample_rate

    if key == "processing.target_fps":
        if value is not None and value <= 0:
            raise ValueError("processing.target_fps must be positive or null")
        warnings.warn(
            "processing.target_fps is deprecated and was mapped to "
            f"processing.sample_rate={value!r}.",
            FutureWarning,
            stacklevel=2,
        )
        return "processing.sample_rate", value

    return key, value


def _parse_value(value: str) -> Any:
    """Parse a string value to its appropriate Python type."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "none" or value.lower() == "null":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
