"""YAML config loading."""

import copy
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
    raw = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))

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

    return [
        _resolve_path(ref, config_path.parent).resolve()
        for ref in refs
    ]


def _load_raw_config(
    yaml_path: str,
    stack: Optional[List[Path]] = None,
) -> Dict[str, Any]:
    """Load a YAML config with optional recursive base-config merging."""
    config_path = _coerce_path(yaml_path).resolve()
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


SOURCE_PATH_SUFFIXES = (
    "_file",
    "_dir",
    "_root",
    "_json",
    "_csv",
    "_tsv",
)


def _resolve_model_path(path_str: str, project_root: Path) -> str:
    """Resolve local model paths relative to project root."""
    if "://" in path_str or "::" in path_str:
        return path_str
    return str(_resolve_path(path_str, project_root))


def _find_project_root(config_dir: Path) -> Path:
    """Resolve the project root for configs at any nesting depth."""
    configs_dir = next(
        (path for path in (config_dir, *config_dir.parents) if path.name == "configs"),
        None,
    )
    return configs_dir.parent if configs_dir else config_dir


def resolve_paths(config: Config, project_root: Path) -> Config:
    """Resolve relative paths to absolute paths based on project root."""
    paths = config.paths
    dataset_name = config.dataset.name

    if not paths.root:
        paths.root = str(project_root / "dataset" / dataset_name)

    root = _resolve_path(paths.root, project_root)
    paths.root = str(root)

    for field, default in {
        "videos": "videos",
        "transcripts": "transcripts",
        "manifest": "manifest.csv",
        "output": "output",
        "webdataset": "webdataset",
    }.items():
        value = getattr(paths, field)
        resolved = _resolve_path(value, project_root) if value else root / default
        setattr(paths, field, str(resolved))

    # Resolve source paths relative to project root
    source = config.dataset.source
    for source_key, val in source.items():
        if source_key.endswith(SOURCE_PATH_SUFFIXES) and isinstance(val, str) and val:
            config.dataset.source[source_key] = str(_resolve_path(val, project_root))

    # Resolve model paths in detection_config and pose_config
    proc = config.processing
    for backend_config in (proc.detection_config, proc.pose_config):
        if not backend_config:
            continue
        for attr in (
            "det_model_config",
            "det_model_checkpoint",
            "pose_model_config",
            "pose_model_checkpoint",
        ):
            val = getattr(backend_config, attr, None)
            if val:
                setattr(backend_config, attr, _resolve_model_path(val, project_root))

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
    config_path = _coerce_path(yaml_path).resolve()
    config_dir = config_path.parent

    project_root = _find_project_root(config_dir)
    raw = _load_raw_config(str(config_path))

    # Apply CLI overrides
    if overrides:
        for override in overrides:
            if "=" not in override:
                raise ValueError(f"Override must be key=value, got: {override}")
            key, value = override.split("=", 1)
            _set_nested(raw, key, _parse_value(value))

    # Apply dict overrides (from experiment layer — values already typed)
    if dict_overrides:
        for key, value in dict_overrides.items():
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

    DATASET_REGISTRY[dataset_name].validate_raw_inputs(raw)

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


def _parse_value(value: str) -> Any:
    """Parse a string value to its appropriate Python type."""
    if not value:
        return value
    return yaml.safe_load("null" if value.lower() == "none" else value)
