"""YOLO model resolver: validates aliases, suggests typo fixes, manages cache."""

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

# Families whose official name drops the "v" (yolo11, yolo26) are listed
# without "v"; families that keep it (yolov8, yolov9) include "v".
SUPPORTED_FAMILIES: dict[str, list[str]] = {
    "yolov8": ["n", "s", "m", "l", "x"],
    "yolov9": ["t", "s", "m", "c", "e"],
    "yolo11": ["n", "s", "m", "l", "x"],
    "yolo26": ["n", "s", "m", "l", "x"],
}

VALID_ALIASES: set[str] = {
    f"{family}{size}.pt"
    for family, sizes in SUPPORTED_FAMILIES.items()
    for size in sizes
}

_HTTP_SCHEMES = {"http", "https"}
_TRITON_SCHEMES = _HTTP_SCHEMES | {"grpc"}
_TYPO_RE = re.compile(r"^yolov(\d+)([a-z])\.pt$")


def _normalize_stem(model: str) -> str:
    """Ultralytics accepts bare stems but always normalizes to .pt internally."""
    if not model.endswith(".pt"):
        return model + ".pt"
    return model


@lru_cache(maxsize=1)
def _hub_web_root() -> str:
    try:
        from ultralytics.hub import HUB_WEB_ROOT
        return HUB_WEB_ROOT
    except Exception:
        return "https://hub.ultralytics.com"


def _suggest_correction(model: str) -> Optional[tuple[str, str]]:
    """Return (corrected_alias, typo_family) for mistyped yolov11*/yolov26*, or None."""
    m = _TYPO_RE.match(model)
    if m:
        digits, size = m.group(1), m.group(2)
        candidate = f"yolo{digits}{size}.pt"
        if candidate in VALID_ALIASES:
            return candidate, f"yolov{digits}"
    return None


def _get_ultralytics_weights_dir() -> Optional[Path]:
    try:
        from ultralytics import settings
        raw = settings.get("weights_dir", None)
        if raw:
            return Path(str(raw))
    except Exception:
        pass
    return None


@lru_cache(maxsize=1)
def _get_ultralytics_asset_stems() -> Optional[frozenset[str]]:
    try:
        from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS
        return frozenset(GITHUB_ASSETS_STEMS)
    except Exception:
        return None


@lru_cache(maxsize=1)
def _get_ultralytics_version() -> Optional[str]:
    try:
        from ultralytics import __version__
        return __version__
    except Exception:
        return None


def _check_installed_alias_support(alias: str) -> None:
    """Raise RuntimeError if the installed Ultralytics package lacks this alias."""
    asset_stems = _get_ultralytics_asset_stems()
    if asset_stems is None:
        return

    stem = alias.removesuffix(".pt")
    if stem not in asset_stems:
        installed = _get_ultralytics_version() or "unknown"
        raise RuntimeError(
            f"YOLO model {alias!r} is an official alias for this backend, "
            f"but the installed ultralytics {installed} package does not "
            "include it in the official asset catalogue. Upgrade ultralytics "
            "to a version that supports this model family."
        )


def is_valid_alias(model: str) -> bool:
    """Check if model string is a recognized Ultralytics detect alias.

    Accepts both bare stems (``yolo11m``) and ``.pt`` form (``yolo11m.pt``).
    """
    return _normalize_stem(model) in VALID_ALIASES


def resolve_yolo_model(
    model: str,
    *,
    allow_download: bool = True,
    weights_dir: Optional[str] = None,
) -> str:
    """Resolve a YOLO model string to a loadable path or alias.

    Args:
        model: Local file path or Ultralytics alias (e.g. ``yolo11m.pt``
            or bare stem ``yolo11m``).
        allow_download: If False, only accept existing local files.
        weights_dir: Extra directory to check for cached weights.
            This is a read-only search path — it does not mutate
            Ultralytics' global settings.

    Returns:
        The resolved model string (always ``.pt`` normalised) to pass to
        ``YOLO()``.

    Raises:
        FileNotFoundError: Model file not found locally when download is
            disabled.
        ValueError: Model alias is not supported or appears to be a typo.
        RuntimeError: Model alias is not supported by the installed
            Ultralytics package.
    """
    model = model.strip()
    local_path = Path(model)

    if local_path.is_file():
        logger.info("YOLO model resolved to local file: %s", local_path)
        return str(local_path)

    url = urlsplit(model)
    if model.startswith(f"{_hub_web_root()}/models/"):
        if not allow_download:
            raise FileNotFoundError(
                f"YOLO model {model!r} is a remote Ultralytics HUB reference. "
                "Set allow_download=True or provide a local weights path."
            )
        return model

    if (
        url.scheme in _TRITON_SCHEMES
        and url.netloc
        and "/v2/models/" in url.path
    ):
        return model

    if url.scheme in _HTTP_SCHEMES and url.path.endswith((".pt", ".pth")):
        if not allow_download:
            raise FileNotFoundError(
                f"YOLO model {model!r} is a remote weights URL. "
                "Set allow_download=True or provide a local weights path."
            )
        return model

    normalised = _normalize_stem(model)

    if weights_dir:
        cached = Path(weights_dir) / normalised
        if cached.is_file():
            logger.info("YOLO model resolved from cache: %s", cached)
            return str(cached)

    suggestion = _suggest_correction(normalised)
    if suggestion:
        corrected, typo_family = suggestion
        official_family = corrected.removesuffix(".pt")[:-1]
        raise ValueError(
            f"Unknown YOLO model {model!r}. Did you mean {corrected!r}? "
            f"{official_family.upper()} uses the naming scheme "
            f"'{official_family}<size>.pt', not '{typo_family}<size>.pt'."
        )

    if normalised in VALID_ALIASES:
        _check_installed_alias_support(normalised)

        if not allow_download:
            ul_dir = _get_ultralytics_weights_dir()
            if ul_dir:
                candidate = ul_dir / normalised
                if candidate.is_file():
                    logger.info(
                        "YOLO model found in Ultralytics cache "
                        "(download disabled): %s",
                        candidate,
                    )
                    return str(candidate)

            raise FileNotFoundError(
                f"YOLO model {normalised!r} is a valid alias but "
                f"allow_download=False and no cached weights found. Either "
                f"set allow_download=True or provide a local path."
            )

        logger.info(
            "YOLO model %r is a valid alias; Ultralytics will download if needed.",
            normalised,
        )
        return normalised

    raise ValueError(
        f"YOLO model {model!r} is not an existing file and not a recognized "
        f"Ultralytics detect alias. Supported aliases: "
        f"{', '.join(sorted(VALID_ALIASES))}"
    )


__all__ = [
    "SUPPORTED_FAMILIES",
    "VALID_ALIASES",
    "is_valid_alias",
    "resolve_yolo_model",
]
