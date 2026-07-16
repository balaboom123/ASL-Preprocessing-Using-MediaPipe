"""Shared CUDA helpers for detection backends."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def is_cuda_device(device: str) -> bool:
    """Return True when the configured device targets CUDA."""
    return str(device).startswith("cuda")


def parse_cuda_device_index(device: str) -> Optional[int]:
    """Parse ``cuda`` / ``cuda:N`` strings into a numeric device index."""
    if not is_cuda_device(device):
        return None

    text = str(device).strip()
    if text == "cuda":
        return 0

    prefix, sep, suffix = text.partition(":")
    if prefix != "cuda" or not sep or not suffix:
        raise RuntimeError(
            f"Invalid CUDA device {device!r}. Use 'cuda' or 'cuda:<index>'."
        )

    try:
        return int(suffix)
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid CUDA device {device!r}. Use 'cuda' or 'cuda:<index>'."
        ) from exc


def validate_cuda_device(device: str) -> None:
    """Raise a clear error when a requested CUDA device is unavailable."""
    if not is_cuda_device(device):
        return

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {device!r} requested but CUDA is unavailable. "
            "Verify your PyTorch CUDA install and GPU visibility, or switch "
            "detection_config.device to 'cpu'."
        )

    index = parse_cuda_device_index(device)
    if index is None:
        return

    device_count = torch.cuda.device_count()
    if index < 0 or index >= device_count:
        raise RuntimeError(
            f"CUDA device {device!r} requested, but only {device_count} visible "
            "CUDA device(s) are available."
        )


def describe_device(device: str) -> str:
    """Return a user-facing device description for logs."""
    text = str(device)
    if not is_cuda_device(text):
        return text

    try:
        import torch

        if not torch.cuda.is_available():
            return text

        index = parse_cuda_device_index(text)
        if index is None:
            return text
        if index < 0 or index >= torch.cuda.device_count():
            return text

        return f"{text} (GPU {index}: {torch.cuda.get_device_name(index)})"
    except Exception:
        logger.debug("Unable to resolve CUDA device name", exc_info=True)
        return text


def is_cuda_oom_error(exc: BaseException) -> bool:
    """Return True when an exception message matches a CUDA OOM failure."""
    message = str(exc).lower()
    return "out of memory" in message or "cuda oom" in message


def clear_cuda_cache(device: str) -> None:
    """Best-effort CUDA cache clear for retry paths, scoped to ``device``."""
    if not is_cuda_device(device):
        return
    try:
        import torch
    except Exception:
        logger.debug("Unable to import torch for CUDA cache clear", exc_info=True)
        return

    index = parse_cuda_device_index(device)
    try:
        if index is not None and torch.cuda.is_available():
            with torch.cuda.device(index):
                torch.cuda.empty_cache()
                return
    except Exception:
        logger.debug(
            "Unable to clear CUDA cache on device %s; falling back to global clear.",
            device,
            exc_info=True,
        )

    try:
        torch.cuda.empty_cache()
    except Exception:
        logger.debug("Unable to clear CUDA cache", exc_info=True)


def format_effective_batch_size_message(
    *,
    backend: str,
    device: str,
    previous_batch_size: int,
    new_batch_size: int,
) -> str:
    """Describe an adaptive batch-size reduction after CUDA OOM."""
    return (
        f"{backend} lowered the effective batch size for this run from "
        f"{previous_batch_size} to {new_batch_size} on {describe_device(device)} "
        "after CUDA out of memory. To reduce VRAM usage and avoid repeated "
        "retries, set processing.detection_config.batch_size to "
        f"{new_batch_size} or lower."
    )


def format_cuda_oom_message(
    *,
    backend: str,
    device: str,
    configured_batch_size: int,
    attempted_batch_size: int,
    model: Optional[str] = None,
    learned_batch_size: Optional[int] = None,
) -> str:
    """Return an actionable terminal CUDA OOM message for users."""
    model_suffix = f" with model {model}" if model else ""
    parts = [
        f"CUDA out of memory during {backend} inference on "
        f"{describe_device(device)}{model_suffix}.",
        (
            "Configured processing.detection_config.batch_size="
            f"{configured_batch_size}; last attempted batch size="
            f"{attempted_batch_size}."
        ),
    ]

    if learned_batch_size is not None and learned_batch_size < configured_batch_size:
        parts.append(
            "This run only fit batches up to "
            f"{learned_batch_size}. Reduce processing.detection_config.batch_size "
            f"to {learned_batch_size} or lower to reduce VRAM usage and avoid "
            "repeated retries."
        )
    elif configured_batch_size > 1:
        parts.append(
            "Reduce processing.detection_config.batch_size to lower VRAM usage."
        )
    else:
        parts.append(
            "VRAM is insufficient even at processing.detection_config.batch_size=1."
        )

    parts.append(
        "If that is still not enough, use a smaller model, lower the input "
        "resolution or sample rate, choose another GPU, or switch to CPU."
    )
    return " ".join(parts)


__all__ = [
    "clear_cuda_cache",
    "describe_device",
    "format_cuda_oom_message",
    "format_effective_batch_size_message",
    "is_cuda_device",
    "is_cuda_oom_error",
    "parse_cuda_device_index",
    "validate_cuda_device",
]
