"""Component registries for datasets and processors."""

from typing import Dict, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .datasets.base import DatasetAdapter
    from .processors.base import BaseProcessor

DATASET_REGISTRY: Dict[str, Type["DatasetAdapter"]] = {}
PROCESSOR_REGISTRY: Dict[str, Type["BaseProcessor"]] = {}


def register_dataset(name: str):
    """Register a dataset class under the given name."""
    def decorator(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator


def register_processor(name: str):
    """Register a processor class under the given name."""
    def decorator(cls):
        PROCESSOR_REGISTRY[name] = cls
        return cls
    return decorator
