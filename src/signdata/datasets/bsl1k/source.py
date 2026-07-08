"""BSL-1K compatibility source config."""

from ..bobsl.source import BOBSLSourceConfig as BSL1KSourceConfig


def get_source_config(config) -> BSL1KSourceConfig:
    source = dict(config.dataset.source)
    source.setdefault("view", "isolated_signs")
    return BSL1KSourceConfig(**source)
