"""BSL-1K compatibility adapter built on the public BOBSL release."""

from ...registry import register_dataset
from ..bobsl.adapter import BOBSLDataset
from . import source as _source


@register_dataset("bsl1k")
class BSL1KDataset(BOBSLDataset):
    """BSL-1K compatibility view over the BOBSL release."""

    name = "bsl1k"

    @classmethod
    def validate_config(cls, config) -> None:
        super().validate_config(config)
        view = config.dataset.source.get("view", "isolated_signs")
        if view != "isolated_signs":
            raise ValueError(
                "bsl1k only supports dataset.source.view='isolated_signs'."
            )

    def get_source_config(self, config) -> _source.BSL1KSourceConfig:
        return _source.get_source_config(config)
