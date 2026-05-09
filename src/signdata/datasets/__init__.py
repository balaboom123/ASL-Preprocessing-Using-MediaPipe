"""Dataset definitions with config validation."""

import importlib
import pkgutil

from .base import DatasetAdapter, BaseDataset

# Auto-discover and import all dataset packages/modules to trigger @register_dataset.
for _, _module_name, _ in pkgutil.iter_modules(__path__):
    if _module_name != "base" and not _module_name.startswith("_"):
        importlib.import_module(f".{_module_name}", __package__)

from .youtube_asl import YouTubeASLDataset
from .how2sign import How2SignDataset
from .openasl import OpenASLDataset
from .autsl import AUTSLDataset
from .lsa64 import LSA64Dataset
from .slovo import SlovoDataset
from .rwth_phoenix_weather import RWTHPhoenixWeatherDataset
from .wlasl import WLASLDataset
from .csl import CSLDataset
from .msasl import MSASLDataset
from .bobsl import BOBSLDataset
from .bsl1k import BSL1KDataset

__all__ = [
    "DatasetAdapter",
    "BaseDataset",
    "YouTubeASLDataset",
    "How2SignDataset",
    "OpenASLDataset",
    "AUTSLDataset",
    "LSA64Dataset",
    "SlovoDataset",
    "RWTHPhoenixWeatherDataset",
    "WLASLDataset",
    "CSLDataset",
    "MSASLDataset",
    "BOBSLDataset",
    "BSL1KDataset",
]
