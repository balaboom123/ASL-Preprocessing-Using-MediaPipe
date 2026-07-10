"""Pipeline processors."""

from .base import BaseProcessor
from .video2pose import Video2PoseProcessor
from .video2crop import Video2CropProcessor
from .video2compression import Video2CompressionProcessor
from .video2parts import Video2PartsProcessor

__all__ = [
    "BaseProcessor",
    "Video2PoseProcessor",
    "Video2CropProcessor",
    "Video2CompressionProcessor",
    "Video2PartsProcessor",
]
