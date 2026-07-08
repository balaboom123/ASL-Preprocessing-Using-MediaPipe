"""BSL-1K compatibility adapter built on the public BOBSL release."""

from pathlib import Path

from ...registry import register_dataset
from ..bobsl.adapter import BOBSLDataset
from . import source as _source


@register_dataset("bsl1k")
class BSL1KDataset(BOBSLDataset):
    """BSL-1K compatibility view over the BOBSL release."""

    name = "bsl1k"

    @classmethod
    def _resolve_release_videos_dir(cls, config) -> Path | None:
        release_dir = str(config.dataset.source.get("release_dir", "") or "").strip()
        if not release_dir:
            return None
        return Path(release_dir) / "videos"

    @classmethod
    def _uses_implicit_videos_dir(cls, config) -> bool:
        videos = str(getattr(config.paths, "videos", "") or "").strip()
        if not videos:
            return True

        root = str(getattr(config.paths, "root", "") or "").strip()
        if not root:
            return False

        return Path(videos) == (Path(root) / "videos")

    @classmethod
    def validate_config(cls, config) -> None:
        super().validate_config(config)
        view = config.dataset.source.get("view", "isolated_signs")
        if view != "isolated_signs":
            raise ValueError(
                "bsl1k only supports dataset.source.view='isolated_signs'."
            )
        if cls._uses_implicit_videos_dir(config):
            release_videos = cls._resolve_release_videos_dir(config)
            if release_videos is not None:
                config.paths.videos = str(release_videos)

    def get_source_config(self, config) -> _source.BSL1KSourceConfig:
        return _source.get_source_config(config)

    def resolve_videos_dir(self, config) -> Path | None:
        if self._uses_implicit_videos_dir(config):
            release_videos = self._resolve_release_videos_dir(config)
            if release_videos is not None:
                return release_videos
        return super().resolve_videos_dir(config)
