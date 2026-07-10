"""SLoVo dataset adapter."""

from pathlib import Path

from ..base import DatasetAdapter
from ...registry import register_dataset
from . import manifest as _manifest
from . import source as _source


@register_dataset("slovo")
class SlovoDataset(DatasetAdapter):
    """SLoVo Russian Sign Language dataset adapter."""

    name = "slovo"

    @classmethod
    def validate_config(cls, config) -> None:
        source = _source.get_source_config(config)
        if not _source.resolve_release_dir(source, config):
            raise ValueError(
                "slovo requires dataset.source.release_dir or paths.videos "
                "pointing to the local SLoVo release directory."
            )

    def get_source_config(self, config) -> _source.SlovoSourceConfig:
        return _source.get_source_config(config)

    def resolve_videos_dir(self, config):
        source = self.get_source_config(config)
        release_dir = _source.resolve_release_dir(source, config)
        return Path(release_dir) if release_dir else None

    def download(self, config, context):
        source = self.get_source_config(config)
        stats = _source.validate(source, config, self.logger)
        context.stats["dataset.download"] = stats
        return context

    def build_manifest(self, config, context):
        source = self.get_source_config(config)
        df = _manifest.build(config, source, self.logger)
        context.manifest_path = Path(config.paths.manifest)
        context.manifest_df = df
        context.stats["dataset.manifest"] = {
            "videos": int(df["VIDEO_ID"].nunique()),
            "segments": len(df),
            "split": source.split,
            "class_map_mode": source.class_map_mode,
        }
        self.logger.info(
            "SLoVo manifest built: %d segments, %d unique signers -> %s",
            len(df), df["SIGNER_ID"].nunique(), config.paths.manifest,
        )
        return context
