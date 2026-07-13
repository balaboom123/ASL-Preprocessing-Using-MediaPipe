"""OpenASL dataset adapter."""

from pathlib import Path

from ..base import DatasetAdapter
from ...registry import register_dataset
from . import manifest as _manifest
from . import source as _source


@register_dataset("openasl")
class OpenASLDataset(DatasetAdapter):
    """OpenASL dataset adapter."""

    name = "openasl"

    @classmethod
    def validate_config(cls, config) -> None:
        source = config.dataset.source
        if not source.get("manifest_tsv"):
            raise ValueError(
                "openasl requires dataset.source.manifest_tsv pointing to "
                "the official openasl-v1.0.tsv file"
            )

    def get_source_config(self, config) -> _source.OpenASLSourceConfig:
        return _source.get_source_config(config)

    def download(self, config, context):
        source = self.get_source_config(config)
        stats = _source.download(source, config, self.logger)
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
        }
        self.logger.info(
            "OpenASL manifest built: %d segments, %d videos -> %s",
            len(df), df["VIDEO_ID"].nunique(), config.paths.manifest,
        )
        return context
