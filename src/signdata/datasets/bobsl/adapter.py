"""BOBSL dataset adapter."""

from ..base import DatasetAdapter
from ...registry import register_dataset
from . import manifest as _manifest
from . import source as _source


@register_dataset("bobsl")
class BOBSLDataset(DatasetAdapter):
    """BOBSL subtitle/signing dataset adapter."""

    name = "bobsl"

    @classmethod
    def validate_config(cls, config) -> None:
        source = _source.get_source_config(config)
        if not source.release_dir:
            raise ValueError(
                "bobsl requires dataset.source.release_dir pointing to the "
                "downloaded BOBSL release root."
            )
        if source.split == "challenge" and (
            config.dataset.download or config.dataset.manifest
        ):
            raise ValueError(
                "The public BOBSL challenge partition does not include the "
                "subtitle or isolated-sign annotation files required by the "
                "current local-validation and manifest-building workflows."
            )

    def get_source_config(self, config) -> _source.BOBSLSourceConfig:
        return _source.get_source_config(config)

    def download(self, config, context):
        source = self.get_source_config(config)
        stats = _source.validate_release(source, config, self.logger)
        context.stats["dataset.download"] = stats
        return context

    def build_manifest(self, config, context):
        source = self.get_source_config(config)
        df = _manifest.build(config, source, self.logger)
        self._set_manifest(context, config.paths.manifest, df)
        context.stats["dataset.manifest"] = {
            "videos": int(df["VIDEO_ID"].nunique()),
            "segments": len(df),
            "splits": list(df["SPLIT"].unique()) if "SPLIT" in df.columns else [],
        }
        self.logger.info(
            "BOBSL manifest built: %d segments, %d videos -> %s",
            len(df), df["VIDEO_ID"].nunique(), config.paths.manifest,
        )
        return context
