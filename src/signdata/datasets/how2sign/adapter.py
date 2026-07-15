"""How2Sign dataset adapter."""

from ..base import DatasetAdapter
from ...registry import register_dataset
from . import manifest as _manifest
from . import source as _source


@register_dataset("how2sign")
class How2SignDataset(DatasetAdapter):
    name = "how2sign"

    def get_source_config(self, config) -> _source.How2SignSourceConfig:
        return _source.get_source_config(config)

    def download(self, config, context):
        source = self.get_source_config(config)
        stats = _source.validate(source, config, self.logger)
        context.stats["dataset.download"] = stats
        return context

    def build_manifest(self, config, context):
        source = self.get_source_config(config)
        manifest_path, df = _manifest.build(config, source)

        self._set_manifest(context, manifest_path, df)
        context.stats["dataset.manifest"] = {
            "videos": df["VIDEO_ID"].nunique() if "VIDEO_ID" in df.columns else 0,
            "segments": len(df),
        }
        self.logger.info(
            "How2Sign manifest loaded: %d segments from %s",
            len(df), manifest_path,
        )
        return context
