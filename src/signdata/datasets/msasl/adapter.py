"""MS-ASL dataset adapter."""

from ..base import DatasetAdapter
from ...registry import register_dataset
from . import manifest as _manifest
from . import source as _source


@register_dataset("msasl")
class MSASLDataset(DatasetAdapter):
    """MS-ASL dataset adapter."""

    name = "msasl"

    @classmethod
    def validate_config(cls, config) -> None:
        source = config.dataset.source
        if not source.get("annotations_dir"):
            raise ValueError(
                "msasl requires dataset.source.annotations_dir pointing to "
                "the directory containing MSASL_train.json, MSASL_val.json, "
                "MSASL_test.json, and MSASL_classes.json"
            )

    def get_source_config(self, config) -> _source.MSASLSourceConfig:
        return _source.get_source_config(config)

    def download(self, config, context):
        source = self.get_source_config(config)

        if source.download_mode == "validate":
            stats = _source.validate(source, config, self.logger)
        else:
            stats = _source.download_missing(source, config, self.logger)

        context.stats["dataset.download"] = stats
        return context

    def build_manifest(self, config, context):
        source = self.get_source_config(config)
        df = _manifest.build(config, source)
        self._set_manifest(context, config.paths.manifest, df)
        context.stats["dataset.manifest"] = {
            "videos": int(df["VIDEO_ID"].nunique()),
            "segments": len(df),
        }
        self.logger.info(
            "MS-ASL manifest built: %d segments, %d videos -> %s",
            len(df), df["VIDEO_ID"].nunique(), config.paths.manifest,
        )
        return context
