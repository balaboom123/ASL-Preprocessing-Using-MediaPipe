"""WLASL dataset adapter."""

from pathlib import Path

from ..base import DatasetAdapter
from ...registry import register_dataset
from . import manifest as _manifest
from . import source as _source


@register_dataset("wlasl")
class WLASLDataset(DatasetAdapter):
    """WLASL isolated sign language dataset adapter."""

    name = "wlasl"

    @classmethod
    def validate_config(cls, config) -> None:
        source = config.dataset.source
        if not source.get("metadata_json") and not source.get("annotation_json"):
            raise ValueError(
                "wlasl requires dataset.source.metadata_json pointing to "
                "the official WLASL_v0.3.json file "
                "(annotation_json is accepted as a compatibility alias)"
            )

    def get_source_config(self, config) -> _source.WLASLSourceConfig:
        return _source.get_source_config(config)

    def download(self, config, context):
        source = self.get_source_config(config)

        if source.download_mode == "validate":
            stats = _source.validate_release(source, config, self.logger)
        elif source.download_mode == "download_missing":
            stats = _source.download_missing(source, config, self.logger)
        else:
            raise ValueError(
                f"Unknown download_mode '{source.download_mode}'. "
                f"Expected 'validate' or 'download_missing'."
            )

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
            "WLASL manifest built: %d segments, %d videos -> %s",
            len(df), df["VIDEO_ID"].nunique(), config.paths.manifest,
        )
        return context
