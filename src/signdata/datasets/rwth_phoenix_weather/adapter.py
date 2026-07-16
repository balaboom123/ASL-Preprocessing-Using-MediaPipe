"""RWTH-PHOENIX-Weather 2014-T dataset adapter."""

from ..base import DatasetAdapter
from ...registry import register_dataset
from . import manifest as _manifest
from . import source as _source


@register_dataset("rwth_phoenix_weather")
class RWTHPhoenixWeatherDataset(DatasetAdapter):
    """RWTH-PHOENIX-Weather 2014-T dataset adapter."""

    name = "rwth_phoenix_weather"

    @classmethod
    def validate_config(cls, config) -> None:
        source = _source.get_source_config(config)
        release_dir = source.release_dir or config.paths.videos
        if not release_dir:
            raise ValueError(
                "rwth_phoenix_weather requires either "
                "dataset.source.release_dir or paths.videos pointing to the "
                "unpacked PHOENIX release directory."
            )

    def get_source_config(self, config) -> _source.RWTHPhoenixWeatherSourceConfig:
        return _source.get_source_config(config)

    def download(self, config, context):
        source = self.get_source_config(config)
        stats = _source.prepare(source, config, self.logger)
        context.stats["dataset.download"] = stats
        return context

    def build_manifest(self, config, context):
        source = self.get_source_config(config)
        df = _manifest.build(config, source, self.logger)
        self._set_manifest(context, config.paths.manifest, df)
        context.stats["dataset.manifest"] = {
            "videos": int(df["VIDEO_ID"].nunique()),
            "segments": len(df),
            "splits": list(df["SPLIT"].unique()),
        }
        self.logger.info(
            "PHOENIX manifest built: %d segments, %d videos -> %s",
            len(df), df["VIDEO_ID"].nunique(), config.paths.manifest,
        )
        return context
