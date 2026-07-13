"""CSL dataset adapter."""

from pathlib import Path

from ..base import DatasetAdapter
from ...registry import register_dataset
from . import manifest as _manifest
from . import source as _source


@register_dataset("csl")
class CSLDataset(DatasetAdapter):
    """CSL Chinese Sign Language dataset adapter."""

    name = "csl"

    @classmethod
    def validate_config(cls, config) -> None:
        source = _source.get_source_config(config)
        release_dir = source.release_dir or config.paths.root
        if not release_dir:
            raise ValueError(
                "csl requires dataset.source.release_dir or paths.root "
                "pointing to the local CSL release directory."
            )
        _source.validate_source_config(source)

    def get_source_config(self, config) -> _source.CSLSourceConfig:
        return _source.get_source_config(config)

    def resolve_videos_dir(self, config) -> Path | None:
        source = self.get_source_config(config)
        return _source.resolve_runtime_video_dir(source, config)

    def download(self, config, context):
        source = self.get_source_config(config)
        stats = _source.prepare(source, config, self.logger)
        runtime_dir = Path(stats["runtime_video_dir"])
        context.videos_dir = runtime_dir
        context.stats["dataset.download"] = stats
        return context

    def build_manifest(self, config, context):
        source = self.get_source_config(config)
        runtime_dir = _source.resolve_runtime_video_dir(source, config)
        context.videos_dir = runtime_dir
        df = _manifest.build(config, source, self.logger)
        context.manifest_path = Path(config.paths.manifest)
        context.manifest_df = df
        context.stats["dataset.manifest"] = {
            "videos": int(df["VIDEO_ID"].nunique()),
            "segments": len(df),
            "signers": int(df["SIGNER_ID"].nunique()) if "SIGNER_ID" in df.columns else 0,
            "protocol": source.protocol,
            "runtime_video_dir": str(runtime_dir),
        }
        self.logger.info(
            "CSL manifest built: %d segments, %d signers -> %s",
            len(df),
            df["SIGNER_ID"].nunique(),
            config.paths.manifest,
        )
        return context
