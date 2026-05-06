"""LSA64 dataset adapter."""

from pathlib import Path

from ..base import DatasetAdapter
from ...registry import register_dataset
from . import manifest as _manifest
from . import source as _source


@register_dataset("lsa64")
class LSA64Dataset(DatasetAdapter):
    """LSA64 Argentinian Sign Language dataset adapter."""

    name = "lsa64"

    @classmethod
    def validate_config(cls, config) -> None:
        source = cls().get_source_config(config)
        if _source.resolve_release_dir(config, source) is None:
            raise ValueError(
                "lsa64 requires dataset.source.release_dir or paths.videos "
                "pointing to the local LSA64 release directory."
            )
        _source.validate_variant_path_consistency(config, source)

    def get_source_config(self, config) -> _source.LSA64SourceConfig:
        return _source.get_source_config(config)

    def resolve_videos_dir(self, config) -> Path | None:
        source = self.get_source_config(config)
        return _source.resolve_video_dir(config, source)

    @staticmethod
    def _sync_video_dir(context, video_dir: Path | None) -> None:
        context.videos_dir = video_dir

    def download(self, config, context):
        source = self.get_source_config(config)
        video_dir = self.resolve_videos_dir(config)
        stats = _source.validate_release(source, video_dir, self.logger)
        self._sync_video_dir(context, video_dir)
        context.stats["dataset.download"] = stats
        return context

    def build_manifest(self, config, context):
        source = self.get_source_config(config)
        video_dir = self.resolve_videos_dir(config)
        self._sync_video_dir(context, video_dir)
        df = _manifest.build(config, source, self.logger)
        context.manifest_path = Path(config.paths.manifest)
        context.manifest_df = df
        context.stats["dataset.manifest"] = {
            "videos": int(df["VIDEO_ID"].nunique()),
            "segments": len(df),
            "classes": int(df["CLASS_ID"].nunique()),
            "signers": int(df["SIGNER_ID"].nunique()),
        }
        self.logger.info(
            "LSA64 manifest built: %d segments, %d classes, %d signers -> %s",
            len(df), df["CLASS_ID"].nunique(), df["SIGNER_ID"].nunique(),
            config.paths.manifest,
        )
        return context

    def validate_loaded_manifest(self, config, context) -> None:
        if context.manifest_df is None:
            return
        source = self.get_source_config(config)
        _source.validate_loaded_manifest_variant(
            context.manifest_df,
            context.manifest_path,
            source,
        )
