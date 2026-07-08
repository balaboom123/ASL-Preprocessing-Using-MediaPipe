"""LSA64 dataset adapter."""

from pathlib import Path
from typing import Any, Dict

from ..base import DatasetAdapter
from ...registry import register_dataset
from . import manifest as _manifest
from . import source as _source


@register_dataset("lsa64")
class LSA64Dataset(DatasetAdapter):
    """LSA64 Argentinian Sign Language dataset adapter."""

    name = "lsa64"

    @classmethod
    def _get_validated_source_config(cls, config) -> _source.LSA64SourceConfig:
        cls.validate_config(config)
        return _source.get_source_config(config)

    @classmethod
    def validate_raw_inputs(cls, raw: Dict[str, Any]) -> None:
        dataset_dict = raw.get("dataset") if isinstance(raw.get("dataset"), dict) else {}
        source_dict = dataset_dict.get("source") if isinstance(dataset_dict.get("source"), dict) else {}
        paths_dict = raw.get("paths") if isinstance(raw.get("paths"), dict) else {}

        release_dir = str(source_dict.get("release_dir", "") or "").strip()
        videos_dir = str(paths_dict.get("videos", "") or "").strip()
        if not (release_dir or videos_dir):
            raise ValueError(
                "lsa64 requires an explicit dataset.source.release_dir or paths.videos "
                "pointing to the local LSA64 release directory."
            )

    @classmethod
    def validate_config(cls, config) -> None:
        source = _source.get_source_config(config)
        if _source.resolve_release_dir(config, source) is None:
            raise ValueError(
                "lsa64 requires dataset.source.release_dir or paths.videos "
                "pointing to the local LSA64 release directory."
            )
        _source.validate_variant_path_consistency(config, source)

    def get_source_config(self, config) -> _source.LSA64SourceConfig:
        return _source.get_source_config(config)

    def resolve_videos_dir(self, config) -> Path | None:
        source = self._get_validated_source_config(config)
        return _source.resolve_video_dir(config, source)

    def download(self, config, context):
        source = self._get_validated_source_config(config)
        video_dir = _source.resolve_video_dir(config, source)
        stats = _source.validate_release(source, video_dir, self.logger)
        context.videos_dir = video_dir
        context.stats["dataset.download"] = stats
        return context

    def build_manifest(self, config, context):
        source = self._get_validated_source_config(config)
        video_dir = _source.resolve_video_dir(config, source)
        context.videos_dir = video_dir
        df = _manifest.build(config, source, video_dir, self.logger)
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
        source = self._get_validated_source_config(config)
        _source.validate_loaded_manifest_variant(
            context.manifest_df,
            context.manifest_path,
            source,
        )
