"""AUTSL dataset adapter."""

from pathlib import Path
from typing import Any, Dict

from ..base import DatasetAdapter
from ...registry import register_dataset
from . import manifest as _manifest
from . import source as _source


@register_dataset("autsl")
class AUTSLDataset(DatasetAdapter):
    """AUTSL isolated sign language dataset adapter."""

    name = "autsl"

    @classmethod
    def validate_raw_inputs(cls, raw: Dict[str, Any]) -> None:
        dataset_dict = raw.get("dataset") if isinstance(raw.get("dataset"), dict) else {}
        source_dict = dataset_dict.get("source") if isinstance(dataset_dict.get("source"), dict) else {}
        paths_dict = raw.get("paths") if isinstance(raw.get("paths"), dict) else {}

        release_dir = str(source_dict.get("release_dir", "") or "").strip()
        videos_dir = str(paths_dict.get("videos", "") or "").strip()
        if not (release_dir or videos_dir):
            raise ValueError(
                "autsl requires an explicit dataset.source.release_dir or paths.videos "
                "pointing to the local AUTSL release directory."
            )

    @classmethod
    def validate_config(cls, config) -> None:
        source = _source.get_source_config(config)
        if not (source.release_dir or config.paths.videos):
            raise ValueError(
                "autsl requires dataset.source.release_dir or paths.videos "
                "pointing to the local AUTSL release directory."
            )
    def get_source_config(self, config) -> _source.AUTSLSourceConfig:
        return _source.get_source_config(config)

    def resolve_videos_dir(self, config) -> Path | None:
        source = self.get_source_config(config)
        return _source.resolve_release_root(source, config)

    def download(self, config, context):
        source = self.get_source_config(config)
        stats = _source.validate(source, config, self.logger)
        context.videos_dir = Path(stats["release_root"])
        context.stats["dataset.download"] = stats
        return context

    def build_manifest(self, config, context):
        source = self.get_source_config(config)
        context.videos_dir = _source.resolve_release_root(source, config)
        df = _manifest.build(config, source, self.logger)
        context.manifest_path = Path(config.paths.manifest)
        context.manifest_df = df
        context.stats["dataset.manifest"] = {
            "videos": int(df["VIDEO_ID"].nunique()),
            "segments": len(df),
        }
        self.logger.info(
            "AUTSL manifest built: %d segments, %d videos -> %s",
            len(df), df["VIDEO_ID"].nunique(), config.paths.manifest,
        )
        return context
