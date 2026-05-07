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
        release_dir = source.release_dir or getattr(config.paths, "root", "")
        if not release_dir:
            raise ValueError(
                "csl requires dataset.source.release_dir or paths.root "
                "pointing to the local CSL release directory."
            )
        if source.variant not in _source.SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unsupported CSL variant: {source.variant!r}. "
                f"Valid options: {sorted(_source.SUPPORTED_VARIANTS)}."
            )
        if source.protocol not in _source.SUPPORTED_PROTOCOLS:
            raise ValueError(
                f"Unsupported CSL protocol: {source.protocol!r}. "
                f"Valid options: {sorted(_source.SUPPORTED_PROTOCOLS)}."
            )
        if source.split not in _source.SUPPORTED_SPLITS:
            raise ValueError(
                f"Unsupported CSL split: {source.split!r}. "
                f"Valid options: {sorted(_source.SUPPORTED_SPLITS)}."
            )
        if source.prepare_mode not in _source.SUPPORTED_PREPARE_MODES:
            raise ValueError(
                f"Unsupported CSL prepare_mode: {source.prepare_mode!r}. "
                f"Valid options: {sorted(_source.SUPPORTED_PREPARE_MODES)}."
            )
        if source.split_spec_file and not Path(source.split_spec_file).exists():
            raise ValueError(
                f"CSL split_spec_file not found: {source.split_spec_file}"
            )

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
            "variant": source.variant,
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
