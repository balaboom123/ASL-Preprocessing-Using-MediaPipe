"""YouTube-ASL dataset adapter."""

from pathlib import Path

from ..base import DatasetAdapter
from ...registry import register_dataset
from . import manifest as _manifest
from . import source as _source


@register_dataset("youtube_asl")
class YouTubeASLDataset(DatasetAdapter):
    name = "youtube_asl"

    @classmethod
    def validate_config(cls, config) -> None:
        source = config.dataset.source
        if not source.get("video_ids_file"):
            raise ValueError(
                "youtube_asl requires dataset.source.video_ids_file to be set"
            )

    def get_source_config(self, config) -> _source.YouTubeASLSourceConfig:
        return _source.get_source_config(config)

    def download(self, config, context):
        source = self.get_source_config(config)
        stats = _source.download(source, config, self.logger)
        context.stats["dataset.download"] = stats
        return context

    def build_manifest(self, config, context):
        source = self.get_source_config(config)
        manifest_path, df, stats = _manifest.build(config, source, self.logger)

        context.manifest_path = Path(manifest_path)
        context.manifest_df = df
        context.stats["dataset.manifest"] = stats
        self.logger.info(
            "Manifest built: %d videos, %d segments -> %s",
            stats["videos"], stats["segments"], manifest_path,
        )
        return context

    @staticmethod
    def _process_segments(*args, **kwargs):
        return _manifest._process_segments(*args, **kwargs)

    def _download_transcripts(self, video_id_file, transcript_dir, source):
        return _source._download_transcripts(
            video_id_file, transcript_dir, source, self.logger
        )

    @staticmethod
    def _build_transcript_proxies(source):
        return _source._build_transcript_proxies(source)

    @staticmethod
    def _build_transcript_client(source):
        return _source._build_transcript_client(source)

    @staticmethod
    def _fetch_transcript(
        transcript_client,
        transcript_api_cls,
        video_id,
        languages,
        proxies=None,
    ):
        return _source._fetch_transcript(
            transcript_client,
            transcript_api_cls,
            video_id,
            languages,
            proxies=proxies,
        )

    @staticmethod
    def _normalize_transcript_payload(transcript):
        return _source._normalize_transcript_payload(transcript)

    def _download_videos(self, video_id_file, video_dir, source):
        return _source._download_videos(
            video_id_file, video_dir, source, self.logger
        )
