"""Tests for dataset adapters (youtube_asl.py, how2sign.py).

Covers validate_config, get_source_config, download, and build_manifest.
"""

import json
import logging
import os
import sys
import types

import pandas as pd
import pytest

from signdata.config.schema import Config
from signdata.datasets.youtube_asl import (
    DEFAULT_DOWNLOAD_FORMAT,
    DEFAULT_TRANSCRIPT_LANGUAGES,
    YouTubeASLDataset,
    YouTubeASLSourceConfig,
)
from signdata.datasets.youtube_asl.source import _download_transcripts
from signdata.datasets._ingestion.youtube import _build_yt_config
from signdata.datasets.how2sign import How2SignDataset, How2SignSourceConfig
from signdata.datasets.base import DatasetAdapter
from signdata.pipeline.context import PipelineContext
from signdata.registry import DATASET_REGISTRY


# ── DatasetAdapter ABC ──────────────────────────────────────────────────────

class TestDatasetAdapterABC:
    def test_cannot_instantiate_abstract(self):
        """Cannot instantiate DatasetAdapter directly."""
        with pytest.raises(TypeError):
            DatasetAdapter()

    def test_registered_in_registry(self):
        assert "youtube_asl" in DATASET_REGISTRY
        assert "how2sign" in DATASET_REGISTRY

    def test_adapter_has_required_methods(self):
        """Adapters implement download, build_manifest, get_source_config."""
        adapter = YouTubeASLDataset()
        assert hasattr(adapter, "download")
        assert hasattr(adapter, "build_manifest")
        assert hasattr(adapter, "get_source_config")
        assert hasattr(adapter, "validate_config")


# ── YouTube-ASL validate_config ─────────────────────────────────────────────

class TestYouTubeASLValidateConfig:
    def test_valid_config_passes(self):
        cfg = Config(
            dataset={
                "name": "youtube_asl",
                "source": {"video_ids_file": "assets/ids.txt"},
            },
        )
        # Should not raise
        YouTubeASLDataset.validate_config(cfg)

    def test_missing_video_ids_file_raises(self):
        cfg = Config(
            dataset={
                "name": "youtube_asl",
                "source": {"video_ids_file": ""},
            },
        )
        with pytest.raises(ValueError, match="video_ids_file"):
            YouTubeASLDataset.validate_config(cfg)

    def test_default_source_raises(self):
        cfg = Config(dataset={"name": "youtube_asl"})
        with pytest.raises(ValueError, match="video_ids_file"):
            YouTubeASLDataset.validate_config(cfg)


# ── YouTube-ASL get_source_config ───────────────────────────────────────────

class TestYouTubeASLSourceConfig:
    def test_source_config_from_source_dict(self):
        """get_source_config parses config.dataset.source into typed model."""
        cfg = Config(
            dataset={
                "name": "youtube_asl",
                "source": {
                    "video_ids_file": "assets/ids.txt",
                    "languages": ["en", "ase"],
                    "rate_limit": "10M",
                    "max_text_length": 500,
                    "min_duration": 0.5,
                },
            },
        )
        adapter = YouTubeASLDataset()
        source = adapter.get_source_config(cfg)

        assert isinstance(source, YouTubeASLSourceConfig)
        assert source.video_ids_file == "assets/ids.txt"
        assert source.languages == ["en", "ase"]
        assert source.rate_limit == "10M"
        assert source.max_text_length == 500
        assert source.min_duration == 0.5
        assert source.max_duration == 60.0  # default

    def test_source_config_defaults(self):
        cfg = Config(dataset={"name": "youtube_asl"})
        adapter = YouTubeASLDataset()
        source = adapter.get_source_config(cfg)

        assert source.languages == DEFAULT_TRANSCRIPT_LANGUAGES
        assert source.download_format == DEFAULT_DOWNLOAD_FORMAT
        assert "worstaudio" in source.download_format
        assert source.concurrent_fragments == 5
        assert source.text_processing.fix_encoding is True
        assert source.text_processing.lowercase is False

    def test_source_config_text_processing(self):
        """Text processing fields flow into source config."""
        cfg = Config(
            dataset={
                "name": "youtube_asl",
                "source": {
                    "text_processing": {
                        "lowercase": True,
                        "strip_punctuation": True,
                    },
                },
            },
        )
        adapter = YouTubeASLDataset()
        source = adapter.get_source_config(cfg)

        assert source.text_processing.lowercase is True
        assert source.text_processing.strip_punctuation is True
        assert source.text_processing.fix_encoding is True  # default preserved

    def test_source_config_transcript_network_options(self):
        cfg = Config(
            dataset={
                "name": "youtube_asl",
                "source": {
                    "transcript_proxy_https": "http://proxy.example:8080",
                    "stop_on_transcript_block": False,
                },
            },
        )
        adapter = YouTubeASLDataset()
        source = adapter.get_source_config(cfg)

        assert source.transcript_proxy_http is None
        assert source.transcript_proxy_https == "http://proxy.example:8080"
        assert source.stop_on_transcript_block is False


def test_yt_dlp_config_uses_python_api_keys(tmp_path):
    config = _build_yt_config(
        str(tmp_path),
        "video-id",
        download_format="best",
        rate_limit=5_000_000,
        concurrent_fragments=3,
    )

    assert config["ratelimit"] == 5_000_000
    assert config["concurrent_fragment_downloads"] == 3
    assert "nocheckcertificate" not in config
    assert all("-" not in key for key in config)


class TestYouTubeASLTranscriptDownload:
    @staticmethod
    def _install_fake_youtube_transcript_api(
        monkeypatch,
        api_cls,
        blocked_error_cls,
    ):
        root = types.ModuleType("youtube_transcript_api")
        root.YouTubeTranscriptApi = api_cls

        errors = types.ModuleType("youtube_transcript_api._errors")
        errors.TranscriptsDisabled = type("TranscriptsDisabled", (Exception,), {})
        errors.NoTranscriptFound = type("NoTranscriptFound", (Exception,), {})
        errors.VideoUnavailable = type("VideoUnavailable", (Exception,), {})
        errors.RequestBlocked = blocked_error_cls
        errors.IpBlocked = blocked_error_cls

        formatters = types.ModuleType("youtube_transcript_api.formatters")

        class JSONFormatter:
            def format_transcript(self, transcript):
                return json.dumps(transcript.to_raw_data())

        formatters.JSONFormatter = JSONFormatter

        proxies = types.ModuleType("youtube_transcript_api.proxies")

        class GenericProxyConfig:
            def __init__(self, http_url=None, https_url=None):
                self.http_url = http_url
                self.https_url = https_url

        proxies.GenericProxyConfig = GenericProxyConfig

        monkeypatch.setitem(sys.modules, "youtube_transcript_api", root)
        monkeypatch.setitem(sys.modules, "youtube_transcript_api._errors", errors)
        monkeypatch.setitem(
            sys.modules, "youtube_transcript_api.formatters", formatters
        )
        monkeypatch.setitem(sys.modules, "youtube_transcript_api.proxies", proxies)

    def test_download_transcripts_accepts_list_payload(self, tmp_path, monkeypatch):
        class RequestBlocked(Exception):
            pass

        class DummyApi:
            def __init__(self, proxy_config=None):
                self.proxy_config = proxy_config

            def fetch(self, video_id, languages):
                return [{"text": "Hello", "start": 0.0, "duration": 1.5}]

        self._install_fake_youtube_transcript_api(
            monkeypatch, DummyApi, RequestBlocked
        )
        monkeypatch.setattr(
            "signdata.datasets.youtube_asl.source.time.sleep", lambda _: None
        )

        ids_file = tmp_path / "ids.txt"
        ids_file.write_text("vid001\n", encoding="utf-8")
        transcript_dir = tmp_path / "transcripts"
        transcript_dir.mkdir()

        source = YouTubeASLSourceConfig(video_ids_file=str(ids_file))
        stats = _download_transcripts(
            str(ids_file), str(transcript_dir), source, logging.getLogger("test")
        )

        assert stats["downloaded"] == 1
        assert stats["errors"] == 0
        assert stats["blocked"] is False
        assert json.loads((transcript_dir / "vid001.json").read_text()) == [
            {"text": "Hello", "start": 0.0, "duration": 1.5},
        ]

    def test_download_transcripts_stops_after_ip_block(
        self, tmp_path, monkeypatch
    ):
        class RequestBlocked(Exception):
            pass

        class DummyApi:
            def __init__(self, proxy_config=None):
                self.proxy_config = proxy_config

            def fetch(self, video_id, languages):
                raise RequestBlocked(video_id)

        self._install_fake_youtube_transcript_api(
            monkeypatch, DummyApi, RequestBlocked
        )
        monkeypatch.setattr(
            "signdata.datasets.youtube_asl.source.time.sleep", lambda _: None
        )

        ids_file = tmp_path / "ids.txt"
        ids_file.write_text("vid001\nvid002\n", encoding="utf-8")
        transcript_dir = tmp_path / "transcripts"
        transcript_dir.mkdir()

        source = YouTubeASLSourceConfig(
            video_ids_file=str(ids_file),
            stop_on_transcript_block=True,
        )
        stats = _download_transcripts(
            str(ids_file), str(transcript_dir), source, logging.getLogger("test")
        )

        assert stats["attempted"] == 1
        assert stats["downloaded"] == 0
        assert stats["errors"] == 1
        assert stats["blocked"] is True
        assert not (transcript_dir / "vid001.json").exists()


# ── YouTube-ASL build_manifest ──────────────────────────────────────────────

class TestYouTubeASLBuildManifest:
    def _make_context(self, config):
        adapter = YouTubeASLDataset()
        return PipelineContext(config=config, dataset=adapter)

    def test_build_manifest_from_transcripts(self, tmp_path):
        """build_manifest produces manifest from transcript JSON files."""
        transcript_dir = tmp_path / "transcripts"
        transcript_dir.mkdir()

        # Write a sample transcript
        transcript = [
            {"text": "Hello world", "start": 0.0, "duration": 2.0},
            {"text": "Second sentence", "start": 3.0, "duration": 1.5},
        ]
        (transcript_dir / "vid001.json").write_text(json.dumps(transcript))

        manifest_path = tmp_path / "manifest.csv"

        cfg = Config(
            dataset={"name": "youtube_asl"},
            paths={
                "transcripts": str(transcript_dir),
                "manifest": str(manifest_path),
            },
        )
        context = self._make_context(cfg)
        context = YouTubeASLDataset().build_manifest(cfg, context)

        assert context.manifest_path == manifest_path
        assert context.manifest_df is not None
        assert len(context.manifest_df) == 2
        assert "VIDEO_ID" in context.manifest_df.columns
        assert "SAMPLE_ID" in context.manifest_df.columns
        assert context.stats["dataset.manifest"]["segments"] == 2

    def test_build_manifest_no_transcripts(self, tmp_path):
        """build_manifest handles empty transcript directory."""
        transcript_dir = tmp_path / "transcripts"
        transcript_dir.mkdir()
        manifest_path = tmp_path / "manifest.csv"

        cfg = Config(
            dataset={"name": "youtube_asl"},
            paths={
                "transcripts": str(transcript_dir),
                "manifest": str(manifest_path),
            },
        )
        context = self._make_context(cfg)
        context = YouTubeASLDataset().build_manifest(cfg, context)

        assert context.stats["dataset.manifest"]["segments"] == 0

    def test_build_manifest_filters_by_duration(self, tmp_path):
        """build_manifest respects min/max duration."""
        transcript_dir = tmp_path / "transcripts"
        transcript_dir.mkdir()

        transcript = [
            {"text": "Too short", "start": 0.0, "duration": 0.05},
            {"text": "OK", "start": 1.0, "duration": 1.0},
            {"text": "Too long", "start": 5.0, "duration": 100.0},
        ]
        (transcript_dir / "vid001.json").write_text(json.dumps(transcript))

        manifest_path = tmp_path / "manifest.csv"
        cfg = Config(
            dataset={"name": "youtube_asl"},
            paths={
                "transcripts": str(transcript_dir),
                "manifest": str(manifest_path),
            },
        )
        context = self._make_context(cfg)
        context = YouTubeASLDataset().build_manifest(cfg, context)

        assert context.stats["dataset.manifest"]["segments"] == 1
        assert context.manifest_df.iloc[0]["TEXT"] == "OK"

    def test_build_manifest_text_processing_wired(self, tmp_path):
        """source.text_processing.lowercase=True flows through to output."""
        transcript_dir = tmp_path / "transcripts"
        transcript_dir.mkdir()

        transcript = [
            {"text": "Hello World!", "start": 0.0, "duration": 2.0},
        ]
        (transcript_dir / "vid001.json").write_text(json.dumps(transcript))

        manifest_path = tmp_path / "manifest.csv"
        cfg = Config(
            dataset={
                "name": "youtube_asl",
                "source": {
                    "text_processing": {
                        "lowercase": True,
                        "strip_punctuation": True,
                    },
                },
            },
            paths={
                "transcripts": str(transcript_dir),
                "manifest": str(manifest_path),
            },
        )
        context = self._make_context(cfg)
        context = YouTubeASLDataset().build_manifest(cfg, context)

        assert context.manifest_df.iloc[0]["TEXT"] == "hello world"


# ── How2Sign validate_config ───────────────────────────────────────────────

class TestHow2SignValidateConfig:
    def test_valid_config_passes(self):
        cfg = Config(dataset={"name": "how2sign"})
        # Should not raise — validate_config is a no-op
        How2SignDataset.validate_config(cfg)


# ── How2Sign get_source_config ──────────────────────────────────────────────

class TestHow2SignSourceConfig:
    def test_source_config_from_existing_config(self):
        cfg = Config(
            dataset={"name": "how2sign"},
            paths={"manifest": "/data/how2sign/manifest.csv"},
        )
        adapter = How2SignDataset()
        source = adapter.get_source_config(cfg)

        assert isinstance(source, How2SignSourceConfig)
        assert source.manifest_csv == "/data/how2sign/manifest.csv"

    def test_source_config_from_source_dict(self):
        cfg = Config(
            dataset={
                "name": "how2sign",
                "source": {"manifest_csv": "/data/manifest.csv"},
            },
        )
        adapter = How2SignDataset()
        source = adapter.get_source_config(cfg)

        assert source.manifest_csv == "/data/manifest.csv"

    def test_unknown_source_option_rejected(self):
        cfg = Config(
            dataset={"name": "how2sign", "source": {"split": "val"}},
        )

        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            How2SignDataset().get_source_config(cfg)


# ── How2Sign download ────────────────────────────────────────────────────────

class TestHow2SignDownload:
    def test_download_validates_existing_dir(self, tmp_path):
        video_dir = tmp_path / "videos"
        video_dir.mkdir()

        cfg = Config(
            dataset={"name": "how2sign"},
            paths={"videos": str(video_dir)},
        )
        adapter = How2SignDataset()
        context = PipelineContext(config=cfg, dataset=adapter)

        # Should not raise
        context = adapter.download(cfg, context)
        assert context.stats["dataset.download"]["validated"] is True

    def test_download_missing_dir_raises(self, tmp_path):
        cfg = Config(
            dataset={"name": "how2sign"},
            paths={"videos": str(tmp_path / "nonexistent")},
        )
        adapter = How2SignDataset()
        context = PipelineContext(config=cfg, dataset=adapter)

        with pytest.raises(FileNotFoundError, match="How2Sign"):
            adapter.download(cfg, context)


# ── How2Sign build_manifest ────────────────────────────────────────────────

class TestHow2SignBuildManifest:
    def test_build_manifest_loads_csv(self, tmp_path):
        """build_manifest loads existing TSV and sets context."""
        manifest_path = tmp_path / "manifest.csv"
        # Write with legacy columns — read_manifest normalizes them
        df = pd.DataFrame({
            "VIDEO_NAME": ["vid1", "vid1", "vid2"],
            "SENTENCE_NAME": ["vid1-000", "vid1-001", "vid2-000"],
            "START_REALIGNED": [0.0, 2.0, 0.0],
            "END_REALIGNED": [2.0, 4.0, 3.0],
            "SENTENCE": ["Hello", "World", "Test"],
        })
        df.to_csv(manifest_path, sep="\t", index=False)

        cfg = Config(
            dataset={"name": "how2sign"},
            paths={"manifest": str(manifest_path)},
        )
        adapter = How2SignDataset()
        context = PipelineContext(config=cfg, dataset=adapter)

        context = adapter.build_manifest(cfg, context)

        assert context.manifest_path == manifest_path
        assert len(context.manifest_df) == 3
        assert context.stats["dataset.manifest"]["videos"] == 2
        assert context.stats["dataset.manifest"]["segments"] == 3

    def test_build_manifest_missing_file_raises(self, tmp_path):
        cfg = Config(
            dataset={"name": "how2sign"},
            paths={"manifest": str(tmp_path / "nope.csv")},
        )
        adapter = How2SignDataset()
        context = PipelineContext(config=cfg, dataset=adapter)

        with pytest.raises(FileNotFoundError, match="manifest"):
            adapter.build_manifest(cfg, context)
