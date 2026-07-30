"""Tests for Video2CompressionProcessor and the encoder argument builder."""

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from signdata.config.schema import Config, VideoProcessingConfig
from signdata.datasets.how2sign import How2SignDataset
from signdata.pipeline.context import PipelineContext
from signdata.processors.video import ffmpeg
from signdata.processors.video.ffmpeg import (
    EncoderOptionError,
    VideoInfo,
    _encoder_args,
    _run_ffmpeg,
    probe_media,
)
from signdata.processors.video2compression import (
    Video2CompressionProcessor,
    _codec_family,
)
import signdata.processors  # noqa: F401 – trigger registrations
from signdata.registry import PROCESSOR_REGISTRY


def test_probe_video_releases_capture(monkeypatch):
    class Capture:
        released = False

        def isOpened(self):
            return True

        def get(self, prop):
            return {
                ffmpeg.cv2.CAP_PROP_FRAME_WIDTH: 640,
                ffmpeg.cv2.CAP_PROP_FRAME_HEIGHT: 480,
                ffmpeg.cv2.CAP_PROP_FPS: 30,
            }[prop]

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(ffmpeg.cv2, "VideoCapture", lambda _: capture)

    assert ffmpeg.probe_video("video.mp4") == (640, 480, 30.0)
    assert capture.released


class TestEncoderArgs:
    """NVENC ignoring -crf is what produced larger-than-source output."""

    def test_nvenc_uses_cq_and_never_crf(self):
        config = VideoProcessingConfig(
            codec="hevc_nvenc", preset="p6", crf=26, aq_strength=10
        )
        args = _encoder_args(config, None)

        assert "-crf" not in args
        assert args[args.index("-cq") + 1] == "26"
        assert args[args.index("-rc") + 1] == "vbr"
        # -cq is inert without an explicit zero average bitrate.
        assert args[args.index("-b:v") + 1] == "0"
        assert args[args.index("-aq-strength") + 1] == "10"
        assert args[args.index("-preset") + 1] == "p6"

    def test_nvenc_gpu_is_selected(self):
        config = VideoProcessingConfig(codec="hevc_nvenc", preset="p7", nvenc_gpu=1)
        args = _encoder_args(config, None)
        assert args[args.index("-gpu") + 1] == "1"

    def test_software_encoder_uses_crf_and_never_cq(self):
        config = VideoProcessingConfig(codec="libx265", preset="slow", crf=22)
        args = _encoder_args(config, None)

        assert "-cq" not in args
        assert "-rc" not in args
        assert args[args.index("-crf") + 1] == "22"
        assert args[args.index("-preset") + 1] == "slow"

    def test_bitrate_cap_sets_a_four_times_buffer(self):
        config = VideoProcessingConfig(codec="libx264")
        args = _encoder_args(config, 1_000_000)

        assert args[args.index("-maxrate") + 1] == "1000000"
        assert args[args.index("-bufsize") + 1] == "4000000"

    def test_no_cap_emits_no_rate_limit(self):
        args = _encoder_args(VideoProcessingConfig(codec="libx264"), None)
        assert "-maxrate" not in args

    def test_audio_is_dropped(self):
        # Nothing downstream reads audio, and copying Opus into mp4 fails the
        # mux; -an removes the failure mode and shrinks the file.
        args = _encoder_args(VideoProcessingConfig(codec="libx264"), None)
        assert "-an" in args
        assert "-c:a" not in args

    def test_ten_bit_source_keeps_its_depth(self):
        config = VideoProcessingConfig(codec="hevc_nvenc", preset="p6")
        args = _encoder_args(config, None, source_pix_fmt="yuv420p10le")
        assert args[args.index("-pix_fmt") + 1] == "p010le"

    def test_eight_bit_source_stays_yuv420p(self):
        config = VideoProcessingConfig(codec="hevc_nvenc", preset="p6")
        args = _encoder_args(config, None, source_pix_fmt="yuv420p")
        assert args[args.index("-pix_fmt") + 1] == "yuv420p"


class TestRunFfmpeg:
    def _proc(self, returncode, stderr):
        return subprocess.CompletedProcess(
            args=[], returncode=returncode, stdout=b"", stderr=stderr.encode()
        )

    def test_silently_ignored_option_raises(self):
        stderr = (
            "Codec AVOption crf (Select the quality for constant quality "
            "mode) specified for output file #0 (out.mp4) has not been used "
            "for any stream."
        )
        with patch.object(
            ffmpeg.subprocess, "run", return_value=self._proc(0, stderr)
        ):
            with pytest.raises(EncoderOptionError, match="crf"):
                _run_ffmpeg(["ffmpeg"], 60)

    def test_nonzero_exit_returns_false(self):
        with patch.object(
            ffmpeg.subprocess, "run", return_value=self._proc(1, "boom")
        ):
            assert _run_ffmpeg(["ffmpeg"], 60) is False

    def test_clean_run_returns_true(self):
        with patch.object(
            ffmpeg.subprocess, "run", return_value=self._proc(0, "")
        ):
            assert _run_ffmpeg(["ffmpeg"], 60) is True


class TestProbeMedia:
    def _run(self, payload):
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(payload).encode(),
            stderr=b"",
        )

    def test_reads_stream_bitrate(self):
        payload = {
            "streams": [{
                "codec_name": "h264",
                "width": 1280,
                "height": 720,
                "pix_fmt": "yuv420p",
                "bit_rate": "2500000",
            }],
            "format": {"duration": "60.0", "bit_rate": "2600000"},
        }
        with patch.object(ffmpeg.shutil, "which", return_value="/ffprobe"), \
                patch.object(
                    ffmpeg.subprocess, "run", return_value=self._run(payload)
                ):
            info = probe_media("video.mp4")

        assert info == VideoInfo("h264", 1280, 720, 60.0, 2_500_000, "yuv420p")

    def test_falls_back_to_container_bitrate(self, tmp_path):
        """WebM VP9/AV1 routinely omit the per-stream bit_rate."""
        payload = {
            "streams": [{"codec_name": "vp9", "width": 1280, "height": 720}],
            "format": {"duration": "60.0", "bit_rate": "1200000"},
        }
        with patch.object(ffmpeg.shutil, "which", return_value="/ffprobe"), \
                patch.object(
                    ffmpeg.subprocess, "run", return_value=self._run(payload)
                ):
            info = probe_media("video.webm")

        assert info.codec == "vp9"
        assert info.bitrate_bps == 1_200_000

    def test_falls_back_to_bytes_over_duration(self, tmp_path):
        video = tmp_path / "video.webm"
        video.write_bytes(b"x" * 1_000_000)
        payload = {
            "streams": [{"codec_name": "av1", "width": 1280, "height": 720}],
            "format": {"duration": "8.0"},
        }
        with patch.object(ffmpeg.shutil, "which", return_value="/ffprobe"), \
                patch.object(
                    ffmpeg.subprocess, "run", return_value=self._run(payload)
                ):
            info = probe_media(str(video))

        assert info.bitrate_bps == 1_000_000

    def test_missing_ffprobe_returns_none(self):
        with patch.object(ffmpeg.shutil, "which", return_value=None):
            assert probe_media("video.mp4") is None


class TestOutputPixFmt:
    """10-bit sources keep their depth; 8-bit layouts must not be upcast."""

    def test_eight_bit_stays_yuv420p(self):
        assert ffmpeg._output_pix_fmt("yuv420p", "hevc_nvenc") == "yuv420p"

    def test_nv12_is_not_mistaken_for_high_depth(self):
        # nv12/nv16 are 8-bit; the digits are a chroma layout, not a depth.
        assert ffmpeg._output_pix_fmt("nv12", "hevc_nvenc") == "yuv420p"

    def test_ten_bit_kept_on_nvenc(self):
        assert ffmpeg._output_pix_fmt("yuv420p10le", "hevc_nvenc") == "p010le"

    def test_ten_bit_kept_on_software(self):
        assert (
            ffmpeg._output_pix_fmt("yuv444p10le", "libx265") == "yuv420p10le"
        )

    def test_h264_cannot_hold_ten_bit(self):
        assert ffmpeg._output_pix_fmt("yuv420p10le", "h264_nvenc") == "yuv420p"


def test_codec_family_maps_probe_and_encoder_names():
    assert _codec_family("hevc") == _codec_family("hevc_nvenc") == "hevc"
    assert _codec_family("av1") == _codec_family("av01") == "av1"
    assert _codec_family("h264") == _codec_family("libx264") == "h264"
    assert _codec_family("vp9") != _codec_family("av1")


class TestVideo2CompressionProcessor:
    def _make_config(self, tmp_path, **video_overrides):
        video_config = {"codec": "libx265", "preset": "medium", "crf": 26}
        video_config.update(video_overrides)
        return Config(
            dataset={"name": "how2sign"},
            processing={
                "enabled": True,
                "processor": "video2compression",
                "video_config": video_config,
            },
            paths={"root": str(tmp_path)},
        )

    def _make_context(self, tmp_path, df, **video_overrides):
        videos_dir = tmp_path / "videos"
        videos_dir.mkdir(exist_ok=True)
        config = self._make_config(tmp_path, **video_overrides)
        context = PipelineContext(
            config=config,
            dataset=How2SignDataset(),
            manifest_df=df,
            videos_dir=videos_dir,
            output_dir=tmp_path / "output" / "compression",
        )
        return config, context, videos_dir

    def _df(self, rows=1):
        return pd.DataFrame({
            "VIDEO_ID": ["vid_a"] * rows,
            "VIDEO_NAME": ["cam_1"] * rows,
            "SAMPLE_ID": [f"seg_{i:03d}" for i in range(rows)],
        })

    def _output(self, tmp_path):
        return (
            tmp_path / "output" / "compression" / "compressed" / "cam_1.mp4"
        )

    def _info(self, codec, **overrides):
        fields = {
            "codec": codec,
            "width": 1280,
            "height": 720,
            "duration": 60.0,
            "bitrate_bps": 2_500_000,
        }
        fields.update(overrides)
        return VideoInfo(**fields)

    def test_deduplicates_rows_and_compresses_once(self, tmp_path):
        config, context, videos_dir = self._make_context(
            tmp_path, self._df(rows=2)
        )
        (videos_dir / "cam_1.mp4").write_bytes(b"s" * 200)

        def write_smaller(_, __, output_path, *, max_bitrate_bps=None,
                          source_pix_fmt=""):
            # 0.8 * 2_500_000, the configured ceiling on the source bitrate.
            assert max_bitrate_bps == 2_000_000
            Path(output_path).write_bytes(b"o" * 100)
            return True

        with patch(
            "signdata.processors.video2compression.probe_media",
            side_effect=[self._info("h264"), self._info("hevc")],
        ), patch(
            "signdata.processors.video2compression.transcode",
            side_effect=write_smaller,
        ) as transcode_mock:
            result = Video2CompressionProcessor(config).run(context)

        assert result.stats["processing"] == {
            "total": 1,
            "source_rows": 2,
            "processed": 1,
            "skipped": 0,
            "already_efficient": 0,
            "not_smaller": 0,
            "errors": 0,
        }
        assert transcode_mock.call_count == 1

        output = self._output(tmp_path)
        assert output.stat().st_size == 100
        metadata = json.loads(
            output.with_suffix(".mp4.compression.json").read_text("utf-8")
        )
        assert metadata["source_codec"] == "h264"
        assert metadata["output_codec"] == "hevc"
        assert (metadata["width"], metadata["height"]) == (1280, 720)
        assert metadata["source_bytes"] == 200
        assert metadata["output_bytes"] == 100

    def test_larger_output_is_rejected(self, tmp_path):
        config, context, videos_dir = self._make_context(tmp_path, self._df())
        (videos_dir / "cam_1.mp4").write_bytes(b"s" * 100)

        def write_larger(_, __, output_path, *, max_bitrate_bps=None,
                         source_pix_fmt=""):
            Path(output_path).write_bytes(b"o" * 150)
            return True

        with patch(
            "signdata.processors.video2compression.probe_media",
            side_effect=[self._info("h264"), self._info("hevc")],
        ), patch(
            "signdata.processors.video2compression.transcode",
            side_effect=write_larger,
        ):
            result = Video2CompressionProcessor(config).run(context)

        stats = result.stats["processing"]
        assert (stats["processed"], stats["not_smaller"]) == (0, 1)
        assert stats["skipped"] == 1
        # Not shrunk, so the original is mirrored through untouched.
        assert self._output(tmp_path).read_bytes() == b"s" * 100

    def test_source_already_in_target_codec_is_left_alone(self, tmp_path):
        config, context, videos_dir = self._make_context(tmp_path, self._df())
        (videos_dir / "cam_1.mp4").write_bytes(b"s" * 200)

        with patch(
            "signdata.processors.video2compression.probe_media",
            return_value=self._info("hevc"),
        ), patch(
            "signdata.processors.video2compression.transcode",
        ) as transcode_mock:
            result = Video2CompressionProcessor(config).run(context)

        stats = result.stats["processing"]
        assert stats["already_efficient"] == 1
        assert stats["processed"] == 0
        transcode_mock.assert_not_called()
        # Already efficient, so mirrored through as-is (no sidecar).
        assert self._output(tmp_path).read_bytes() == b"s" * 200
        assert not self._output(tmp_path).with_suffix(
            ".mp4.compression.json"
        ).exists()

    def test_geometry_change_is_an_error(self, tmp_path):
        """Downstream crops are re-derived from these pixels; losing any is fatal."""
        config, context, videos_dir = self._make_context(tmp_path, self._df())
        (videos_dir / "cam_1.mp4").write_bytes(b"s" * 200)

        def write_smaller(_, __, output_path, *, max_bitrate_bps=None,
                          source_pix_fmt=""):
            Path(output_path).write_bytes(b"o" * 100)
            return True

        with patch(
            "signdata.processors.video2compression.probe_media",
            side_effect=[
                self._info("h264"),
                self._info("hevc", width=640, height=360),
            ],
        ), patch(
            "signdata.processors.video2compression.transcode",
            side_effect=write_smaller,
        ):
            result = Video2CompressionProcessor(config).run(context)

        assert result.stats["processing"]["errors"] == 1
        # The bad encode is discarded; the original is mirrored through.
        assert self._output(tmp_path).read_bytes() == b"s" * 200

    def test_truncated_encode_is_an_error(self, tmp_path):
        config, context, videos_dir = self._make_context(tmp_path, self._df())
        (videos_dir / "cam_1.mp4").write_bytes(b"s" * 200)

        def write_smaller(_, __, output_path, *, max_bitrate_bps=None,
                          source_pix_fmt=""):
            Path(output_path).write_bytes(b"o" * 100)
            return True

        with patch(
            "signdata.processors.video2compression.probe_media",
            side_effect=[
                self._info("h264"),
                self._info("hevc", duration=42.0),
            ],
        ), patch(
            "signdata.processors.video2compression.transcode",
            side_effect=write_smaller,
        ):
            result = Video2CompressionProcessor(config).run(context)

        assert result.stats["processing"]["errors"] == 1
        # Truncated encode discarded; the original is mirrored through.
        assert self._output(tmp_path).read_bytes() == b"s" * 200

    def test_resize_is_refused(self, tmp_path):
        config, context, _ = self._make_context(
            tmp_path, self._df(), resize=[640, 360]
        )
        with pytest.raises(ValueError, match="must not rescale"):
            Video2CompressionProcessor(config).run(context)

    def test_encoder_misconfiguration_aborts_the_run(self, tmp_path):
        """A silently-ignored option is a config bug, not a bad input file."""
        config, context, videos_dir = self._make_context(tmp_path, self._df())
        (videos_dir / "cam_1.mp4").write_bytes(b"s" * 200)

        with patch(
            "signdata.processors.video2compression.probe_media",
            return_value=self._info("h264"),
        ), patch(
            "signdata.processors.video2compression.transcode",
            side_effect=EncoderOptionError("crf ignored"),
        ):
            with pytest.raises(EncoderOptionError):
                Video2CompressionProcessor(config).run(context)

    def test_unreadable_source_counts_as_an_error(self, tmp_path):
        config, context, videos_dir = self._make_context(tmp_path, self._df())
        (videos_dir / "cam_1.mp4").write_bytes(b"s" * 200)

        with patch(
            "signdata.processors.video2compression.probe_media",
            return_value=None,
        ), patch(
            "signdata.processors.video2compression.transcode",
        ) as transcode_mock:
            result = Video2CompressionProcessor(config).run(context)

        assert result.stats["processing"]["errors"] == 1
        transcode_mock.assert_not_called()
        # Unreadable metadata, but the file exists, so it is mirrored through.
        assert self._output(tmp_path).read_bytes() == b"s" * 200

    def test_missing_source_leaves_no_mirror_file(self, tmp_path):
        """A source that is not on disk cannot be mirrored; it stays a hole."""
        config, context, _ = self._make_context(tmp_path, self._df())
        # cam_1.mp4 is deliberately never written to videos/.

        result = Video2CompressionProcessor(config).run(context)

        assert result.stats["processing"]["errors"] == 1
        assert not self._output(tmp_path).exists()


class TestVideo2CompressionRegistration:
    def test_registered(self):
        assert "video2compression" in PROCESSOR_REGISTRY
