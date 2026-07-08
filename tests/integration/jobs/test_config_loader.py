"""Tests for config loading and merging (loader.py)."""

import os
from pathlib import Path

import pytest
import yaml

from signdata.config.loader import (
    _parse_value,
    _load_raw_config,
    _resolve_base_config_paths,
    _set_nested,
    deep_merge,
    load_config,
    resolve_paths,
)
from signdata.config.schema import Config


# ── deep_merge ──────────────────────────────────────────────────────────────

class TestDeepMerge:
    def test_nested_override(self):
        base = {"a": {"b": 1, "c": 2}}
        override = {"a": {"b": 99}}
        result = deep_merge(base, override)
        assert result == {"a": {"b": 99, "c": 2}}

    def test_non_overlapping_keys(self):
        base = {"x": 1}
        override = {"y": 2}
        result = deep_merge(base, override)
        assert result == {"x": 1, "y": 2}

    def test_list_replacement_not_append(self):
        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}
        result = deep_merge(base, override)
        assert result["items"] == [4, 5]

    def test_does_not_mutate_base(self):
        base = {"a": {"b": 1}}
        override = {"a": {"b": 2}}
        deep_merge(base, override)
        assert base["a"]["b"] == 1

    def test_empty_override(self):
        base = {"a": 1}
        result = deep_merge(base, {})
        assert result == {"a": 1}

    def test_empty_base(self):
        result = deep_merge({}, {"a": 1})
        assert result == {"a": 1}


# ── _parse_value ────────────────────────────────────────────────────────────

class TestParseValue:
    def test_true(self):
        assert _parse_value("true") is True
        assert _parse_value("True") is True
        assert _parse_value("TRUE") is True

    def test_false(self):
        assert _parse_value("false") is False
        assert _parse_value("False") is False

    def test_none(self):
        assert _parse_value("none") is None
        assert _parse_value("None") is None
        assert _parse_value("null") is None

    def test_int(self):
        assert _parse_value("42") == 42
        assert isinstance(_parse_value("42"), int)

    def test_float(self):
        assert _parse_value("3.14") == pytest.approx(3.14)
        assert isinstance(_parse_value("3.14"), float)

    def test_string(self):
        assert _parse_value("hello") == "hello"
        assert isinstance(_parse_value("hello"), str)

    def test_negative_int(self):
        assert _parse_value("-5") == -5

    def test_negative_float(self):
        assert _parse_value("-1.5") == pytest.approx(-1.5)


# ── _set_nested ─────────────────────────────────────────────────────────────

class TestSetNested:
    def test_creates_nested_dict(self):
        d = {}
        _set_nested(d, "a.b.c", 1)
        assert d == {"a": {"b": {"c": 1}}}

    def test_overwrites_existing(self):
        d = {"a": {"b": {"c": 10}}}
        _set_nested(d, "a.b.c", 99)
        assert d["a"]["b"]["c"] == 99

    def test_simple_key(self):
        d = {}
        _set_nested(d, "x", 5)
        assert d == {"x": 5}

    def test_partial_existing(self):
        d = {"a": {"x": 1}}
        _set_nested(d, "a.y", 2)
        assert d == {"a": {"x": 1, "y": 2}}


# ── resolve_paths ───────────────────────────────────────────────────────────

class TestResolvePaths:
    def test_empty_paths_get_defaults(self):
        cfg = Config(dataset={"name": "youtube_asl"})
        project_root = Path("/proj")
        cfg = resolve_paths(cfg, project_root)

        assert cfg.paths.root == str(project_root / "dataset" / "youtube_asl")
        assert cfg.paths.videos == str(
            project_root / "dataset" / "youtube_asl" / "videos"
        )
        assert cfg.paths.transcripts == str(
            project_root / "dataset" / "youtube_asl" / "transcripts"
        )

    def test_output_and_webdataset_defaults(self):
        cfg = Config(dataset={"name": "test"})
        project_root = Path("/proj")
        cfg = resolve_paths(cfg, project_root)

        expected_root = project_root / "dataset" / "test"
        assert cfg.paths.output == str(expected_root / "output")
        assert cfg.paths.webdataset == str(expected_root / "webdataset")

    def test_relative_paths_resolve(self):
        cfg = Config(
            dataset={"name": "test"},
            paths={"root": "data/test", "videos": "data/test/vids"},
        )
        project_root = Path("/proj")
        cfg = resolve_paths(cfg, project_root)

        assert cfg.paths.root == str(project_root / "data" / "test")
        assert cfg.paths.videos == str(project_root / "data" / "test" / "vids")

    def test_absolute_paths_unchanged(self):
        cfg = Config(
            dataset={"name": "test"},
            paths={"root": "/abs/root", "videos": "/abs/videos"},
        )
        project_root = Path("/proj")
        cfg = resolve_paths(cfg, project_root)

        assert Path(cfg.paths.root).as_posix() == "/abs/root"
        assert Path(cfg.paths.videos).as_posix() == "/abs/videos"

    def test_video_ids_file_relative_resolved(self):
        """dataset.source.video_ids_file is resolved relative to project root."""
        cfg = Config(
            dataset={
                "name": "test",
                "source": {"video_ids_file": "assets/ids.txt"},
            },
        )
        project_root = Path("/proj")
        cfg = resolve_paths(cfg, project_root)
        assert cfg.dataset.source["video_ids_file"] == str(
            project_root / "assets" / "ids.txt"
        )

    def test_video_ids_file_absolute_unchanged(self):
        cfg = Config(
            dataset={
                "name": "test",
                "source": {"video_ids_file": "/abs/ids.txt"},
            },
        )
        project_root = Path("/proj")
        cfg = resolve_paths(cfg, project_root)
        assert Path(cfg.dataset.source["video_ids_file"]).as_posix() == "/abs/ids.txt"

    def test_windows_absolute_paths_are_normalized(self):
        cfg = Config(
            dataset={
                "name": "test",
                "source": {"video_ids_file": r"C:\assets\ids.txt"},
            },
            paths={"root": r"D:\dataset\test"},
        )
        project_root = Path("/proj")
        cfg = resolve_paths(cfg, project_root)

        assert cfg.paths.root == str(Path("/mnt/d/dataset/test"))
        assert cfg.paths.videos == str(Path("/mnt/d/dataset/test/videos"))
        assert cfg.paths.transcripts == str(Path("/mnt/d/dataset/test/transcripts"))
        assert cfg.dataset.source["video_ids_file"] == str(Path("/mnt/c/assets/ids.txt"))

    def test_video_ids_file_empty_unchanged(self):
        cfg = Config(dataset={"name": "test"})
        project_root = Path("/proj")
        cfg = resolve_paths(cfg, project_root)
        assert cfg.dataset.source.get("video_ids_file", "") == ""

    def test_annotations_dir_relative_resolved(self):
        cfg = Config(
            dataset={
                "name": "test",
                "source": {"annotations_dir": "dataset/msasl/annotations"},
            },
        )
        project_root = Path("/proj")
        cfg = resolve_paths(cfg, project_root)
        assert cfg.dataset.source["annotations_dir"] == str(
            project_root / "dataset" / "msasl" / "annotations"
        )

    def test_autsl_optional_source_files_relative_resolved(self):
        cfg = Config(
            dataset={
                "name": "test",
                "source": {
                    "class_id_file": "dataset/autsl/SignList_ClassId_TR_EN.csv",
                    "train_labels_file": "dataset/autsl/train_labels.csv",
                    "val_labels_file": "dataset/autsl/val_labels.csv",
                    "test_labels_file": "dataset/autsl/test_labels.csv",
                },
            },
        )
        project_root = Path("/proj")
        cfg = resolve_paths(cfg, project_root)
        assert cfg.dataset.source["class_id_file"] == str(
            project_root / "dataset" / "autsl" / "SignList_ClassId_TR_EN.csv"
        )
        assert cfg.dataset.source["train_labels_file"] == str(
            project_root / "dataset" / "autsl" / "train_labels.csv"
        )
        assert cfg.dataset.source["val_labels_file"] == str(
            project_root / "dataset" / "autsl" / "val_labels.csv"
        )
        assert cfg.dataset.source["test_labels_file"] == str(
            project_root / "dataset" / "autsl" / "test_labels.csv"
        )

    def test_corpus_file_relative_resolved(self):
        cfg = Config(
            dataset={
                "name": "test",
                "source": {"corpus_file": "dataset/csl/corpus.txt"},
            },
        )
        project_root = Path("/proj")
        cfg = resolve_paths(cfg, project_root)
        assert cfg.dataset.source["corpus_file"] == str(
            project_root / "dataset" / "csl" / "corpus.txt"
        )

    def test_split_spec_file_relative_resolved(self):
        cfg = Config(
            dataset={
                "name": "test",
                "source": {"split_spec_file": "dataset/csl/splits.tsv"},
            },
        )
        project_root = Path("/proj")
        cfg = resolve_paths(cfg, project_root)
        assert cfg.dataset.source["split_spec_file"] == str(
            project_root / "dataset" / "csl" / "splits.tsv"
        )

    def test_generic_source_path_suffixes_relative_resolved(self):
        cfg = Config(
            dataset={
                "name": "test",
                "source": {
                    "metadata_file": "dataset/bobsl/metadata/subset2episode.json",
                    "annotation_root": "dataset/bobsl/annotations",
                    "subtitles_root": "dataset/bobsl/subtitles/manually-aligned",
                },
            },
        )
        project_root = Path("/proj")
        cfg = resolve_paths(cfg, project_root)
        assert cfg.dataset.source["metadata_file"] == str(
            project_root / "dataset" / "bobsl" / "metadata" / "subset2episode.json"
        )
        assert cfg.dataset.source["annotation_root"] == str(
            project_root / "dataset" / "bobsl" / "annotations"
        )
        assert cfg.dataset.source["subtitles_root"] == str(
            project_root / "dataset" / "bobsl" / "subtitles" / "manually-aligned"
        )

    def test_detection_model_paths_resolved(self):
        """processing.detection_config model paths resolve relative to project root."""
        cfg = Config(
            dataset={"name": "test"},
            processing={
                "enabled": True,
                "processor": "video2pose",
                "detection": "mmdet",
                "pose": "mmpose",
                "detection_config": {
                    "det_model_config": "resources/detection_models/rtmdet/configs/det.py",
                    "det_model_checkpoint": "resources/detection_models/rtmdet/checkpoints/det.pth",
                },
                "pose_config": {
                    "pose_model_config": "resources/pose_models/mmpose/configs/model.py",
                    "pose_model_checkpoint": "resources/pose_models/mmpose/checkpoints/model.pth",
                },
            },
        )
        project_root = Path("/proj")
        cfg = resolve_paths(cfg, project_root)
        assert cfg.processing.detection_config.det_model_config == str(
            project_root / "resources/detection_models/rtmdet/configs/det.py"
        )
        assert cfg.processing.pose_config.pose_model_config == str(
            project_root / "resources/pose_models/mmpose/configs/model.py"
        )

    def test_detection_model_paths_absolute_unchanged(self):
        cfg = Config(
            dataset={"name": "test"},
            processing={
                "enabled": True,
                "processor": "video2pose",
                "detection": "mmdet",
                "pose": "mmpose",
                "detection_config": {
                    "det_model_config": "/abs/det.py",
                    "det_model_checkpoint": "/abs/det.pth",
                },
                "pose_config": {
                    "pose_model_config": "/abs/model.py",
                    "pose_model_checkpoint": "/abs/model.pth",
                },
            },
        )
        project_root = Path("/proj")
        cfg = resolve_paths(cfg, project_root)
        assert Path(cfg.processing.detection_config.det_model_config).as_posix() == "/abs/det.py"
        assert Path(cfg.processing.pose_config.pose_model_config).as_posix() == "/abs/model.py"

    def test_run_name_not_in_static_paths(self):
        """run_name is NOT embedded in paths by resolve_paths.
        Run isolation is handled by PipelineContext.resolve_paths() at runtime."""
        cfg = Config(dataset={"name": "test"}, run_name="exp1")
        project_root = Path("/proj")
        cfg = resolve_paths(cfg, project_root)
        # paths.output is the template, not scoped by run_name
        expected_root = project_root / "dataset" / "test"
        assert cfg.paths.output == str(expected_root / "output")


# ── load_config ─────────────────────────────────────────────────────────────

class TestLoadConfig:
    """Load real YAML files from configs/datasets/ directory."""

    def test_load_youtube_asl_pose_mmpose(self, project_root):
        import signdata.datasets
        import signdata.processors

        yaml_path = str(
            project_root / "configs" / "datasets" / "youtube_asl" / "pose_mmpose.yaml"
        )
        if not os.path.exists(yaml_path):
            pytest.skip("Config file not found")

        cfg = load_config(yaml_path)
        assert cfg.dataset.name == "youtube_asl"
        assert cfg.processing.processor == "video2pose"
        assert cfg.processing.pose == "mmpose"
        assert cfg.processing.detection == "mmdet"
        assert cfg.processing.pose_config.pose_model_config == str(
            project_root
            / "resources"
            / "pose_models"
            / "mmpose"
            / "configs"
            / "rtmw3d-l_8xb64_cocktail14-384x288.py"
        )
        assert cfg.processing.detection_config.det_model_config == str(
            project_root
            / "resources"
            / "detection_models"
            / "rtmdet"
            / "configs"
            / "rtmdet_nano_320-8xb32_coco-person.py"
        )

    def test_load_how2sign_pose_mediapipe(self, project_root):
        import signdata.datasets
        import signdata.processors

        yaml_path = str(
            project_root / "configs" / "datasets" / "how2sign" / "pose_mediapipe.yaml"
        )
        if not os.path.exists(yaml_path):
            pytest.skip("Config file not found")

        cfg = load_config(yaml_path)
        assert cfg.dataset.name == "how2sign"
        assert cfg.processing.processor == "video2pose"
        assert cfg.processing.pose == "mediapipe"

    def test_load_youtube_asl_video_yolo(self, project_root):
        import signdata.datasets
        import signdata.processors

        yaml_path = str(
            project_root / "configs" / "datasets" / "youtube_asl" / "video_yolo.yaml"
        )
        if not os.path.exists(yaml_path):
            pytest.skip("Config file not found")

        cfg = load_config(yaml_path)
        assert cfg.dataset.name == "youtube_asl"
        assert cfg.processing.processor == "video2crop"
        assert cfg.processing.detection == "yolo"

    def test_load_msasl_pose_mediapipe(self, project_root):
        import signdata.datasets
        import signdata.processors

        yaml_path = str(project_root / "configs" / "jobs" / "msasl" / "mediapipe.yaml")
        cfg = load_config(yaml_path)

        assert cfg.dataset.name == "msasl"
        assert cfg.dataset.source["annotations_dir"] == str(
            project_root / "dataset" / "msasl" / "annotations"
        )
        assert cfg.processing.processor == "video2pose"
        assert cfg.processing.pose == "mediapipe"

    def test_load_msasl_video_yolo(self, project_root):
        import signdata.datasets
        import signdata.processors

        yaml_path = str(project_root / "configs" / "jobs" / "msasl" / "video.yaml")
        cfg = load_config(yaml_path)

        assert cfg.dataset.name == "msasl"
        assert cfg.processing.processor == "video2crop"
        assert cfg.processing.detection == "yolo"

    def test_load_autsl_pose_mediapipe(self, project_root):
        import signdata.datasets
        import signdata.processors

        yaml_path = str(project_root / "configs" / "jobs" / "autsl" / "mediapipe.yaml")
        cfg = load_config(yaml_path)

        assert cfg.dataset.name == "autsl"
        assert cfg.dataset.source["release_dir"] == str(project_root / "dataset" / "autsl")
        assert cfg.dataset.source["variant"] == "challenge_2021"
        assert cfg.dataset.source["modality"] == "rgb"
        assert cfg.dataset.source["allow_unlabeled"] is False
        assert cfg.processing.processor == "video2pose"
        assert cfg.processing.pose == "mediapipe"
        assert cfg.paths.videos == str(project_root / "dataset" / "autsl")

    def test_load_autsl_pose_mmpose(self, project_root):
        import signdata.datasets
        import signdata.processors

        yaml_path = str(project_root / "configs" / "jobs" / "autsl" / "mmpose.yaml")
        cfg = load_config(yaml_path)

        assert cfg.dataset.name == "autsl"
        assert cfg.processing.processor == "video2pose"
        assert cfg.processing.detection == "mmdet"
        assert cfg.processing.pose == "mmpose"

    def test_load_autsl_video_yolo(self, project_root):
        import signdata.datasets
        import signdata.processors

        yaml_path = str(project_root / "configs" / "jobs" / "autsl" / "video.yaml")
        cfg = load_config(yaml_path)

        assert cfg.dataset.name == "autsl"
        assert cfg.processing.processor == "video2crop"
        assert cfg.processing.detection == "yolo"

    def test_load_lsa64_pose_mediapipe(self, project_root):
        import signdata.datasets
        import signdata.processors

        yaml_path = str(project_root / "configs" / "jobs" / "lsa64" / "mediapipe.yaml")
        cfg = load_config(yaml_path)

        assert cfg.dataset.name == "lsa64"
        assert cfg.dataset.source["release_dir"] == str(
            project_root / "dataset" / "lsa64"
        )
        assert cfg.dataset.source["variant"] == "cut"
        assert cfg.dataset.source["split_strategy"] == "none"
        assert cfg.processing.processor == "video2pose"
        assert cfg.processing.pose == "mediapipe"

    def test_load_lsa64_pose_mmpose(self, project_root):
        import signdata.datasets
        import signdata.processors

        yaml_path = str(project_root / "configs" / "jobs" / "lsa64" / "mmpose.yaml")
        cfg = load_config(yaml_path)

        assert cfg.dataset.name == "lsa64"
        assert cfg.processing.processor == "video2pose"
        assert cfg.processing.detection == "mmdet"
        assert cfg.processing.pose == "mmpose"

    def test_load_lsa64_video_yolo(self, project_root):
        import signdata.datasets
        import signdata.processors

        yaml_path = str(project_root / "configs" / "jobs" / "lsa64" / "video.yaml")
        cfg = load_config(yaml_path)

        assert cfg.dataset.name == "lsa64"
        assert cfg.processing.processor == "video2crop"
        assert cfg.processing.detection == "yolo"

    def test_load_csl_pose_mediapipe(self, project_root):
        import signdata.datasets
        import signdata.processors

        yaml_path = str(project_root / "configs" / "jobs" / "csl" / "mediapipe.yaml")
        cfg = load_config(yaml_path)

        assert cfg.dataset.name == "csl"
        assert cfg.dataset.source["release_dir"] == str(project_root / "dataset" / "csl")
        assert cfg.dataset.source["variant"] == "continuous_2015"
        assert cfg.dataset.source["prepare_mode"] == "materialize_missing"
        assert cfg.processing.processor == "video2pose"
        assert cfg.processing.pose == "mediapipe"

    def test_load_csl_pose_mmpose(self, project_root):
        import signdata.datasets
        import signdata.processors

        yaml_path = str(project_root / "configs" / "jobs" / "csl" / "mmpose.yaml")
        cfg = load_config(yaml_path)

        assert cfg.dataset.name == "csl"
        assert cfg.processing.processor == "video2pose"
        assert cfg.processing.detection == "mmdet"
        assert cfg.processing.pose == "mmpose"

    def test_load_csl_video_yolo(self, project_root):
        import signdata.datasets
        import signdata.processors

        yaml_path = str(project_root / "configs" / "jobs" / "csl" / "video.yaml")
        cfg = load_config(yaml_path)

        assert cfg.dataset.name == "csl"
        assert cfg.processing.processor == "video2crop"
        assert cfg.processing.detection == "yolo"

    def test_load_bobsl_pose_mediapipe(self, project_root):
        import signdata.datasets
        import signdata.processors

        yaml_path = str(project_root / "configs" / "jobs" / "bobsl" / "mediapipe.yaml")
        cfg = load_config(yaml_path)

        assert cfg.dataset.name == "bobsl"
        assert cfg.dataset.source["release_dir"] == str(project_root / "dataset" / "bobsl")
        assert cfg.dataset.source["view"] == "subtitle_slt"
        assert cfg.dataset.source["subtitle_alignment"] == "manual"
        assert cfg.processing.processor == "video2pose"
        assert cfg.processing.pose == "mediapipe"

    def test_load_bobsl_video_yolo(self, project_root):
        import signdata.datasets
        import signdata.processors

        yaml_path = str(project_root / "configs" / "jobs" / "bobsl" / "video.yaml")
        cfg = load_config(yaml_path)

        assert cfg.dataset.name == "bobsl"
        assert cfg.processing.processor == "video2crop"
        assert cfg.processing.detection == "yolo"

    def test_load_bsl1k_pose_mediapipe(self, project_root):
        import signdata.datasets
        import signdata.processors

        yaml_path = str(project_root / "configs" / "jobs" / "bsl1k" / "mediapipe.yaml")
        cfg = load_config(yaml_path)

        assert cfg.dataset.name == "bsl1k"
        assert cfg.dataset.source["release_dir"] == str(project_root / "dataset" / "bobsl")
        assert cfg.dataset.source["view"] == "isolated_signs"
        assert cfg.processing.processor == "video2pose"
        assert cfg.processing.pose == "mediapipe"

    def test_load_bsl1k_video_yolo(self, project_root):
        import signdata.datasets
        import signdata.processors

        yaml_path = str(project_root / "configs" / "jobs" / "bsl1k" / "video.yaml")
        cfg = load_config(yaml_path)

        assert cfg.dataset.name == "bsl1k"
        assert cfg.processing.processor == "video2crop"
        assert cfg.processing.detection == "yolo"

    def test_load_bsl1k_base_config_uses_bobsl_videos(self, project_root):
        import signdata.datasets

        yaml_path = str(project_root / "configs" / "base" / "datasets" / "bsl1k.yaml")
        cfg = load_config(yaml_path)

        assert cfg.dataset.name == "bsl1k"
        assert cfg.dataset.source["release_dir"] == str(project_root / "dataset" / "bobsl")
        assert cfg.paths.videos == str(project_root / "dataset" / "bobsl" / "videos")

    def test_load_lsa64_requires_explicit_release_dir_or_paths_videos(self, tmp_path):
        import signdata.datasets

        yaml_path = tmp_path / "configs" / "jobs" / "lsa64.yaml"
        yaml_path.parent.mkdir(parents=True)
        yaml_path.write_text(yaml.dump({
            "dataset": {"name": "lsa64"},
            "processing": {"enabled": False},
            "post_processing": {"enabled": False},
            "output": {"enabled": False},
        }))

        with pytest.raises(ValueError, match="release_dir|paths\\.videos"):
            load_config(str(yaml_path))

    def test_load_autsl_requires_explicit_release_dir_or_paths_videos(self, tmp_path):
        import signdata.datasets

        yaml_path = tmp_path / "configs" / "jobs" / "autsl.yaml"
        yaml_path.parent.mkdir(parents=True)
        yaml_path.write_text(yaml.dump({
            "dataset": {"name": "autsl"},
            "processing": {"enabled": False},
            "post_processing": {"enabled": False},
            "output": {"enabled": False},
        }))

        with pytest.raises(ValueError, match="release_dir|paths\\.videos"):
            load_config(str(yaml_path))

    def test_cli_overrides_apply(self, project_root):
        import signdata.datasets
        import signdata.processors

        yaml_path = str(
            project_root / "configs" / "datasets" / "how2sign" / "pose_mediapipe.yaml"
        )
        if not os.path.exists(yaml_path):
            pytest.skip("Config file not found")

        cfg = load_config(yaml_path, overrides=["processing.max_workers=16"])
        assert cfg.processing.max_workers == 16

    def test_missing_dataset_raises(self, tmp_path):
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text(yaml.dump({
            "processing": {"enabled": False},
        }))
        with pytest.raises(ValueError, match="dataset"):
            load_config(str(yaml_path))

    def test_shorthand_dataset_string(self, tmp_path):
        """dataset: 'name' shorthand is expanded to dataset: {name: 'name'}."""
        import signdata.datasets

        configs_dir = tmp_path / "configs" / "datasets"
        configs_dir.mkdir(parents=True)
        yaml_path = configs_dir / "test.yaml"
        yaml_path.write_text(yaml.dump({
            "dataset": "youtube_asl",
        }))
        # Create assets for validation
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir(exist_ok=True)

        # youtube_asl validate_config requires video_ids_file — should raise
        with pytest.raises(ValueError, match="video_ids_file"):
            load_config(str(yaml_path))

    def test_unknown_dataset_raises(self, tmp_path):
        configs_dir = tmp_path / "configs" / "datasets"
        configs_dir.mkdir(parents=True)
        yaml_path = configs_dir / "test.yaml"
        yaml_path.write_text(yaml.dump({
            "dataset": {"name": "nonexistent_dataset"},
        }))
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_config(str(yaml_path))

    def test_base_config_is_merged_before_validation(self, tmp_path):
        import signdata.datasets

        base_dir = tmp_path / "configs" / "base" / "datasets"
        base_dir.mkdir(parents=True)
        base_path = base_dir / "youtube_asl.yaml"
        base_path.write_text(yaml.dump({
            "dataset": {
                "name": "youtube_asl",
                "source": {"video_ids_file": "assets/ids.txt"},
            },
        }))

        job_dir = tmp_path / "configs" / "jobs" / "youtube_asl"
        job_dir.mkdir(parents=True)
        yaml_path = job_dir / "video.yaml"
        yaml_path.write_text(yaml.dump({
            "base": "../../base/datasets/youtube_asl.yaml",
            "processing": {
                "enabled": True,
                "processor": "video2crop",
                "detection": "yolo",
                "detection_config": {"model": "yolov8n.pt"},
            },
        }))

        cfg = load_config(str(yaml_path))

        assert cfg.dataset.name == "youtube_asl"
        assert cfg.dataset.source["video_ids_file"] == str(
            tmp_path / "assets" / "ids.txt"
        )
        assert cfg.processing.processor == "video2crop"

    def test_base_config_dataset_shorthand_is_normalized_before_merge(self, tmp_path):
        import signdata.datasets

        base_dir = tmp_path / "configs" / "base" / "datasets"
        base_dir.mkdir(parents=True)
        base_path = base_dir / "youtube_asl.yaml"
        base_path.write_text(yaml.dump({
            "dataset": "youtube_asl",
        }))

        job_dir = tmp_path / "configs" / "jobs" / "youtube_asl"
        job_dir.mkdir(parents=True)
        yaml_path = job_dir / "video.yaml"
        yaml_path.write_text(yaml.dump({
            "base": "../../base/datasets/youtube_asl.yaml",
            "dataset": {
                "source": {"video_ids_file": "assets/ids.txt"},
            },
        }))

        cfg = load_config(str(yaml_path))

        assert cfg.dataset.name == "youtube_asl"
        assert cfg.dataset.source["video_ids_file"] == str(
            tmp_path / "assets" / "ids.txt"
        )

    @pytest.mark.parametrize("document", ["[]\n", "false\n", "0\n"])
    def test_falsey_non_mapping_job_config_raises_mapping_error(
        self, tmp_path, document
    ):
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text(document)

        with pytest.raises(ValueError, match="YAML mapping"):
            load_config(str(yaml_path))

    @pytest.mark.parametrize("document", ["[]\n", "false\n", "0\n"])
    def test_falsey_non_mapping_base_config_raises_mapping_error(
        self, tmp_path, document
    ):
        import signdata.datasets

        base_path = tmp_path / "configs" / "base.yaml"
        base_path.parent.mkdir(parents=True)
        base_path.write_text(document)

        job_path = tmp_path / "configs" / "jobs" / "job.yaml"
        job_path.parent.mkdir(parents=True)
        job_path.write_text(yaml.dump({
            "base": "../base.yaml",
            "dataset": {
                "name": "youtube_asl",
                "source": {"video_ids_file": "assets/ids.txt"},
            },
        }))

        with pytest.raises(ValueError, match="YAML mapping"):
            load_config(str(job_path))

    def test_base_config_cycles_raise(self, tmp_path):
        config_a = tmp_path / "a.yaml"
        config_b = tmp_path / "b.yaml"
        config_a.write_text(yaml.dump({"base": "b.yaml", "dataset": {"name": "x"}}))
        config_b.write_text(yaml.dump({"base": "a.yaml", "dataset": {"name": "x"}}))

        with pytest.raises(ValueError, match="Circular config base reference"):
            _load_raw_config(str(config_a))

    def test_windows_absolute_base_paths_resolve_without_config_dir_prefix(self, tmp_path):
        config_path = tmp_path / "configs" / "jobs" / "job.yaml"
        config_path.parent.mkdir(parents=True)

        resolved = _resolve_base_config_paths(r"C:\shared\base.yaml", config_path)

        assert resolved == [Path("/mnt/c/shared/base.yaml").resolve()]
