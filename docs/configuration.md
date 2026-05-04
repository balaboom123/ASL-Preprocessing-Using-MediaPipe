# Configuration Reference

## Job and Experiment Files

Runnable job YAMLs live under `configs/jobs/<dataset>/`.
Experiment YAMLs live under `configs/experiments/` and reference job paths relative to `configs/`.
Jobs may also declare a top-level `base:` key to inherit one or more shared YAML files before local values are applied.

```bash
python -m signdata run configs/jobs/youtube_asl/mediapipe.yaml \
  --override processing.max_workers=8 post_processing.normalize.remove_z=true

python -m signdata experiment configs/experiments/baseline_youtube_asl.yaml
```

## Merge Order

Config values are applied in this order, with later values winning:

1. Pydantic defaults from `src/signdata/config/schema.py`
2. Base YAML values referenced by top-level `base:`
3. Job YAML values
4. CLI overrides passed to `python -m signdata run`
5. Per-job `overrides` from an experiment YAML

## Minimal Working Config

The smallest practical job config needs the fields required by the
dataset adapter and the selected processor:

```yaml
# configs/jobs/how2sign/video.yaml
dataset:
  name: how2sign
  download: true
  manifest: true
  source:
    manifest_csv: dataset/how2sign/text/how2sign_realigned_val.csv

processing:
  enabled: true
  processor: video2crop
  detection: yolo
  detection_config:
    model: yolov8n.pt

run_name: default

paths:
  root: dataset/how2sign
  videos: dataset/how2sign/videos
```

The loader will derive `paths.transcripts`, `paths.manifest`, `paths.output`,
and `paths.webdataset` from `paths.root` when they are omitted.

## File Layout

```text
configs/
├── experiments/
│   ├── baseline_youtube_asl.yaml
│   └── privacy_aware_slt.yaml
└── jobs/
    ├── youtube_asl/
    │   ├── mediapipe.yaml
    │   ├── mmpose.yaml
    │   └── video.yaml
    ├── wlasl/
    │   ├── mediapipe.yaml
    │   ├── mmpose.yaml
    │   └── video.yaml
    └── how2sign/
        ├── mediapipe.yaml
        ├── mmpose.yaml
        └── video.yaml
```

## Top-Level Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `dataset` | `DatasetConfig` | none | Dataset adapter settings |
| `processing` | `ProcessingConfig` | disabled | Main processor selection and backend configs |
| `post_processing` | `PostProcessingConfig` | enabled | Post-processing recipes such as `normalize` |
| `output` | `OutputConfig` | `webdataset` | Output writer settings |
| `run_name` | `str` | `"default"` | Run namespace appended under `paths.output` and `paths.webdataset` |
| `paths` | `PathsConfig` | empty strings | Artifact root paths, resolved relative to the project root |

## `paths`

All relative paths are resolved from the project root.

| Field | Type | Default | Description |
|---|---|---|---|
| `root` | `str` | `""` | Dataset root directory |
| `videos` | `str` | `""` | Source videos |
| `transcripts` | `str` | `""` | Transcript JSON files |
| `manifest` | `str` | `""` | Base manifest TSV/CSV path |
| `output` | `str` | `""` | Base processing output directory; run outputs land in `{paths.output}/{run_name}` |
| `webdataset` | `str` | `""` | Base shard directory; shards land in `{paths.webdataset}/{run_name}` |

If omitted, the loader derives these defaults from `paths.root`:

- `paths.videos` → `{root}/videos`
- `paths.transcripts` → `{root}/transcripts`
- `paths.manifest` → `{root}/manifest.csv`
- `paths.output` → `{root}/output`
- `paths.webdataset` → `{root}/webdataset`

## `dataset`

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | none | Dataset name registered in `DATASET_REGISTRY` |
| `download` | `bool` | `true` | Run `dataset.download` |
| `manifest` | `bool` | `true` | Run `dataset.manifest`; if false, the runner loads `paths.manifest` directly |
| `source` | `dict` | `{}` | Dataset-specific source settings |

## `dataset.source`

`dataset.source` is adapter-specific. Common keys used by the built-in datasets are:

| Field | Type | Default | Description |
|---|---|---|---|
| `video_ids_file` | `str` | `""` | Video ID list for YouTube-style datasets |
| `metadata_json` | `str` | `""` | Canonical WLASL metadata JSON path (`annotation_json` is accepted as a compatibility alias) |
| `languages` | `list[str]` | adapter defaults | Transcript language codes |
| `availability_policy` | `str` | `"drop_unavailable"` | Availability handling policy for datasets that may have missing clips |
| `download_mode` | `str` | adapter defaults | Dataset acquisition mode such as `validate` or `download_missing`; WLASL uses `validate` for local preprocessed clips and `download_missing` for raw-source URL fetches |
| `download_format` | `str` | `"worstvideo[...]+worstaudio/.../best"` | yt-dlp format selector |
| `rate_limit` | `str` | `"5M"` | Download rate limit |
| `concurrent_fragments` | `int` | `5` | Parallel download fragments |
| `transcript_proxy_http` | `str` | `null` | Optional HTTP proxy for `youtube-transcript-api` |
| `transcript_proxy_https` | `str` | `null` | Optional HTTPS proxy for `youtube-transcript-api` |
| `stop_on_transcript_block` | `bool` | `true` | Stop transcript downloading after `RequestBlocked`/`IpBlocked` |
| `max_text_length` | `int` | `300` | Max caption length |
| `min_duration` | `float` | `0.2` | Min segment duration |
| `max_duration` | `float` | `60.0` | Max segment duration |
| `manifest_csv` | `str` | `""` | Existing manifest path for datasets such as How2Sign |
| `split` | `str` | `"all"` | Split label for datasets such as How2Sign and WLASL |
| `subset` | `int` | `0` | Optional class-count subset for datasets such as WLASL |
| `text_processing` | `dict` | adapter defaults | Text cleanup options for transcript-derived manifests |

Relative file paths in `dataset.source`, such as `video_ids_file`, `manifest_csv`,
`metadata_json`, and `annotation_json`, are resolved from the project root.

## `processing`

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `true` | Run the processing stage |
| `processor` | `"video2pose"` \| `"video2crop"` | `"video2pose"` | Main processor |
| `detection` | `"yolo"` \| `"mediapipe"` \| `"mmdet"` \| `"null"` | `"null"` | Detection backend |
| `pose` | `"mediapipe"` \| `"mmpose"` \| `null` | `null` | Pose backend; required for `video2pose` |
| `sample_rate` | `float?` | `0.5` | `null` = native FPS, `0 < value < 1` = keep ratio, `value >= 1` = absolute FPS |
| `max_workers` | `int` | `1` | Worker count used by processors and post-processors |
| `detection_config` | backend-specific model | `null` | Required for non-`null` detection backends |
| `pose_config` | backend-specific model | `null` | Required for `video2pose` |
| `video_config` | `VideoProcessingConfig?` | auto-created for `video2crop` | Crop/output settings for `video2crop` |

`video2pose` requires both `processing.pose` and `processing.pose_config`.
`video2crop` auto-fills a default `processing.video_config` when omitted.

### `sample_rate` semantics

`processing.sample_rate` uses three modes:

- `null` — keep the native video FPS
- `0 < value < 1` — keep that fraction of frames uniformly
- `value >= 1` — downsample to that absolute FPS

Examples:

```yaml
processing:
  sample_rate: null   # keep native FPS
```

```yaml
processing:
  sample_rate: 0.5    # keep half the frames
```

```yaml
processing:
  sample_rate: 24.0   # downsample to 24 FPS
```

Legacy sampling keys are deprecated:

- `processing.frame_skip`
- `processing.target_fps`

They are still accepted for migration, but new configs should use only
`processing.sample_rate`. Legacy `video2crop` configs that combined both old
keys cannot be reproduced exactly with a single scalar because the old pipeline
used `target_fps` for clip cadence and `frame_skip` as a separate detection-only
stride.

If the source video FPS is lower than an absolute target, the pipeline keeps the
source FPS. It does not upsample or invent frames.

## `processing.detection_config`

### YOLO

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `"yolov8n.pt"` | YOLO model id or path |
| `device` | `str` | `"cpu"` | Inference device |
| `confidence_threshold` | `float` | `0.5` | Detection threshold |
| `min_bbox_area` | `float` | `0.05` | Minimum normalized bbox area |

### MMDet

| Field | Type | Default | Description |
|---|---|---|---|
| `det_model_config` | `str` | none | Detector config path |
| `det_model_checkpoint` | `str` | none | Detector checkpoint path |
| `device` | `str` | `"cuda:0"` | Inference device |

### MediaPipe

| Field | Type | Default | Description |
|---|---|---|---|
| `min_detection_confidence` | `float` | `0.5` | Detection threshold |

## `processing.pose_config`

### MediaPipe

| Field | Type | Default | Description |
|---|---|---|---|
| `model_complexity` | `int` | `1` | MediaPipe model complexity |
| `min_detection_confidence` | `float` | `0.5` | Initial detection threshold |
| `min_tracking_confidence` | `float` | `0.5` | Tracking threshold |
| `refine_face_landmarks` | `bool` | `true` | Use refined face landmarks |
| `batch_size` | `int` | `16` | Frames per inference batch |

### MMPose

| Field | Type | Default | Description |
|---|---|---|---|
| `pose_model_config` | `str` | none | Pose model config path |
| `pose_model_checkpoint` | `str` | none | Pose checkpoint path |
| `device` | `str` | `"cuda:0"` | Inference device |
| `bbox_threshold` | `float` | `0.5` | Person-box threshold |
| `keypoint_threshold` | `float` | `0.3` | Keypoint threshold |
| `batch_size` | `int` | `16` | Frames per inference batch |

## `processing.video_config`

| Field | Type | Default | Description |
|---|---|---|---|
| `codec` | `str` | `"libx264"` | ffmpeg video codec for cropped outputs |
| `padding` | `float` | `0.0` | Extra crop padding around the detected bbox |
| `resize` | `list[int]?` | `null` | Optional `[width, height]` resize after crop |

## `post_processing`

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `true` | Run post-processing recipes |
| `recipes` | `list[str]` | `[]` | Ordered recipe names, e.g. `["normalize"]` |
| `normalize` | `NormalizeConfig?` | `null` | Config block for the built-in `normalize` recipe |

## `post_processing.normalize`

| Field | Type | Default | Description |
|---|---|---|---|
| `mode` | `str` | `"xy_isotropic_z_minmax"` | Normalization mode |
| `remove_z` | `bool` | `false` | Drop z after normalization |
| `select_keypoints` | `bool` | `true` | Reduce keypoints before flattening |
| `keypoint_preset` | `str?` | `null` | Named preset from `signdata.processors.pose` |
| `keypoint_indices` | `list[int]?` | `null` | Explicit keypoint indices |
| `mask_empty_frames` | `bool` | `true` | Mask all-zero frames |
| `mask_low_confidence` | `bool` | `false` | Mask low-visibility landmarks |
| `visibility_threshold` | `float` | `0.3` | Visibility cutoff |
| `missing_value` | `float` | `-999.0` | Fill value for masked data |

## `output`

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `true` | Run the output stage |
| `type` | `"webdataset"` | `"webdataset"` | Output writer type |
| `config` | `dict` | `{}` | Writer-specific settings |

## `output.config`

For the built-in WebDataset writer:

| Field | Type | Runtime Default | Description |
|---|---|---|---|
| `max_shard_count` | `int` | `10000` | Max samples per shard |
| `max_shard_size` | `int?` | `null` | Max shard size in bytes |

## CLI Examples

```bash
# Change number of workers
python -m signdata run configs/jobs/youtube_asl/mediapipe.yaml \
  --override processing.max_workers=8

# Change normalization behavior
python -m signdata run configs/jobs/youtube_asl/mediapipe.yaml \
  --override post_processing.normalize.remove_z=true

# Override run name for isolated outputs
python -m signdata run configs/jobs/youtube_asl/video.yaml \
  --run-name privacy_v1

# List built-in keypoint presets
python -m signdata run --list-presets
```

## See Also

- [Architecture](architecture.md) -- runner, registries, and package layout
- [Pipeline Stages](pipeline-stages.md) -- stage behavior and outputs
- [Installation Guide](installation.md) -- environment setup and MMPose dependencies
