# Pipeline Stages

`PipelineRunner` executes the pipeline in fixed order:

1. `dataset.download`
2. `dataset.manifest`
3. `processing.<processor>`
4. `post_processing.<recipe>`
5. `output.<type>`

## Stage Summary

| Stage | Output | Activation |
|---|---|---|
| `dataset.download` | videos and transcripts, or validation of local data | `dataset.download: true` |
| `dataset.manifest` | manifest TSV/CSV at `paths.manifest` | `dataset.manifest: true` |
| `processing.video2pose` | raw landmarks at `{paths.output}/{run_name}/raw/{sample_id}.npy` | `processing.enabled: true` and `processing.processor: video2pose` |
| `processing.video2crop` | cropped clips at `{paths.output}/{run_name}/raw/{sample_id}.mp4` | `processing.enabled: true` and `processing.processor: video2crop` |
| `processing.video2parts` | part streams at `{paths.output}/{run_name}/raw/{sample_id}/` | `processing.enabled: true` and `processing.processor: video2parts` |
| `post_processing.normalize` | normalized landmarks at `{paths.output}/{run_name}/normalized/{sample_id}.npy` | `post_processing.enabled: true` with a `post_processing.normalize` block |
| `output.webdataset` | shards at `{paths.webdataset}/{run_name}/shard-000000.tar` | `output.enabled: true` |

If `dataset.manifest` is `false` and `paths.manifest` is set, the runner loads
the existing manifest before continuing to downstream stages.

## `dataset.download`

Handles dataset acquisition.

- YouTube-ASL reads `dataset.source.video_ids_file`, downloads videos into `paths.videos`, and downloads transcript JSON into `paths.transcripts`.
- OpenASL reads `dataset.source.manifest_tsv` and downloads raw YouTube videos by `yid`.
- WLASL and MS-ASL validate local preprocessed clips by default; `download_mode=download_missing` fetches missing raw source videos.
- How2Sign does not download data; it validates that local inputs exist.
- Local-release datasets such as BOBSL/BSL-1K, AUTSL, LSA64, SLoVo, CSL, and RWTH-PHOENIX-Weather validate local files. CSL and RWTH-PHOENIX-Weather can materialize frame folders into `.mp4` clips.

Key config paths:

- `dataset.source.video_ids_file`
- `dataset.source.manifest_tsv`
- `dataset.source.release_dir`
- `dataset.source.download_format`
- `dataset.source.languages`
- `dataset.source.rate_limit`
- `dataset.source.concurrent_fragments`

## `dataset.manifest`

Builds or loads the base manifest.

- YouTube-ASL parses transcript JSON into a manifest.
- OpenASL maps the official TSV columns into canonical rows and optionally merges `dataset.source.bbox_json`.
- Local-release adapters discover videos/annotations and write canonical TSV rows with `SAMPLE_ID`, `VIDEO_ID`, `REL_PATH` where needed, timing columns, text/gloss labels, split labels, and dataset-specific metadata.
- How2Sign loads an existing CSV from `paths.manifest` or `dataset.source.manifest_csv`.

Key config paths:

- `dataset.source.manifest_csv`
- `dataset.source.manifest_tsv`
- `dataset.source.annotations_csv`
- `dataset.source.max_text_length`
- `dataset.source.min_duration`
- `dataset.source.max_duration`
- `paths.manifest`

## `processing.video2pose`

Reads source videos, samples frames, optionally runs person detection, then
writes landmark arrays into:

```text
{paths.output}/{run_name}/raw/{sample_id}.npy
```

Supporting modules live in:

- `src/signdata/utils/video.py`
- `src/signdata/processors/detection/`
- `src/signdata/processors/pose/`

`processing.sample_rate` controls frame selection:

- `null` keeps native FPS
- `0 < value < 1` keeps that fraction of frames uniformly
- `value >= 1` downsamples to that FPS without upsampling low-FPS sources

Key config paths:

- `processing.detection`
- `processing.detection_config`
- `processing.pose`
- `processing.pose_config`
- `processing.sample_rate`

## `processing.video2crop`

Runs a two-pass ffmpeg crop pipeline:

1. decode frames for detection
2. compute a union bbox
3. clip and crop the source segment into:

```text
{paths.output}/{run_name}/raw/{sample_id}.mp4
```

The shared ffmpeg helpers live in `src/signdata/processors/video/ffmpeg.py`.

`processing.sample_rate` controls both ffmpeg decode cadence and output clip
FPS:

- `null` keeps native FPS
- `0 < value < 1` keeps that fraction of frames uniformly
- `value >= 1` downsamples to that FPS without upsampling low-FPS sources

Key config paths:

- `processing.detection`
- `processing.detection_config`
- `processing.video_config.codec`
- `processing.video_config.padding`
- `processing.video_config.resize`
- `processing.sample_rate`

## `processing.video2parts`

Builds the SignMusketeers-style representation from MediaPipe Holistic output:

```text
{paths.output}/{run_name}/raw/{sample_id}/face.mp4
{paths.output}/{run_name}/raw/{sample_id}/left_hand.mp4
{paths.output}/{run_name}/raw/{sample_id}/right_hand.mp4
{paths.output}/{run_name}/raw/{sample_id}/pose.npz
{paths.output}/{run_name}/raw/{sample_id}/meta.json
```

`pose.npz` contains `body_pose`, frame timing/index arrays, per-stream bboxes,
and validity flags. The three MP4 files are synchronized 224 x 224 crop streams
by default.

Key config paths:

- `processing.pose: mediapipe`
- `processing.pose_config`
- `processing.parts_config.crop_size`
- `processing.parts_config.bbox_scale`
- `processing.parts_config.codec`
- `processing.sample_rate`

## `post_processing.normalize`

Reads raw landmark arrays from:

```text
{paths.output}/{run_name}/raw/{sample_id}.npy
```

and writes normalized arrays to:

```text
{paths.output}/{run_name}/normalized/{sample_id}.npy
```

Key config paths:

- `post_processing.normalize.mode`
- `post_processing.normalize.remove_z`
- `post_processing.normalize.keypoint_preset`
- `post_processing.normalize.keypoint_indices`
- `post_processing.normalize.mask_empty_frames`
- `post_processing.normalize.mask_low_confidence`
- `post_processing.normalize.visibility_threshold`
- `post_processing.normalize.missing_value`

## `output.webdataset`

Packages the manifest plus the active artifact directory into shards:

```text
{paths.webdataset}/{run_name}/shard-000000.tar
```

- `video2pose` packages normalized `.npy` files when present, otherwise raw `.npy`.
- `video2crop` packages raw `.mp4` files from `{paths.output}/{run_name}/raw/`.
- `video2parts` packages `face_mp4`, `left_hand_mp4`, `right_hand_mp4`,
  `pose_npz`, `json`, and `txt` per sample so standard WebDataset readers
  group every modality under one key.

## See Also

- [Architecture](architecture.md) -- runner and routing
- [Configuration Reference](configuration.md) -- config layout and fields
- [Datasets](datasets.md) -- dataset-specific setup
