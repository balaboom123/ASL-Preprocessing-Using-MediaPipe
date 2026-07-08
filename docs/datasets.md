# Datasets

## YouTube-ASL

A large-scale, open-domain ASL-English parallel corpus with 11,000+ YouTube videos and 73,000+ segments ([Uthus et al., 2023](https://arxiv.org/abs/2306.15162)).

**Default pose job:** `dataset.download → dataset.manifest → processing.video2pose → post_processing.normalize → output.webdataset`

```bash
python -m signdata run configs/jobs/youtube_asl/mediapipe.yaml
```

**Default video job:** `dataset.download → dataset.manifest → processing.video2crop → output.webdataset`

```bash
python -m signdata run configs/jobs/youtube_asl/video.yaml
```

Requires `dataset.source.video_ids_file` pointing to the video ID list
(included at `assets/youtube-asl_youtube_asl_video_ids.txt`). The dataset
download stage fetches videos via yt-dlp and transcripts via
`youtube-transcript-api`. If transcript requests start failing with
`RequestBlocked` or `IpBlocked`, configure
`dataset.source.transcript_proxy_http` / `dataset.source.transcript_proxy_https`
or retry from a non-blocked residential IP.

## How2Sign

80+ hours of instructional "how-to" videos with continuous ASL, recorded in a controlled environment with professional signers ([Duarte et al., CVPR 2021](https://how2sign.github.io/)).

**Default pose job:** `dataset.download (validation only) → dataset.manifest → processing.video2pose → post_processing.normalize → output.webdataset`

```bash
python -m signdata run configs/jobs/how2sign/mediapipe.yaml
```

**Default video job:** `dataset.download (validation only) → dataset.manifest → processing.video2crop → output.webdataset`

```bash
python -m signdata run configs/jobs/how2sign/video.yaml
```

**Setup:**
1. Download the dataset from [how2sign.github.io](https://how2sign.github.io/)
2. Place videos in the `videos` path (default: `dataset/how2sign/videos/`)
3. Place the alignment CSV (e.g. `how2sign_realigned_val.csv`) at `paths.manifest` or `dataset.source.manifest_csv`

The How2Sign dataset adapter uses `dataset.download` as a validation step for
local files; it does not fetch remote data.

## BOBSL

Broadcast BSL corpus released with subtitle-aligned interpreter videos and manual isolated-sign annotations ([Albanie et al., ICCV 2021](https://arxiv.org/abs/2111.03635)).

**Default pose job:** `dataset.download (local validation) → dataset.manifest → processing.video2pose → post_processing.normalize → output.webdataset`

```bash
python -m signdata run configs/jobs/bobsl/mediapipe.yaml
```

**Default video job:** `dataset.download (local validation) → dataset.manifest → processing.video2crop → output.webdataset`

```bash
python -m signdata run configs/jobs/bobsl/video.yaml
```

**Setup:**
1. Download and unpack the public BOBSL release under `dataset/bobsl/`, or override `dataset.source.release_dir`.
2. Place interpreter-cropped MP4s under `paths.videos` (the shipped jobs use `dataset/bobsl/videos/`).
3. Keep the release metadata JSON (for example `metadata/subset2episode.json`) under the release root, or override `dataset.source.metadata_file`.
4. Keep subtitle files under the release root. The adapter auto-discovers the common `subtitles/manually-aligned/` and `subtitles/audio-aligned-heuristic-correction/` layouts, or you can override `dataset.source.subtitles_root`.
5. Keep manual isolated-sign annotations under the release root. The adapter auto-discovers common `annotations/` layouts, or you can override `dataset.source.annotation_root`.

The BOBSL adapter exposes two manifest views:

- `dataset.source.view=subtitle_slt` builds one row per subtitle segment with `TEXT`, `START`, and `END`.
- `dataset.source.view=isolated_signs` builds one row per manual isolated-sign annotation with `GLOSS`, `TEXT`, `CLASS_ID`, `START`, and `END`.

Available BOBSL-specific overrides:

- `dataset.source.subtitle_alignment=manual|original` chooses between manually aligned subtitles and the original/audio-aligned subtitle release.
- `dataset.source.split=train|val|test|all` filters the selected release partition.
- `dataset.source.metadata_file`, `subtitles_root`, and `annotation_root` let you point at repackaged local layouts when auto-discovery is not enough.

## BSL-1K

Compatibility lexical view built from the public BOBSL release rather than a separate downloader ([BSL-1K project page](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/)).

**Default pose job:** `dataset.download (local validation) → dataset.manifest → processing.video2pose → post_processing.normalize → output.webdataset`

```bash
python -m signdata run configs/jobs/bsl1k/mediapipe.yaml
```

**Default video job:** `dataset.download (local validation) → dataset.manifest → processing.video2crop → output.webdataset`

```bash
python -m signdata run configs/jobs/bsl1k/video.yaml
```

The shipped BSL-1K configs reuse `dataset.source.release_dir: dataset/bobsl` and force `dataset.source.view: isolated_signs`, so outputs are organized under `dataset/bsl1k/` while the raw release and default video root stay under `dataset/bobsl/`.

## WLASL

Word-level ASL dataset with 2,000 glosses and 12,000+ isolated sign videos ([Li et al., WACV 2020](https://dxli94.github.io/WLASL/)).

**Default pose job:** `dataset.download (local validation) → dataset.manifest → processing.video2pose → post_processing.normalize → output.webdataset`

```bash
python -m signdata run configs/jobs/wlasl/mediapipe.yaml
```

**Default video job:** `dataset.download (local validation) → dataset.manifest → processing.video2crop → output.webdataset`

```bash
python -m signdata run configs/jobs/wlasl/video.yaml
```

**Setup:**
1. Download and preprocess WLASL clips with the official [start-kit repository](https://github.com/dxli94/WLASL)
2. Place `WLASL_v0.3.json` at `dataset/wlasl/WLASL_v0.3.json` or override `dataset.source.metadata_json`
3. Place one preprocessed clip per `video_id` under `paths.videos` (default: `dataset/wlasl/videos/`)
4. The provided base config keeps `dataset.source.download_mode: validate` for local preprocessed clips; optionally tune `dataset.source.split`, `dataset.source.subset`, and `dataset.source.availability_policy`

The WLASL dataset adapter supports two acquisition modes:

- `download_mode: validate` treats files under `paths.videos` as preprocessed clips. Manifest rows keep the original `FRAME_START` / `FRAME_END` metadata, but `START=0.0` and `END` is taken from the isolated clip duration when available.
- `download_mode: download_missing` fetches missing raw source videos from each instance `url` in `WLASL_v0.3.json` and keeps source-aligned `START` / `END` timing in the manifest.

## MS-ASL

Large-scale isolated ASL dataset with signer-diverse lexical clips ([Joze and Koller, CVPR 2019](https://arxiv.org/abs/1812.01053)).

**Default pose job:** `dataset.download (local validation) → dataset.manifest → processing.video2pose → post_processing.normalize → output.webdataset`

```bash
python -m signdata run configs/jobs/msasl/mediapipe.yaml
```

**Default video job:** `dataset.download (local validation) → dataset.manifest → processing.video2crop → output.webdataset`

```bash
python -m signdata run configs/jobs/msasl/video.yaml
```

**Setup:**
1. Download the official MS-ASL annotation release from the [Microsoft Download Center](https://www.microsoft.com/en-us/download/details.aspx?id=100121)
2. Place `MSASL_train.json`, `MSASL_val.json`, `MSASL_test.json`, and `MSASL_classes.json` under `dataset/msasl/annotations/` or override `dataset.source.annotations_dir`
3. Place local clips under `paths.videos` (default: `dataset/msasl/videos/`); both flat layouts and nested-by-class layouts are supported
4. The provided base config keeps `dataset.source.download_mode: validate`; optionally tune `dataset.source.split`, `dataset.source.subset`, and `dataset.source.availability_policy`

The MS-ASL dataset adapter supports two acquisition modes:

- `download_mode: validate` treats files under `paths.videos` as the local clip corpus and writes per-sample `REL_PATH` values into the manifest.
- `download_mode: download_missing` extracts YouTube IDs from the selected split JSON files and downloads any missing videos into `paths.videos`.

## AUTSL

Automatic Turkish Sign Language dataset used in the CVPR/ICCV 2021 isolated sign challenge. The release contains signer-independent `train`, `val`/`validation`, and `test` splits with paired RGB/depth clips and class-ID correspondence files.

**Default pose job:** `dataset.download (local validation) → dataset.manifest → processing.video2pose → post_processing.normalize → output.webdataset`

```bash
python -m signdata run configs/jobs/autsl/mediapipe.yaml
```

**Default video job:** `dataset.download (local validation) → dataset.manifest → processing.video2crop → output.webdataset`

```bash
python -m signdata run configs/jobs/autsl/video.yaml
```

**Setup:**
1. Download and extract the official AUTSL challenge release under `dataset/autsl/`, or override `dataset.source.release_dir`.
2. Keep the split directories under the release root, for example `dataset/autsl/train/`, `dataset/autsl/validation/`, and `dataset/autsl/test/`.
3. Ensure the release contains the class-ID correspondence CSV (for example `SignList*.csv`) plus `train_labels.csv` and `val_labels.csv`. The shipped configs assume the common challenge layout and auto-discover these files.
4. The shipped AUTSL jobs intentionally set `paths.videos: dataset/autsl` instead of `dataset/autsl/videos` because manifest rows use split-relative paths such as `train/signer0_sample1_color.mp4`.

The shipped AUTSL configs default to RGB clips via `dataset.source.modality: rgb` and keep `allow_unlabeled: false`, so unlabeled public `test` rows are skipped by default even when `dataset.source.split: all`.

Available AUTSL-specific overrides:

- `dataset.source.modality=rgb|depth` chooses the `_color.mp4` or `_depth.mp4` files.
- `dataset.source.split=train|val|test|all` filters which challenge split(s) feed the manifest.
- `dataset.source.allow_unlabeled=true` includes unlabeled split rows such as the public challenge `test` set.
- `dataset.source.class_id_file`, `train_labels_file`, `val_labels_file`, and `test_labels_file` let you override auto-discovered metadata files when the release uses custom filenames.

## CSL

Continuous Chinese Sign Language dataset released by USTC with 100 sentence prompts and 50 signers. The release page describes RGB/depth/skeleton modalities, while the benchmark papers commonly use one RGB clip per signer-sentence pair for Split I / Split II evaluation ([USTC CSL release](https://ustc-slr.github.io/datasets/2015_csl/)).

**Default pose job:** `dataset.download (validate/materialize local release) → dataset.manifest → processing.video2pose → post_processing.normalize → output.webdataset`

```bash
python -m signdata run configs/jobs/csl/mediapipe.yaml
```

**Default video job:** `dataset.download (validate/materialize local release) → dataset.manifest → processing.video2crop → output.webdataset`

```bash
python -m signdata run configs/jobs/csl/video.yaml
```

**Setup:**
1. Request access from the official USTC page: download the release agreement, sign it, and email it as instructed at [ustc-slr.github.io/datasets/2015_csl](https://ustc-slr.github.io/datasets/2015_csl/).
2. Unpack the release under `dataset/csl/` so the corpus file lives at `dataset/csl/corpus.txt` and RGB data lives under `dataset/csl/color/`, or override `dataset.source.release_dir`.
3. Run the normal CSL job. `dataset.download` is intentionally simple:
   it only validates a local release and, when needed, materializes frame folders into `paths.videos`.
4. `dataset.manifest` follows the paper-aligned split logic:
   `split_i` uses signers 1-40 for train and 41-50 for test; `split_ii` uses sentences 1-94 for train and 95-100 for test.

The shipped CSL configs default to `dataset.source.prepare_mode: materialize_missing`, so both common local layouts are supported:

- RGB video clips already present under `dataset.source.rgb_subdir` such as `dataset/csl/color/000000/*.mp4`
- per-sample frame folders such as `dataset/csl/color/000000/<sample>/000001.jpg`, which are converted to `.mp4` clips before preprocessing

The adapter targets the **continuous 2015 CSL release** and currently uses only the RGB modality in this preprocessing pipeline. The native depth and Kinect skeleton files from the release are not ingested directly by SignDATA.

## LSA64

Argentinian Sign Language isolated-sign dataset with 64 glosses, 10 signers, and 3,200 RGB clips ([Ronchetti et al., CACIC 2016](https://facundoq.github.io/datasets/lsa64/)).

**Default pose job:** `dataset.download (local validation) → dataset.manifest → processing.video2pose → post_processing.normalize → output.webdataset`

```bash
python -m signdata run configs/jobs/lsa64/mediapipe.yaml
```

**Default video job:** `dataset.download (local validation) → dataset.manifest → processing.video2crop → output.webdataset`

```bash
python -m signdata run configs/jobs/lsa64/video.yaml
```

**Setup:**
1. Download the official dataset from [facundoq.github.io/datasets/lsa64](https://facundoq.github.io/datasets/lsa64/)
2. Extract either the **cut** or **raw** RGB release under `dataset/lsa64/`
3. The shipped configs default to `dataset.source.variant: cut`; override with `dataset.source.variant=raw` to target the raw release
4. Keep `dataset.source.release_dir` pointed at the release root (`dataset/lsa64/`) unless you want to bypass variant resolution and point directly at a flat clip directory

The LSA64 adapter supports both official RGB layouts:

- release root with variant subdirectories such as `dataset/lsa64/cut/*.mp4` and `dataset/lsa64/raw/*.mp4`
- flat clip directories containing `*.mp4` files directly

The canonical file naming convention is `{CLASS_ID}_{SIGNER_ID}_{REPETITION_ID}.mp4`, for example `01_09_05.mp4`.

The shipped configs use `split_strategy: none` and emit `SPLIT=all` by default. For signer-independent evaluation, enable:

```bash
python -m signdata run configs/jobs/lsa64/mediapipe.yaml \
  --override dataset.source.split_strategy=community_signer_8_1_1 \
  --override dataset.source.split=train
```

By default, the adapter loads the bundled `assets/lsa64_class_map.tsv` gloss map derived from the official dataset website. You can override it with `dataset.source.class_map_file`.

LSA64 is licensed under [CC BY-NC-SA 4.0](https://facundoq.github.io/datasets/lsa64/). If you use or redistribute derivatives of the dataset, the dataset authors request that you cite the official website or paper.
## RWTH-PHOENIX-Weather

Continuous German Sign Language (DGS) weather-report dataset with ~8,000 sentence-level clips across train / dev / test splits ([Camgoz et al., CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Camgoz_Neural_Sign_Language_CVPR_2018_paper.html)).

**Default pose job:** `dataset.download (materialize missing videos) → dataset.manifest → processing.video2pose → post_processing.normalize → output.webdataset`

```bash
python -m signdata run configs/jobs/rwth_phoenix_weather/mediapipe.yaml \
    dataset.source.release_dir=/path/to/PHOENIX-2014-T-release3
```

**Default video job:** `dataset.download (materialize missing videos) → dataset.manifest → processing.video2crop → output.webdataset`

```bash
python -m signdata run configs/jobs/rwth_phoenix_weather/video.yaml \
    dataset.source.release_dir=/path/to/PHOENIX-2014-T-release3
```

**Setup:**
1. Request and download the dataset from [RWTH-PHOENIX-Weather](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/)
2. Unpack the archive. The official release usually stores annotations under `annotations/manual/PHOENIX-2014-T.{train,dev,test}.corpus.csv` and frames under `features/fullFrame-210x260px/<split>/<clip>/`.
3. Set `dataset.source.release_dir` to the unpacked release root, either via CLI override or in YAML.
4. Set `paths.videos` to the directory where materialized `.mp4` files should be written.

Repackaged layouts with top-level corpus CSVs and simpler relative frame paths are also accepted for compatibility, but the official release layout is the primary target.

The shipped RWTH-PHOENIX-Weather configs default to `dataset.source.prepare_mode: materialize_missing` so a fresh unpacked release produces `.mp4` clips before manifest filtering.

The adapter supports three prepare modes via `dataset.source.prepare_mode`:

- `validate` checks that `release_dir` exists and skips frame materialization.
- `materialize_missing` (default) encodes frame directories into `.mp4` files for clips that do not yet have a video file.
- `rematerialize_all` force re-encodes all clips regardless of existing files.

Frame directories are read in lexicographic order of their `.png` filenames. The default frame rate is 25 fps via `dataset.source.video_fps`.

## Adding a New Dataset

All datasets must use the package layout
`src/signdata/datasets/<dataset_name>/` with `adapter.py`, `source.py`, and
`manifest.py` as the default entry files.

See [CONTRIBUTING.md](../CONTRIBUTING.md#adding-a-new-dataset) for the
required structure, responsibilities, and code template.

---

## See Also

- [Pipeline Stages](pipeline-stages.md) -- what each stage does and its I/O
- [Configuration Reference](configuration.md) -- full config schema and CLI overrides
- [Research-Aligned Preprocessing](research-preprocessing.md) -- paper-aligned methodology notes
