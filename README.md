<!-- H1 -->
# SignDATA: Data Pipeline for Sign Language Translation

<!-- Animated Header -->
<img src="https://balaboom123-capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=180&section=header&text=SignDATA&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=32&desc=Config-driven%20Pose/Video%20Preprocessing%20Pipeline&descAlignY=52&descSize=18" alt="SignDATA – Data Pipeline for Sign Language Translation"/>

<p align="center">
  <a href="https://arxiv.org/pdf/2604.20357"><img src="https://img.shields.io/badge/arXiv-2604.20357-b31b1b?style=flat" alt="arXiv"/></a> &nbsp;
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-10B981?style=flat" alt="License"/></a> &nbsp;
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue?style=flat" alt="Python 3.11+"/></a>
</p>

A config-driven, modular pipeline for preprocessing multiple **Sign Language** datasets.
Supports multiple extractors including **MediaPipe Holistic**, **MMPose**, **MMDet**, and **YOLO**. Supports two pipeline modes including **Pose Landmarks** and **Video Clips**.

---

## Key Features

- **Config-Driven** — YAML job configs, experiment configs, and CLI overrides
- **Multiple Extractors** — MediaPipe Holistic, MMPose, MMDet, and YOLO
- **Two Pipeline Modes** — `pose` (landmarks) and `video` (clip extraction)
- **WebDataset Output** — sharded tar archives for efficient training data loading

---

## Supported Datasets

| Dataset | Venue | Description | License |
|:--------|:------|:------------|:--------|
| **[YouTube-ASL](docs/datasets.md#youtube-asl)** | NeurIPS 2023 | 11,000+ videos, 73,000+ segments -- open-domain ASL-English parallel corpus | [Apache-2.0](https://github.com/google-research/google-research/tree/master/youtube_asl) |
| **[OpenASL](docs/datasets.md#openasl)** | EMNLP 2022 | Open-domain ASL-English translation dataset with official TSV and bbox metadata | [CC BY-NC-ND 4.0](https://github.com/chevalierNoir/OpenASL) |
| **[How2Sign](docs/datasets.md#how2sign)** | CVPR 2021 | 80+ hours of instructional ASL in a controlled studio environment | [CC BY-NC 4.0](https://how2sign.github.io/) |
| **[BOBSL](docs/datasets.md#bobsl)** | ICCV 2021 | Broadcast subtitle-aligned BSL corpus with continuous subtitle segments and isolated-sign annotations | Dataset access required |
| **[BSL-1K](docs/datasets.md#bsl-1k)** | arXiv 2020 | Compatibility lexical view over the public BOBSL release for isolated-sign style preprocessing | Follows BOBSL release terms |
| **[WLASL](docs/datasets.md#wlasl)** | WACV 2020 | 12,000+ isolated sign clips across 2,000 ASL glosses | [Dataset site](https://dxli94.github.io/WLASL/) |
| **[MS-ASL](docs/datasets.md#ms-asl)** | CVPR 2019 | Large-scale isolated ASL dataset with signer-diverse lexical clips | [Microsoft Download Center terms](https://www.microsoft.com/en-us/download/details.aspx?id=100121) |
| **[AUTSL](docs/datasets.md#autsl)** | ICCV 2021 challenge | Turkish Sign Language isolated-sign benchmark with RGB/depth clips and signer-independent train/val/test splits | Dataset access required |
| **[CSL](docs/datasets.md#csl)** | USTC release 2015 | 100 continuous Chinese sign sentences from 50 signers; RGB/depth/skeleton release with paper-aligned Split I / Split II evaluation | [CSL release agreement](https://ustc-slr.github.io/datasets/2015_csl/Release-Agreement-csl2015.pdf) |
| **[LSA64](docs/datasets.md#lsa64)** | CACIC 2016 | 3,200 isolated Argentinian Sign Language clips across 64 glosses | [CC BY-NC-SA 4.0](https://facundoq.github.io/datasets/lsa64/) |
| **[SLoVo](docs/datasets.md#slovo)** | ICCVS 2023 | Russian Sign Language isolated-sign clips with official `annotations.csv` | [Dataset license](https://github.com/hukenovs/slovo) |
| **[RWTH-PHOENIX-Weather](docs/datasets.md#rwth-phoenix-weather)** | CVIU 2015 | German Sign Language weather corpus distributed as image sequences and annotations | Dataset access required |

For paper-aligned preprocessing methodology, see [Research-Aligned Preprocessing](docs/research-preprocessing.md).

---

## Installation

```bash
git clone https://github.com/balaboom123/signdata-slt.git
cd signdata-slt
python -m venv venv
source venv/bin/activate  # Linux/macOS — use venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Optional: GPU-based Extractors (MMPose, MMDet)

MediaPipe and the shipped YOLO jobs work on CPU after the base install. MMPose
and MMDet require a CUDA-capable GPU and additional dependencies -- see the
[Installation Guide](docs/installation.md) for setup instructions.

---

## Quick Start

```bash
# YouTube-ASL: download, extract MediaPipe landmarks, normalize, package
python -m signdata run configs/jobs/youtube_asl/mediapipe.yaml

# How2Sign: extract MMPose landmarks (CUDA required)
python -m signdata run configs/jobs/how2sign/mmpose.yaml

# BOBSL: validate local release, build subtitle-aligned manifest, extract MediaPipe landmarks
python -m signdata run configs/jobs/bobsl/mediapipe.yaml

# BSL-1K compatibility view: build isolated-sign manifest from the BOBSL release
python -m signdata run configs/jobs/bsl1k/mediapipe.yaml

# MS-ASL: validate local clips, extract MediaPipe landmarks, normalize, package
python -m signdata run configs/jobs/msasl/mediapipe.yaml

# AUTSL: validate local release, extract MediaPipe landmarks, normalize, package
python -m signdata run configs/jobs/autsl/mediapipe.yaml

# CSL: validate/materialize the continuous release, extract MediaPipe landmarks, normalize, package
python -m signdata run configs/jobs/csl/mediapipe.yaml

# LSA64: validate local clips, extract MediaPipe landmarks, normalize, package
python -m signdata run configs/jobs/lsa64/mediapipe.yaml

# Override config values from the command line
python -m signdata run configs/jobs/youtube_asl/mediapipe.yaml \
  --override processing.max_workers=8 stop_at=extract
```

Both modes produce [WebDataset](https://github.com/webdataset/webdataset) tar shards for efficient training data loading. See [Pipeline Stages](docs/pipeline-stages.md) for detailed output formats and data shapes.

---

## Documentation

- [Installation Guide](docs/installation.md) -- base setup and MMPose GPU dependencies
- [Architecture](docs/architecture.md) -- system design, registry, pipeline flow
- [Configuration](docs/configuration.md) -- job/experiment layout and CLI overrides
- [Pipeline Stages](docs/pipeline-stages.md) -- recipe stages and optional stages
- [Datasets](docs/datasets.md) -- setup for all built-in dataset adapters
- [Contributing](CONTRIBUTING.md) -- required dataset package structure and extension guide
- [Research-Aligned Preprocessing](docs/research-preprocessing.md) -- paper-aligned preprocessing notes

## Citation

If you use SignDATA in your research, please cite:

```bibtex
@Article{chen2026signdata,
    author  = {Kuanwei Chen and Tingyi Lin},
    journal = {arXiv:2604.20357},
    title   = {SignDATA: Data Pipeline for Sign Language Translation},
    year    = {2026},
}
```

## License

The MIT license in this repository applies to the code and documentation in this project. Use of external datasets, research artifacts, and upstream repos referenced above must comply with their original licenses and usage terms.

MIT -- see [LICENSE](LICENSE).
