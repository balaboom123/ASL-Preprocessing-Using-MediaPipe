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
| **[How2Sign](docs/datasets.md#how2sign)** | CVPR 2021 | 80+ hours of instructional ASL in a controlled studio environment | [CC BY-NC 4.0](https://how2sign.github.io/) |

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

### Optional: GPU-based Extractors (MMPose, MMDet, YOLO)

MediaPipe works on CPU out of the box. MMPose, MMDet, and YOLO require a CUDA-capable GPU and additional dependencies -- see the [Installation Guide](docs/installation.md) for full setup instructions.

---

## Quick Start

```bash
# YouTube-ASL: download, extract MediaPipe landmarks, normalize, package
python -m signdata run configs/jobs/youtube_asl/mediapipe.yaml

# How2Sign: extract MMPose landmarks (CUDA required)
python -m signdata run configs/jobs/how2sign/mmpose.yaml

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
- [Datasets](docs/datasets.md) -- YouTube-ASL vs How2Sign setup
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
