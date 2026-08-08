# Installation Guide

## Base Installation

```bash
git clone https://github.com/balaboom123/signdata-slt.git
cd signdata-slt
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

This installs dependencies for MediaPipe pose jobs and the shipped CPU YOLO
jobs. MMPose and MMDet are installed separately below because they require a
matching CUDA and OpenMMLab stack.

### FFmpeg availability

`video2pose` and `video2parts` prefer FFmpeg for decoding when the executable
is available on `PATH`, but automatically fall back to OpenCV under the base
installation.

`video2crop` and `video2compression` create encoded MP4 outputs and require a
system FFmpeg installation. Install FFmpeg through the operating system's
package manager, then verify it is visible:

```bash
ffmpeg -version
```

## MMPose (GPU Required)

MMPose whole-body extraction requires a CUDA-capable GPU. Follow these steps after the base installation:

### 1. Install MMPose dependencies

```bash
pip install -U openmim
mim install mmcv==2.0.1 mmengine==0.10.7 mmdet==3.1.0
```

### 2. Install MMPose from pip

```bash
pip install mmpose==1.3.2
```

Do not install MMPose in editable mode from a sibling checkout; the default configs use the package resources shipped with `mmpose==1.3.2`.

### 3. Download the detector checkpoint

```bash
mkdir -p resources/detection_models/rtmdet/checkpoints

wget -P resources/detection_models/rtmdet/checkpoints/ \
  https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth
```

The default MMPose pose checkpoint is referenced by URL in the job configs, so
MMPose downloads it through its normal checkpoint loader. The RTMDet person
detector still uses the local `resources/detection_models/rtmdet/checkpoints/`
path above.

### 4. Verify installation

```bash
python -c "from mmpose.apis import init_model; print('MMPose OK')"
python -c "from mmdet.apis import init_detector; print('MMDet OK')"
```

Both commands should print without errors. If you see CUDA-related issues, verify your GPU driver and PyTorch CUDA version match.

---

## See Also

- [Architecture](architecture.md) -- system design and pipeline flow
- [Configuration Reference](configuration.md) -- full config schema and CLI overrides
- [Datasets](datasets.md) -- dataset-specific setup guides
