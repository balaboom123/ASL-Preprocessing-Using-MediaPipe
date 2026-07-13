# SignMusketeers Research Preprocessing Plan

This note summarizes the preprocessing and storage plan for:

- Gueuwou, Du, Shakhnarovich, and Livescu. "SignMusketeers: An Efficient Multi-Stream Approach for Sign Language Translation at Scale"
- Paper: https://arxiv.org/abs/2406.06907
- Project page: https://signmusketeers.pals.ttic.edu/

The paper's useful idea for this repo is not "make another video". It is a
multi-stream representation where each frame is split into the signer parts
that matter most for signed languages:

- face crop
- left hand crop
- right hand crop
- normalized upper-body pose

The downstream model then encodes the three image streams with sign-specialized
DINOv2 encoders, projects the body-pose vector, concatenates all four streams
per frame, and sends the frame sequence to T5 for translation.

## 1. Paper Method

The paper's preprocessing stage parses each source frame with MediaPipe
Holistic, then builds four synchronized channels.

### Face stream

For every sampled frame:

1. Read face landmarks.
2. Create the smallest square bounding box covering the face landmarks.
3. Enlarge the box by 1.2x.
4. If face landmarks are missing, estimate the face box from upper-body pose
   landmarks.
5. Clip the box to the image boundary.
6. Resize the crop to 224 x 224 with bicubic interpolation.

### Hand streams

For left and right hands:

1. Read hand landmarks.
2. Create the smallest square bounding box covering the hand landmarks.
3. Enlarge the box by 1.2x.
4. If hand landmarks are missing, estimate the hand center from MediaPipe pose
   finger landmarks:
   - left hand: pose indices 17, 19, 21
   - right hand: pose indices 18, 20, 22
5. Use a square box with the same size as the face box.
6. If a hand box leaves the frame, shift it inward or reuse the last valid box.
7. Resize the crop to 224 x 224 with bicubic interpolation.

### Body-pose stream

The body channel is not the full pose. It is only seven upper-body landmarks:

- nose: 0
- left shoulder: 11
- right shoulder: 12
- left elbow: 13
- right elbow: 14
- left wrist: 15
- right wrist: 16

These 2D points are normalized into a signer-relative signing space:

1. Define a head unit as half the shoulder distance.
2. Define signing-space width as 6 head units.
3. Define signing-space height as 7 head units.
4. Anchor the signing space with the nose and eye/upper-face pose landmarks.
5. Scale coordinates into a unit box centered around `(0.5, 0.5)`.
6. Flatten the seven `(x, y)` points into a 14-value vector.
7. If pose is missing, reuse the previous valid pose; if none exists, write a
   negative placeholder vector.

### Feature stage

The paper then uses:

- DINOv2-Face: frozen face encoder, producing `T x 384`
- DINOv2-Hand: frozen hand encoder, reused for both hands, producing two
  `T x 384` matrices
- body pose: `T x 14`
- stream-specific projections and a final projection to T5 input size

For this repo, the preprocessing option should stop at either the synchronized
part-stream bundle or the optional feature cache. Translation training belongs
outside the preprocessing pipeline.

## 2. Recommended Storage

Use WebDataset shards as the distribution layer because the repo already writes
WebDataset tar shards. Inside each sample, store a synchronized multi-part
bundle.

Recommended canonical sample:

```text
{sample_id}.face_mp4
{sample_id}.left_hand_mp4
{sample_id}.right_hand_mp4
{sample_id}.pose_npz
{sample_id}.json
{sample_id}.txt
```

The raw processor directory keeps files named `face.mp4`, `left_hand.mp4`,
`right_hand.mp4`, and `pose.npz`. WebDataset shards use the single-suffix
names above so all modalities group under one `{sample_id}`. The three crop
streams are 224 x 224 RGB and aligned frame-for-frame. `pose_npz` stores
numeric arrays:

```text
body_pose:       float32[T, 14]
frame_time_s:    float32[T]
frame_index:     int64[T]
face_bbox_xyxy:  float32[T, 4]
left_bbox_xyxy:  float32[T, 4]
right_bbox_xyxy: float32[T, 4]
valid:           bool[T, 4]      # face, left_hand, right_hand, body_pose
```

`json` stores metadata:

```json
{
  "format": "signdata.parts.v1",
  "sample_id": "...",
  "video_id": "...",
  "source_dataset": "...",
  "sign_language": "ase",
  "spoken_language": "en",
  "country": "US",
  "start": 0.0,
  "end": 4.2,
  "sample_rate": 25.0,
  "crop_size": 224,
  "bbox_scale": 1.2,
  "pose_backend": "mediapipe_holistic",
  "pose_landmark_layout": "mediapipe_553",
  "body_pose_layout": "signmusketeers_upper_body_14",
  "license": "...",
  "source_url": "..."
}
```

`txt` stores the aligned translation or gloss text when available.

Why this is the default:

- It keeps the visual evidence reusable for future encoders, not only DINOv2.
- It is much smaller than raw `uint8[T, 224, 224, 3]` arrays in `.npz`.
- It works with the current WebDataset package stage.
- It keeps all streams aligned under one sample key.
- It can be shared internationally without assuming ASL-only labels.

## 3. Optional Training Cache

For fast SignMusketeers-style training, add an optional derived artifact:

```text
{sample_id}.features.npz
```

Arrays:

```text
face:       float16[T, 384]
left_hand:  float16[T, 384]
right_hand: float16[T, 384]
body_pose:  float32[T, 14]
valid:      bool[T, 4]
```

This is not the canonical storage format because it is tied to exact encoder
weights, preprocessing version, and feature dimension. It is a cache.

## 4. Storage Options To Decide

| Option | What is stored | Pros | Cons | Use when |
|---|---|---|---|---|
| Canonical part bundle | three crop videos plus pose arrays | reusable, compact, inspectable | must decode crop videos during training | default public format |
| Feature cache | DINO features plus pose arrays | small and fast | model-version locked | repeated training with fixed encoders |
| Raw crop arrays | `npz` with crop tensors | simple and lossless | too large for global datasets | tiny experiments only |
| Zarr corpus | chunked arrays by dataset/split/stream | cloud random access | new dependency and more ops | WebDataset becomes the bottleneck |

Start with the canonical part bundle. Add feature cache next. Defer Zarr until
there is evidence that shard-level loading is too slow or cloud random access is
required.

## 5. Pipeline Mapping

Add one processor option:

```yaml
processing:
  enabled: true
  processor: video2parts
  pose: mediapipe
  sample_rate: 25
  parts_config:
    crop_size: 224
    bbox_scale: 1.2
```

The processor should:

1. Load the manifest row and resolve the source video.
2. Sample frames with the existing sampler.
3. Run the existing MediaPipe holistic extractor.
4. Split the MediaPipe output into pose, face, left hand, and right hand
   sections.
5. Compute face and hand boxes with SignMusketeers fallbacks.
6. Crop and resize the three image streams to 224 x 224.
7. Normalize upper-body pose to `T x 14`.
8. Write the canonical part bundle.
9. Let `output.webdataset` package all bundle files with the sample metadata.

Do not add a translation model to this repo for this option. The preprocessing
pipeline should produce reusable artifacts; model training should consume them.

## 6. International Reuse Rules

The format must not assume one country, one spoken language, or one sign
language. Required metadata:

- `sign_language`: BCP-47 style language code where possible, for example
  `ase` for ASL.
- `spoken_language`: translation text language, for example `en`.
- `country`: ISO 3166-1 alpha-2 when known.
- `source_dataset`: dataset adapter name.
- `license`: source license or access terms.
- `consent/privacy`: free-form field for consent notes, face handling, or
  release restrictions.
- `pose_landmark_layout`: exact landmark layout and version.
- `format`: stable format version, starting with `signdata.parts.v1`.

This keeps the same part-stream data usable for ASL, BSL, Libras, CSL, ISL,
and other sign languages without changing the arrays.

## 7. Implementation Boundary

The smallest useful implementation is now represented by:

1. `video2parts` processor writing sample directories under
   `{paths.output}/{run_name}/raw/{sample_id}/`.
2. WebDataset output support for multi-file samples.
3. `configs/jobs/how2sign/parts.yaml`.
4. Unit coverage for crop/pose packing helpers.

Skip DINOv2 feature extraction in the first implementation unless the storage
decision is explicitly "feature cache first".
