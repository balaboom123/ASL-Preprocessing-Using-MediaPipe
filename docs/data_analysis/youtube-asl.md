I audited `/home/gorden/dataset/YouTube-ASL/origin`.

Scope: 10,655 `.mp4` files; 10,643 valid probes and 12 invalid/corrupt files. The repository lists 11,096 IDs, so 441 IDs are not present locally.

Video codec distribution:

| Codec | Files | Share |
|---|---:|---:|
| H.264/AVC | 10,348 | 97.23% |
| VP9 | 227 | 2.13% |
| AV1 | 68 | 0.64% |

Effective bitrate was calculated as `file_size × 8 / duration` because stream-level bitrate metadata is unreliable for VP9/AV1.

- `<0.25 Mb/s`: 252 files, 2.37%
- `0.25–0.5 Mb/s`: 955, 8.97%
- `0.5–1 Mb/s`: 3,475, 32.65%
- `1–2 Mb/s`: 5,454, 51.24%
- `2–4 Mb/s`: 499, 4.69%
- `4–8 Mb/s`: 8, 0.08%

Median effective bitrate: **1.068 Mb/s**.

Dominant exact resolutions:

| Resolution | Files | Share |
|---|---:|---:|
| 1280×720 | 8,673 | 81.49% |
| 640×480 | 422 | 3.97% |
| 854×480 | 289 | 2.72% |
| 640×360 | 221 | 2.08% |
| 1920×1080 | 165 | 1.55% |
| 1080×720 | 147 | 1.38% |
| 406×720 | 146 | 1.37% |
| Other dimensions | 580 | 5.45% |

Conclusion: VP9+AV1 account for only **295 files / 2.77%** of the valid corpus, about **1.47% of storage**.
They are generally modest bitrate—**287 of 295 are below 1 Mb/s**—so preserve/remux those files rather than transcoding them unnecessarily.
However, this is not a large enough fraction to eliminate the broader tuning exercise: **97.23% of the corpus is H.264**.

## Compression Influence Evaluation

Because no sign-language recognizer was available, MediaPipe Holistic was run independently on the original videos and three compressed versions. The resulting hand-detection and landmark measurements are a visual/landmark-stability proxy; they do not establish whether the translated English meaning is preserved.

The evaluation covered **616 segments per condition**, with **0 errors**:

| Condition | Measured storage reduction | Either-hand detection |
|---|---:|---:|
| Original | 0.00% | 93.68% |
| Small (target 25%) | 23.58% | 93.19% |
| Base (target 50%) | 43.58% | 93.03% |
| Large (target 60%) | 60.23% | 92.65% |

### Interpretation

In this initial three-level HEVC-only pilot, **Base** was the best storage/quality trade-off. **Large** saved more space but caused the greatest decrease in hand detection. The later cross-codec benchmark below supersedes this preliminary recommendation with AV1 CQ34 as the overall default. A translation model, human signer judgment, or hand-focused perceptual metric is still required before making a semantic-accuracy claim.

### Reproducibility artifacts

- Results directory: `/home/gorden/dataset/YouTube-ASL/test_output/mediapipe_eval/`
- [Final analysis report](/home/gorden/dataset/YouTube-ASL/test_output/mediapipe_eval/mediapipe_compression_analysis.md)
- [CSV summary](/home/gorden/dataset/YouTube-ASL/test_output/mediapipe_eval/mediapipe_compression_summary.csv)
- [JSON results](/home/gorden/dataset/YouTube-ASL/test_output/mediapipe_eval/mediapipe_compression_results.json)
- [Evaluation script](../../scripts/evaluate_mediapipe_compression.py)

## NVENC profile benchmark and GPU-1 MediaPipe check

The follow-up benchmark tested nine NVENC profiles on the three-video test set. Encoding and MediaPipe extraction were routed to physical GPU 1 (RTX 2080 Ti); GPU 0 was not selected. The benchmark preserved the source geometry and frame rate and dropped audio, so the size comparison isolates video compression.

The focused MediaPipe comparison covered **616 segments per condition**, **36,291 sampled frames**, and **0 processing errors**. Compared with the original, the balanced profile below saved 28.65% of storage while changing either-hand detection by only **−0.41 percentage points**, with 95.89% relative hand-landmark capture and 0.008666 mean 3D landmark error.

| Profile | Codec parameters | Output size | Reduction | Either-hand detection | Hand capture vs. original | Hand error 3D |
|---|---|---:|---:|---:|---:|---:|
| Original | source | 414,741,162 B | 0.00% | 93.68% | — | — |
| **Recommended** | `hevc_nvenc`, CQ28, p7, maxrate 0.75×, AQ8 | **295,934,177 B** | **28.65%** | **93.27%** | **95.89%** | **0.008666** |
| Higher reduction | `hevc_nvenc`, CQ30, p7, maxrate 0.70×, AQ8 | 275,751,551 B | 33.51% | 93.12% | 95.76% | 0.008828 |
| Storage-first candidate | `hevc_nvenc`, CQ32, p7, maxrate 0.60×, AQ8 | 233,745,694 B | 43.64% | 92.94% | 95.27% | 0.009076 |

### Recommendation

Within this GPU-1 HEVC/H.264 benchmark, use **HEVC CQ28 / p7 / maxrate 0.75× / AQ8** as the HEVC default. It is the strongest tested HEVC setting that keeps the either-hand detection loss below 0.5 percentage points while retaining high landmark capture. The later cross-codec benchmark supersedes it as the global default with AV1 CQ34. CQ30 is a reasonable HEVC storage-priority alternative; CQ32 and the CQ34/CQ36 profiles should be treated as storage-first options because their larger compression gains come with progressively worse landmark proxies. AQ12 did not improve the CQ30 result: it produced 33.22% reduction, 93.06% either-hand detection, and 0.008981 hand error.

These numbers measure MediaPipe landmark stability, not translation accuracy. Validate the selected setting later with a sign-language recognizer, human signer review, or a hand-focused perceptual/temporal metric before making a semantic claim.

### NVENC artifacts

- [Benchmark report](/home/gorden/dataset/YouTube-ASL/test_output/nvenc_benchmark/README.md)
- [Compression summary CSV](/home/gorden/dataset/YouTube-ASL/test_output/nvenc_benchmark/compression_summary.csv)
- [Per-video size CSV](/home/gorden/dataset/YouTube-ASL/test_output/nvenc_benchmark/compression_files.csv)
- [GPU-1 MediaPipe analysis](/home/gorden/dataset/YouTube-ASL/test_output/mediapipe_eval_nvenc_gpu1/mediapipe_compression_analysis.md)
- [GPU-1 MediaPipe summary](/home/gorden/dataset/YouTube-ASL/test_output/mediapipe_eval_nvenc_gpu1/mediapipe_compression_summary.csv)
- [GPU-1 MediaPipe JSON results](/home/gorden/dataset/YouTube-ASL/test_output/mediapipe_eval_nvenc_gpu1/mediapipe_compression_results.json)
- [NVENC benchmark script](../../scripts/benchmark_nvenc_profiles.py)

## CUDA-0 AV1 versus HEVC comparison

AV1 NVENC and comparable HEVC NVENC profiles were encoded with **GPU 0 (RTX 4060 Ti)**. The source set contained **3 videos / 616 matched segments**, and all 15 tested encoding profiles completed successfully. The protocol preserved the source geometry and frame rate, dropped audio, and used VBR, temporal AQ, spatial AQ, AQ strength 8 or 12, and a 32-frame lookahead.

| Settings | HEVC reduction | HEVC detection / capture | AV1 reduction | AV1 detection / capture |
|---|---:|---:|---:|---:|
| CQ26 / p6 / 0.80× / AQ8 | 23.58% | 93.26% / 95.85%† | 21.63% | 93.50% / 96.40% |
| CQ28 / p7 / 0.75× / AQ8 | 28.61% | 93.27% / 95.89%† | 26.47% | 93.53% / 96.39% |
| CQ30 / p7 / 0.70× / AQ8 | 33.46% | 93.12% / 95.76%† | 31.34% | 93.46% / 96.31% |
| CQ30 / p7 / 0.70× / AQ12 | 33.17% | 93.06% / 95.60%† | 31.33% | 93.44% / 96.22% |
| CQ32 / p7 / 0.60× / AQ8 | 43.58% | 92.94% / 95.27%† | 41.33% | 93.34% / 96.10% |
| CQ34 / p7 / 0.52× / AQ8 | 52.44% | 92.78% / 95.03%‡ | 49.10% | 93.36% / 95.81% |
| CQ36 / p7 / 0.45× / AQ8 | 60.23% | 92.65% / 94.53%‡ | 55.66% | 93.30% / 95.56% |

Detection is the percentage of frames with either hand detected; capture is relative hand-landmark capture versus the original. The original baseline was **93.68% either-hand detection**. The dagger values are the previously verified HEVC MediaPipe measurements from GPU 1 (RTX 2080 Ti); the double-dagger values are new HEVC CQ34/CQ36 MediaPipe measurements from GPU 0. HEVC sizes in this table were re-encoded on GPU 0, and all AV1 MediaPipe measurements were run on GPU 0. Every listed condition now has 616/616 segment outputs with zero processing errors.

### Selection

- **Best AV1 storage-first choice:** `av1_nvenc`, CQ36, preset p7, maxrate 0.45× source bitrate, AQ8. It reduced storage by **55.66%**, with **93.30%** either-hand detection (−0.37 points from original) and **95.56%** landmark capture.
- **Recommended balanced AV1 choice:** CQ32 / p7 / 0.60× / AQ8. It reduces storage by **41.33%** while retaining **93.34%** detection and **96.10%** capture; use this when a larger quality margin is preferred. AQ12 did not improve the CQ30 result.
- **Best HEVC default:** `hevc_nvenc`, CQ28, preset p7, maxrate 0.75× source bitrate, AQ8. It reduced storage by **28.61%**, with **93.27%** either-hand detection and **95.89%** landmark capture. HEVC remains the safer deployment choice when decoder compatibility matters.

At equal numeric settings, AV1 produced roughly **2.0–4.6 percentage points less storage reduction** than HEVC in this NVENC test, but its MediaPipe proxy was better. HEVC CQ34/CQ36 lost **0.90/1.02 detection points** and captured **95.03%/94.53%** of original hand landmarks, so they are not recommended for landmark-preserving compression. Choose AV1 CQ36 for maximum tested storage reduction when AV1 playback is available; choose AV1 CQ32 for a safer quality margin, or HEVC CQ28 for compatibility.

### CUDA-0 artifacts

- [GPU-0 benchmark README](/home/gorden/dataset/YouTube-ASL/test_output/nvenc_benchmark_gpu0/README.md)
- [GPU-0 compression summary](/home/gorden/dataset/YouTube-ASL/test_output/nvenc_benchmark_gpu0/compression_summary.csv)
- [GPU-0 per-video sizes](/home/gorden/dataset/YouTube-ASL/test_output/nvenc_benchmark_gpu0/compression_files.csv)
- [GPU-0 AV1 MediaPipe analysis](/home/gorden/dataset/YouTube-ASL/test_output/mediapipe_eval_gpu0_av1_hevc/mediapipe_compression_analysis.md)
- [GPU-0 AV1 MediaPipe summary](/home/gorden/dataset/YouTube-ASL/test_output/mediapipe_eval_gpu0_av1_hevc/mediapipe_compression_summary.csv)


## Consolidated cross-codec comparison and top five

The table below combines every unique profile with completed MediaPipe results. The earlier Small, Base, and Large labels duplicate the HEVC CQ26, CQ32, and CQ36 profiles, so they are not counted as separate tests. Detection change is measured against the same **93.68% original either-hand detection** baseline. Each evaluated profile covered **616 segments / 36,291 sampled frames with zero processing errors**.

| Codec and parameters | Storage reduction | Either-hand detection | Detection change | Hand capture | Hand error 3D |
|---|---:|---:|---:|---:|---:|
| AV1 CQ26 / p6 / 0.80× / AQ8 | 21.63% | 93.50% | −0.18 pp | 96.40% | 0.007988 |
| AV1 CQ28 / p6 / 0.75× / AQ8 | 26.47% | 93.51% | −0.17 pp | 96.30% | 0.007962 |
| AV1 CQ28 / p7 / 0.75× / AQ8 | 26.47% | 93.53% | −0.15 pp | 96.39% | 0.008078 |
| AV1 CQ30 / p7 / 0.70× / AQ8 | 31.34% | 93.46% | −0.22 pp | 96.31% | 0.008150 |
| AV1 CQ30 / p7 / 0.70× / AQ12 | 31.33% | 93.44% | −0.24 pp | 96.22% | 0.008277 |
| AV1 CQ32 / p7 / 0.60× / AQ8 | 41.33% | 93.34% | −0.34 pp | 96.10% | 0.008548 |
| AV1 CQ34 / p7 / 0.52× / AQ8 | 49.10% | 93.36% | −0.32 pp | 95.81% | 0.008757 |
| AV1 CQ36 / p7 / 0.45× / AQ8 | 55.66% | 93.30% | −0.37 pp | 95.56% | 0.009228 |
| HEVC CQ26 / p6 / 0.80× / AQ8 | 23.58% | 93.26% | −0.41 pp | 95.85% | 0.008469 |
| HEVC CQ28 / p7 / 0.75× / AQ8 | 28.61% | 93.27% | −0.41 pp | 95.89% | 0.008666 |
| HEVC CQ30 / p7 / 0.70× / AQ8 | 33.46% | 93.12% | −0.56 pp | 95.76% | 0.008828 |
| HEVC CQ30 / p7 / 0.70× / AQ12 | 33.17% | 93.06% | −0.62 pp | 95.60% | 0.008981 |
| HEVC CQ32 / p7 / 0.60× / AQ8 | 43.58% | 92.94% | −0.74 pp | 95.27% | 0.009076 |
| HEVC CQ34 / p7 / 0.52× / AQ8 | 52.44% | 92.78% | −0.90 pp | 95.03% | 0.009501 |
| HEVC CQ36 / p7 / 0.45× / AQ8 | 60.23% | 92.65% | −1.02 pp | 94.53% | 0.010176 |
| H.264 CQ26 / p7 / 0.80× / AQ8 | 22.75% | 93.41% | −0.26 pp | 95.70% | 0.009480 |
| H.264 CQ30 / p7 / 0.70× / AQ8 | 32.64% | not evaluated | — | — | — |

HEVC CQ26–CQ32 and H.264 were evaluated on GPU 1 (RTX 2080 Ti); AV1 and HEVC CQ34–CQ36 were evaluated on GPU 0 (RTX 4060 Ti). The HEVC CQ26–CQ32 storage values above are the matching GPU-0 encodes. Small differences should therefore be treated as experimental variation, especially differences of only a few hundredths of a percentage point. H.264 CQ30 is excluded from the ranking because only its output size was measured.

### Recommended top five profiles

The ranking first rejects profiles with more than **0.5 percentage-point hand-detection loss**, then considers storage reduction, landmark capture/error, and decoder compatibility. It is a practical ranking rather than a claim of translation accuracy.

| Rank | Recommended use | Parameters | Why |
|---:|---|---|---|
| **1** | Best overall balance | `av1_nvenc`, CQ34, p7, maxrate 0.52×, AQ8 | 49.10% smaller with only −0.32 pp detection change and 95.81% capture |
| **2** | Safer AV1 default | `av1_nvenc`, CQ32, p7, maxrate 0.60×, AQ8 | 41.33% smaller, 96.10% capture, and lower error than CQ34/CQ36 |
| **3** | Maximum recommended storage saving | `av1_nvenc`, CQ36, p7, maxrate 0.45×, AQ8 | 55.66% smaller while detection loss remains below 0.5 pp; use after visual review |
| **4** | Quality-first AV1 | `av1_nvenc`, CQ30, p7, maxrate 0.70×, AQ8 | 31.34% smaller with only −0.22 pp detection change and 96.31% capture |
| **5** | Compatibility-oriented default | `hevc_nvenc`, CQ28, p7, maxrate 0.75×, AQ8 | Best tested HEVC compromise; broader decoder support than AV1 |

Use **AV1 CQ34** as the new general recommendation when AV1 decoding is available. Use **AV1 CQ32** when preserving a larger landmark-quality margin matters, **AV1 CQ36** for storage-first experiments, and **HEVC CQ28** for wider deployment compatibility. H.264 CQ26 remains a legacy compatibility fallback, but it is not a top-five metric winner: it saves less storage and has higher landmark error than the preferred AV1 profiles. AQ12 is not recommended because it did not improve either codec at CQ30.

The best-overall profile has been adopted as the default in [`configs/jobs/tools/compression.yaml`](../../configs/jobs/tools/compression.yaml):

```yaml
codec: av1_nvenc
preset: p7
crf: 34
aq_strength: 8
nvenc_gpu: 0
max_bitrate_ratio: 0.52
```

These rankings only establish MediaPipe landmark stability on the three-video pilot. Before applying a preset to the full corpus, add signer review or a sign-language recognition/translation test and validate playback support on the target devices.
