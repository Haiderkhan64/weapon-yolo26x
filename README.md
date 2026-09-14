---
license: bigscience-openrail-m
tags:
  - object-detection
  - yolo
  - computer-vision
  - weapon-detection
  - tensorrt
  - real-time
  - ultralytics
task_categories:
  - object-detection
language:
  - en
---

# Weapon Detection YOLO26x

A 7-class threat detector trained on 104,697 images using a progressive fine-tuning curriculum (3 training phases + a non-training TTA/export step). Final mAP@50 of **0.8913** with TTA. TensorRT FP16 export runs at ~2ms/image on H100.

This README documents what was done, why each decision was made, what the numbers actually mean, and how to reproduce everything from scratch. Where a claim is a design rationale rather than something directly measured, it's labeled as such — see the [companion paper](#citation) for the full experimental treatment, including which claims are supported by ablations and which remain open questions.

---

## What this is

A fine-tuned [YOLO26x](https://github.com/ultralytics/ultralytics) checkpoint for detecting weapons and related threat objects in images and video. The seven classes are: `Blunt Weapon`, `Explosive`, `Fire Smoke`, `Firearm`, `Melee_Weapon`, `Person`, `Tool`.

`Person` is intentionally included as a context class, not a standalone person detector. Its lower mAP@50 (0.747) is expected — person annotations in the dataset are sparse and only label people in weapon-adjacent scenes. Don't use this model as a people counter.

The training pipeline is the main contribution here. Most public YOLO uploads are a single training run with default hyperparameters. This one uses a progressive curriculum motivated by two problems observed during development: a training-time numerical instability (NaN losses) when fine-tuning at full resolution in a single stage, and the resolution gap between the pretrained checkpoint and the 1024px deployment target. Details — including what was and wasn't established about the instability's root cause — are in the [Training](#training) section.

---

## Dataset

**Weapons-130K** · [Kaggle](https://www.kaggle.com/datasets/haiderkhan6410/weapons-130k) · CC BY 4.0

*(Note: the dataset is CC BY 4.0; this repository's code/model is licensed separately under OpenRAIL-M — see [License](#license).)*

| Property | Value |
|---|---|
| Total images | 130,763 |
| Total annotations | 294,950 |
| Train / Val / Test | 104,697 / 13,186 / 12,880 |
| Image size | 800 × 800 px |
| Format | YOLO bounding box (normalized `class cx cy w h`) |
| Background frames | 12,342 (9.4% overall; 8.6% in the training split) |

### Class distribution

| ID | Class | Annotations | Share |
|---|---|---|---|
| 0 | Blunt_Weapon | 9,527 | 3.2% |
| 1 | Explosive | 25,599 | 8.7% |
| 2 | Fire_Smoke | 78,171 | 26.5% |
| 3 | Firearm | 117,864 | 40.0% |
| 4 | Melee_Weapon | 40,166 | 13.6% |
| 5 | Person | 11,801 | 4.0% |
| 6 | Tool | 11,822 | 4.0% |

The dataset is imbalanced by design. Per the paper's stated rationale, this is intended to reflect real-world prevalence and operational priority rather than enforcing uniform class frequency — this is the design intent, not an independently measured claim about true real-world class distributions. Firearm and Fire_Smoke dominate. Applications requiring a different class distribution may consider class-weighting or targeted resampling, though these strategies were not evaluated in this study.

### Cleaning pipeline

Raw source data (76 heterogeneous source classes) was consolidated and cleaned in 4 audited stages before training:

1. **Annotation repair** — 1,052 segmentation polygon lines stripped and converted to bbox format across 772 files.
2. **Deduplication** — 5,169 near-duplicate images removed via perceptual hashing, applied **independently within each split** (train/val/test), not globally.
3. **Degenerate bbox removal** — 153 zero-area/invalid boxes dropped.
4. **Background frame audit** — 12,342 empty-label frames individually confirmed as genuine negatives and retained. Background frames are kept to expose the detector to negative scenes and reduce the risk of false-positive predictions on weapon-free imagery; this rationale was not separately validated by a full-scale ablation in this study.

**Structural integrity checks (all passed):** zero corrupt images · zero out-of-bounds coordinates · zero orphan labels · zero degenerate bounding boxes · zero malformed entries. This is a check of annotation *format* integrity, not a semantic verification that every label is the correct class — that was not independently human-audited at full scale.

**Known limitation — cross-split leakage not audited.** Deduplication was performed within each split only. This prevents images meeting the applied perceptual-hash similarity criterion from remaining duplicated *within* an individual split, but it cannot detect a near-duplicate image (e.g., an adjacent video frame) that landed in two different splits. A global cross-split perceptual-hash audit has not yet been performed and is the single highest-priority follow-up for this dataset; treat reported validation/test metrics with this in mind until that audit is done.

Net annotation delta: 302,550 → 294,950 (−7,600), accounted for by removed duplicates (−7,447, back-calculated from image-level counts, not logged per-annotation) and degenerate-box removal (−153, logged directly).

---

## Live Demo

[![Open in Spaces](https://img.shields.io/badge/🤗%20Open%20in%20Spaces-weapon--yolo26x--demo-yellow)](https://huggingface.co/spaces/HaiderKhan6410/weapon-yolo26x-demo)

Upload a video, adjust confidence and IoU thresholds, get annotated output with a per-class breakdown. No GPU, no install.

**7 classes:** Blunt Weapon · Explosive · Fire/Smoke · Firearm · Melee Weapon · Person · Tool

> **Speed warning:** The Space runs CPU-only (free HF tier). Expect 1–5s/image. Not a model issue — re-export and run locally if you need real throughput.

| Environment | Latency |
|---|---|
| HF Space (CPU, free tier) | ~1–5s / image |
| PyTorch · NVIDIA GPU (H100) | ~5ms / image |
| TensorRT FP16 · H100 | ~2ms / image |
| TensorRT FP16 · RTX PRO 6000 Blackwell | 3.61ms / image (277 FPS) |

The reported **0.8923 test-set mAP@50** was measured on the RTX PRO 6000 (the primary evaluation GPU), while the **~2ms latency** figure comes from a separate H100 run. These are not a single hardware pairing — don't combine the H100 latency with the RTX PRO 6000 accuracy figure as if they describe one deployment configuration.

For real-time use, see [Quick Start](#quick-start).

---

## Example Output

### Original footage
[![Input](https://img.youtube.com/vi/1PXWfTKSMkM/0.jpg)](https://www.youtube.com/shorts/1PXWfTKSMkM)

### YOLO26x Detection Output
https://github.com/user-attachments/assets/ffa2e47d-bd3b-4e57-be44-c952fb0af09d

---

## Numbers

### Final (Phase 3 checkpoint, TTA validated on the 13,186-image validation split)

| Metric | Value |
|---|---|
| mAP@50 | **0.8913** |
| mAP@50-95 | **0.683** |
| Precision | 0.889 |
| Recall | 0.820 |
| Best F1 | 0.8528 @ conf=0.10 |
| Inference (PyTorch, H100) | ~5ms / image |
| Inference (TRT FP16, H100) | ~2ms / image |

These are single-run results (seed 42). Multi-seed variance at full training scale has not been established — each full run costs ~35 GPU-hours, so seed-to-seed spread is left to future work.

The ~2ms/image TensorRT figure is pure model inference latency (batch size 1). It excludes video decode, preprocessing, host-device transfer, NMS/post-processing, tracking, and any application-level I/O — all of which bound real end-to-end pipeline throughput and were not measured here. Don't read this as "500 FPS end-to-end video processing."

Held-out test set (12,880 images, TensorRT FP16): mAP@50 = **0.8923**, mAP@50-95 = 0.7179. The comparable validation and test scores don't show an obvious generalization gap in these particular metrics, though no dedicated train/validation loss-divergence analysis was performed, and the cross-split leakage caveat above applies to this figure as well.

### Per-class mAP@50

| Class | Precision | Recall | mAP@50 |
|---|---|---|---|
| Explosive | 0.950 | 0.903 | **0.959** |
| Melee_Weapon | 0.937 | 0.892 | **0.949** |
| Firearm | 0.916 | 0.868 | **0.932** |
| Blunt_Weapon | 0.875 | 0.834 | **0.896** |
| Tool | 0.871 | 0.802 | **0.881** |
| Fire_Smoke | 0.879 | 0.802 | **0.875** |
| Person | 0.794 | 0.641 | 0.747 |

The strong scores on Explosive and Melee_Weapon may partly reflect that those classes have visually distinctive signatures (grenades, blades) relative to their backgrounds — this is a plausible interpretation, not something separately quantified. Firearm's 0.932 is achieved despite firearms appearing in more contextually varied, partially-occluded scenes across the dataset.

## Baseline Comparison: YOLOv8x

A YOLOv8x model was evaluated on the same dataset using the same curriculum *structure* (freeze → full fine-tune → high-res refinement). **This is a budget-constrained comparison, not an architecture-isolated ablation:** YOLO26x used 10 epochs/phase across all three phases (30 total) while YOLOv8x used 5 epochs/phase (15 total), and the two were trained on different hardware. The performance gap below reflects the combined effect of architecture, training budget, and hardware — it should not be read as evidence that YOLO26x is architecturally superior in isolation. A matched-epoch-budget run is the natural follow-up to isolate the architectural contribution. Full configs and per-epoch metrics are in [`baselines/yolov8x_comparison/`](baselines/yolov8x_comparison/).

| Model | Phase 3 val mAP@50 | Phase 3 test mAP@50 | Total epochs |
|-------|-------------------|---------------------|--------------|
| YOLOv8x (5 epochs/phase) | 0.7755 | 0.785 | 15 |
| YOLO26x (10 epochs/phase) | 0.891 | 0.8923 | 30 |

Under this comparison, YOLO26x reaches +11.6 pp higher mAP@50 than YOLOv8x — a budget-constrained empirical result, not an architecture-only one.

### Phase progression

| Phase | Epochs | imgsz | Frozen layers | mAP@50 |
|---|---|---|---|---|
| 1 Stabilization | 10 | 800 | 10 (backbone) | 0.865 |
| 2 Full backbone | 10 | 800 | 0 (all) | 0.881 |
| 3 High-res refinement | 10 | 1024 | 0 (all) | 0.891 |
| — TTA (non-training) | — | 1024 | — | **0.8913** |

30 epochs total across the three training phases (10 each), matching the paper's reported curriculum.

Phase 2→3 changes resolution (800→1024px) *together with* batch size, learning rate, mosaic probability, and augmentation strength (see hyperparameter table below). The observed +1.0 pp gain is associated with Phase 3 as a whole; it has not been isolated to resolution alone via a component-level ablation, so it shouldn't be attributed entirely to the resolution increase. It's plausible that the larger feature map at 1024px (more spatial positions per object) contributes to the small-object gains, but this is a reasonable hypothesis rather than a demonstrated causal result.

---

## Quick start

### Install

```bash
pip install ultralytics>=8.3.0 opencv-python numpy==1.26.4
```

NumPy is pinned to 1.26.x. Ultralytics 8.3 dropped support for NumPy 2.x in some ops. Check [requirements.txt](requirements.txt) for the full pinned list.

### Download weights

```bash
# Using huggingface_hub CLI
huggingface-cli download HaiderKhan6410/weapon-yolo26x \
  model/best.pt \
  --local-dir .
```

Or via Python:

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download("HaiderKhan6410/weapon-yolo26x", "model/best.pt")
```

### Inference

```python
from ultralytics import YOLO

model = YOLO("model/best.pt")

# Single image
results = model("image.jpg", conf=0.35, iou=0.45, imgsz=1024)
results[0].show()

# Video — stream=True is important, loads one frame at a time
for r in model("video.mp4", conf=0.35, iou=0.45, imgsz=1024, stream=True):
    print(r.boxes)
```

### TensorRT (fastest, GPU only)

The included `.engine` was compiled on H100 with TensorRT 10.15.1. **It will not load on a different GPU architecture** — TensorRT engines are GPU-architecture-specific. Re-export it:

```python
from ultralytics import YOLO
model = YOLO("model/best.pt")
model.export(format="engine", imgsz=1024, half=True, device=0, workspace=6)
```

Then load the exported `.engine` the same way as `best.pt`. The ONNX export (also included) can be recompiled on any NVIDIA platform without needing the original weights.

### CLI scripts

```bash
# PyTorch inference
python inference/infer.py --source image.jpg --weights model/best.pt

# TensorRT inference
python inference/infer_trt.py --source image.jpg --engine model/best_fp16.engine

# Webcam
python inference/infer.py --source 0 --no-save --show
```

---

## Files

```
.
├── flake.nix / flake.lock       # Reproducible Nix dev environment — primary entry point
├── assets
│   └── demo_output.mp4
├── baselines/
│   └── yolov8x_comparison/     # YOLOv8x budget-constrained comparison (see caveats above)
│       ├── phase2/              # args.yaml, results.csv
│       ├── phase3/              # args.yaml, results.csv
│       └── README.md
├── requirements.txt             # Fallback: pip-based dependency resolution
├── README.md                    # Project documentation
├── train.py               
├── app.py                       # Gradio demo (Hugging Face Spaces)
├── model/
│   └── best.pt                  # Final PyTorch weights (Phase 3, 1024px input)
│   └── best_fp16.engine 
├── inference/
│   ├── infer.py                 # PyTorch inference: images, video, webcam
│   ├── infer_trt.py             # TensorRT-optimized inference (GPU only)
│   └── _common.py               # Shared post-processing & visualization utilities
├── config/
│   ├── deploy_config.json       # Runtime thresholds, class mapping, metadata
│   └── validate.py              # Schema validation for config integrity
└── tests/
    └── test_smoke.py            # CPU-only sanity checks (CI/CD friendly)
```

---

## Training

### Dataset

| Split | Images |
|---|---|
| Train | 104,697 |
| Val | 13,186 |
| Background (train) | 9,032 (8.6%) |

Background images are retained intentionally, at a rate consistent with Ultralytics' recommended 0–10% guideline for training splits, to expose the detector to negative scenes.

### Why a multi-phase curriculum

A single-stage fine-tuning run of the pretrained YOLO26x checkpoint directly at 1024px (full backbone unfrozen, default augmentation) produced NaN loss divergence partway through training, reproducibly across independent attempts. Two design decisions were made in response:

1. **Numerical instability during training.** The naive single-stage run diverged with NaN losses. We initially hypothesized this was caused by Mosaic augmentation occasionally assembling a composite image out of four background (empty-label) tiles, producing a batch with no positive training signal — plausible given the dataset's background rate. However, targeted diagnostics (a small-scale Mosaic×AMP grid, an instrumented full-scale run with per-batch NaN localization, a component-freezing test, and disabling AMP entirely) told a more complicated story: the specific batch that triggered the crash in the instrumented run contained **zero** degenerate or empty-box images, which argues against the fully-empty-Mosaic-composite mechanism as the direct trigger. Disabling AMP did eliminate the instability in the tested runs, which points to AMP-related numerical behavior as the more likely proximate factor — but the exact low-level mechanism (which operator, why it varies between runs) was not conclusively identified. **In short: the instability is real and reproducible, AMP is implicated, but the precise causal chain remains an open question.** The proposed curriculum remained stable in the reported training runs (below), even though it doesn't disable AMP — why the staged schedule avoids the instability isn't fully understood either.

2. **Resolution gap.** The pretrained checkpoint was developed around lower-resolution inputs, and the deployment target here is 1024px. Training first at 800px, then refining at 1024px, provides an intermediate adaptation stage rather than jumping straight to the highest resolution.

### Phase details

**Phase 1 — Stabilization (10 epochs, 800px)**

Freeze the first 10 backbone layers. Only the neck and head train. `AdamW` with `lr0=8e-5`, no mixup, no copy_paste. Light augmentation (`degrees=10`, `scale=0.5`, `erasing=0.3`). This allows the detection head and neck to adapt while limiting early gradient updates to the pretrained backbone, minimizing the blast radius of any single anomalous batch during the still-unstable early phase. Loss is stable from epoch 1 in this configuration.

**Phase 2 — Full backbone (10 epochs, 800px)**

Unfreeze everything. Drop LR to `5e-5`. Add `mixup=0.15` and `copy_paste=0.3`. These augmentations are intended to increase scene and contextual diversity — mixup for overlapping weapon/person scenes, copy_paste for weapon-in-new-context combinations the dataset underrepresents — though their individual contribution has not been isolated via component-level ablation at full scale (a smaller supplementary ablation on a 20% data subset suggested mixup contributes a modest additional gain over unfreezing alone, while copy-paste's contribution was not measurably distinguishable from zero at that scale; see the paper for details). `degrees=12`, `scale=0.6` — heavier geometric augmentation now that training has stabilized.

**Phase 3 — High-res refinement (10 epochs, 1024px)**

Load Phase 2 `best.pt`. Drop batch from 32→12 to fit H100 memory at 1024px. `lr0=2e-5` — conservative, intended as fine adjustment rather than relearning. Reduce mosaic to 0.8. This phase's primary intent is improving sensitivity to small objects at higher resolution; recall does move from 0.808 → 0.819 alongside the mAP gain, consistent with (but not proof of) that intent, since other hyperparameters changed simultaneously (see [Phase progression](#phase-progression) above).

**Export / TTA (non-training)**

TTA evaluation with `augment=True`, `conf=0.001`, `iou=0.6`, applied to the Phase 3 checkpoint. This step performs no weight updates and is not a training phase — it generates multiple augmented views per image and merges predictions via weighted box fusion, giving a +0.0003 absolute mAP improvement (+0.03 percentage points — not 3%) at zero training cost. The TRT export took 447s on H100 (workspace=6GB, half=True, imgsz=1024, batch=1) — normal for this configuration.

### Hyperparameter table

| | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|
| epochs | 10 | 10 | 10 |
| imgsz | 800 | 800 | 1024 |
| batch | 32 | 32 | 12 |
| optimizer | AdamW | AdamW | AdamW |
| lr0 | 8e-5 | 5e-5 | 2e-5 |
| lrf | 0.01 | 0.01 | 0.005 |
| freeze | 10 | 0 | 0 |
| mosaic | 1.0 | 1.0 | 0.8 |
| mixup | 0.0 | 0.15 | 0.1 |
| copy_paste | 0.0 | 0.3 | 0.2 |
| degrees | 10 | 12 | 8 |
| scale | 0.5 | 0.6 | 0.5 |
| erasing | 0.3 | 0.4 | 0.2 |
| label_smoothing | 0.05 | 0.05 | 0.05 |
| patience | 10 | 15 | 10 |
| cos_lr | ✓ | ✓ | ✓ |
| amp | ✓ | ✓ | ✓ |

### Reproducing

```bash
# Phase 1 through 3 + TRT export in one shot
python train.py \
  --data dataset.yaml \
  --base-weights yolo26x.pt \
  --work-dir runs/weapon_yolo26x

# Resume from a checkpoint
python train.py \
  --data dataset.yaml \
  --resume-from runs/weapon_yolo26x/phase2/weights/best.pt \
  --start-phase 3

# Export only (if you already have best.pt)
python train.py \
  --export-only \
  --weights model/best.pt \
  --data dataset.yaml
```

The full pipeline takes ~6 hours on H100 (10+10+10 epochs at the respective resolutions, plus the 447s TRT build). This reflects a single training run — reproducing the exact reported numbers to several decimal places is not expected given normal training variance across seeds/hardware.

---

## Model card

| | |
|---|---|
| Architecture | YOLO26x |
| Parameters | 58.8M |
| GFLOPs (train) | 208.6 |
| GFLOPs (fused) | 193.4 |
| Framework | PyTorch / Ultralytics ≥ 8.3 |
| TRT engine | TensorRT 10.15.1 FP16 |
| Training GPU | NVIDIA H100 80GB |
| Python | 3.12.12 |
| PyTorch | 2.9.0+cu126 |
| CUDA | 12.6 |

---

## Limitations

To be honest about what this model doesn't do well.

**Person class is weak.** mAP@50 of 0.747 vs 0.93+ for weapon classes — this is a designed consequence of context-restricted annotation, not a bug: `Person` was labeled only in weapon-adjacent scenes, not exhaustively. Don't use this as a general person-detector or for headcount/occupancy tasks. If you need robust general person detection, run a dedicated, exhaustively-annotated person detector in parallel.

**Low-light performance is unverified.** The dataset skews toward daylight/indoor security footage. Qualitative spot-checks suggest recall degrades on night-time CCTV footage, but this has not been quantitatively benchmarked.

**Extreme occlusion is hard.** A half-visible handgun behind a jacket will likely be missed. The model has no depth or shape-completion capability — it's 2D texture-and-shape matching.

**The `.engine` file is H100-specific.** TensorRT engines are not portable across GPU architectures. Re-export from `best.pt` for any other GPU.

**Cross-split leakage has not been ruled out.** See the [dataset section](#dataset) above — deduplication was per-split only, so a small amount of train/val or train/test near-duplicate leakage is possible and unquantified.

**Single-seed results.** All full-scale numbers reflect one training run. Seed-to-seed variance is not established.

**Not a safety system.** Do not deploy this as a sole gate in any safety-critical pipeline without human review and proper evaluation on your specific deployment domain. All detections should be reviewed by trained personnel before any action is taken.

---

## Threshold guidance (exploratory)

Default thresholds (`conf=0.35`, `iou=0.45`) are a starting point, not a validated deployment recommendation. The suggestions below are heuristics for where to start experimenting, not thresholds validated against a deployment-representative dataset. Adjust based on your use case:

- **Security screening / high recall needed:** try lowering conf toward 0.15–0.25. Expect more false positives. The best F1 on the validation set occurs near conf=0.10, which suggests the default of 0.35 is conservative relative to that metric — but the right threshold for your deployment depends on your specific false-positive/false-negative tolerance and should be calibrated on your own data, not assumed from this dataset's F1 curve.
- **Alert systems / low false-positive budget:** raise conf to 0.5–0.6. You will miss more real detections but the ones you get will be higher-confidence.
- **Overlapping objects / dense scenes:** try lowering iou toward 0.35. Higher iou (0.6+) is more aggressive at suppressing boxes and can merge nearby distinct weapons.

---

## Development environment

The `flake.nix` provides two shells:

```bash
# Full dev shell (torch, ultralytics, gradio via pip venv)
nix develop

# Download-only shell (just huggingface-hub)
nix develop .#download
```

Python 3.12 is pinned — 3.13 is excluded because NumPy 1.26.x dropped Py3.13 support and ultralytics 8.3 hasn't been verified against NumPy 2.x.

The pip venv is intentional. Torch and TensorRT are too GPU-specific and too large to package cleanly in nixpkgs. Nix provides the Python interpreter and system libs (libstdc++, libGL, glib); pip owns the ML stack inside `.venv/`.

---

## Tests

```bash
pytest tests/ -v
```

The smoke tests don't require a GPU or the model weights. They cover: config loading and validation, CLI argument parsing, and the output-path collision-avoidance logic in `_common.py`.

---

## Citation

If citing the underlying research (dataset construction, curriculum design, and the NaN/AMP diagnostic investigation), please cite the paper rather than this repository alone. If citing this specific software/model artifact:

```bibtex
@misc{haiderkhan6410_yolo26x_2026,
  author       = {Haider Khan},
  title        = {Weapon Detection YOLO26x},
  year         = {2026},
  publisher    = {Hugging Face},
  url          = {https://huggingface.co/HaiderKhan6410/weapon-yolo26x},
  note         = {Available on Hugging Face and GitHub.}
}
```

---

## License

**Code and model weights in this repository:** [BigScience OpenRAIL-M](https://huggingface.co/spaces/bigscience/license).

Free to use, modify, and distribute, including commercially, provided the use-based
restrictions in Attachment A of the license are respected. Key restrictions: no use for
illegal purposes, no generating or disseminating disinformation, no use in fully automated
decision systems that affect legal rights without human oversight.

**Weapons-130K dataset:** CC BY 4.0 (separate from the code/model license above — see [Dataset](#dataset)).

Any deployment in a real-world security context should be done in compliance with local
laws, with appropriate human oversight, and with proper evaluation on the target domain
before going live.
