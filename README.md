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

A 7-class weapon detector trained on 104,697 images using a 4-phase progressive fine-tuning pipeline. Final mAP@50 of **0.8913** with TTA. TensorRT FP16 export runs at ~5ms/image on H100.

This README documents exactly what was done, why each decision was made, what the numbers actually mean, and how to reproduce everything from scratch.

---

## What this is

A fine-tuned [YOLO26x](https://github.com/ultralytics/ultralytics) checkpoint for detecting weapons and related threat objects in images and video. The seven classes are: `Blunt Weapon`, `Explosive`, `Fire Smoke`, `Firearm`, `Melee_Weapon`, `Person`, `Tool`.

`Person` is intentionally included as a context class, not a standalone person detector. Its lower mAP@50 (0.747) is expected person annotations in the dataset are sparse and only label people in weapon-adjacent scenes. Don't use this model as a people counter.

The training pipeline is the main contribution here. Most public YOLO uploads are a single training run with default hyperparameters. This one uses a 4-phase curriculum that addresses two real problems: catastrophic gradient updates from mosaic augmentation on background-heavy data early in training, and the resolution gap between pretraining (640px) and deployment (1024px). Details in the [Training](#training) section.

---

## Dataset

**Weapons-130K** · [Kaggle](https://www.kaggle.com/datasets/haiderkhan6410/weapons-130k) · CC BY 4.0

| Property | Value |
|---|---|
| Total images | 130,763 |
| Total annotations | 294,950 |
| Train / Val / Test | 104,697 / 13,186 / 12,880 |
| Image size | 800 × 800 px |
| Format | YOLO bounding box (normalized `class cx cy w h`) |
| Background frames | 12,342 (9.9% intentionally retained) |

### Class distribution

| ID | Class | Annotations | Share |
|---|---|---|---|
| 0 | Blunt_Weapon | 9,527 | 3.2% |
| 1 | Explosive | 25,599 | 8.7% |
| 2 | Fire_Smoke | 78,171 | 26.5% |
| 3 | Firearm | 117,864 | 39.9% |
| 4 | Melee_Weapon | 40,166 | 13.6% |
| 5 | Person | 11,801 | 4.0% |
| 6 | Tool | 11,822 | 4.0% |

The dataset is imbalanced by design. Firearm and Fire_Smoke dominate because these are the most common signals in real surveillance footage. If your use case requires class balance, apply weighted loss or oversample minority classes.

### Cleaning pipeline

Raw source data from Roboflow Universe (76 heterogeneous classes) was consolidated and cleaned in 4 audited stages before training:

1. **Annotation repair** 1,052 segmentation polygon lines stripped and converted to bbox format across 772 files.
2. **Deduplication** 5,169 near-duplicate images removed via perceptual hashing (per-split, not global, to prevent cross-split leakage).
3. **Degenerate bbox removal** 153 zero-area or out-of-bounds boxes dropped; these cause NaN losses under AMP.
4. **Background frame audit** 12,342 empty-label frames confirmed as genuine negatives and retained. A detector trained without background frames learns to always predict something; on real CCTV feeds where most frames are weapon-free, this produces unacceptable false-positive rates.

**Integrity checks (all passed):** zero corrupt images · zero out-of-bounds coordinates · zero orphan labels · zero missing labels · zero invalid class IDs.

Net annotation delta: 302,550 → 294,950 (−7,600), fully accounted for by removed duplicates and degenerate boxes.

---

## Live Demo

[![Open in Spaces](https://img.shields.io/badge/🤗%20Open%20in%20Spaces-weapon--yolo26x--demo-yellow)](https://huggingface.co/spaces/HaiderKhan6410/weapon-yolo26x-demo)

Upload a video, adjust confidence and IoU thresholds, get annotated output with a per-class breakdown. No GPU, no install.

**7 classes:** Blunt Weapon · Explosive · Fire/Smoke · Firearm · Melee Weapon · Person · Tool

> **Speed warning:** The Space runs CPU-only (free HF tier). Expect 1–5s/image. Not a model issue re-export and run locally if you need real throughput.

| Environment | Latency |
|---|---|
| HF Space (CPU, free tier) | ~1–5s / image |
| PyTorch · NVIDIA GPU | ~5ms / image |
| TensorRT FP16 · H100 | ~2ms / image |

For real-time use, see [Quick Start](#quick-start).

---

## Example Output

### Original footage
[![Input](https://img.youtube.com/vi/1PXWfTKSMkM/0.jpg)](https://www.youtube.com/shorts/1PXWfTKSMkM)

### YOLO26x Detection Output
https://github.com/user-attachments/assets/ffa2e47d-bd3b-4e57-be44-c952fb0af09d

---





## Numbers

### Final (Phase 3 checkpoint, TTA validated)

| Metric | Value |
|---|---|
| mAP@50 | **0.8913** |
| mAP@50-95 | **0.6836** |
| Precision | 0.890 |
| Recall | 0.819 |
| Best F1 | 0.8528 @ conf=0.10 |
| Inference (PyTorch, H100) | ~5ms / image |
| Inference (TRT FP16, H100) | ~2ms / image |

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

The high numbers on Explosive and Melee_Weapon are partly a dataset artifact those classes have distinctive visual signatures (grenades, blades) relative to their backgrounds. Firearm at 0.932 is more meaningful because firearms appear in more varied contexts with more partial occlusions.

## Controlled Baseline

A YOLOv8x model was trained on the same dataset and curriculum to isolate
backbone depth as the sole variable. Full configs and per-epoch metrics are in
[`baselines/yolov8x_comparison/`](baselines/yolov8x_comparison/).

| Model | Phase 3 val mAP@50 | Phase 3 test mAP@50 | Δ vs YOLO26x |
|-------|-------------------|---------------------|--------------|
| YOLOv8x (baseline, 5 epochs/phase) | 0.775 | 0.785 | — |
| YOLO26x (ours, 10 epochs/phase) | 0.891 | 0.892 | **+11.6 pp** |

### Phase progression

| Phase | Epochs | imgsz | Frozen layers | mAP@50 |
|---|---|---|---|---|
| 1 Stabilization | 10 | 800 | 10 | 0.865 |
| 2 Full backbone | 15 | 800 | 0 | 0.881 |
| 3 High-res refinement | 10 | 1024 | 0 | 0.891 |
| 4 TTA validation | — | 1024 | — | **0.8913** |

The +1.6% jump from Phase 2→3 comes entirely from the resolution increase. Small objects (knives at distance, grenade pins) that were borderline at 800px become unambiguous at 1024px. This is consistent with what you'd expect from the receptive field math.

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

# Video stream=True is important, loads one frame at a time
for r in model("video.mp4", conf=0.35, iou=0.45, imgsz=1024, stream=True):
    print(r.boxes)
```

### TensorRT (fastest, GPU only)

The included `.engine` was compiled on H100 with TensorRT 10.15.1. **It will not load on a different GPU architecture.** Re-export it:

```python
from ultralytics import YOLO
model = YOLO("model/best.pt")
model.export(format="engine", imgsz=1024, half=True, device=0, workspace=6)
```

Then load the exported `.engine` the same way as `best.pt`.

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
├── flake.nix / flake.lock       # Reproducible Nix dev environment primary entry point
├── assets
│   └── demo_output.mp4
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

The 8.6% background images are intentional and important. They teach the model to suppress false positives in clean scenes. Removing them consistently hurts precision on real-world footage where most frames contain no weapons.

### Why 4 phases

Fine-tuning YOLO26x from scratch (i.e., with a single `model.train()` call at 1024px) produces an unstable run. Two things go wrong:

1. **Mosaic + background images + AMP = NaN.** Mosaic augmentation at 1024px assembles 4 images into one. When a background tile (no annotations) gets combined with weapon tiles, the AMP GradScaler occasionally sees a NaN gradient and skips the update. At 8.6% background frequency this happens enough in the first few epochs to destabilize the optimizer. The fix is to start with freeze=10 (backbone frozen), smaller LR, and no mixup or copy_paste until the head is stable.

2. **Resolution gap.** The base YOLO26x checkpoint was pretrained at 640px. Jumping directly to 1024px triples the spatial resolution. The attention patterns and anchor statistics are mismatched. Training at 800px first lets the model re-learn the feature scales before the final resolution bump.

### Phase details

**Phase 1 Stabilization (10 epochs, 800px)**

Freeze the first 10 backbone layers. Only the neck and head train. `AdamW` with `lr0=8e-5`, no mixup, no copy_paste. Light augmentation (`degrees=10`, `scale=0.5`, `erasing=0.3`). This gets the head calibrated without corrupting the pretrained backbone features. Loss is stable from epoch 1.

**Phase 2 Full backbone (15 epochs, 800px)**

Unfreeze everything. Drop LR to `5e-5`. Add `mixup=0.15` and `copy_paste=0.3`. These two augmentations are high-value for weapon detection specifically: mixup teaches the model to handle overlapping weapon/person scenes, copy_paste synthesizes uncommon weapon-in-new-context combinations that the dataset underrepresents. `degrees=12`, `scale=0.6` heavier geometric augmentation now that the backbone is stable enough to handle it.

**Phase 3 High-res refinement (10 epochs, 1024px)**

Load Phase 2 `best.pt`. Drop batch from 32→12 to fit H100 memory at 1024px. `lr0=2e-5` very conservative, we're making fine adjustments to existing good features, not relearning. Reduce mosaic to 0.8 (full mosaic at 1024px is expensive and we're in a refinement phase). This phase's job is purely recall improvement on small objects. It does exactly that: recall goes from 0.808 → 0.819.

**Phase 4 Export**

TTA validation with `augment=True`, `conf=0.001`, `iou=0.6`. TTA adds ~3× inference cost but gives a more honest mAP estimate than single-pass. The TRT export took 447s on H100 this is normal for a workspace=6GB, half=True, imgsz=1024 build.

### Hyperparameter table

| | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|
| epochs | 10 | 15 | 10 |
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

The full pipeline takes ~6 hours on H100 (10+15+10 epochs at the respective resolutions, plus the 447s TRT build).

---

## Model card

| | |
|---|---|
| Architecture | YOLO26x |
| Parameters (train) | 58.8M |
| Parameters (fused) | 55.6M |
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

**Person class is weak.** mAP@50 of 0.747 vs 0.93+ for weapon classes. The dataset labels people only in weapon-adjacent contexts and the annotations are incomplete. If you need robust person detection, use a dedicated model (e.g., YOLOv8n pretrained on COCO) in parallel.

**Low-light degrades recall meaningfully.** The dataset skews toward daylight/indoor security footage. Dark scenes drop recall on Melee_Weapon and Blunt_Weapon noticeably. No quantified benchmarks for this yet.

**Extreme occlusion is hard.** A half-visible handgun behind a jacket will likely be missed. The model has no depth or shape-completion capability it's purely 2D texture-and-shape matching.

**The `.engine` file is H100-specific.** TensorRT engines are not portable across GPU architectures. Re-export from `best.pt` for any other GPU. The re-export takes ~8 minutes on a T4, ~15 minutes on an older V100.

**Not a safety system.** Detection accuracy in the low-to-mid 90s means false negatives at non-trivial rates. Do not deploy this as a sole gate in any safety-critical pipeline without human review and proper evaluation on your specific deployment domain.

---

## Threshold guidance

Default thresholds (`conf=0.35`, `iou=0.45`) are a balanced starting point. Adjust based on your use case:

- **Security screening / high recall needed:** lower conf to 0.15–0.25. Expect more false positives. The Best F1 is actually at conf=0.10, so the model's natural operating point is lower than the default.
- **Alert systems / low false-positive budget:** raise conf to 0.5–0.6. You will miss more real detections but the ones you get will be high-confidence.
- **Overlapping objects / dense scenes:** lower iou to 0.35. Higher iou (0.6+) is more aggressive at suppressing boxes and can merge nearby distinct weapons.

---

## Development environment

The `flake.nix` provides two shells:

```bash
# Full dev shell (torch, ultralytics, gradio via pip venv)
nix develop

# Download-only shell (just huggingface-hub)
nix develop .#download
```

Python 3.12 is pinned 3.13 is excluded because NumPy 1.26.x dropped Py3.13 support and ultralytics 8.3 hasn't been verified against NumPy 2.x.

The pip venv is intentional. Torch and TensorRT are too GPU-specific and too large to package cleanly in nixpkgs. Nix provides the Python interpreter and system libs (libstdc++, libGL, glib); pip owns the ML stack inside `.venv/`.

---

## Tests

```bash
pytest tests/ -v
```

The smoke tests don't require a GPU or the model weights. They cover: config loading and validation, CLI argument parsing, and the output-path collision-avoidance logic in `_common.py`. These are the classes of bugs most likely to surface silently in a long inference run.

---

## Citation

```bibtex
@misc{haiderkhan6410_yolo26x_2026,
  author       = {Haider Khan},
  title        = {Weapon YOLO26x: Multi-Phase Real-Time Weapon Detection},
  year         = {2026},
  publisher    = {Hugging Face},
  url          = {https://huggingface.co/HaiderKhan6410/weapon-yolo26x},
  note         = {Available on Hugging Face and GitHub. Accessed: 2026-03-26}
}
```

---

## License

[BigScience OpenRAIL-M](https://huggingface.co/spaces/bigscience/license)

Free to use, modify, and distribute, including commercially, provided the use-based
restrictions in Attachment A of the license are respected. Key restrictions: no use for
illegal purposes, no generating or disseminating disinformation, no use in fully automated
decision systems that affect legal rights without human oversight.

Any deployment in a real-world security context should be done in compliance with local
laws, with appropriate human oversight, and with proper evaluation on the target domain
before going live.
