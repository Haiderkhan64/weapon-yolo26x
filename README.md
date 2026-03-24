---
license: cc0-1.0
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

# Weapon Detection YOLO26x — Real-Time, Multi-Phase Trained

A production-grade weapon detection model built on **YOLO26x**, trained using a
4-phase progressive pipeline on 104,697 images and optimized for deployment
with **TensorRT FP16**.

---

## Overview

| Property | Value |
|---|---|
| Architecture | YOLO26x (Ultralytics 8.4.22) |
| Parameters | 58.8M (train) / 55.6M (fused) |
| GFLOPs | 208.6 (train) / 193.4 (fused) |
| Input Size | 800px (Ph1–2) → 1024px (Ph3) |
| Framework | PyTorch (Ultralytics ≥ 8.3) |
| Optimization | TensorRT 10.15.1 FP16 |
| GPU used | NVIDIA H100 80GB |
| License | CC0 — Public Domain |

---

## Final Performance (Phase 3 — TTA validated)

| Metric | Value |
|---|---|
| **mAP@50** | **0.8913** |
| **mAP@50-95** | **0.6836** |
| Precision | 0.890 |
| Recall | 0.819 |
| Best F1 | 0.8528 (@ conf=0.10) |
| Inference speed (PyTorch) | 5.0ms / image (H100) |

### Per-Class mAP@50

| Class | P | R | mAP@50 |
|---|---|---|---|
| Explosive | 0.950 | 0.903 | **0.959** |
| Melee_Weapon | 0.937 | 0.892 | **0.949** |
| Firearm | 0.916 | 0.868 | **0.932** |
| Blunt_Weapon | 0.875 | 0.834 | **0.896** |
| Tool | 0.871 | 0.802 | **0.881** |
| Fire_Smoke | 0.879 | 0.802 | **0.875** |
| Person | 0.794 | 0.641 | 0.747 |

### Phase-by-Phase Progression

| Phase | Epochs | imgsz | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| Phase 1 (Stabilization) | 10 | 800 | 0.865 | 0.657 |
| Phase 2 (Full backbone) | 15 | 800 | 0.881 | 0.683 |
| Phase 3 (High-res) | 10 | **1024** | **0.891** | **0.684** |
| Phase 4 (TTA validated) | — | 1024 | **0.8913** | **0.6836** |

---

## Files

| File | Size | Description |
|---|---|---|
| `model/best.pt` | 118.4 MB | Final PyTorch model (Phase 3, 1024px) |
| `model/best_fp16.engine` | 110.0 MB | TensorRT 10.15 FP16 engine (H100) |
| `config/deploy_config.json` | — | Inference thresholds and runtime config |
| `inference/infer.py` | — | PyTorch inference (images + video) |
| `inference/infer_trt.py` | — | TensorRT inference |
| `train.py` | — | Reproducible 4-phase training pipeline |
| `requirements.txt` | — | Pinned dependency list |

---

## Quick Start

### Install

```bash
pip install ultralytics>=8.3.0 opencv-python numpy==1.26.4
```

### PyTorch Inference (images)

```python
from ultralytics import YOLO

model = YOLO("model/best.pt")
results = model("your_image.jpg", conf=0.35, iou=0.45)
results[0].show()
results[0].save("output.jpg")
```

### PyTorch Inference (video)

```python
from ultralytics import YOLO

model = YOLO("model/best.pt")
results = model("your_video.mp4", conf=0.35, stream=True)
for r in results:
    print(r.boxes)
```

### TensorRT Inference (fastest — GPU only)

```python
from ultralytics import YOLO

model = YOLO("model/best_fp16.engine")
results = model("your_image.jpg", conf=0.35, iou=0.45)
results[0].show()
```

### Using the inference scripts

```bash
# PyTorch — image
python inference/infer.py --source your_image.jpg --weights model/best.pt

# PyTorch — video
python inference/infer.py --source your_video.mp4 --weights model/best.pt

# TensorRT — image
python inference/infer_trt.py --source your_image.jpg --engine model/best_fp16.engine
```

---

## Training Strategy (What Makes This Different)

Most public YOLO models are trained once and uploaded.
This model uses a **4-phase progressive pipeline** to eliminate training instability
and maximize accuracy at deployment resolution.

### Phase 1 — Stabilization (10 epochs, imgsz=800)
- Starts from a pretrained YOLO26x checkpoint
- `optimizer=AdamW`, `lr0=8e-5`, `freeze=5`, `label_smoothing=0.05`
- Explicit handling of the 8.6% background image problem (9,032 / 104,697)
  which causes mosaic NaN in the AMP GradScaler without mitigation
- Result: **mAP@50 = 0.865**

### Phase 2 — Full Backbone Training (15 epochs, imgsz=800)
- Loads Phase 1 best weights
- `freeze=0` (full backbone), `lr0=5e-5`, `mixup=0.15`, `copy_paste=0.3`
- `degrees=12`, `scale=0.6`, heavier augmentation
- Result: **mAP@50 = 0.881**

### Phase 3 — High-Resolution Refinement (10 epochs, imgsz=1024)
- Loads Phase 2 best weights
- 1024px input sharpens small-object recall (knives, grenades at distance)
- `batch=12` to fit GPU memory at higher resolution
- Result: **mAP@50 = 0.891**

### Phase 4 — Optimization & Export
- TTA validation (`augment=True`): mAP@50 = **0.8913**, mAP@50-95 = **0.6836**
- TensorRT 10.15.1 FP16 export: 447.7s build time, 110MB engine
- `deploy_config.json` written with final runtime thresholds

---

## Dataset

| Property | Value |
|---|---|
| Train images | 104,697 |
| Val images | 13,186 |
| Background images | 9,032 (8.6%) — intentional, teaches FP suppression |
| Classes | 7 |
| Source | Kaggle — `weapon-detection-yolo-v3-cleaned` |

**Classes:** Blunt_Weapon, Explosive, Fire_Smoke, Firearm, Melee_Weapon, Person, Tool

---

## ⚠️ Limitations

- Person class mAP@50 is lower (0.747) — "Person" is a secondary context label
  in weapon scenes, not a standalone person detector
- Performance may degrade in very low-light or extreme occlusion scenarios
- The `.engine` file is compiled for NVIDIA H100 with TensorRT 10.15.1 —
  other GPUs require re-exporting from `best.pt`

---

## Re-export TensorRT for Your GPU

```python
from ultralytics import YOLO

model = YOLO("model/best.pt")
model.export(
    format="engine",
    imgsz=1024,
    half=True,
    device=0,
    workspace=6,
)
```

---

## Environment

| Library | Version |
|---|---|
| Python | 3.12.12 |
| PyTorch | 2.9.0+cu126 |
| Ultralytics | 8.4.22 |
| NumPy | 1.26.4 |
| OpenCV | 4.10.0.84 |
| CUDA | 12.6 |
| TensorRT | 10.15.1.29 |

---

## Citation

```
Haider Khan (2026). Weapon YOLO26x — Multi-Phase Real-Time Detection Model.
Hugging Face. https://huggingface.co/haiderkhan6410/weapon-yolo26x
```

---

## License

CC0 1.0 Universal — Public Domain Dedication.

> Please use responsibly. Weapon detection systems should be deployed
> in compliance with local laws and with appropriate ethical oversight.
