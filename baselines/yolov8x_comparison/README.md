# YOLOv8x Baseline

A YOLOv8x weapon detector trained on Weapons-130K using a 4-phase progressive
curriculum. This directory contains training configs and metrics for Phases 2 & 3.

## Available Artifacts

| Phase | Epochs | imgsz | Frozen Layers | Files |
|-------|--------|-------|---------------|-------|
| 2 — Full Backbone | 15 | 800 | 0 | args.yaml, results.csv |
| 3 — High-Res | 10 | 1024 | 0 | args.yaml, results.csv |

> **Note:** Phase 1 artifacts were not retained. Phase 2 was initialized from
> the standard YOLOv8x COCO-pretrained checkpoint.

## Weights

Hosted on Hugging Face due to file size:

- Phase 2: `HaiderKhan6410/yolov8x-baseline` → `phase2/best.pt`
- Phase 3: `HaiderKhan6410/yolov8x-baseline` → `phase3/best.pt`

```bash
from huggingface_hub import hf_hub_download
path = hf_hub_download("HaiderKhan6410/yolov8x-baseline", "phase3/best.pt")
```

## Reproducing

```bash
pip install ultralytics>=8.3.0

# Phase 2
yolo train cfg=baselines/yolov8x_comparison/phase2/args.yaml

# Phase 3 — initialize from Phase 2 weights
yolo train cfg=baselines/yolov8x_comparison/phase3/args.yaml \
           model=path/to/phase2/best.pt
```
