# YOLOv8x Baseline

A YOLOv8x weapon detector trained on Weapons-130K using a 4-phase progressive
curriculum. This directory contains training configs and metrics for Phases 2 & 3.

## Available Artifacts

| Phase | Epochs | imgsz | Frozen Layers | Files |
|-------|--------|-------|---------------|-------|
| 1 Stabilization | 5 | 800 | 10 | not retained |
| 2 Full Backbone | 5 | 800 | 0 | args.yaml, results.csv |
| 3 High-Res | 5 | 1024 | 0 | args.yaml, results.csv |

> **Phase 1 note:** Phase 1 ran for 5 epochs with the first 10 backbone layers
> frozen (`freeze=10`), stabilizing the detection head before full fine-tuning.
> Weights were not saved — Phase 2 was initialized from Phase 1's best.pt
> during the same Kaggle session. Phase 1 val mAP@50 = **0.650**.

## Results

| Phase | val mAP@50 | val mAP@50-95 |
|-------|------------|---------------|
| 1 | 0.650 | 0.439 |
| 2 | 0.750 | 0.536 |
| 3 (val) | 0.775 | 0.556 |
| 3 (test) | 0.785 | 0.567 |

> YOLOv8x ran 5 epochs/phase vs 10 for YOLO26x due to compute constraints.
> Validation mAP plateaued by epoch 5, confirming convergence.

## Reproducing

```bash
pip install ultralytics>=8.3.0

# Phase 2
yolo train cfg=baselines/yolov8x_comparison/phase2/args.yaml

# Phase 3 — initialize from Phase 2 weights
yolo train cfg=baselines/yolov8x_comparison/phase3/args.yaml \
           model=path/to/phase2/best.pt
```
