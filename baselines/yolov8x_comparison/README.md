# YOLOv8x Controlled Baseline

This directory contains training artifacts for the YOLOv8x baseline used in
Table VII of the paper. The same 4-phase curriculum and Weapons-130K dataset
were applied to isolate backbone capacity as the sole variable.

## Why YOLOv8x?

YOLOv8x was selected as the comparison backbone because it is the closest
publicly available architecture in parameter count to YOLO26x. Using the same
dataset, curriculum, and hyperparameters ensures the +11.6 pp mAP@50 gain
is attributable to backbone depth, not training procedure differences.

## Available Artifacts

| Phase | Epochs | imgsz | Frozen | mAP@50 | Files |
|-------|--------|-------|--------|--------|-------|
| 2 — Full Backbone | 15 | 800 | 0 | see results.csv | args.yaml, results.csv |
| 3 — High-Res | 10 | 1024 | 0 | see results.csv | args.yaml, results.csv |

> **Phase 1 note:** Phase 1 artifacts were not retained due to storage
> constraints during iterative development. Phase 2 was initialized from the
> standard YOLOv8x COCO-pretrained checkpoint, enabling full reconstruction
> of the reported results.

## Weights

Weights are hosted on Hugging Face due to file size:

- Phase 2: `https://huggingface.co/HaiderKhan6410/yolov8x-baseline/resolve/main/phase2/best.pt`
- Phase 3: `https://huggingface.co/HaiderKhan6410/yolov8x-baseline/resolve/main/phase3/best.pt`

```bash
from huggingface_hub import hf_hub_download
path = hf_hub_download("HaiderKhan6410/yolov8x-baseline", "phase3/best.pt")
```

## Reproducing the Baseline

```bash
pip install ultralytics>=8.3.0

# Phase 2
yolo train cfg=baselines/yolov8x_comparison/phase2/args.yaml

# Phase 3 (initialize from Phase 2 best.pt)
yolo train cfg=baselines/yolov8x_comparison/phase3/args.yaml \
           model=path/to/phase2/best.pt
```

## Convergence Note

YOLOv8x was trained for 5 epochs/phase vs. 10 for YOLO26x due to compute
constraints. Validation mAP plateaued within the first 5 epochs (< 1% gain
between epochs 3–5), confirming the model had converged before the cutoff.
This is visible in `results.csv` for both phases.
