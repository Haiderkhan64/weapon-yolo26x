"""
infer.py — PyTorch inference for Weapon YOLO26x

Supports: single image, directory of images, video file, webcam (--source 0)

Usage:
    python infer.py --source image.jpg
    python infer.py --source video.mp4
    python infer.py --source images/
    python infer.py --source 0                          (webcam)
    python infer.py --source image.jpg --conf 0.4
    python infer.py --source video.mp4 --no-save --show
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
from ultralytics import YOLO

from _common import log_summary, process_results, setup_logging

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS = Path(__file__).parent.parent / "model" / "best.pt"
DEFAULT_CONF    = 0.35
DEFAULT_IOU     = 0.45
DEFAULT_IMGSZ   = 1024
DEFAULT_OUTDIR  = Path("runs/detect")

logger = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Weapon YOLO26x — PyTorch inference")
    p.add_argument("--source",   required=True,
                   help="Image / video / folder / webcam index")
    p.add_argument("--weights",  default=str(DEFAULT_WEIGHTS),
                   help="Path to best.pt  (default: model/best.pt)")
    p.add_argument("--conf",     type=float, default=DEFAULT_CONF,
                   help=f"Confidence threshold (default {DEFAULT_CONF})")
    p.add_argument("--iou",      type=float, default=DEFAULT_IOU,
                   help=f"NMS IoU threshold (default {DEFAULT_IOU})")
    p.add_argument("--imgsz",    type=int,   default=DEFAULT_IMGSZ,
                   help=f"Inference image size (default {DEFAULT_IMGSZ})")
    p.add_argument("--no-save",  action="store_true",
                   help="Disable saving annotated output")
    p.add_argument("--show",     action="store_true",
                   help="Display results in a window (requires display)")
    p.add_argument("--outdir",   default=str(DEFAULT_OUTDIR),
                   help="Output directory for saved frames")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(weights: str) -> YOLO:
    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights_path}\n"
            "Download from: https://huggingface.co/haiderkhan6410/weapon-yolo26x"
        )
    logger.info("Loading model: %s", weights_path)
    model  = YOLO(str(weights_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device.upper())
    return model


# ── entry point ───────────────────────────────────────────────────────────────

def run_inference(args: argparse.Namespace) -> None:
    model  = load_model(args.weights)
    save   = not args.no_save
    source: int | str = int(args.source) if args.source.isdigit() else args.source

    logger.info("Source : %s", source)
    logger.info("Conf   : %.2f  |  IoU: %.2f  |  imgsz: %d",
                args.conf, args.iou, args.imgsz)
    logger.info("Backend: PyTorch")

    out_dir = Path(args.outdir)
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)

    t0      = time.perf_counter()
    results = model.predict(
        source  = source,
        conf    = args.conf,
        iou     = args.iou,
        imgsz   = args.imgsz,
        stream  = True,
        verbose = False,
    )

    frame_count, det_count = process_results(
        results     = results,
        out_dir     = out_dir,
        save        = save,
        show        = args.show,
        window_name = "Weapon Detection",
    )

    log_summary(frame_count, det_count, time.perf_counter() - t0,
                out_dir if save else None, save)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    run_inference(args)


if __name__ == "__main__":
    main()
