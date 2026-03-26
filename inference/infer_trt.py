"""
Requires: NVIDIA GPU, TensorRT 10.x, CUDA 12.x

IMPORTANT: The included .engine file was compiled on H100.
Re-export for other GPUs:

    python -c "
    from ultralytics import YOLO
    model = YOLO('model/best.pt')
    model.export(format='engine', imgsz=1024, half=True, device=0, workspace=6)
    "

Usage:
    python infer_trt.py --source image.jpg
    python infer_trt.py --source video.mp4
    python infer_trt.py --source image.jpg --engine path/to/best_fp16.engine
    python infer_trt.py --source video.mp4 --no-save --show
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
from _common import log_summary, process_results, setup_logging
from ultralytics import YOLO

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_ENGINE = Path(__file__).parent.parent / "model" / "best_fp16.engine"
DEFAULT_CONF   = 0.35
DEFAULT_IOU    = 0.45
DEFAULT_IMGSZ  = 1024
DEFAULT_OUTDIR = Path("runs/detect_trt")

logger = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Weapon YOLO26x — TensorRT FP16 inference")
    p.add_argument("--source",   required=True,
                   help="Image / video / folder")
    p.add_argument("--engine",   default=str(DEFAULT_ENGINE),
                   help="Path to .engine file  (default: model/best_fp16.engine)")
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


# ── GPU validation ────────────────────────────────────────────────────────────

def validate_gpu() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "TensorRT inference requires a CUDA-capable GPU.\n"
            "Use infer.py (PyTorch) for CPU inference."
        )
    logger.info("GPU: %s", torch.cuda.get_device_name(0))


# ── engine loading ────────────────────────────────────────────────────────────

def load_engine(engine_path: str, fallback_path: str = "model/best.pt", auto_export: bool = False) -> YOLO:
    engine_file = Path(engine_path)
    pt_file = Path(fallback_path)

    # 1. Try to load the Engine if it exists
    if engine_file.exists():
        try:
            logger.info("Loading TensorRT engine: %s", engine_path)
            return YOLO(str(engine_file), task="detect")
        except Exception as e: # Catching Exception, NOT a bare except
            logger.warning("TensorRT engine load failed (incompatible version or corrupt file): %s", e)
    else:
        logger.warning("Engine not found at %s.", engine_path)

    # 2. Check if Fallback exists before proceeding
    if not pt_file.exists():
        raise FileNotFoundError(f"Critical Error: Both {engine_path} and {fallback_path} are missing.")

    # 3. Handle Auto-Export or Graceful Fallback
    model = YOLO(str(pt_file))

    if auto_export:
        logger.info("Auto-export enabled. Exporting PyTorch model to TensorRT (this may take a few minutes)...")
        try:
            # Note: dynamic=True handles varying image sizes, half=True uses FP16 for speed
            model.export(format="engine", dynamic=True, half=True)
            logger.info("Export complete! Returning the optimized engine.")
            return YOLO(str(engine_file), task="detect")
        except Exception as e:
            logger.error("Auto-export failed: %s. Continuing with PyTorch model.", e)
            return model
    else:
        logger.info("Falling back to PyTorch model: %s", fallback_path)
        return model


# ── entry point ───────────────────────────────────────────────────────────────

def run_inference(args: argparse.Namespace) -> None:
    validate_gpu()
    model = load_engine(args.engine)
    save  = not args.no_save

    logger.info("Source : %s", args.source)
    logger.info("Conf   : %.2f  |  IoU: %.2f  |  imgsz: %d",
                args.conf, args.iou, args.imgsz)
    logger.info("Backend: TensorRT FP16")

    out_dir = Path(args.outdir)
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)

    t0      = time.perf_counter()
    results = model.predict(
        source  = args.source,
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
        window_name = "Weapon Detection (TRT)",
    )

    log_summary(frame_count, det_count, time.perf_counter() - t0,
                out_dir if save else None, save)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    run_inference(args)


if __name__ == "__main__":
    main()
