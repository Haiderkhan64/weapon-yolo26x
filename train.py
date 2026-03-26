"""
All hyperparameters match the exact values used to produce best.pt (mAP@50 = 0.8913).

Phase 1 — Stabilization      : 10 epochs @ 800px  | AdamW   | freeze=10
Phase 2 — Full backbone       : 15 epochs @ 800px  | AdamW | freeze=0
Phase 3 — High-res refinement : 10 epochs @ 1024px | AdamW | freeze=0
Phase 4 — Export              : TTA validation + TensorRT FP16

Usage:
    python train.py --data dataset.yaml --base-weights yolo26x.pt
    python train.py --data dataset.yaml --resume-from phase1/weights/best.pt --start-phase 2
    python train.py --export-only --weights model/best.pt --data dataset.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from ultralytics import YOLO

# ── logging setup ─────────────────────────────────────────────────────────────

def setup_logging(work_dir: Path, level: str = "INFO") -> logging.Logger:
    log_dir = work_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file  = log_dir / f"train_{timestamp}.log"

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler    = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    for h in (file_handler, console_handler):
        h.setFormatter(fmt)

    logger = logging.getLogger("weapon_yolo")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info("Logging initialised → %s", log_file)
    return logger


# ── config dataclasses ─────────────────────────────────────────────────────────

@dataclass
class PhaseConfig:
    phase:           int
    epochs:          int
    imgsz:           int
    batch:           int
    optimizer:       str
    lr0:             float
    lrf:             float
    weight_decay:    float
    warmup_epochs:   float
    warmup_bias_lr:  float
    warmup_momentum: float
    freeze:          int
    label_smoothing: float
    mosaic:          float
    mixup:           float
    copy_paste:      float
    close_mosaic:    int
    patience:        int
    degrees:         float   = 0.0
    scale:           float   = 0.5
    erasing:         float   = 0.0
    cos_lr:          bool    = True
    amp:             bool    = True


@dataclass
class DeployConfig:
    model_pt:       str
    model_engine:   str
    conf_threshold: float
    iou_threshold:  float
    imgsz:          int
    half:           bool
    device:         str
    map50_tta:      float
    map50_95_tta:   float


# ── phase hyperparameter presets ───────────────────────────────────────────────

PHASE_CONFIGS: dict[int, PhaseConfig] = {
    1: PhaseConfig(
        phase=1, epochs=10, imgsz=800, batch=32,
        optimizer="AdamW",   lr0=8e-5,  lrf=0.01,
        weight_decay=5e-4, warmup_epochs=2.0,
        warmup_bias_lr=0.01, warmup_momentum=0.8,
        freeze=10, label_smoothing=0.05,
        mosaic=1.0, mixup=0.0, copy_paste=0.0,
        close_mosaic=5, patience=10,
        degrees=10.0, scale=0.5, erasing=0.3,
    ),
    2: PhaseConfig(
        phase=2, epochs=15, imgsz=800, batch=32,
        optimizer="AdamW", lr0=5e-5,  lrf=0.01,
        weight_decay=5e-4, warmup_epochs=1.0,
        warmup_bias_lr=0.005, warmup_momentum=0.8,
        freeze=0, label_smoothing=0.05,
        mosaic=1.0, mixup=0.15, copy_paste=0.3,
        close_mosaic=5, patience=15,
        degrees=12.0, scale=0.6, erasing=0.4,
    ),
    3: PhaseConfig(
        phase=3, epochs=10, imgsz=1024, batch=12,
        optimizer="AdamW", lr0=2e-5,  lrf=0.005,
        weight_decay=5e-4, warmup_epochs=1.0,
        warmup_bias_lr=0.002, warmup_momentum=0.8,
        freeze=0, label_smoothing=0.05,
        mosaic=0.8, mixup=0.1, copy_paste=0.2,
        close_mosaic=3, patience=10,
        degrees=8.0, scale=0.5, erasing=0.2,
    ),
}


# ── helpers ────────────────────────────────────────────────────────────────────

def get_best_pt(phase_dir: Path, logger: logging.Logger) -> Path:
    """Return the best checkpoint from a phase directory."""
    canonical = phase_dir / "weights" / "best.pt"
    if canonical.exists():
        return canonical
    candidates = sorted(
        phase_dir.rglob("best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        logger.warning(
            "Canonical best.pt not found; falling back to most-recent: %s",
            candidates[0],
        )
        return candidates[0]
    raise FileNotFoundError(f"No best.pt found under {phase_dir}")


def phase_header(cfg: PhaseConfig, logger: logging.Logger) -> None:
    bar = "═" * 60
    logger.info(bar)
    logger.info(
        "  PHASE %d | %s | %d epochs | imgsz=%d | freeze=%d",
        cfg.phase, cfg.optimizer, cfg.epochs, cfg.imgsz, cfg.freeze,
    )
    logger.info(bar)


# ── core training function (DRY) ───────────────────────────────────────────────

def run_phase(
    cfg:      PhaseConfig,
    data:     str,
    weights:  str | Path,
    work_dir: Path,
    workers:  int,
    device:   str,
    logger:   logging.Logger,
) -> Path:
    """
    Execute a single training phase and return the path to best.pt.

    All phase-specific behaviour is encoded in *cfg*; this function never
    needs to be forked per phase.
    """
    phase_header(cfg, logger)
    torch.cuda.empty_cache()

    model = YOLO(str(weights))
    phase_name = f"phase{cfg.phase}"

    try:
        model.train(
            data             = data,
            epochs           = cfg.epochs,
            imgsz            = cfg.imgsz,
            batch            = cfg.batch,
            workers          = workers,
            device           = device,
            project          = str(work_dir),
            name             = phase_name,
            exist_ok         = True,
            optimizer        = cfg.optimizer,
            lr0              = cfg.lr0,
            lrf              = cfg.lrf,
            momentum         = 0.937,
            weight_decay     = cfg.weight_decay,
            warmup_epochs    = cfg.warmup_epochs,
            warmup_momentum  = cfg.warmup_momentum,
            warmup_bias_lr   = cfg.warmup_bias_lr,
            cos_lr           = cfg.cos_lr,
            label_smoothing  = cfg.label_smoothing,
            mosaic           = cfg.mosaic,
            mixup            = cfg.mixup,
            copy_paste       = cfg.copy_paste,
            degrees          = cfg.degrees,
            scale            = cfg.scale,
            erasing          = cfg.erasing,
            close_mosaic     = cfg.close_mosaic,
            amp              = cfg.amp,
            freeze           = cfg.freeze,
            patience         = cfg.patience,
            save_period      = 1,
            val              = True,
            plots            = True,
            verbose          = True,
        )
    except Exception:
        logger.exception("Phase %d training failed — checkpoints preserved in %s",
                         cfg.phase, work_dir / phase_name)
        raise

    best = get_best_pt(work_dir / phase_name, logger)
    logger.info("Phase %d complete. Best checkpoint: %s", cfg.phase, best)
    return best


# ── phase 4 — export ───────────────────────────────────────────────────────────

def phase4_export(
    best_pt:  Path,
    work_dir: Path,
    data:     str,
    device:   str,
    logger:   logging.Logger,
) -> None:
    bar = "═" * 60
    logger.info(bar)
    logger.info("  PHASE 4 — TTA Validation + TensorRT FP16 Export")
    logger.info(bar)

    export_dir = work_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    work_pt = export_dir / "best.pt"
    shutil.copy(str(best_pt), str(work_pt))
    logger.info("Copied checkpoint → %s", work_pt)

    model = YOLO(str(work_pt))

    # TTA validation
    logger.info("Running TTA validation …")
    try:
        metrics = model.val(
            data    = data,
            imgsz   = 1024,
            augment = True,
            conf    = 0.001,
            iou     = 0.6,
            device  = device,
            split   = "val",
            verbose = True,
        )
        map50    = float(metrics.box.map50)
        map50_95 = float(metrics.box.map)
    except Exception:
        logger.exception("TTA validation failed")
        raise

    logger.info("TTA mAP@50     : %.4f", map50)
    logger.info("TTA mAP@50-95  : %.4f", map50_95)

    # TensorRT FP16 export
    logger.info("Exporting TensorRT FP16 engine …")
    t0 = time.perf_counter()
    try:
        model.export(
            format    = "engine",
            imgsz     = 1024,
            half      = True,
            device    = 0,
            workspace = 6,
            verbose   = False,
        )
    except Exception:
        logger.exception("TensorRT export failed")
        raise

    elapsed     = time.perf_counter() - t0
    engine_path = work_pt.with_suffix(".engine")
    logger.info("Engine build time : %.0fs", elapsed)
    logger.info("Engine path       : %s  (%.1f MB)",
                engine_path, engine_path.stat().st_size / 1e6)

    # Write deploy config
    deploy = DeployConfig(
        model_pt       = str(work_pt),
        model_engine   = str(engine_path),
        conf_threshold = 0.35,
        iou_threshold  = 0.45,
        imgsz          = 1024,
        half           = True,
        device         = "cuda:0",
        map50_tta      = round(map50, 4),
        map50_95_tta   = round(map50_95, 4),
    )
    config_path = export_dir / "deploy_config.json"
    config_path.write_text(json.dumps(asdict(deploy), indent=2))
    logger.info("Deploy config written → %s", config_path)
    logger.info("Phase 4 complete.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Weapon YOLO26x — multi-phase training")
    p.add_argument("--data",          default="dataset.yaml",   help="Path to dataset YAML")
    p.add_argument("--base-weights",  default="yolo26x.pt",     help="Starting checkpoint (Phase 1)")
    p.add_argument("--resume-from",   default=None,             help="Checkpoint to resume from")
    p.add_argument("--start-phase",   type=int, default=1,      help="Start from phase N (1–4)")
    p.add_argument("--export-only",   action="store_true",      help="Skip training, export only")
    p.add_argument("--weights",       default=None,             help="Weights for --export-only")
    p.add_argument("--work-dir",      default="runs/weapon_yolo26x", help="Output directory")
    p.add_argument("--workers",       type=int,
                   default=min(8, (os.cpu_count() or 4) // 2),  help="DataLoader workers")
    p.add_argument("--log-level",     default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity")
    return p.parse_args()


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    args     = parse_args()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(work_dir, args.log_level)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device != "cuda":
        logger.warning("No GPU detected — training on CPU will be extremely slow.")

    if args.export_only:
        weights = args.weights or "model/best.pt"
        phase4_export(Path(weights), work_dir, args.data, device, logger)
        return

    # ── resolve starting checkpoint ───────────────────────────────────────────
    current_weights: Path | str = args.base_weights

    for phase_num in range(1, 4):
        if args.start_phase > phase_num:
            # Skip this phase — resolve its best checkpoint for the next phase
            resume = args.resume_from if args.start_phase == phase_num + 1 else None
            current_weights = Path(resume) if resume else get_best_pt(
                work_dir / f"phase{phase_num}", logger
            )
            logger.info("Skipping phase %d, using checkpoint: %s",
                        phase_num, current_weights)
            continue

        cfg             = PHASE_CONFIGS[phase_num]
        current_weights = run_phase(cfg, args.data, current_weights,
                                    work_dir, args.workers, device, logger)

    phase4_export(Path(current_weights), work_dir, args.data, device, logger)

    bar = "═" * 60
    logger.info(bar)
    logger.info("  PIPELINE COMPLETE")
    logger.info("  model  : %s", work_dir / "exports" / "best.pt")
    logger.info("  engine : %s", work_dir / "exports" / "best.engine")
    logger.info(bar)


if __name__ == "__main__":
    main()
