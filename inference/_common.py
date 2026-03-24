"""
inference/_common.py — Shared utilities for PyTorch and TensorRT inference.

Not a public entry point. Import from infer.py or infer_trt.py.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterator

import cv2

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def resolve_output_path(out_dir: Path, r: object, frame_count: int) -> Path:
    """
    Derive a collision-safe output filename for a result frame.

    Falls back to a zero-padded frame index for webcam / streaming sources.
    Appends a numeric suffix when the target file already exists.
    """
    raw_path = getattr(r, "path", None)
    stem_name = Path(raw_path).name if raw_path else f"frame_{frame_count:06d}.jpg"
    out_path  = out_dir / stem_name

    # Avoid silently overwriting previous runs in the same outdir
    if out_path.exists():
        base   = out_path.stem
        suffix = out_path.suffix
        idx    = 1
        while out_path.exists():
            out_path = out_dir / f"{base}_{idx:04d}{suffix}"
            idx += 1

    return out_path


def process_results(
    results:     Iterator,
    out_dir:     Path,
    save:        bool,
    show:        bool,
    window_name: str,
) -> tuple[int, int]:
    """
    Iterate over a YOLO results stream, optionally saving / displaying frames.

    Returns:
        (frame_count, detection_count)
    """
    frame_count = 0
    det_count   = 0

    for r in results:
        frame_count += 1
        boxes = r.boxes
        det_count += len(boxes) if boxes is not None else 0

        if save or show:
            annotated = r.plot()

            if save:
                out_path = resolve_output_path(out_dir, r, frame_count)
                cv2.imwrite(str(out_path), annotated)

            if show:
                cv2.imshow(window_name, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Display window closed by user.")
                    break

    if show:
        cv2.destroyAllWindows()

    return frame_count, det_count


def log_summary(
    frame_count: int,
    det_count:   int,
    elapsed:     float,
    out_dir:     Path | None,
    save:        bool,
) -> None:
    fps = frame_count / elapsed if elapsed > 0 else 0.0
    logger.info("Frames    : %d", frame_count)
    logger.info("Detections: %d", det_count)
    logger.info("Time      : %.2fs  (%.1f FPS)", elapsed, fps)
    if save and out_dir:
        logger.info("Output    : %s", out_dir.resolve())
