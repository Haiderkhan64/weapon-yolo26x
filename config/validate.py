from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = {
    "model_pt", "model_engine", "conf_threshold",
    "iou_threshold", "imgsz", "half", "device",
}


@dataclass(frozen=True)
class DeployConfig:
    model_pt:       str
    model_engine:   str
    conf_threshold: float
    iou_threshold:  float
    imgsz:          int
    half:           bool
    device:         str
    map50_tta:      float = 0.0
    map50_95_tta:   float = 0.0
    notes:          str   = ""

    def __post_init__(self) -> None:
        if not (0.0 < self.conf_threshold < 1.0):
            raise ValueError(f"conf_threshold must be in (0, 1), got {self.conf_threshold}")
        if not (0.0 < self.iou_threshold < 1.0):
            raise ValueError(f"iou_threshold must be in (0, 1), got {self.iou_threshold}")
        if self.imgsz <= 0:
            raise ValueError(f"imgsz must be positive, got {self.imgsz}")


def load_deploy_config(path: str | Path) -> DeployConfig:
    """
    Parse and validate deploy_config.json.

    Raises:
        FileNotFoundError: if the config file is missing.
        KeyError: if any required key is absent.
        ValueError: if any value is out of acceptable range.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Deploy config not found: {config_path}")

    raw = json.loads(config_path.read_text())

    missing = _REQUIRED_KEYS - raw.keys()
    if missing:
        raise KeyError(f"deploy_config.json is missing required keys: {missing}")

    cfg = DeployConfig(
        model_pt       = str(raw["model_pt"]),
        model_engine   = str(raw["model_engine"]),
        conf_threshold = float(raw["conf_threshold"]),
        iou_threshold  = float(raw["iou_threshold"]),
        imgsz          = int(raw["imgsz"]),
        half           = bool(raw["half"]),
        device         = str(raw["device"]),
        map50_tta      = float(raw.get("map50_tta", 0.0)),
        map50_95_tta   = float(raw.get("map50_95_tta", 0.0)),
        notes          = str(raw.get("notes", "")),
    )
    logger.info("Deploy config loaded: conf=%.2f  iou=%.2f  imgsz=%d  device=%s",
                cfg.conf_threshold, cfg.iou_threshold, cfg.imgsz, cfg.device)
    return cfg
