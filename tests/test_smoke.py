"""
Run with:
    pytest tests/ -v

These tests do NOT require a GPU or the full model weights.
They validate config loading, CLI argument parsing, and output
path collision logic — the parts most likely to break silently.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "inference"))
sys.path.insert(0, str(Path(__file__).parent.parent / "config"))


# ── deploy config validation ───────────────────────────────────────────────────

class TestDeployConfig:
    def test_valid_config_loads(self, tmp_path: Path) -> None:
        from validate import load_deploy_config

        cfg_path = tmp_path / "deploy_config.json"
        cfg_path.write_text(json.dumps({
            "model_pt":       "model/best.pt",
            "model_engine":   "model/best_fp16.engine",
            "conf_threshold": 0.35,
            "iou_threshold":  0.45,
            "imgsz":          1024,
            "half":           True,
            "device":         "cuda:0",
            "map50_tta":      0.8913,
            "map50_95_tta":   0.6836,
        }))
        cfg = load_deploy_config(cfg_path)
        assert cfg.conf_threshold == pytest.approx(0.35)
        assert cfg.iou_threshold  == pytest.approx(0.45)
        assert cfg.imgsz == 1024
        assert cfg.half is True

    def test_missing_required_key_raises(self, tmp_path: Path) -> None:
        from validate import load_deploy_config

        cfg_path = tmp_path / "bad.json"
        cfg_path.write_text(json.dumps({"model_pt": "x"}))  # missing many keys
        with pytest.raises(KeyError):
            load_deploy_config(cfg_path)

    def test_invalid_conf_threshold_raises(self, tmp_path: Path) -> None:
        from validate import load_deploy_config

        cfg_path = tmp_path / "bad.json"
        cfg_path.write_text(json.dumps({
            "model_pt": "x", "model_engine": "y",
            "conf_threshold": 1.5,   # invalid
            "iou_threshold": 0.45,
            "imgsz": 1024,
            "half": True,
            "device": "cpu",
        }))
        with pytest.raises(ValueError, match="conf_threshold"):
            load_deploy_config(cfg_path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        from validate import load_deploy_config

        with pytest.raises(FileNotFoundError):
            load_deploy_config(tmp_path / "nonexistent.json")


# ── output path collision avoidance ────────────────────────────────────────────

class TestResolveOutputPath:
    def test_no_collision(self, tmp_path: Path) -> None:
        from _common import resolve_output_path

        class FakeResult:
            path = "image.jpg"

        out = resolve_output_path(tmp_path, FakeResult(), 1)
        assert out == tmp_path / "image.jpg"

    def test_collision_increments_suffix(self, tmp_path: Path) -> None:
        from _common import resolve_output_path

        class FakeResult:
            path = "image.jpg"

        # Pre-create the file to simulate a collision
        (tmp_path / "image.jpg").write_bytes(b"")

        out = resolve_output_path(tmp_path, FakeResult(), 1)
        assert out.name == "image_0001.jpg"

    def test_fallback_for_missing_path(self, tmp_path: Path) -> None:
        from _common import resolve_output_path

        class FakeResult:
            path = None

        out = resolve_output_path(tmp_path, FakeResult(), 42)
        assert out.name == "frame_000042.jpg"


# ── train CLI argument parsing ─────────────────────────────────────────────────

class TestTrainArgParse:
    def test_defaults(self) -> None:
        import train

        args = train.parse_args.__wrapped__() if hasattr(train.parse_args, "__wrapped__") \
               else _parse_with_args(train.parse_args, [])
        assert args.start_phase == 1
        assert args.export_only is False

    def test_export_only_flag(self) -> None:
        import train

        args = _parse_with_args(train.parse_args, ["--export-only", "--weights", "x.pt"])
        assert args.export_only is True
        assert args.weights == "x.pt"


# ── phase config completeness ──────────────────────────────────────────────────

class TestPhaseConfigs:
    def test_all_three_phases_defined(self) -> None:
        from train import PHASE_CONFIGS

        assert set(PHASE_CONFIGS.keys()) == {1, 2, 3}

    def test_phase_lr_descends(self) -> None:
        from train import PHASE_CONFIGS

        assert PHASE_CONFIGS[1].lr0 > PHASE_CONFIGS[2].lr0 > PHASE_CONFIGS[3].lr0

    def test_phase3_high_resolution(self) -> None:
        from train import PHASE_CONFIGS

        assert PHASE_CONFIGS[3].imgsz == 1024
        assert PHASE_CONFIGS[1].imgsz == 800


# ── helpers ────────────────────────────────────────────────────────────────────

def _parse_with_args(parse_fn, argv: list[str]):
    """Call an argparse parse_args function with a fixed argv list."""
    import sys
    old = sys.argv
    sys.argv = ["prog"] + argv
    try:
        return parse_fn()
    finally:
        sys.argv = old
