"""
app.py — Gradio Space for weapon-yolo26x
Deploy at: https://huggingface.co/spaces/HaiderKhan6410/weapon-yolo26x-demo
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO

# ── model loading ─────────────────────────────────────────────────────────────

MODEL_REPO = "HaiderKhan6410/weapon-yolo26x"
MODEL_FILE = "model/best.pt"

CONF_DEFAULT = 0.35
IOU_DEFAULT  = 0.45
IMGSZ        = 1024

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model on {DEVICE.upper()} …")
_model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
model = YOLO(_model_path)
model.to(DEVICE)
print("Model ready.")

# ── class colours (matches YOLO26x class order) ───────────────────────────────

CLASS_LABELS = {
    "Blunt_Weapon":  "🪓",
    "Explosive":     "💣",
    "Fire_Smoke":    "🔥",
    "Firearm":       "🔫",
    "Melee_Weapon":  "🗡️",
    "Person":        "🧍",
    "Tool":          "🔧",
}

# ── inference ─────────────────────────────────────────────────────────────────

def detect(
    image:     Image.Image,
    conf:      float,
    iou:       float,
    show_conf: bool,
) -> tuple[Image.Image, str]:
    """
    Run inference on a PIL image.

    Returns:
        annotated image, markdown summary string
    """
    if image is None:
        return None, "⚠️ No image provided."

    t0      = time.perf_counter()
    results = model(image, conf=conf, iou=iou, imgsz=IMGSZ, verbose=False)
    elapsed = time.perf_counter() - t0

    r     = results[0]
    boxes = r.boxes

    # Annotated frame
    annotated = Image.fromarray(r.plot())

    # Build detection summary
    if boxes is None or len(boxes) == 0:
        summary = "**No weapons detected.**"
    else:
        counts: dict[str, list[float]] = {}
        names  = r.names  # {int: str}

        for box in boxes:
            cls_id  = int(box.cls[0])
            conf_val = float(box.conf[0])
            label   = names.get(cls_id, f"class_{cls_id}")
            counts.setdefault(label, []).append(conf_val)

        lines = ["**Detections:**\n"]
        for label, confs in sorted(counts.items()):
            icon  = CLASS_LABELS.get(label, "•")
            count = len(confs)
            avg_c = sum(confs) / count
            conf_str = f"  avg conf: `{avg_c:.2f}`" if show_conf else ""
            lines.append(f"- {icon} **{label}** × {count}{conf_str}")

        lines.append(f"\n_Inference: `{elapsed * 1000:.1f} ms` on {DEVICE.upper()}_")
        summary = "\n".join(lines)

    return annotated, summary


# ── examples ──────────────────────────────────────────────────────────────────

EXAMPLES_DIR = Path("examples")
examples = [
    [str(p), CONF_DEFAULT, IOU_DEFAULT, True]
    for p in sorted(EXAMPLES_DIR.glob("*.jpg"))
] if EXAMPLES_DIR.exists() else []


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(theme=gr.themes.Soft(), title="Weapon Detection YOLO26x") as demo:
    gr.Markdown(
        """
# 🔫 Weapon Detection — YOLO26x
**Multi-phase trained** on 104,697 images · mAP@50 = **0.8913** · 7 classes

> Upload an image to detect weapons. Adjust confidence and IoU thresholds below.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            inp_image = gr.Image(type="pil", label="Input image")

            with gr.Accordion("⚙️ Detection settings", open=False):
                inp_conf = gr.Slider(
                    minimum=0.10, maximum=0.90, value=CONF_DEFAULT, step=0.05,
                    label="Confidence threshold",
                    info="Lower → more detections (more false positives)",
                )
                inp_iou = gr.Slider(
                    minimum=0.10, maximum=0.90, value=IOU_DEFAULT, step=0.05,
                    label="IoU (NMS) threshold",
                    info="Lower → fewer overlapping boxes",
                )
                inp_show_conf = gr.Checkbox(
                    value=True, label="Show confidence scores in summary"
                )

            btn_detect = gr.Button("🔍 Detect", variant="primary")

        with gr.Column(scale=1):
            out_image   = gr.Image(label="Detection result", type="pil")
            out_summary = gr.Markdown()

    btn_detect.click(
        fn      = detect,
        inputs  = [inp_image, inp_conf, inp_iou, inp_show_conf],
        outputs = [out_image, out_summary],
    )

    # Auto-run on image upload as well
    inp_image.change(
        fn      = detect,
        inputs  = [inp_image, inp_conf, inp_iou, inp_show_conf],
        outputs = [out_image, out_summary],
    )

    if examples:
        gr.Examples(
            examples        = examples,
            inputs          = [inp_image, inp_conf, inp_iou, inp_show_conf],
            outputs         = [out_image, out_summary],
            fn              = detect,
            cache_examples  = True,
        )

    gr.Markdown(
        """
---
**Classes:** Blunt_Weapon · Explosive · Fire_Smoke · Firearm · Melee_Weapon · Person · Tool

**Model:** [`HaiderKhan6410/weapon-yolo26x`](https://huggingface.co/HaiderKhan6410/weapon-yolo26x) · License: CC0
        """
    )

# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch()
