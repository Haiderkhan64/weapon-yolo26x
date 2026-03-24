"""
app.py — Gradio Space for weapon-yolo26x (Video Edition)
Deploy at: https://huggingface.co/spaces/HaiderKhan6410/weapon-yolo26x-demo
"""

from __future__ import annotations

import os
import time
import tempfile
from pathlib import Path
from collections import defaultdict

import cv2
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

# ── class labels ──────────────────────────────────────────────────────────────

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

def detect_video(
    video_path: str,
    conf:       float,
    iou:        float,
    show_conf:  bool,
    frame_skip: int,
) -> tuple[str, str]:
    """
    Run inference on every Nth frame of a video file.

    Returns:
        path to annotated output video, markdown summary string
    """
    if video_path is None:
        return None, "⚠️ No video provided."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "❌ Could not open video file."

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output temp file
    out_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = out_file.name
    out_file.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    all_counts: dict[str, list[float]] = defaultdict(list)
    frame_idx   = 0
    processed   = 0
    total_time  = 0.0

    last_annotated = None  # reuse annotation on skipped frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % max(1, frame_skip) == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)

            t0      = time.perf_counter()
            results = model(pil, conf=conf, iou=iou, imgsz=IMGSZ, verbose=False)
            total_time += time.perf_counter() - t0

            r     = results[0]
            boxes = r.boxes
            annotated_rgb = r.plot()
            last_annotated = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

            if boxes is not None:
                names = r.names
                for box in boxes:
                    cls_id   = int(box.cls[0])
                    conf_val = float(box.conf[0])
                    label    = names.get(cls_id, f"class_{cls_id}")
                    all_counts[label].append(conf_val)

            processed += 1

        writer.write(last_annotated if last_annotated is not None else frame)
        frame_idx += 1

    cap.release()
    writer.release()

    # Re-encode with H.264 for browser compatibility (if ffmpeg available)
    h264_path = out_path.replace(".mp4", "_h264.mp4")
    ret_code = os.system(
        f'ffmpeg -y -i "{out_path}" -vcodec libx264 -acodec aac "{h264_path}" -loglevel quiet'
    )
    final_path = h264_path if ret_code == 0 and Path(h264_path).exists() else out_path

    # Build summary
    avg_ms = (total_time / processed * 1000) if processed > 0 else 0.0
    lines  = [
        f"**Video stats:** {total} frames · {processed} processed "
        f"(every {frame_skip} frame{'s' if frame_skip > 1 else ''}) · "
        f"avg inference `{avg_ms:.1f} ms` on {DEVICE.upper()}\n"
    ]

    if not all_counts:
        lines.append("**No weapons detected** across the entire video.")
    else:
        lines.append("**Aggregated detections (all frames):**\n")
        for label, confs in sorted(all_counts.items()):
            icon    = CLASS_LABELS.get(label, "•")
            count   = len(confs)
            avg_c   = sum(confs) / count
            conf_str = f"  avg conf: `{avg_c:.2f}`" if show_conf else ""
            lines.append(f"- {icon} **{label}** × {count} detections{conf_str}")

    summary = "\n".join(lines)
    return final_path, summary


# ── examples ──────────────────────────────────────────────────────────────────

EXAMPLES_DIR = Path("examples")
examples = [
    [str(p), CONF_DEFAULT, IOU_DEFAULT, True, 2]
    for p in sorted(EXAMPLES_DIR.glob("*.mp4"))
] if EXAMPLES_DIR.exists() else []


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(theme=gr.themes.Soft(), title="Weapon Detection YOLO26x — Video") as demo:
    gr.Markdown(
        """
# 🔫 Weapon Detection — YOLO26x · Video Mode
**Multi-phase trained** on 104,697 images · mAP@50 = **0.8913** · 7 classes

> Upload a video to detect weapons frame-by-frame. Adjust thresholds and frame-skip below.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            inp_video = gr.Video(label="Input video", sources=["upload"])

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
                inp_frame_skip = gr.Slider(
                    minimum=1, maximum=10, value=2, step=1,
                    label="Frame skip (process every Nth frame)",
                    info="Higher → faster processing, lower temporal resolution",
                )
                inp_show_conf = gr.Checkbox(
                    value=True, label="Show confidence scores in summary"
                )

            btn_detect = gr.Button("🔍 Detect", variant="primary")

        with gr.Column(scale=1):
            out_video   = gr.Video(label="Detection result", autoplay=True)
            out_summary = gr.Markdown()

    btn_detect.click(
        fn      = detect_video,
        inputs  = [inp_video, inp_conf, inp_iou, inp_show_conf, inp_frame_skip],
        outputs = [out_video, out_summary],
    )

    if examples:
        gr.Examples(
            examples       = examples,
            inputs         = [inp_video, inp_conf, inp_iou, inp_show_conf, inp_frame_skip],
            outputs        = [out_video, out_summary],
            fn             = detect_video,
            cache_examples = True,
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
