from __future__ import annotations

import os
os.environ['YOLO_CONFIG_DIR'] = '/tmp/Ultralytics'

import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

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
# Keys MUST match exact model output names — do not change these.

CLASS_META = {
    "Blunt_Weapon":  {"icon": "🪓", "color": "#f59e0b", "risk": "MEDIUM"},
    "Explosive":     {"icon": "💣", "color": "#ef4444", "risk": "CRITICAL"},
    "Fire_Smoke":    {"icon": "🔥", "color": "#f97316", "risk": "HIGH"},
    "Firearm":       {"icon": "🔫", "color": "#ef4444", "risk": "CRITICAL"},
    "Melee_Weapon":  {"icon": "🗡️",  "color": "#f59e0b", "risk": "MEDIUM"},
    "Person":        {"icon": "🧍", "color": "#64748b", "risk": "INFO"},
    "Tool":          {"icon": "🔧", "color": "#22d3ee", "risk": "LOW"},
}

# UI display names — separate from internal model labels (presentation layer only)
DISPLAY_NAMES = {
    "Blunt_Weapon": "Blunt Weapon",
    "Fire_Smoke":   "Fire / Smoke",
    "Melee_Weapon": "Melee Weapon",
}

RISK_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}

# ── inference ─────────────────────────────────────────────────────────────────

def detect_video(
    video_path: str,
    conf:       float,
    iou:        float,
    show_conf:  bool,
    frame_skip: int,
    progress=gr.Progress(track_tqdm=True),
) -> tuple[str, str]:
    if video_path is None:
        return None, _error_card("No video provided. Please upload a video file first.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, _error_card("Could not open video file. Please check the format.")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur_s  = total / fps

    out_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = out_file.name
    out_file.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    all_counts: dict[str, list[float]] = defaultdict(list)
    frame_idx   = 0
    processed   = 0
    total_time  = 0.0
    last_annotated = None

    # FIX: ceiling division so the final partial block is counted correctly
    frames_to_process = max(1, (total + frame_skip - 1) // max(1, frame_skip))

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
            progress(processed / frames_to_process, desc=f"Processing frame {frame_idx}/{total}")

        writer.write(last_annotated if last_annotated is not None else frame)
        frame_idx += 1

    cap.release()
    writer.release()

    # FIX: guard against zero processed frames (corrupted file / extreme frame_skip)
    if processed == 0:
        return None, _error_card(
            "No frames were processed. Try lowering Frame Skip or check the video file."
        )

    # Re-encode with H.264 for browser compatibility — FIX: use subprocess, not os.system
    h264_path = out_path.replace(".mp4", "_h264.mp4")
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", out_path,
            "-vcodec", "libx264", "-acodec", "aac",
            h264_path, "-loglevel", "quiet",
        ],
        check=False,
    )
    final_path = (
        h264_path
        if result.returncode == 0 and Path(h264_path).exists()
        else out_path
    )

    avg_ms = total_time / processed * 1000
    summary_html = _build_summary_html(
        total, processed, frame_skip, avg_ms, dur_s,
        width, height, fps, all_counts, show_conf
    )

    return final_path, summary_html


# ── HTML builders ─────────────────────────────────────────────────────────────

def _error_card(msg: str) -> str:
    return f"""
<div class="result-card error-card">
  <span class="error-icon">⚠</span>
  <span class="error-msg">{msg}</span>
</div>"""


def _build_summary_html(
    total, processed, frame_skip, avg_ms,
    dur_s, width, height, fps,
    all_counts, show_conf
) -> str:

    # Threat level
    has_critical = any(
        CLASS_META.get(k, {}).get("risk") in ("CRITICAL", "HIGH")
        for k in all_counts
    )
    threat_level = "🔴 HIGH THREAT DETECTED" if has_critical else (
        "🟡 MEDIUM THREAT" if all_counts else "🟢 CLEAR — No Threats Found"
    )
    threat_cls = "threat-high" if has_critical else ("threat-med" if all_counts else "threat-clear")

    # Stats row
    stats_html = f"""
<div class="stats-grid">
  <div class="stat-card">
    <div class="stat-label">DURATION</div>
    <div class="stat-value">{dur_s:.1f}s</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">RESOLUTION</div>
    <div class="stat-value">{width}×{height}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">FPS</div>
    <div class="stat-value">{fps:.0f}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">FRAMES</div>
    <div class="stat-value">{processed}<span class="stat-sub">/{total}</span></div>
  </div>
  <div class="stat-card">
    <div class="stat-label">AVG INFER</div>
    <div class="stat-value">{avg_ms:.0f}<span class="stat-sub">ms</span></div>
  </div>
  <div class="stat-card">
    <div class="stat-label">DEVICE</div>
    <div class="stat-value">{DEVICE.upper()}</div>
  </div>
</div>"""

    # Detection rows
    if not all_counts:
        detections_html = '<div class="no-detect">No objects detected across all processed frames.</div>'
    else:
        sorted_classes = sorted(
            all_counts.items(),
            key=lambda x: RISK_ORDER.get(CLASS_META.get(x[0], {}).get("risk", "INFO"), 99)
        )
        rows = ""
        for label, confs in sorted_classes:
            meta    = CLASS_META.get(label, {"icon": "•", "color": "#64748b", "risk": "INFO"})
            icon    = meta["icon"]
            color   = meta["color"]
            risk    = meta["risk"]
            count   = len(confs)
            avg_c   = sum(confs) / count
            max_c   = max(confs)
            bar_w   = int(avg_c * 100)

            # FIX: map internal model label → clean UI display name (presentation layer only)
            display_label = DISPLAY_NAMES.get(label, label.replace("_", " "))

            conf_block = f"""
  <div class="conf-bar-wrap" title="avg {avg_c:.2f} | max {max_c:.2f}">
    <div class="conf-bar" style="width:{bar_w}%; background:{color};"></div>
    <span class="conf-label">{avg_c:.2f}</span>
  </div>""" if show_conf else ""

            rows += f"""
<div class="detect-row">
  <div class="detect-icon">{icon}</div>
  <div class="detect-info">
    <div class="detect-label">{display_label}</div>
    <span class="risk-badge risk-{risk.lower()}">{risk}</span>
  </div>
  <div class="detect-count">×{count}</div>
  {conf_block}
</div>"""

        detections_html = f'<div class="detections-list">{rows}</div>'

    return f"""
<div class="summary-root">
  <div class="threat-banner {threat_cls}">{threat_level}</div>
  {stats_html}
  <div class="section-title">DETECTIONS</div>
  {detections_html}
</div>"""


# ── CSS ───────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* ── Imports ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Barlow+Condensed:wght@300;500;700;900&display=swap');

/* ── Root / Theme ── */
:root {
  --bg-base:    #09090b;
  --bg-surface: #111114;
  --bg-card:    #18181c;
  --bg-hover:   #222228;
  --border:     #2a2a32;
  --accent:     #ef4444;
  --accent2:    #f59e0b;
  --accent3:    #22d3ee;
  --text-pri:   #f4f4f5;
  --text-sec:   #71717a;
  --text-muted: #3f3f46;
  --mono:       'Space Mono', monospace;
  --display:    'Barlow Condensed', sans-serif;
}

/* ── Global overrides ── */
.gradio-container {
  background: var(--bg-base) !important;
  font-family: var(--display) !important;
  letter-spacing: 0.01em;
}
.dark { background: var(--bg-base) !important; }

footer { display: none !important; }

/* ── Header ── */
.header-wrap {
  padding: 32px 0 20px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 24px;
  position: relative;
}
.header-eyebrow {
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 0.2em;
  color: var(--accent);
  text-transform: uppercase;
  margin-bottom: 8px;
}
.header-title {
  font-family: var(--display);
  font-size: 42px;
  font-weight: 900;
  color: var(--text-pri);
  letter-spacing: -0.01em;
  line-height: 1;
  margin-bottom: 8px;
}
.header-title span { color: var(--accent); }
.header-sub {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--text-sec);
  letter-spacing: 0.1em;
}
.header-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
}
.header-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(239,68,68,0.1);
  border: 1px solid rgba(239,68,68,0.3);
  color: var(--accent);
  font-family: var(--mono);
  font-size: 10px;
  padding: 4px 10px;
  border-radius: 2px;
}
.header-badge-warn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(245,158,11,0.08);
  border: 1px solid rgba(245,158,11,0.3);
  color: #f59e0b;
  font-family: var(--mono);
  font-size: 10px;
  padding: 4px 10px;
  border-radius: 2px;
}
.status-dot {
  width: 6px; height: 6px;
  background: var(--accent);
  border-radius: 50%;
  animation: pulse-dot 1.5s ease-in-out infinite;
}
@keyframes pulse-dot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50%       { opacity: 0.4; transform: scale(0.7); }
}

/* ── Panel cards ── */
.panel-card {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
}

/* ── Section label ── */
.section-label {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 0.25em;
  color: var(--text-muted);
  text-transform: uppercase;
  margin-bottom: 10px;
  padding-bottom: 6px;
  border-bottom: 1px solid var(--border);
}

/* ── Detect button ── */
.detect-btn button {
  background: var(--accent) !important;
  color: #fff !important;
  font-family: var(--mono) !important;
  font-size: 12px !important;
  letter-spacing: 0.15em !important;
  text-transform: uppercase !important;
  border: none !important;
  border-radius: 2px !important;
  padding: 14px 28px !important;
  transition: background 0.15s ease, transform 0.1s ease !important;
}
.detect-btn button:hover {
  background: #dc2626 !important;
  transform: translateY(-1px) !important;
}
.detect-btn button:active { transform: translateY(0) !important; }

/* ── Settings accordion ── */
.settings-acc > .label-wrap {
  font-family: var(--mono) !important;
  font-size: 10px !important;
  letter-spacing: 0.1em !important;
  color: var(--text-sec) !important;
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 2px !important;
}

/* ── Sliders ── */
input[type=range]::-webkit-slider-thumb { background: var(--accent) !important; }
input[type=range]::-webkit-slider-runnable-track { background: var(--border) !important; }

/* ── Summary root ── */
.summary-root {
  font-family: var(--display);
  color: var(--text-pri);
  padding: 4px 0;
}

/* ── Threat banner ── */
.threat-banner {
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.2em;
  padding: 10px 16px;
  margin-bottom: 16px;
  border-radius: 2px;
  border-left: 3px solid currentColor;
}
.threat-high   { background: rgba(239,68,68,0.1);  color: #ef4444; border-color: #ef4444; }
.threat-med    { background: rgba(245,158,11,0.1); color: #f59e0b; border-color: #f59e0b; }
.threat-clear  { background: rgba(34,197,94,0.1);  color: #22c55e; border-color: #22c55e; }

/* ── Stats grid ── */
.stats-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
  margin-bottom: 20px;
}
.stat-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 2px;
  padding: 10px 12px;
}
.stat-label {
  font-family: var(--mono);
  font-size: 8px;
  letter-spacing: 0.2em;
  color: var(--text-muted);
  margin-bottom: 4px;
}
.stat-value {
  font-family: var(--mono);
  font-size: 18px;
  font-weight: 700;
  color: var(--text-pri);
  line-height: 1;
}
.stat-sub { font-size: 11px; color: var(--text-sec); margin-left: 2px; }

/* ── Section title ── */
.section-title {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 0.25em;
  color: var(--text-muted);
  text-transform: uppercase;
  margin-bottom: 10px;
  padding-bottom: 6px;
  border-bottom: 1px solid var(--border);
}

/* ── Detection rows ── */
.detections-list { display: flex; flex-direction: column; gap: 6px; }
.detect-row {
  display: flex;
  align-items: center;
  gap: 10px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 2px;
  padding: 10px 14px;
  transition: background 0.15s;
}
.detect-row:hover { background: var(--bg-hover); }
.detect-icon { font-size: 18px; flex-shrink: 0; }
.detect-info { flex: 1; }
.detect-label { font-size: 15px; font-weight: 700; letter-spacing: 0.03em; }
.detect-count {
  font-family: var(--mono);
  font-size: 13px;
  color: var(--text-sec);
  flex-shrink: 0;
}

/* ── Risk badges ── */
.risk-badge {
  font-family: var(--mono);
  font-size: 8px;
  letter-spacing: 0.15em;
  padding: 2px 6px;
  border-radius: 1px;
  display: inline-block;
  margin-top: 2px;
}
.risk-critical { background: rgba(239,68,68,0.15); color: #ef4444; }
.risk-high     { background: rgba(249,115,22,0.15); color: #f97316; }
.risk-medium   { background: rgba(245,158,11,0.15); color: #f59e0b; }
.risk-low      { background: rgba(34,211,238,0.15); color: #22d3ee; }
.risk-info     { background: rgba(100,116,139,0.15); color: #94a3b8; }

/* ── Confidence bar ── */
.conf-bar-wrap {
  position: relative;
  width: 80px;
  height: 16px;
  background: var(--bg-base);
  border: 1px solid var(--border);
  border-radius: 1px;
  flex-shrink: 0;
  overflow: hidden;
}
.conf-bar {
  height: 100%;
  border-radius: 1px;
  opacity: 0.7;
  transition: width 0.4s ease;
}
.conf-label {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: var(--mono);
  font-size: 9px;
  color: #fff;
  font-weight: 700;
  text-shadow: 0 0 4px rgba(0,0,0,0.8);
}

/* ── No detections ── */
.no-detect {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--text-muted);
  text-align: center;
  padding: 24px;
  border: 1px dashed var(--border);
  border-radius: 2px;
}

/* ── Error card ── */
.error-card {
  display: flex;
  align-items: center;
  gap: 12px;
  background: rgba(239,68,68,0.08);
  border: 1px solid rgba(239,68,68,0.3);
  border-radius: 2px;
  padding: 14px 16px;
  font-family: var(--mono);
  font-size: 11px;
  color: #ef4444;
}
.error-icon { font-size: 18px; }

/* ── Footer ── */
.footer-strip {
  margin-top: 32px;
  padding-top: 16px;
  border-top: 1px solid var(--border);
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 16px;
  justify-content: space-between;
}
.footer-left {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--text-muted);
  letter-spacing: 0.1em;
}
.footer-left a {
  color: var(--text-sec);
  text-decoration: none;
}
.footer-left a:hover { color: var(--accent); }
.footer-right {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}
.footer-link {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--text-sec);
  text-decoration: none;
  letter-spacing: 0.08em;
  padding: 4px 10px;
  border: 1px solid var(--border);
  border-radius: 2px;
  transition: color 0.15s, border-color 0.15s;
}
.footer-link:hover { color: var(--accent); border-color: rgba(239,68,68,0.4); }

/* ── Classes strip ── */
.class-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 16px;
}
.class-pill {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 0.1em;
  color: var(--text-sec);
  background: var(--bg-card);
  border: 1px solid var(--border);
  padding: 4px 10px;
  border-radius: 2px;
}

/* ── Gradio video component ── */
video { border-radius: 2px !important; }
"""

# ── Examples ──────────────────────────────────────────────────────────────────

EXAMPLES_DIR = Path("examples")
examples = [
    [str(p), CONF_DEFAULT, IOU_DEFAULT, True, 2]
    for p in sorted(EXAMPLES_DIR.glob("*.mp4"))
] if EXAMPLES_DIR.exists() else []

# ── UI ────────────────────────────────────────────────────────────────────────

HEADER_HTML = f"""
<div class="header-wrap">
  <div class="header-title">YOLO26x <span>Threat Detection</span></div>
  <div class="header-sub">YOLO26X · mAP@50 = 0.8913 · 104,697 TRAINING IMAGES · 7 CLASSES</div>
  <div class="header-badges">
    <div class="header-badge-warn">
      ⚠ Not a safety-critical system — human review required
    </div>
  </div>
</div>
"""

# FIX: correct OpenRAIL-M license link (was pointing to wrong HF page)
FOOTER_HTML = """
<div class="footer-strip">
  <div class="footer-left">
    © HaiderKhan6410 · WEAPON-YOLO26X ·
    <a href="https://huggingface.co/spaces/bigscience/openrail" target="_blank" rel="noopener">
      BigScience OpenRAIL-M
    </a> · Use subject to OpenRAIL-M restrictions
  </div>
  <div class="footer-right">
    <a class="footer-link" href="https://huggingface.co/HaiderKhan6410/weapon-yolo26x" target="_blank" rel="noopener">
      ↗ MODEL ON HF
    </a>
    <a class="footer-link" href="https://github.com/ultralytics/ultralytics" target="_blank" rel="noopener">
      ↗ ULTRALYTICS
    </a>
    <a class="footer-link" href="https://huggingface.co/spaces/HaiderKhan6410/weapon-yolo26x-demo" target="_blank" rel="noopener">
      ↗ THIS SPACE
    </a>
  </div>
</div>
<div class="class-strip">
  <span class="class-pill">🪓 BLUNT WEAPON</span>
  <span class="class-pill">💣 EXPLOSIVE</span>
  <span class="class-pill">🔥 FIRE / SMOKE</span>
  <span class="class-pill">🔫 FIREARM</span>
  <span class="class-pill">🗡️ MELEE WEAPON</span>
  <span class="class-pill">🧍 PERSON</span>
  <span class="class-pill">🔧 TOOL</span>
</div>
"""

with gr.Blocks(title="Weapon Detection YOLO26x") as demo:

    gr.HTML(HEADER_HTML)

    with gr.Row(equal_height=False):

        # ── Left column: inputs ──────────────────────────────────────────────
        with gr.Column(scale=5, elem_classes="panel-card"):
            gr.HTML('<div class="section-label">INPUT VIDEO</div>')
            inp_video = gr.Video(
                label="",
                sources=["upload"],
                show_label=False,
            )

            with gr.Accordion(
                "⚙  DETECTION PARAMETERS",
                open=False,
                elem_classes="settings-acc",
            ):
                inp_conf = gr.Slider(
                    minimum=0.10, maximum=0.90,
                    value=CONF_DEFAULT, step=0.05,
                    label="Confidence Threshold (0–1)",
                    # FIX: show both directions so users understand the trade-off
                    info="Lower → more detections (↑ false positives) | Higher → fewer detections (↑ missed detections)",
                )
                inp_iou = gr.Slider(
                    minimum=0.10, maximum=0.90,
                    value=IOU_DEFAULT, step=0.05,
                    label="IoU / NMS Threshold (0–1)",
                    # FIX: clearer wording for both directions
                    info="Lower → more aggressive suppression (fewer overlapping boxes) | Higher → keeps more overlapping detections",
                )
                inp_frame_skip = gr.Slider(
                    minimum=1, maximum=10, value=2, step=1,
                    label="Frame Skip (process every N-th frame)",
                    # FIX: warn about missing brief events
                    info="Higher → faster processing, but may miss brief events between frames",
                )
                inp_show_conf = gr.Checkbox(
                    value=True,
                    label="Show per-class confidence bars in results",
                )

            btn_detect = gr.Button(
                "◉  RUN DETECTION",
                variant="primary",
                elem_classes="detect-btn",
            )

        # ── Right column: outputs ────────────────────────────────────────────
        with gr.Column(scale=5, elem_classes="panel-card"):
            gr.HTML('<div class="section-label">ANNOTATED OUTPUT</div>')
            out_video = gr.Video(
                label="",
                autoplay=True,
                show_label=False,
            )
            gr.HTML('<div class="section-label" style="margin-top:16px;">ANALYSIS REPORT</div>')
            out_summary = gr.HTML()

    btn_detect.click(
        fn      = detect_video,
        inputs  = [inp_video, inp_conf, inp_iou, inp_show_conf, inp_frame_skip],
        outputs = [out_video, out_summary],
    )

    if examples:
        gr.HTML('<div class="section-label" style="margin-top:24px;">EXAMPLE VIDEOS</div>')
        gr.Examples(
            examples       = examples,
            inputs         = [inp_video, inp_conf, inp_iou, inp_show_conf, inp_frame_skip],
            outputs        = [out_video, out_summary],
            fn             = detect_video,
            cache_examples = True,
        )

    gr.HTML(FOOTER_HTML)

# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        ssr_mode=False,
        theme=gr.themes.Base(
            primary_hue="red",
            neutral_hue="zinc",
            font=gr.themes.GoogleFont("Space Mono"),
        ),
        css=CUSTOM_CSS,
    )
