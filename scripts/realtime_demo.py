"""Real-time detection demo (Tkinter GUI) for CCSKDE.

Rolls ShanghaiTech clips at video framerate with the pedestrian skeleton and
oriented vehicle keypoints overlaid, plus a live hazard-score gauge that fires
when our CCSKDE car-matrix model flags a pedestrian-vehicle hazard. The score is
the model's actual per-frame anomaly score (confidence-weighted NLL) produced by
the inference pass in run_phase_a.py and saved to colab_results/phase_a/; the GUI
plays it back synced to the frames so you watch detections happen in real time.

Controls: clip dropdown, Play/Pause, Restart, speed and threshold sliders, model
selector (ours / baseline), and Auto-advance (rolls through a hazard reel for
~1 min). Pure standard-library Tkinter + Pillow + NumPy -- no web stack, no GPU.

Run:
    set PYTHONPATH=%CD%
    .venv\\Scripts\\python scripts\\realtime_demo.py
    .venv\\Scripts\\python scripts\\realtime_demo.py --selftest   # render 1 frame, no GUI
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ccskde.context.config import HAZARD_CLASSES   # noqa: E402

DATA = os.path.join(ROOT, "data", "ShanghaiTech")
FRAMES = os.path.join(DATA, "shanghaitech", "testing", "frames")
ORIENTED = os.path.join(DATA, "context_oriented", "test")
POSE = os.path.join(DATA, "pose", "test")
GT = os.path.join(DATA, "gt", "test_frame_mask")
FS = os.path.join(ROOT, "colab_results", "phase_a", "frame_scores")
W, H = 856, 480
KEEP = set(HAZARD_CLASSES.values())
ID2NAME = {v: k for k, v in HAZARD_CLASSES.items()}
FPS = 24

# COCO-17 skeleton edges (AlphaPose / STG-NF release order)
EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6), (5, 7), (7, 9),
         (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
         (12, 14), (14, 16)]

# hazard reel: vehicle-present hazard clip first, then high-gain clips
REEL = ["06_0155", "01_0056", "08_0079", "01_0027", "07_0048", "01_0135"]

CYAN = (0, 229, 255)
YELLOW = (255, 234, 0)
RED = (255, 59, 48)
GREEN = (52, 199, 89)


def _font(sz, bold=False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    try:
        return ImageFont.truetype(rf"C:\Windows\Fonts\{name}", sz)
    except Exception:
        return ImageFont.load_default()


F_BIG = _font(34, bold=True)
F_MED = _font(18, bold=True)
F_SM = _font(14)


# ----------------------------------------------------------------- data loading
_CLIP_ORDER = None
_OFFSETS = None


def _clip_index():
    global _CLIP_ORDER, _OFFSETS
    if _CLIP_ORDER is None:
        clips = sorted(f[:-4] for f in os.listdir(GT) if f.endswith(".npy"))
        fc = [int(np.load(os.path.join(GT, c + ".npy")).shape[0]) for c in clips]
        _CLIP_ORDER = clips
        _OFFSETS = dict(zip(clips, np.concatenate([[0], np.cumsum(fc)])[:-1]))
    return _CLIP_ORDER, _OFFSETS


def available_clips():
    clips, _ = _clip_index()
    have_scores = os.path.exists(os.path.join(FS, "vehicle.npy"))
    out = [c for c in clips if os.path.isdir(os.path.join(FRAMES, c)) and have_scores]
    # hazard reel first, then the rest
    reel = [c for c in REEL if c in out]
    return reel + [c for c in out if c not in reel]


def load_poses(clip):
    import json
    pj = os.path.join(POSE, f"{clip}_alphapose_tracked_person.json")
    out = {}
    if not os.path.exists(pj):
        return out
    d = json.load(open(pj))
    for _pid, frames in d.items():
        if isinstance(frames, list):
            m = {}
            [m.update(s) for s in frames]
            frames = m
        for fk, rec in frames.items():
            kp = np.array(rec["keypoints"], dtype=np.float64).reshape(-1, 3)
            out.setdefault(int(fk), []).append(kp)
    return out


def _norm(x):
    return (x - x.min()) / (np.ptp(x) + 1e-9)


class ClipData:
    """All per-frame data for one clip, cached."""

    def __init__(self, clip):
        self.clip = clip
        self.gt = np.load(os.path.join(GT, f"{clip}.npy")).astype(int)
        self.n = len(self.gt)
        self.dets = np.load(os.path.join(ORIENTED, f"{clip}.npy"), allow_pickle=True)
        self.poses = load_poses(clip)
        _, off = _clip_index()
        o = off[clip]
        self.scores = {}
        for name in ("vehicle", "baseline"):
            p = os.path.join(FS, f"{name}.npy")
            s = np.load(p)[o:o + self.n]
            self.scores[name] = _norm(s)
        self.frame_paths = [os.path.join(FRAMES, clip, f"{i:03d}.jpg") for i in range(self.n)]


# ----------------------------------------------------------------- rendering
def draw_frame(data: ClipData, i: int, model: str, thr: float) -> Image.Image:
    img = Image.open(data.frame_paths[i]).convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")
    # skeletons
    for kp in data.poses.get(i, []):
        xy, c = kp[:, :2], kp[:, 2]
        for a, b in EDGES:
            if a < len(kp) and b < len(kp) and c[a] > 0.1 and c[b] > 0.1:
                d.line([tuple(xy[a]), tuple(xy[b])], fill=CYAN, width=3)
        for j in range(len(kp)):
            if c[j] > 0.1:
                x, y = xy[j]
                d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=CYAN, outline=(0, 0, 0))
    # vehicles (oriented keypoints)
    rows = [r for r in data.dets[i] if int(r[0]) in KEEP] if i < len(data.dets) and data.dets[i] is not None else []
    for r in rows:
        kps = (np.array(r[2:14], dtype=np.float64).reshape(6, 2) * [W, H])
        corners, center, heading = kps[:4], kps[4], kps[5]
        d.polygon([tuple(p) for p in corners], outline=YELLOW, width=3)
        for p in corners:
            d.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], fill=YELLOW, outline=(0, 0, 0))
        d.line([tuple(center), tuple(heading)], fill=RED, width=3)
        d.ellipse([center[0] - 5, center[1] - 5, center[0] + 5, center[1] + 5], fill=RED, outline=(0, 0, 0))
        d.text((center[0] + 6, center[1] - 18), ID2NAME.get(int(r[0]), "veh"), font=F_SM, fill=RED)

    score = float(data.scores[model][i])
    fired = score >= thr
    col = RED if fired else GREEN

    # --- HUD: score gauge (top-left) ---
    d.rectangle([12, 12, 312, 64], fill=(0, 0, 0, 150))
    d.text((20, 16), f"hazard score ({'ours' if model == 'vehicle' else 'baseline'})",
           font=F_SM, fill=(255, 255, 255))
    d.rectangle([20, 38, 304, 58], outline=(255, 255, 255), width=1)
    d.rectangle([20, 38, 20 + int(284 * score), 58], fill=col)
    d.text((310 - 52, 38), f"{score:.2f}", font=F_MED, fill=(255, 255, 255))

    # --- GT + frame counter (top-right) ---
    gt_lab = "ANOMALY" if data.gt[i] == 1 else "normal"
    gt_col = RED if data.gt[i] == 1 else GREEN
    d.text((W - 250, 14), f"clip {data.clip}  frame {i+1}/{data.n}", font=F_SM, fill=(255, 255, 255))
    d.text((W - 250, 32), f"ground truth: ", font=F_SM, fill=(255, 255, 255))
    d.text((W - 250 + 92, 32), gt_lab, font=F_SM, fill=gt_col)

    # --- big banner when fired ---
    if fired:
        tw = d.textlength("HAZARD DETECTED", font=F_BIG)
        d.rectangle([(W - tw) / 2 - 16, 74, (W + tw) / 2 + 16, 120], fill=(255, 59, 48, 210))
        d.text(((W - tw) / 2, 78), "HAZARD DETECTED", font=F_BIG, fill=(255, 255, 255))

    # --- rolling sparkline (bottom) ---
    win = 140
    lo = max(0, i - win + 1)
    hist = data.scores[model][lo:i + 1]
    gx0, gy0, gw, gh = 12, H - 80, 360, 60
    d.rectangle([gx0, gy0, gx0 + gw, gy0 + gh], fill=(0, 0, 0, 150))
    # threshold line
    ty = gy0 + gh - int(gh * thr)
    d.line([gx0, ty, gx0 + gw, ty], fill=(255, 255, 255, 120), width=1)
    d.text((gx0 + 4, gy0 + 2), f"score history (thr={thr:.2f})", font=F_SM, fill=(255, 255, 255))
    if len(hist) > 1:
        pts = [(gx0 + int(gw * k / (len(hist) - 1)), gy0 + gh - int(gh * float(v)))
               for k, v in enumerate(hist)]
        d.line(pts, fill=CYAN, width=2)
        d.ellipse([pts[-1][0] - 3, pts[-1][1] - 3, pts[-1][0] + 3, pts[-1][1] + 3],
                  fill=col, outline=(255, 255, 255))
    return img


# ----------------------------------------------------------------- GUI
def run_gui():
    import tkinter as tk
    from tkinter import ttk
    from PIL import ImageTk

    clips = available_clips()
    if not clips:
        raise SystemExit("No clips with frames + saved scores found. Run scripts/run_phase_a.py first.")

    root = tk.Tk()
    root.title("CCSKDE — real-time pedestrian-vehicle hazard detection")
    root.configure(bg="#1e1e1e")

    state = {"data": None, "i": 0, "playing": False, "after": None}

    canvas = tk.Label(root, bg="black")
    canvas.grid(row=0, column=0, columnspan=8, padx=8, pady=8)

    bar = tk.Frame(root, bg="#1e1e1e")
    bar.grid(row=1, column=0, columnspan=8, sticky="we", padx=8, pady=(0, 8))

    clip_var = tk.StringVar(value=clips[0])
    model_var = tk.StringVar(value="vehicle")
    speed_var = tk.DoubleVar(value=1.0)
    thr_var = tk.DoubleVar(value=0.55)
    auto_var = tk.BooleanVar(value=True)
    status = tk.StringVar(value="")

    def load(clip):
        state["data"] = ClipData(clip)
        state["i"] = 0
        render()

    def render():
        data = state["data"]
        img = draw_frame(data, state["i"], model_var.get(), float(thr_var.get()))
        tkimg = ImageTk.PhotoImage(img)
        canvas.configure(image=tkimg)
        canvas.image = tkimg
        s = data.scores[model_var.get()][state["i"]]
        status.set(f"{data.clip}  frame {state['i']+1}/{data.n}   score={s:.2f}   "
                   f"{'FIRED' if s >= float(thr_var.get()) else '--'}")

    def tick():
        if not state["playing"]:
            return
        data = state["data"]
        state["i"] += 1
        if state["i"] >= data.n:
            if auto_var.get():
                cur = clips.index(clip_var.get())
                nxt = clips[(cur + 1) % len(clips)]
                clip_var.set(nxt)
                load(nxt)
            else:
                state["i"] = 0
        render()
        delay = max(1, int(1000 / (FPS * float(speed_var.get()))))
        state["after"] = root.after(delay, tick)

    def play_pause():
        state["playing"] = not state["playing"]
        btn_play.config(text="⏸ Pause" if state["playing"] else "▶ Play")
        if state["playing"]:
            tick()

    def restart():
        state["i"] = 0
        render()

    def on_clip(*_):
        if state["after"]:
            root.after_cancel(state["after"])
        load(clip_var.get())
        if state["playing"]:
            tick()

    tk.Label(bar, text="Clip:", fg="white", bg="#1e1e1e").pack(side="left")
    ttk.Combobox(bar, textvariable=clip_var, values=clips, width=10,
                 state="readonly").pack(side="left", padx=4)
    btn_play = tk.Button(bar, text="▶ Play", width=9, command=play_pause)
    btn_play.pack(side="left", padx=4)
    tk.Button(bar, text="⟲ Restart", command=restart).pack(side="left", padx=4)
    tk.Label(bar, text="Model:", fg="white", bg="#1e1e1e").pack(side="left", padx=(12, 0))
    ttk.Combobox(bar, textvariable=model_var, values=["vehicle", "baseline"], width=9,
                 state="readonly").pack(side="left", padx=4)
    tk.Label(bar, text="Speed", fg="white", bg="#1e1e1e").pack(side="left", padx=(12, 0))
    tk.Scale(bar, from_=0.25, to=2.0, resolution=0.25, orient="horizontal",
             variable=speed_var, length=90, bg="#1e1e1e", fg="white",
             highlightthickness=0).pack(side="left")
    tk.Label(bar, text="Threshold", fg="white", bg="#1e1e1e").pack(side="left", padx=(12, 0))
    tk.Scale(bar, from_=0.0, to=1.0, resolution=0.05, orient="horizontal",
             variable=thr_var, length=110, bg="#1e1e1e", fg="white",
             highlightthickness=0, command=lambda *_: render() if state["data"] else None).pack(side="left")
    tk.Checkbutton(bar, text="Auto-advance", variable=auto_var, fg="white", bg="#1e1e1e",
                   selectcolor="#333").pack(side="left", padx=10)

    model_var.trace_add("write", lambda *_: render() if state["data"] else None)
    clip_var.trace_add("write", on_clip)

    tk.Label(root, textvariable=status, fg="#9cdcfe", bg="#1e1e1e",
             font=("Consolas", 10)).grid(row=2, column=0, columnspan=8, sticky="w", padx=10, pady=(0, 8))

    load(clips[0])
    root.bind("<space>", lambda e: play_pause())
    root.bind("q", lambda e: root.destroy())
    play_pause()        # autostart
    root.mainloop()


def selftest():
    clips = available_clips()
    print("available clips:", len(clips), "->", clips[:6], "...")
    data = ClipData(clips[0])
    idx = int(np.argmax(data.scores["vehicle"]))
    img = draw_frame(data, idx, "vehicle", 0.55)
    out = os.path.join(ROOT, "colab_results", "phase_a", "realtime_selftest.png")
    img.save(out)
    print(f"rendered {clips[0]} frame {idx} (score={data.scores['vehicle'][idx]:.2f}) -> "
          f"{os.path.relpath(out, ROOT)}  size={img.size}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="render one frame and exit (no GUI)")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        run_gui()
