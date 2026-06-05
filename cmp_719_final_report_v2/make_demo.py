"""Detection demo for the CMP719 report: predictions in action on one clip.

Picks a hazard clip where a vehicle is genuinely present during the anomaly and
shows (a) the per-frame anomaly-score timeline for the baseline vs. our
car-matrix model, with the ground-truth anomaly window shaded and vehicle-present
frames marked, and (b) annotated keyframes (pedestrian skeleton + oriented
vehicle keypoints) with each model's score. Also writes an animated GIF.

CPU-only: reads the per-frame scores saved by run_phase_a.py plus the cached
frames/poses/detections. Run:
    set PYTHONPATH=%CD% && .venv\\Scripts\\python cmp_719_final_report_v2\\make_demo.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from make_qualitative import (   # noqa: E402  (reuse the overlay helpers)
    W, H, KEEP, load_poses, draw_skeleton, draw_vehicle, FIG, FRAMES, ORIENTED, GT,
)
from ccskde.eval.hazard import clip_frame_proximity   # noqa: E402
from ccskde.context.config import HAZARD_CLASSES       # noqa: E402

FS = os.path.join(ROOT, "colab_results", "phase_a", "frame_scores")
CLIP = "06_0155"          # hazard clip with a vehicle present during the anomaly
COCO_IDS = tuple(HAZARD_CLASSES.values())


def clip_slice(name):
    """Per-frame score for CLIP from the concatenated saved scores."""
    clips = sorted(f[:-4] for f in os.listdir(GT) if f.endswith(".npy"))
    fc = [int(np.load(os.path.join(GT, c + ".npy")).shape[0]) for c in clips]
    off = np.concatenate([[0], np.cumsum(fc)])
    i = clips.index(CLIP)
    s = np.load(os.path.join(FS, f"{name}.npy"))[off[i]:off[i + 1]]
    return (s - s.min()) / (np.ptp(s) + 1e-9)        # per-clip min-max for display


def vehicle_present(dets, n):
    out = np.zeros(n, bool)
    for f in range(min(n, len(dets))):
        d = dets[f]
        if d is not None and len(d) and any(int(r[0]) in KEEP for r in d):
            out[f] = True
    return out


def main():
    g = np.load(os.path.join(GT, f"{CLIP}.npy")).astype(int)
    n = len(g)
    base = clip_slice("baseline")
    veh = clip_slice("vehicle")
    dets = np.load(os.path.join(ORIENTED, f"{CLIP}.npy"), allow_pickle=True)
    vp = vehicle_present(dets, n)
    poses = load_poses(CLIP)
    t = np.arange(n)

    # anomaly window (contiguous block)
    idx = np.where(g == 1)[0]
    a0, a1 = int(idx.min()), int(idx.max())

    # keyframes: normal before -> anomaly onset -> the vehicle-present hazard peak
    # (so an oriented car is actually shown) -> normal after.
    vp_anom = [f for f in range(a0, a1 + 1) if vp[f]]
    vehframe = int(max(vp_anom, key=lambda f: veh[f])) if vp_anom else \
        int(a0 + np.argmax(veh[a0:a1 + 1]))
    keyframes = sorted({max(0, a0 - 50), a0 + 3, vehframe, min(n - 1, a1 + 40)})

    # ---------------- figure: timeline + keyframes ----------------
    fig = plt.figure(figsize=(11, 6.2))
    gs = fig.add_gridspec(2, len(keyframes), height_ratios=[1.05, 1.0], hspace=0.32, wspace=0.06)

    axt = fig.add_subplot(gs[0, :])
    axt.axvspan(a0, a1, color="tab:red", alpha=0.12, label="ground-truth anomaly")
    axt.plot(t, base, color="tab:gray", lw=1.6, label="SeeKer baseline (score)")
    axt.plot(t, veh, color="tab:blue", lw=1.9, label="CCSKDE car-matrix (score)")
    # mark vehicle-present frames along the bottom
    axt.plot(t[vp], np.full(vp.sum(), -0.04), "|", color="tab:green", ms=7,
             label="vehicle detected")
    for kf in keyframes:
        axt.axvline(kf, color="k", ls=":", lw=0.8, alpha=0.6)
        axt.text(kf, 1.02, f"{kf}", ha="center", va="bottom", fontsize=8)
    axt.set_xlim(0, n - 1); axt.set_ylim(-0.08, 1.08)
    axt.set_xlabel("frame"); axt.set_ylabel("normalised anomaly score")
    axt.set_title(f"Detection timeline on clip {CLIP}: the car-matrix score rises on the "
                  f"pedestrian-vehicle hazard window; the baseline stays low", fontsize=10)
    axt.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)

    for j, kf in enumerate(keyframes):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(plt.imread(os.path.join(FRAMES, CLIP, f"{kf:03d}.jpg")))
        for kp in poses.get(kf, []):
            draw_skeleton(ax, kp)
        rows = [r for r in dets[kf] if int(r[0]) in KEEP] if kf < len(dets) and dets[kf] is not None else []
        for r in rows:
            draw_vehicle(ax, r)
        lab = "ANOMALY" if g[kf] == 1 else "normal"
        col = "tab:red" if g[kf] == 1 else "tab:green"
        ax.set_title(f"frame {kf} | {lab}\nours={veh[kf]:.2f}  base={base[kf]:.2f}",
                     fontsize=8, color=col)
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")

    p = os.path.join(FIG, "demo_detection.pdf")
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote", os.path.relpath(p, ROOT))

    # ---------------- animated GIF ----------------
    try:
        make_gif(CLIP, g, base, veh, dets, poses, n)
    except Exception as e:   # noqa: BLE001
        print("GIF skipped:", e)


def make_gif(clip, g, base, veh, dets, poses, n, step=3, lo=40, hi=235):
    from PIL import Image
    frames = []
    hi = min(hi, n)
    for f in range(lo, hi, step):
        fig, ax = plt.subplots(figsize=(6.6, 3.9))
        ax.imshow(plt.imread(os.path.join(FRAMES, clip, f"{f:03d}.jpg")))
        for kp in poses.get(f, []):
            draw_skeleton(ax, kp)
        rows = [r for r in dets[f] if int(r[0]) in KEEP] if f < len(dets) and dets[f] is not None else []
        for r in rows:
            draw_vehicle(ax, r)
        # live score bar (our model) + GT flag
        ax.barh(20, veh[f] * 300, height=22, left=20, color="tab:blue", alpha=0.85)
        ax.text(20, 8, f"hazard score {veh[f]:.2f}", color="white", fontsize=9,
                weight="bold", va="center",
                bbox=dict(boxstyle="round", fc="tab:blue", ec="none", alpha=0.7))
        if g[f] == 1:
            ax.text(W - 20, 20, "ANOMALY", color="white", fontsize=11, weight="bold",
                    ha="right", va="center",
                    bbox=dict(boxstyle="round", fc="tab:red", ec="none"))
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
        ax.set_title(f"clip {clip}  frame {f}", fontsize=9)
        fig.tight_layout(pad=0.2)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(Image.fromarray(buf.copy()))
        plt.close(fig)
    out = os.path.join(FIG, "demo_detection.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=120, loop=0)
    print("wrote", os.path.relpath(out, ROOT), f"({len(frames)} frames)")


if __name__ == "__main__":
    main()
