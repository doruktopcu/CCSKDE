"""Qualitative figure for the CMP719 report: the scene representation in action.

Renders, on real ShanghaiTech test frames, the pedestrian COCO-17 skeleton and
each vehicle's oriented-keypoint set (4 corners + centre + heading) that our
Scene-SKDE consumes. Left: an anomalous near-vehicle (hazard) frame; right: a
normal frame for contrast. CPU-only (matplotlib + the cached detections/poses).

Run:  set PYTHONPATH=%CD% && .venv\\Scripts\\python cmp_719_final_report_v2\\make_qualitative.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ccskde.context.config import HAZARD_CLASSES   # noqa: E402

FIG = os.path.join(ROOT, "cmp_719_final_report_v2", "cmp719_final_latex", "figures")
FRAMES = os.path.join(ROOT, "data", "ShanghaiTech", "shanghaitech", "testing", "frames")
ORIENTED = os.path.join(ROOT, "data", "ShanghaiTech", "context_oriented", "test")
POSE = os.path.join(ROOT, "data", "ShanghaiTech", "pose", "test")
GT = os.path.join(ROOT, "data", "ShanghaiTech", "gt", "test_frame_mask")
W, H = 856, 480
KEEP = set(HAZARD_CLASSES.values())
ID2NAME = {v: k for k, v in HAZARD_CLASSES.items()}

# COCO-17 skeleton edges (AlphaPose/STG-NF release order)
COCO17_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6),
                (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
                (11, 13), (13, 15), (12, 14), (14, 16)]


def load_poses(key):
    """{frame_idx: [ (17,3) keypoint array, ... ]} in pixels."""
    pj = os.path.join(POSE, f"{key}_alphapose_tracked_person.json")
    d = json.load(open(pj))
    out = {}
    for _pid, frames in d.items():
        if isinstance(frames, list):
            m = {}
            [m.update(s) for s in frames]
            frames = m
        for fk, rec in frames.items():
            kp = np.array(rec["keypoints"], dtype=np.float64).reshape(-1, 3)
            out.setdefault(int(fk), []).append(kp)
    return out


def draw_skeleton(ax, kp, color="#00e5ff"):
    xy, c = kp[:, :2], kp[:, 2]
    for i, j in COCO17_EDGES:
        if i < len(kp) and j < len(kp) and c[i] > 0.1 and c[j] > 0.1:
            ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]],
                    "-", color=color, lw=2.0, alpha=0.9)
    vis = c > 0.1
    ax.scatter(xy[vis, 0], xy[vis, 1], s=14, color=color, edgecolors="k",
               linewidths=0.4, zorder=5)


def draw_vehicle(ax, row):
    """row = [cls, conf, x0,y0,...,x3,y3, cx,cy, hx,hy] normalised."""
    kps = np.array(row[2:14], dtype=np.float64).reshape(6, 2) * np.array([W, H])
    corners, center, heading = kps[:4], kps[4], kps[5]
    ax.add_patch(Polygon(corners, closed=True, fill=False,
                         edgecolor="#ffea00", lw=2.4, zorder=4))
    ax.scatter(corners[:, 0], corners[:, 1], s=36, color="#ffea00",
               edgecolors="k", linewidths=0.5, zorder=6)
    ax.scatter([center[0]], [center[1]], s=70, marker="P", color="#ff3b30",
               edgecolors="k", linewidths=0.6, zorder=7)
    ax.annotate("", xy=heading, xytext=center,
                arrowprops=dict(arrowstyle="-|>", color="#ff3b30", lw=2.2))
    name = ID2NAME.get(int(row[0]), str(int(row[0])))
    ax.text(center[0], center[1] - 10, name, color="#ff3b30", fontsize=8,
            ha="center", va="bottom", weight="bold")


def ped_centroid(kp):
    xy, w = kp[:, :2], kp[:, 2]
    return (xy * w[:, None]).sum(0) / w.sum() if w.sum() > 0 else xy.mean(0)


def proximity(peds, rows):
    if not peds or not rows:
        return 0.0
    centers = np.array([np.array(r[10:12]) * [W, H] for r in rows])
    diag = np.hypot(W, H)
    best = 0.0
    for kp in peds:
        c = ped_centroid(kp)
        d = np.linalg.norm(centers - c[None, :], axis=1) / diag
        best = max(best, 1.0 / (float(d.min()) + 1e-3))
    return best


def panel(ax, key, f, title_tag):
    img = plt.imread(os.path.join(FRAMES, key, f"{f:03d}.jpg"))
    ax.imshow(img)
    poses = load_poses(key).get(f, [])
    dets = np.load(os.path.join(ORIENTED, f"{key}.npy"), allow_pickle=True)
    rows = [r for r in dets[f] if int(r[0]) in KEEP] if f < len(dets) and dets[f] is not None else []
    for kp in poses:
        draw_skeleton(ax, kp)
    for r in rows:
        draw_vehicle(ax, r)
    g = np.load(os.path.join(GT, f"{key}.npy")).astype(int)
    lab = "ANOMALY" if g[f] == 1 else "normal"
    prox = proximity(poses, rows)
    ax.set_title(f"{title_tag}: {key} frame {f}\nlabel = {lab} | proximity = {prox:.2f} | "
                 f"{len(rows)} vehicle(s), {len(poses)} pedestrian(s)", fontsize=9)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")


def find_normal_frame(key):
    """A normal frame in the same clip with a pedestrian present (vehicle far/absent)."""
    g = np.load(os.path.join(GT, f"{key}.npy")).astype(int)
    poses = load_poses(key)
    for f in range(len(g)):
        if g[f] == 0 and poses.get(f):
            return f
    return 0


def main():
    haz_key, haz_f = "01_0014", 154          # anomalous near-vehicle (auto-found)
    norm_key = haz_key
    norm_f = find_normal_frame(norm_key)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    panel(axes[0], haz_key, haz_f, "Hazard")
    panel(axes[1], norm_key, norm_f, "Normal")
    # legend (proxy artists)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#00e5ff", lw=2, marker="o", label="pedestrian skeleton (COCO-17)"),
        Line2D([0], [0], color="#ffea00", lw=2, marker="s", label="vehicle oriented box + corners"),
        Line2D([0], [0], color="#ff3b30", lw=2, marker="P", label="vehicle centre + heading"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.06))
    p = os.path.join(FIG, "qualitative.pdf")
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote", os.path.relpath(p, ROOT))


if __name__ == "__main__":
    main()
