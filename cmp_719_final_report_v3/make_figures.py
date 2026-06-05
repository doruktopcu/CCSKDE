"""Generate all EDA + results figures for the CMP719 v2 final report.

CPU-only (matplotlib); reads existing result JSONs + ShanghaiTech test caches/
poses. Writes vector PDFs into cmp719_final_latex/figures/.

Run:  set PYTHONPATH=%CD% && .venv\\Scripts\\python cmp_719_final_report_v2\\make_figures.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FIG = os.path.join(ROOT, "cmp_719_final_report_v2", "cmp719_final_latex", "figures")
os.makedirs(FIG, exist_ok=True)

from ccskde.eval.hazard import clip_frame_proximity        # noqa: E402
from ccskde.context.config import HAZARD_CLASSES           # noqa: E402

DATA = os.path.join(ROOT, "data", "ShanghaiTech")
GT_DIR = os.path.join(DATA, "gt", "test_frame_mask")
POSE_DIR = os.path.join(DATA, "pose", "test")
BOX_CACHE = os.path.join(ROOT, "colab_results", "results", "context_cache", "test")
W, H = 856, 480
COCO_IDS = tuple(HAZARD_CLASSES.values())
ID2NAME = {v: k for k, v in HAZARD_CLASSES.items()}
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.bbox": "tight"})


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p)
    plt.close(fig)
    print("wrote", os.path.relpath(p, ROOT))


def load_ped_centroids(pose_json):
    d = json.load(open(pose_json))
    out = {}
    for _pid, frames in d.items():
        if isinstance(frames, list):
            m = {}
            [m.update(s) for s in frames]
            frames = m
        for fk, rec in frames.items():
            kp = np.array(rec["keypoints"], dtype=np.float64).reshape(-1, 3)
            w = kp[:, 2]
            c = (kp[:, :2] * w[:, None]).sum(0) / w.sum() if w.sum() > 0 else kp[:, :2].mean(0)
            out.setdefault(int(fk), []).append(np.array([c[0] / W, c[1] / H]))
    return out


# --------------------------------------------------------------- dataset EDA
def compute_dataset_stats():
    prox, gt, vehs_per_frame = [], [], []
    class_counts = {c: 0 for c in COCO_IDS}
    clips = sorted(f for f in os.listdir(GT_DIR) if f.endswith(".npy"))
    for clip in clips:
        key = clip.split(".")[0]
        g = np.load(os.path.join(GT_DIR, clip)).astype(int)
        n = g.shape[0]
        det_p = os.path.join(BOX_CACHE, f"{key}.npy")
        pj = os.path.join(POSE_DIR, f"{key}_alphapose_tracked_person.json")
        if not (os.path.exists(det_p) and os.path.exists(pj)):
            prox.append(np.zeros(n)); gt.append(g); continue
        dets = list(np.load(det_p, allow_pickle=True))
        peds = load_ped_centroids(pj)
        prox.append(clip_frame_proximity(dets, peds, n, COCO_IDS))
        gt.append(g)
        for f in range(min(n, len(dets))):
            d = dets[f]
            keep = [r for r in d if int(r[0]) in class_counts] if len(d) else []
            vehs_per_frame.append(len(keep))
            for r in keep:
                class_counts[int(r[0])] += 1
    return (np.concatenate(prox), np.concatenate(gt),
            np.array(vehs_per_frame), class_counts)


def fig_proximity_hist(prox, gt):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    m = prox > 0                      # frames with a hazard present
    pn = prox[m & (gt == 0)]
    pa = prox[m & (gt == 1)]
    bins = np.logspace(np.log10(1.4), np.log10(1000), 30)
    ax.hist(pn, bins=bins, alpha=0.6, density=True, label=f"normal (n={pn.size})")
    ax.hist(pa, bins=bins, alpha=0.6, density=True, label=f"anomalous (n={pa.size})")
    ax.set_xscale("log")
    ax.set_xlabel(r"pedestrian–vehicle proximity  $1/(d_{\min}+\epsilon)$  (higher = closer)")
    ax.set_ylabel("density")
    ax.set_title("Per-frame pedestrian–vehicle proximity (frames with a hazard present)")
    ax.legend()
    save(fig, "eda_proximity_hist.pdf")


def fig_vehicle_classes(vehs, class_counts):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    names = [ID2NAME[c] for c in COCO_IDS]
    vals = [class_counts[c] for c in COCO_IDS]
    axes[0].bar(names, vals, color="tab:blue")
    axes[0].set_ylabel("detections (test split)")
    axes[0].set_title("Hazard-class detections")
    axes[0].tick_params(axis="x", rotation=30)
    mx = int(vehs.max()) if vehs.size else 0
    axes[1].hist(vehs, bins=np.arange(-0.5, mx + 1.5, 1), color="tab:orange")
    axes[1].set_xlabel("hazard objects per frame")
    axes[1].set_ylabel("# frames")
    axes[1].set_title(f"Objects/frame (mean {vehs.mean():.2f})")
    save(fig, "eda_vehicle_classes.pdf")


def fig_regime_breakdown(prox, gt, tau=3.0):
    near = prox >= tau
    counts = {
        "normal\nfar": int(((gt == 0) & ~near).sum()),
        "normal\nnear": int(((gt == 0) & near).sum()),
        "anomaly\nfar": int(((gt == 1) & ~near).sum()),
        "anomaly\nnear": int(((gt == 1) & near).sum()),
    }
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars = ax.bar(list(counts), list(counts.values()),
                  color=["tab:green", "tab:olive", "tab:red", "tab:purple"])
    ax.set_ylabel("# test frames")
    n_near = counts["normal\nnear"] + counts["anomaly\nnear"]
    frac = counts["anomaly\nnear"] / max(n_near, 1)
    ax.set_title(f"Frame regimes (tau={tau}); near-vehicle frames are {frac:.0%} anomalous")
    for b, v in zip(bars, counts.values()):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    save(fig, "eda_regime_breakdown.pdf")


# --------------------------------------------------------------- results
def fig_epoch_curves():
    j = json.load(open(os.path.join(ROOT, "colab_results", "results_v2", "experiment_results.json")))
    r = j["results"]
    order = ["baseline", "proximity", "vehicle", "vehicle_film", "vehicle_shuffled", "scene"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in order:
        if name in r and "aucs" in r[name]:
            a = r[name]["aucs"]
            ax.plot(range(1, len(a) + 1), a, marker="o", ms=4, label=name.replace("_", " "))
    ax.set_xlabel("epoch"); ax.set_ylabel("validation AUROC")
    ax.set_title("Per-epoch validation AUROC (single seed)")
    ax.legend(fontsize=8, ncol=2)
    save(fig, "res_epoch_curves.pdf")


def fig_multiseed():
    j = json.load(open(os.path.join(ROOT, "colab_results", "results_v3_multiseed", "experiment_results.json")))
    r = j["results"]
    names, means, stds = [], [], []
    for name in ["baseline", "vehicle", "vehicle_shuffled"]:
        if name in r and "mean_mean" in r[name]:
            names.append(name.replace("_", "\n")); means.append(r[name]["mean_mean"]); stds.append(r[name]["mean_std"])
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=6, color=["tab:gray", "tab:blue", "tab:cyan"][:len(names)])
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("mean AUROC (5 seeds)")
    ax.set_ylim(0.70, 0.80)
    ax.set_title("Multi-seed mean AUROC (error bars = std)")
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.001, f"{m:.3f}\n±{s:.3f}", ha="center", va="bottom", fontsize=8)
    save(fig, "res_multiseed.pdf")


def fig_counterfactual():
    # from the counterfactual probe on vehicle_film (sigma=0)
    variants = ["with context\n(real)", "vehicle removed\n(counterfactual)", "interaction\n(real − cf)"]
    full = [0.7971, 0.7411, 0.6868]
    vehh = [0.9658, 0.7875, 0.9672]
    rho = [0.640, 0.151, 0.698]
    x = np.arange(len(variants)); w = 0.27
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w, full, w, label="full-set AUROC")
    ax.bar(x, vehh, w, label="vehicle-hazard AUROC")
    ax.bar(x + w, rho, w, label=r"score$\leftrightarrow$proximity $\rho$")
    ax.set_xticks(x); ax.set_xticklabels(variants)
    ax.set_ylabel("metric value")
    ax.set_title("Counterfactual interaction probe (vehicle = car-matrix + FiLM)")
    ax.legend(fontsize=8); ax.axhline(0.5, color="k", lw=0.7, ls=":")
    save(fig, "res_counterfactual.pdf")


def fig_sigma():
    sig = [0, 3, 5, 10, 15, 20, 30, 40, 50]
    base = [0.7808, 0.7839, 0.7874, 0.7923, 0.7961, 0.7971, 0.7926, 0.7833, 0.7743]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].plot(sig, base, marker="o")
    ax[0].axhline(0.855, color="tab:red", ls="--", label="published 0.855")
    ax[0].set_xlabel(r"temporal smoothing $\sigma$"); ax[0].set_ylabel("AUROC")
    ax[0].set_title("Baseline AUROC vs smoothing"); ax[0].legend(fontsize=8)
    cfgs = ["baseline", "vehicle", "vehicle_film"]
    s0 = [0.7351, 0.8023, np.nan]
    s20 = [0.7573, 0.7652, 0.7485]
    x = np.arange(len(cfgs)); w = 0.35
    ax[1].bar(x - w / 2, s0, w, label=r"$\sigma=0$")
    ax[1].bar(x + w / 2, s20, w, label=r"$\sigma=20$")
    ax[1].set_xticks(x); ax[1].set_xticklabels([c.replace("_", "\n") for c in cfgs])
    ax[1].set_ylabel("AUROC"); ax[1].set_ylim(0.70, 0.82)
    ax[1].set_title("Operating-point sensitivity"); ax[1].legend(fontsize=8)
    save(fig, "res_sigma.pdf")


def main():
    print("== dataset EDA ==")
    prox, gt, vehs, cc = compute_dataset_stats()
    print(f"frames={gt.size} anomalous={int(gt.sum())} ({gt.mean():.1%}) "
          f"with-hazard={int((prox>0).sum())}")
    fig_proximity_hist(prox, gt)
    fig_vehicle_classes(vehs, cc)
    fig_regime_breakdown(prox, gt)
    print("== results ==")
    fig_epoch_curves()
    fig_multiseed()
    fig_counterfactual()
    fig_sigma()
    print("done. figures in", os.path.relpath(FIG, ROOT))


if __name__ == "__main__":
    main()
