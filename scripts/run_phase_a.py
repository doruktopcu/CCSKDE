"""Phase A: eval-only metrics + significance on the existing checkpoints.

One sequential process (frees each model between configs, so peak memory is one
model). For every results_v2 config it computes the full frame-level metric suite
(AUROC, AUPRC, EER, false-alarm@90% recall, per-scene macro/micro) plus the
hazard-subset metrics, and stores the per-frame scores. It then runs DeLong's
correlated-AUC test and a paired bootstrap of baseline-vs-each-context-config on
the identical test frames. Saves everything to colab_results/phase_a/.

Run (after `set PYTHONPATH=%CD%`):
    .venv\\Scripts\\python scripts\\run_phase_a.py --device cuda --sigma 0
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEKER = os.path.join(ROOT, "seeker")
for p in (SEEKER, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from args import init_parser, init_sub_args                  # noqa: E402
from dataset import SkeletonSequenceDataset                  # noqa: E402
from validation import score_anomalies                       # noqa: E402

from ccskde.context.config import HAZARD_CLASSES             # noqa: E402
from ccskde.eval.hazard import (                             # noqa: E402
    hazard_subset_metrics, score_proximity_correlation, balanced_hazard_auroc,
    clip_frame_proximity, auroc_ap, equal_error_rate, false_alarm_at_recall,
    per_scene_auroc, delong_roc_test, paired_bootstrap_auroc,
)
from scripts.compare_models import build_model_loader        # noqa: E402
from scripts.evaluate_hazard import (                        # noqa: E402
    score_test, score_test_scene, load_ped_centroids,
)

CONFIGS = [   # (name, mode, film_cov)
    ("baseline", "none", False),
    ("proximity", "proximity", False),
    ("vehicle", "vehicle", False),
    ("vehicle_film", "vehicle", True),
    ("vehicle_shuffled", "vehicle", False),
    ("scene", "scene", False),
]


def resolve_ckpt(name):
    hits = sorted(glob.glob(os.path.join(
        ROOT, "colab_results", "results_v2", f"ShanghaiTech_{name}", "*", "checkpoint_best.pth")))
    if not hits:
        raise FileNotFoundError(name)
    return hits[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sigma", type=int, default=0)
    ap.add_argument("--seg_len", type=int, default=24)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--max_vehicles", type=int, default=2)
    ap.add_argument("--kv", type=int, default=6)
    ap.add_argument("--scene_readout", default="joint")
    ap.add_argument("--proximity_cache", default="colab_results/results/context_cache")
    ap.add_argument("--oriented_cache", default="data/ShanghaiTech/context_oriented")
    ap.add_argument("--tau", type=float, default=3.0)
    a = ap.parse_args()

    outdir = os.path.join(ROOT, "colab_results", "phase_a")
    os.makedirs(os.path.join(outdir, "frame_scores"), exist_ok=True)

    parser = init_parser()
    args = parser.parse_args(["--dataset", "ShanghaiTech", "--data_dir", "data",
                              "--device", a.device, "--seg_len", str(a.seg_len),
                              "--batch_size", str(a.batch_size), "--num_workers", "0"])
    args, _ = init_sub_args(args)
    base = SkeletonSequenceDataset(args.pose_path["test"], path_to_vid_dir=args.vid_path["test"],
                                   evaluate=True, filter_conf=args.filter_conf,
                                   seg_len=a.seg_len, dataset="ShanghaiTech",
                                   train_seg_conf_th=0.0, seg_stride=1,
                                   vid_path=args.vid_path["test"], split="test")

    # ---- per-frame proximity + per-scene bookkeeping (once) ----
    coco_ids = tuple(HAZARD_CLASSES.values())
    W, H = 856, 480
    gt_dir = os.path.join("data", "ShanghaiTech", "gt", "test_frame_mask")
    pcache = os.path.join(a.proximity_cache, "test")
    prox_chunks, clip_keys, frame_counts = [], [], []
    for clip in sorted(f for f in os.listdir(gt_dir) if f.endswith(".npy")):
        key = clip.split(".")[0]
        n = int(np.load(os.path.join(gt_dir, clip)).shape[0])
        clip_keys.append(key); frame_counts.append(n)
        det_path = os.path.join(pcache, f"{key}.npy")
        pj = os.path.join(args.pose_path["test"], f"{key}_alphapose_tracked_person.json")
        if not (os.path.exists(det_path) and os.path.exists(pj)):
            prox_chunks.append(np.zeros(n)); continue
        dets = list(np.load(det_path, allow_pickle=True))
        peds = load_ped_centroids(pj, W, H)
        prox_chunks.append(clip_frame_proximity(dets, peds, n, coco_ids))
    proximity = np.concatenate(prox_chunks)

    results, frame_scores, gt_ref = {}, {}, None
    for name, mode, film in CONFIGS:
        t0 = time.time()
        ckpt = resolve_ckpt(name)
        na = SimpleNamespace(seg_len=a.seg_len, batch_size=a.batch_size, device=a.device,
                             max_vehicles=a.max_vehicles, kv=a.kv,
                             oriented_cache=a.oriented_cache, proximity_cache=a.proximity_cache,
                             scene_readout=a.scene_readout, sigma=a.sigma)
        model, loader, kind = build_model_loader(mode, ckpt, base, args, na, film_cov=film)
        kp = (score_test_scene(model, loader, a.device, a.scene_readout) if kind == "scene"
              else score_test(model, loader, a.device, kind == "ctx"))
        auc, scores, gt = score_anomalies(kp, base.metadata, args=args, split="test",
                                          ret_gt=True, sigma=a.sigma)
        gt_ref = gt if gt_ref is None else gt_ref
        assert len(scores) == len(proximity) == len(gt)
        frame_scores[name] = scores
        np.save(os.path.join(outdir, "frame_scores", f"{name}.npy"), scores)

        ap_m = auroc_ap(scores, gt)
        eer = equal_error_rate(scores, gt)
        far = false_alarm_at_recall(scores, gt, 0.90)
        ps = per_scene_auroc(scores, gt, clip_keys, frame_counts)
        hz = hazard_subset_metrics(scores, gt, proximity, a.tau)
        bal = balanced_hazard_auroc(scores, gt, proximity, a.tau)
        corr = score_proximity_correlation(scores, proximity, gt)
        results[name] = dict(
            ckpt=os.path.relpath(ckpt, ROOT), sigma=a.sigma,
            auroc=ap_m["auroc"], auprc=ap_m["ap"], base_rate=ap_m["base_rate"],
            eer=eer["eer"], far_at_90recall=far["fpr"],
            per_scene_micro=ps["micro"], per_scene_macro=ps["macro"],
            per_scene=ps["per_scene"],
            vehicle_hazard_auroc=hz["vehicle_hazard"]["auroc"],
            interaction_auroc=hz["interaction_regime"]["auroc"],
            balanced_near_auroc=bal["auroc"], balanced_lo=bal["lo"], balanced_hi=bal["hi"],
            spearman=corr["spearman"], pearson=corr["pearson"],
        )
        dt = time.time() - t0
        print(f"[{name:16}] AUROC={ap_m['auroc']:.4f} AUPRC={ap_m['ap']:.4f} "
              f"EER={eer['eer']:.4f} per-scene macro={ps['macro']:.4f} "
              f"veh-haz={hz['vehicle_hazard']['auroc']:.4f} rho={corr['spearman']:.3f} "
              f"({dt:.0f}s)", flush=True)
        del model, loader
        torch.cuda.empty_cache()

    # ---- significance: baseline vs each context config (same frames) ----
    sig = {}
    base_s = frame_scores["baseline"]
    for name in ("proximity", "vehicle", "vehicle_film", "scene"):
        d = delong_roc_test(gt_ref, base_s, frame_scores[name])   # auc_a=base, auc_b=ctx
        pb = paired_bootstrap_auroc(gt_ref, base_s, frame_scores[name], n_boot=2000)
        sig[f"baseline_vs_{name}"] = dict(
            auc_baseline=d["auc_a"], auc_ctx=d["auc_b"], delta=d["auc_b"] - d["auc_a"],
            delong_z=d["z"], delong_p=d["p_value"],
            boot_mean_delta=pb["mean_diff"], boot_lo=pb["lo"], boot_hi=pb["hi"],
            boot_p=pb["p_b_gt_a"])
        print(f"[DeLong] baseline vs {name:13} delta={d['auc_b']-d['auc_a']:+.4f} "
              f"z={d['z']:+.2f} p={d['p_value']:.2e}  boot CI[{pb['lo']:+.4f},{pb['hi']:+.4f}]",
              flush=True)

    with open(os.path.join(outdir, "phase_a_results.json"), "w") as f:
        json.dump(dict(sigma=a.sigma, tau=a.tau, configs=results, significance=sig),
                  f, indent=2)
    print("\nwrote", os.path.relpath(os.path.join(outdir, "phase_a_results.json"), ROOT))


if __name__ == "__main__":
    main()
