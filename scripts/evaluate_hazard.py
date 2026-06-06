"""Hazard-subset evaluation driver.

Loads a trained checkpoint, scores the ShanghaiTech test set, computes per-frame
pedestrian-vehicle proximity, and reports:
  * full / interaction-regime / vehicle-hazard AUROC (ccskde.eval.hazard)
  * Spearman/Pearson score-vs-proximity correlation over anomalous frames
  * a tau sweep

Works for both the vanilla baseline and CCSKDE context models.

Example:
    set PYTHONPATH=%CD%
    .venv\\Scripts\\python scripts\\evaluate_hazard.py ^
      --checkpoint colab_results\\results\\checkpoints\\baseline_best.pth ^
      --data_dir data --proximity_cache colab_results\\results\\context_cache ^
      --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEKER = os.path.join(ROOT, "seeker")
for p in (SEEKER, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from args import init_parser, init_sub_args            # noqa: E402
from dataset import SkeletonSequenceDataset            # noqa: E402
from validation import score_anomalies                 # noqa: E402
from models.partial_autoregressive import PartialAutoregressiveFC  # noqa: E402

from ccskde.context import ContextSpec                 # noqa: E402
from ccskde.context.config import HAZARD_CLASSES       # noqa: E402
from ccskde.context.scene import PED_KP                # noqa: E402
from ccskde.data import ContextualSkeletonSequenceDataset  # noqa: E402
from ccskde.data.scene_dataset import SceneSkeletonDataset  # noqa: E402
from ccskde.models import (                            # noqa: E402
    PartialAutoregressiveContextFC, PartialAutoregressiveSceneFC,
)
from ccskde.eval.hazard import (                       # noqa: E402
    hazard_subset_metrics, score_proximity_correlation, clip_frame_proximity, sweep_tau,
)
from torch.utils.data import DataLoader                # noqa: E402


def joint_lnp(model, x, c=None):
    B, T, N, A = x.shape
    out = model(x, c) if c is not None else model(x)
    mu, logvar = torch.chunk(out, 2, dim=1)
    mu = mu.reshape(B, T * N, A)[:, (-N - 1):-1]
    logvar = logvar.reshape(B, T * N, A)[:, (-N - 1):-1].clamp(-10, 10)
    var = torch.exp(logvar)
    E_inv = torch.diag_embed(1.0 / var)
    x = x.reshape(B, T * N, A)
    x_tgt = x[:, -N:]
    return -0.5 * (torch.einsum("bti,btij,btj->bt", x_tgt - mu, E_inv, x_tgt - mu)
                   + torch.sum(logvar, dim=-1) + A * math.log(2 * math.pi))


def score_test(model, loader, device, is_ctx, zero_ctx=False):
    """Score the test set. If zero_ctx, the context vector is zeroed — the
    'vehicle-removed' counterfactual (the model's pedestrian-only behaviour)."""
    model.eval().to(device)
    probs = torch.empty(0).to(device)
    for data_arr in loader:
        xc = torch.permute(data_arr[0], (0, 2, 3, 1))
        x = xc[..., :2].to(device).float()
        conf = xc[..., -1].to(device)
        B, T, N, _ = x.shape
        c = data_arr[2].to(device).float() if is_ctx else None
        if zero_ctx and c is not None:
            c = torch.zeros_like(c)
        with torch.no_grad():
            lnp = joint_lnp(model, x, c)
            pad = torch.randn(B, (T - 1) * N, device=device)
            nll = -torch.cat((pad, lnp), dim=1)
            nll = (nll.view(B, T, N) * conf).flatten(start_dim=1)
        probs = torch.cat((probs, nll), dim=0)
    return probs.cpu().numpy().squeeze().copy(order="C")


def score_test_scene(model, loader, device, readout="ped", ped_first=False):
    """Scene model scoring. `readout`:
      'ped'   — pedestrian keypoints only (SeeKer-comparable; misses anomalies
                that live in the VEHICLE, e.g. a car in a pedestrian zone).
      'joint' — pedestrian + vehicle keypoint NLL (the vehicle term is folded
                into the 18 pedestrian slots so SeeKer's scorer still applies),
                so vehicle-intrinsic hazards are captured.
    `ped_first` matches the trained ordering: default (False) is vehicles-first
    (pedestrian = last 18); True is the pedestrian-first ablation (first 18)."""
    model.eval().to(device)
    probs = torch.empty(0).to(device)
    for data_arr in loader:
        xc = torch.permute(data_arr[0], (0, 2, 3, 1))
        x = xc[..., :2].to(device).float()                 # (B,T,N',2)
        conf = xc[..., -1].to(device)                      # (B,T,N')
        B, T, Np, _ = x.shape
        psl = slice(0, PED_KP) if ped_first else slice(Np - PED_KP, Np)
        vsl = slice(PED_KP, Np) if ped_first else slice(0, Np - PED_KP)
        with torch.no_grad():
            lnp = joint_lnp(model, x)                       # (B, N')  log-prob
            lnp_ped = lnp[:, psl].clone()
            if readout == "joint" and Np > PED_KP:
                # fold the vehicle log-prob into the pedestrian slots so the
                # per-frame NLL = -(sum ped + sum vehicle) = joint scene NLL.
                lnp_veh = lnp[:, vsl]
                lnp_ped = lnp_ped + lnp_veh.sum(dim=1, keepdim=True) / PED_KP
            pad = torch.randn(B, (T - 1) * PED_KP, device=device)
            nll = -torch.cat((pad, lnp_ped), dim=1)
            nll = (nll.view(B, T, PED_KP) * conf[..., psl]).flatten(start_dim=1)
        probs = torch.cat((probs, nll), dim=0)
    return probs.cpu().numpy().squeeze().copy(order="C")


def load_ped_centroids(pose_json, W, H):
    """{frame_idx: [ped centroid (2,) in [0,1], ...]} from a tracked_person.json."""
    d = json.load(open(pose_json))
    out: dict[int, list] = {}
    for pid, frames in d.items():
        if isinstance(frames, list):
            merged = {}
            for sub in frames:
                merged.update(sub)
            frames = merged
        for fk, rec in frames.items():
            kp = np.array(rec["keypoints"], dtype=np.float64).reshape(-1, 3)
            xy = kp[:, :2]
            w = kp[:, 2]
            c = (xy * w[:, None]).sum(0) / w.sum() if w.sum() > 0 else xy.mean(0)
            out.setdefault(int(fk), []).append(np.array([c[0] / W, c[1] / H]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--proximity_cache", default="colab_results/results/context_cache")
    ap.add_argument("--oriented_cache", default="data/ShanghaiTech/context_oriented")
    ap.add_argument("--context_mode", choices=["none", "proximity", "vehicle", "scene"], default="none")
    ap.add_argument("--max_vehicles", type=int, default=2)
    ap.add_argument("--kv", type=int, default=6)
    ap.add_argument("--scene_readout", choices=["ped", "joint"], default="joint",
                    help="scene model: joint scene NLL (default; captures "
                         "vehicle-intrinsic hazards) or pedestrian keypoints only")
    ap.add_argument("--ped_first", action="store_true",
                    help="scene model trained with pedestrian-first ordering "
                         "(ablation that removes pedestrian-given-vehicles "
                         "conditioning); matches the trained block layout")
    ap.add_argument("--counterfactual", action="store_true",
                    help="context models: compare score with real vs zeroed "
                         "context (vehicle removed); the difference isolates the "
                         "vehicle's effect on pedestrian likelihood (interaction "
                         "vs mere object presence).")
    ap.add_argument("--film_cov", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seg_len", type=int, default=24)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--taus", type=float, nargs="+", default=[2.0, 3.0, 5.0, 10.0])
    ap.add_argument("--sigma", type=int, default=0,
                    help="test-time temporal smoothing of the anomaly score")
    a = ap.parse_args()

    parser = init_parser()
    args = parser.parse_args(["--dataset", "ShanghaiTech", "--data_dir", a.data_dir,
                              "--device", a.device, "--seg_len", str(a.seg_len),
                              "--batch_size", str(a.batch_size), "--num_workers", "0"])
    args, _ = init_sub_args(args)

    is_ctx = a.context_mode in ("proximity", "vehicle")
    is_scene = a.context_mode == "scene"
    base = SkeletonSequenceDataset(args.pose_path["test"], path_to_vid_dir=args.vid_path["test"],
                                   evaluate=True, filter_conf=args.filter_conf,
                                   seg_len=a.seg_len, dataset="ShanghaiTech",
                                   train_seg_conf_th=0.0, seg_stride=1,
                                   vid_path=args.vid_path["test"], split="test")
    n_kp = 2 * 18 * a.seg_len
    hidden = [n_kp - 2]
    spec = ContextSpec(mode="vehicle" if (a.context_mode in ("vehicle", "scene")) else "proximity",
                       max_vehicles=a.max_vehicles, kv=a.kv, ped_first=a.ped_first)

    if is_scene:
        n_prime = PED_KP + a.max_vehicles * a.kv
        ds = SceneSkeletonDataset(base, os.path.join(a.oriented_cache, "test"), spec=spec).precompute()
        n_in = 2 * n_prime * a.seg_len
        model = PartialAutoregressiveSceneFC(dim=n_in, n_current_kp=n_prime,
                                             hidden_dims=[n_in - 2], droppout=0.0)
        loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False)
    elif is_ctx:
        cache = a.oriented_cache if a.context_mode == "vehicle" else a.proximity_cache
        ds = ContextualSkeletonSequenceDataset(base, os.path.join(cache, "test"), spec=spec).precompute()
        model = PartialAutoregressiveContextFC(dim=n_kp, ctx_dim=spec.dim * a.seg_len,
                                               hidden_dims=hidden, droppout=0.0, film_cov=a.film_cov)
        loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False)
    else:
        model = PartialAutoregressiveFC(dim=n_kp, hidden_dims=hidden, droppout=0.0)
        loader = DataLoader(base, batch_size=a.batch_size, shuffle=False)

    ckpt = torch.load(a.checkpoint, map_location=a.device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    print(f"[eval] loaded {a.checkpoint} (epoch {ckpt.get('epoch','?')})")

    # --- per-frame proximity in the SAME clip order score_anomalies uses ---
    W, H = spec.img_wh
    coco_ids = tuple(HAZARD_CLASSES.values())
    gt_dir = os.path.join(a.data_dir, "ShanghaiTech", "gt", "test_frame_mask")
    pcache = os.path.join(a.proximity_cache, "test")
    prox_chunks = []
    clip_keys, frame_counts = [], []     # for the per-scene AUROC breakdown
    for clip in sorted(f for f in os.listdir(gt_dir) if f.endswith(".npy")):
        key = clip.split(".")[0]
        n_frames = int(np.load(os.path.join(gt_dir, clip)).shape[0])
        clip_keys.append(key)
        frame_counts.append(n_frames)
        det_path = os.path.join(pcache, f"{key}.npy")
        pj = os.path.join(args.pose_path["test"], f"{key}_alphapose_tracked_person.json")
        if not (os.path.exists(det_path) and os.path.exists(pj)):
            prox_chunks.append(np.zeros(n_frames))
            continue
        dets = list(np.load(det_path, allow_pickle=True))
        peds = load_ped_centroids(pj, W, H)
        prox_chunks.append(clip_frame_proximity(dets, peds, n_frames, coco_ids))
    proximity = np.concatenate(prox_chunks)

    # --- counterfactual interaction probe (W1/W2) ---
    if a.counterfactual and is_ctx:
        kp_real = score_test(model, loader, a.device, is_ctx, zero_ctx=False)
        kp_cf = score_test(model, loader, a.device, is_ctx, zero_ctx=True)
        _, s_real, gt = score_anomalies(kp_real, base.metadata, args=args, split="test", ret_gt=True)
        _, s_cf, _ = score_anomalies(kp_cf, base.metadata, args=args, split="test", ret_gt=True)
        s_int = s_real - s_cf            # the vehicle's effect on pedestrian likelihood
        assert len(proximity) == len(s_real) == len(gt)
        print("\n=== Counterfactual interaction probe (vehicle removed = zeroed context) ===")
        print(f"{'variant':<26}{'full AUROC':>12}{'veh-hazard':>12}{'prox rho':>10}")
        for name, s in [("with context (real)", s_real),
                        ("vehicle removed (cf)", s_cf),
                        ("interaction (real-cf)", s_int)]:
            m = hazard_subset_metrics(s, gt, proximity, tau=3.0)
            rho = score_proximity_correlation(s, proximity, gt)["spearman"]
            print(f"{name:<26}{m['full']['auroc']:>12.4f}"
                  f"{m['vehicle_hazard']['auroc']:>12.4f}{rho:>10.3f}")
        print("\nReading: if 'interaction (real-cf)' carries real vehicle-hazard "
              "AUROC and proximity rho, the\ncontext encodes pedestrian-vehicle "
              "interaction beyond object presence; if ~chance, the gain\nis not "
              "interaction-specific.")
        return

    scores_kp = (score_test_scene(model, loader, a.device, a.scene_readout, a.ped_first) if is_scene
                 else score_test(model, loader, a.device, is_ctx))
    auc, scores, gt = score_anomalies(scores_kp, base.metadata, args=args, split="test",
                                      ret_gt=True, sigma=a.sigma)
    print(f"[eval] full-set AUROC (seeker scorer, sigma={a.sigma}): {auc:.4f}")
    assert len(proximity) == len(scores) == len(gt), (len(proximity), len(scores), len(gt))

    # balanced near-vehicle metric (W8): 50/50 normal vs anomalous among near frames
    from ccskde.eval.hazard import balanced_hazard_auroc
    bal = balanced_hazard_auroc(scores, gt, proximity, tau=3.0)

    print("\n=== Hazard-subset metrics ===")
    for t in a.taus:
        m = hazard_subset_metrics(scores, gt, proximity, t)
        ir, vh = m["interaction_regime"], m["vehicle_hazard"]
        print(f"  tau={t:5.1f} frac_near={m['frac_near']:.3f} | "
              f"interaction AUROC={ir['auroc']:.4f} (n={ir['n']}) | "
              f"vehicle_hazard AUROC={vh['auroc']:.4f} (n_pos={vh['n_pos']})")
    print(f"  balanced near-vehicle AUROC (tau=3): {bal['auroc']:.4f} "
          f"[{bal['lo']:.4f}, {bal['hi']:.4f}] 95% CI, n_per_class={bal['n_per_class']}")
    corr = score_proximity_correlation(scores, proximity, gt, anomalous_only=True)
    print(f"\n  score-vs-proximity (anomalous frames): "
          f"spearman={corr['spearman']:.3f} pearson={corr['pearson']:.3f} n={corr['n']}")

    # --- extra frame-level report metrics (AUPRC / EER / FAR / per-scene) ---
    from ccskde.eval.hazard import (
        auroc_ap, equal_error_rate, false_alarm_at_recall, per_scene_auroc,
    )
    ap = auroc_ap(scores, gt)
    eer = equal_error_rate(scores, gt)
    far90 = false_alarm_at_recall(scores, gt, 0.90)
    ps = per_scene_auroc(scores, gt, clip_keys, frame_counts)
    print("\n=== Extra frame-level metrics ===")
    print(f"  AUROC={ap['auroc']:.4f}  AUPRC={ap['ap']:.4f} "
          f"(base rate {ap['base_rate']:.3f})  EER={eer['eer']:.4f}")
    print(f"  false-alarm rate @ 90% recall: {far90['fpr']:.4f}")
    print(f"  per-scene AUROC: micro={ps['micro']:.4f}  macro={ps['macro']:.4f}  "
          f"(over {len(ps['per_scene'])} scenes)")
    worst = sorted((v, k) for k, v in ps["per_scene"].items() if not math.isnan(v))[:3]
    print("    weakest scenes: " + ", ".join(f"{k}={v:.3f}" for v, k in worst))


if __name__ == "__main__":
    main()
