"""Statistically compare two trained models on the SAME ShanghaiTech test frames.

Produces the report's significance numbers:
  * each model's frame-level AUROC / AUPRC / EER,
  * DeLong's test for two CORRELATED AUCs (the correct paired test — accounts
    for the shared test frames, unlike an unpaired t-test over seed means),
  * a paired bootstrap of the AUROC difference with a 95% CI.

Typical use (baseline vs car-matrix):
    set PYTHONPATH=%CD%
    .venv\\Scripts\\python scripts\\compare_models.py ^
      --ckpt_a colab_results\\results_v2\\checkpoints\\baseline_best.pth --mode_a none ^
      --ckpt_b colab_results\\results_v2\\checkpoints\\vehicle_best.pth  --mode_b vehicle ^
      --sigma 0 --device cuda

Both models are scored through the identical SeeKer pipeline, so the comparison
is apples-to-apples.
"""
from __future__ import annotations

import argparse
import os
import sys

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
from models.partial_autoregressive import PartialAutoregressiveFC  # noqa: E402
from torch.utils.data import DataLoader                      # noqa: E402

from ccskde.context import ContextSpec                       # noqa: E402
from ccskde.context.scene import PED_KP                      # noqa: E402
from ccskde.data import ContextualSkeletonSequenceDataset    # noqa: E402
from ccskde.data.scene_dataset import SceneSkeletonDataset   # noqa: E402
from ccskde.models import (                                  # noqa: E402
    PartialAutoregressiveContextFC, PartialAutoregressiveSceneFC,
)
from ccskde.eval.hazard import (                             # noqa: E402
    delong_roc_test, paired_bootstrap_auroc, auroc_ap, equal_error_rate,
)
# reuse the exact scorers used by evaluate_hazard.py (no duplication of logic)
from scripts.evaluate_hazard import score_test, score_test_scene  # noqa: E402


def build_model_loader(mode, ckpt_path, base, args, a, film_cov=False):
    """Construct the model + test loader for one checkpoint, mirroring
    evaluate_hazard.py exactly so scores are directly comparable."""
    n_kp = 2 * 18 * a.seg_len
    hidden = [n_kp - 2]
    spec = ContextSpec(mode="vehicle" if mode in ("vehicle", "scene") else "proximity",
                       max_vehicles=a.max_vehicles, kv=a.kv)
    if mode == "scene":
        n_prime = PED_KP + a.max_vehicles * a.kv
        ds = SceneSkeletonDataset(base, os.path.join(a.oriented_cache, "test"), spec=spec).precompute()
        n_in = 2 * n_prime * a.seg_len
        model = PartialAutoregressiveSceneFC(dim=n_in, n_current_kp=n_prime,
                                             hidden_dims=[n_in - 2], droppout=0.0)
        loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False)
        kind = "scene"
    elif mode in ("proximity", "vehicle"):
        cache = a.oriented_cache if mode == "vehicle" else a.proximity_cache
        ds = ContextualSkeletonSequenceDataset(base, os.path.join(cache, "test"), spec=spec).precompute()
        model = PartialAutoregressiveContextFC(dim=n_kp, ctx_dim=spec.dim * a.seg_len,
                                               hidden_dims=hidden, droppout=0.0, film_cov=film_cov)
        loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False)
        kind = "ctx"
    else:
        model = PartialAutoregressiveFC(dim=n_kp, hidden_dims=hidden, droppout=0.0)
        loader = DataLoader(base, batch_size=a.batch_size, shuffle=False)
        kind = "base"
    ckpt = torch.load(ckpt_path, map_location=a.device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    return model, loader, kind


def frame_scores(model, loader, kind, a):
    kp = (score_test_scene(model, loader, a.device, a.scene_readout) if kind == "scene"
          else score_test(model, loader, a.device, kind == "ctx"))
    _, scores, gt = score_anomalies(kp, model_meta, args=ARGS, split="test",
                                    ret_gt=True, sigma=a.sigma)
    return scores, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_a", required=True)
    ap.add_argument("--mode_a", choices=["none", "proximity", "vehicle", "scene"], default="none")
    ap.add_argument("--film_a", action="store_true")
    ap.add_argument("--ckpt_b", required=True)
    ap.add_argument("--mode_b", choices=["none", "proximity", "vehicle", "scene"], default="vehicle")
    ap.add_argument("--film_b", action="store_true")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--proximity_cache", default="colab_results/results/context_cache")
    ap.add_argument("--oriented_cache", default="data/ShanghaiTech/context_oriented")
    ap.add_argument("--max_vehicles", type=int, default=2)
    ap.add_argument("--kv", type=int, default=6)
    ap.add_argument("--scene_readout", choices=["ped", "joint"], default="joint")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seg_len", type=int, default=24)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--sigma", type=int, default=0)
    ap.add_argument("--n_boot", type=int, default=2000)
    a = ap.parse_args()

    parser = init_parser()
    args = parser.parse_args(["--dataset", "ShanghaiTech", "--data_dir", a.data_dir,
                              "--device", a.device, "--seg_len", str(a.seg_len),
                              "--batch_size", str(a.batch_size), "--num_workers", "0"])
    args, _ = init_sub_args(args)
    base = SkeletonSequenceDataset(args.pose_path["test"], path_to_vid_dir=args.vid_path["test"],
                                   evaluate=True, filter_conf=args.filter_conf,
                                   seg_len=a.seg_len, dataset="ShanghaiTech",
                                   train_seg_conf_th=0.0, seg_stride=1,
                                   vid_path=args.vid_path["test"], split="test")
    global ARGS, model_meta
    ARGS, model_meta = args, base.metadata

    out = {}
    for tag, mode, ckpt, film in [("A", a.mode_a, a.ckpt_a, a.film_a),
                                  ("B", a.mode_b, a.ckpt_b, a.film_b)]:
        model, loader, kind = build_model_loader(mode, ckpt, base, args, a, film_cov=film)
        s, gt = frame_scores(model, loader, kind, a)
        out[tag] = (s, gt, mode, ckpt)
        m = auroc_ap(s, gt)
        e = equal_error_rate(s, gt)
        print(f"[{tag}] {mode:<9} {os.path.basename(ckpt):<28} "
              f"AUROC={m['auroc']:.4f} AUPRC={m['ap']:.4f} EER={e['eer']:.4f}")
        del model, loader
        torch.cuda.empty_cache()

    (sa, gt, _, _), (sb, gtb, _, _) = out["A"], out["B"]
    assert len(sa) == len(sb) == len(gt) and np.array_equal(gt, gtb), "frame mismatch"

    d = delong_roc_test(gt, sa, sb)               # B - A in our convention -> negate diff
    pb = paired_bootstrap_auroc(gt, sa, sb, n_boot=a.n_boot)
    print("\n=== Paired comparison (B vs A) on identical test frames ===")
    print(f"  AUROC  A={d['auc_a']:.4f}  B={d['auc_b']:.4f}  "
          f"deltaB-A={d['auc_b']-d['auc_a']:+.4f}")
    print(f"  DeLong:  z={d['z']:+.3f}  p={d['p_value']:.3e}  "
          f"({'significant' if d['p_value']<0.05 else 'n.s.'} at 0.05)")
    print(f"  paired bootstrap deltaB-A: {pb['mean_diff']:+.4f} "
          f"[{pb['lo']:+.4f}, {pb['hi']:+.4f}] 95% CI  "
          f"(P(B<=A)={pb['p_b_gt_a']:.4f})")
    print("\n(Sigma={}; both models scored through the identical SeeKer pipeline.)"
          .format(a.sigma))


if __name__ == "__main__":
    main()
