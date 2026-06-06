"""W4 probe: sweep test-time temporal smoothing (sigma) for a baseline checkpoint.

ShanghaiTech AUROC is highly sensitive to temporal score smoothing, which our
default runs effectively skip (sigma=0). This re-scores a checkpoint at several
sigmas WITHOUT retraining, to see how much of the gap to the published 0.855 is
explained by smoothing.

Run:  set PYTHONPATH=%CD% && .venv\\Scripts\\python scripts\\baseline_repro.py --checkpoint <best.pth>
"""
from __future__ import annotations

import argparse
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

from args import init_parser, init_sub_args                          # noqa: E402
from dataset import SkeletonSequenceDataset                          # noqa: E402
from validation import score_anomalies                               # noqa: E402
from models.partial_autoregressive import PartialAutoregressiveFC    # noqa: E402
from torch.utils.data import DataLoader                              # noqa: E402


def joint_lnp(model, x):
    B, T, N, A = x.shape
    out = model(x)
    mu, logvar = torch.chunk(out, 2, dim=1)
    mu = mu.reshape(B, T * N, A)[:, (-N - 1):-1]
    logvar = logvar.reshape(B, T * N, A)[:, (-N - 1):-1].clamp(-10, 10)
    var = torch.exp(logvar)
    E_inv = torch.diag_embed(1.0 / var)
    x = x.reshape(B, T * N, A)
    x_tgt = x[:, -N:]
    return -0.5 * (torch.einsum("bti,btij,btj->bt", x_tgt - mu, E_inv, x_tgt - mu)
                   + torch.sum(logvar, dim=-1) + A * math.log(2 * math.pi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seg_len", type=int, default=24)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--sigmas", type=int, nargs="+", default=[0, 3, 5, 10, 15, 20, 30, 40, 50])
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
    n_kp = 2 * 18 * a.seg_len
    model = PartialAutoregressiveFC(dim=n_kp, hidden_dims=[n_kp - 2], droppout=0.0)
    ckpt = torch.load(a.checkpoint, map_location=a.device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval().to(a.device)
    print(f"[repro] {a.checkpoint} (epoch {ckpt.get('epoch','?')})")

    loader = DataLoader(base, batch_size=a.batch_size, shuffle=False)
    probs = torch.empty(0).to(a.device)
    for data_arr in loader:
        xc = torch.permute(data_arr[0], (0, 2, 3, 1))
        x = xc[..., :2].to(a.device).float()
        conf = xc[..., -1].to(a.device)
        B, T, N, _ = x.shape
        with torch.no_grad():
            lnp = joint_lnp(model, x)
            pad = torch.randn(B, (T - 1) * N, device=a.device)
            nll = -torch.cat((pad, lnp), dim=1)
            nll = (nll.view(B, T, N) * conf).flatten(start_dim=1)
        probs = torch.cat((probs, nll), dim=0)
    scores_kp = probs.cpu().numpy().squeeze().copy(order="C")

    print(f"\n{'sigma':>6}{'AUROC':>10}")
    print("-" * 16)
    for s in a.sigmas:
        auc = score_anomalies(scores_kp.copy(), base.metadata, args=args, split="test", sigma=s)
        print(f"{s:>6}{auc:>10.4f}")
    print("\n(published SeeKer ShanghaiTech AUROC = 0.855)")


if __name__ == "__main__":
    main()
