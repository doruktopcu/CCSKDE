"""Efficiency comparison (W11): parameter counts + per-frame inference latency
for the baseline, context (car-matrix + FiLM), and unified scene models.

Run:  set PYTHONPATH=%CD% && .venv\\Scripts\\python scripts\\measure_efficiency.py
"""
from __future__ import annotations

import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEKER = os.path.join(ROOT, "seeker")
for p in (SEEKER, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.partial_autoregressive import PartialAutoregressiveFC      # noqa: E402
from ccskde.models import (                                            # noqa: E402
    PartialAutoregressiveContextFC, PartialAutoregressiveSceneFC,
)
from ccskde.context import ContextSpec                                 # noqa: E402
from ccskde.context.scene import PED_KP                                # noqa: E402


def n_params(m):
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def latency_ms(fn, x, c, device, iters=30, warmup=5):
    for _ in range(warmup):
        fn(x, c) if c is not None else fn(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(x, c) if c is not None else fn(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T, B = 24, 1024
    spec = ContextSpec(mode="vehicle", max_vehicles=2, kv=6)
    n_kp = 2 * 18 * T                       # 864
    n_prime = PED_KP + spec.max_vehicles * spec.kv   # 30
    hid = [n_kp - 2]

    base = PartialAutoregressiveFC(dim=n_kp, hidden_dims=hid, droppout=0.0).to(device).eval()
    ctx = PartialAutoregressiveContextFC(dim=n_kp, ctx_dim=spec.dim * T,
                                         hidden_dims=hid, droppout=0.0, film_cov=True).to(device).eval()
    n_in_scene = 2 * n_prime * T            # 1440
    scene = PartialAutoregressiveSceneFC(dim=n_in_scene, n_current_kp=n_prime,
                                         hidden_dims=[n_in_scene - 2], droppout=0.0).to(device).eval()

    x_kp = torch.randn(B, T, 18, 2, device=device)
    c_ctx = torch.randn(B, T, spec.dim, device=device)
    x_scene = torch.randn(B, T, n_prime, 2, device=device)

    rows = [
        ("baseline (SeeKer)", base, x_kp, None),
        ("context (car-matrix + FiLM)", ctx, x_kp, c_ctx),
        ("unified scene-SKDE", scene, x_scene, None),
    ]

    print(f"device={device}  batch={B}  seg_len={T}\n")
    print(f"{'model':<32}{'params':>12}{'ms/batch':>12}{'frames/s':>14}")
    print("-" * 70)
    for name, m, x, c in rows:
        p = n_params(m)
        ms = latency_ms(m, x, c, device)
        # each batch item is one (pedestrian) segment of T frames -> B*T frame-scores
        fps = (B * T) / (ms / 1000.0)
        print(f"{name:<32}{p/1e6:>10.2f}M{ms:>12.2f}{fps:>14,.0f}")
    print("\n(frames/s = batch_size * seg_len / forward time; offline YOLO "
          "extraction excluded — it is one-time preprocessing.)")


if __name__ == "__main__":
    main()
