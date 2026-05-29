"""Autoregressive-preservation tests for the context-conditioned MADE.

The central methodological claim of CCSKDE is that injecting the context
(input-layer shortcut + optional FiLM covariance gate) does NOT leak future
keypoints into the prediction of an earlier keypoint — MADE's autoregressive
structure must be preserved bit-for-bit on the keypoint inputs, while the
context inputs are fully reachable.

We verify this empirically with autograd Jacobians.

Run:  python tests/test_ar_mask.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ccskde.models import PartialAutoregressiveContextFC

torch.manual_seed(0)

T, N, A = 2, 18, 2          # 2 frames so there is a past frame + current frame
D = A * N * T               # 72 keypoint coords
DCTX = 10 * T               # arbitrary context width
HIDDEN = [D - 2]


def _build(film_cov: bool):
    m = PartialAutoregressiveContextFC(
        dim=D, ctx_dim=DCTX, hidden_dims=HIDDEN, droppout=0.0, film_cov=film_cov
    ).double()
    m.eval()
    return m


def _current_frame_degrees():
    # SeeKer degree assignment on keypoint coords: repeat(arange(0,D,2),2)
    deg = np.repeat(np.arange(0, D, 2), 2)
    return deg  # length D; last frame = indices [D-36, D)


def _mu_jacobian(model):
    """Return d mu / d x_kp as (D_out, D_in) for one random sample."""
    x = torch.randn(1, T, N, A, dtype=torch.float64, requires_grad=True)
    c = torch.randn(1, T, 10, dtype=torch.float64, requires_grad=True)
    out = model(x, c)                       # (1, 2D)
    mu = out[0, :D]                          # keypoint means
    J = torch.zeros(D, D, dtype=torch.float64)
    for o in range(D):
        g, = torch.autograd.grad(mu[o], x, retain_graph=True)
        J[o] = g.reshape(-1)
    return J.detach().numpy()


def test_ar_no_future_leakage():
    """mu of a keypoint coord must not depend on equal/higher-degree coords
    within the SAME (last) frame."""
    deg = _current_frame_degrees()
    model = _build(film_cov=False)
    J = _mu_jacobian(model)

    last0 = D - A * N                       # first index of the last frame
    violations = 0
    for o in range(last0, D):               # outputs in the last frame
        for q in range(last0, D):           # inputs in the last frame
            if deg[q] >= deg[o]:            # q is "at or after" o -> forbidden
                if abs(J[o, q]) > 1e-9:
                    violations += 1
    print(f"AR forbidden-region violations: {violations}")
    assert violations == 0


def test_past_frame_reachable():
    """mu of the last frame may depend on past-frame keypoints (reachable)."""
    model = _build(film_cov=False)
    J = _mu_jacobian(model)
    last0 = D - A * N
    past = J[last0:D, :last0]               # last-frame outputs vs past inputs
    frac = (np.abs(past) > 1e-9).mean()
    print(f"past-frame reachability: {frac:.2%} of entries non-zero")
    assert frac > 0.5


def test_context_reachable():
    """Every context input must influence the output (shortcut wired)."""
    model = _build(film_cov=False)
    x = torch.randn(1, T, N, A, dtype=torch.float64, requires_grad=True)
    c = torch.randn(1, T, 10, dtype=torch.float64, requires_grad=True)
    out = model(x, c)
    out.sum().backward()
    nz = (c.grad.abs() > 1e-12).sum().item()
    print(f"context inputs reachable: {nz}/{c.numel()}")
    assert nz == c.numel()


def test_film_identity_init_preserves_ar():
    """With FiLM (zero-init), output equals the shortcut-only model at init,
    and the keypoint AR pattern is unchanged."""
    torch.manual_seed(0)
    base = _build(film_cov=False)
    torch.manual_seed(0)
    film = _build(film_cov=True)
    # copy the shared MADE weights so only the FiLM head differs
    film.model.load_state_dict(base.model.state_dict())

    x = torch.randn(1, T, N, A, dtype=torch.float64)
    c = torch.randn(1, T, 10, dtype=torch.float64)
    with torch.no_grad():
        assert torch.allclose(base(x, c), film(x, c), atol=1e-10), \
            "FiLM zero-init must be identity"

    # FiLM must still let context modulate the covariance (logvar) gradient
    c2 = torch.randn(1, T, 10, dtype=torch.float64, requires_grad=True)
    out = film(torch.randn(1, T, N, A, dtype=torch.float64), c2)
    logvar = out[0, D:]
    logvar.sum().backward()
    # after a step the FiLM weights are zero so grad wrt c is zero AT init;
    # instead verify the FiLM head exists and is wired by perturbing its weights
    for p in film.film.parameters():
        torch.nn.init.normal_(p, std=0.1)
    c3 = torch.randn(1, T, 10, dtype=torch.float64, requires_grad=True)
    out = film(torch.randn(1, T, N, A, dtype=torch.float64), c3)
    out[0, D:].sum().backward()
    assert (c3.grad.abs() > 1e-12).any(), "FiLM must route context to covariance"
    print("FiLM identity-init + covariance routing OK")


if __name__ == "__main__":
    test_ar_no_future_leakage()
    test_past_frame_reachable()
    test_context_reachable()
    test_film_identity_init_preserves_ar()
    print("\nAll AR-mask tests passed.")
