"""Tests for the unified scene-SKDE: generalised MADE AR-preservation + builder.

Run:  python tests/test_scene_model.py
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ccskde.models import PartialAutoregressiveSceneFC
from ccskde.context import ContextSpec
from ccskde.context.scene import build_scene_skeleton, PED_KP
from ccskde.context.vehicle import oriented_box_keypoints

torch.manual_seed(0)


def test_scene_made_ar_no_future_leakage():
    """Generalised MADE with N' keypoints must preserve autoregression: a
    current-frame keypoint coord must not depend on equal/higher-degree coords
    of the same frame (so a pedestrian keypoint cannot see 'future' keypoints)."""
    T, M, Kv = 2, 2, 6
    Np = PED_KP + M * Kv                 # 30
    A = 2
    D = A * Np * T
    model = PartialAutoregressiveSceneFC(dim=D, n_current_kp=Np,
                                         hidden_dims=[D - 2], droppout=0.0).double()
    model.eval()
    deg = np.repeat(np.arange(0, D, 2), 2)
    cur0 = D - A * Np                    # first index of current frame

    x = torch.randn(1, T, Np, A, dtype=torch.float64, requires_grad=True)
    out = model(x)
    mu = torch.chunk(out, 2, dim=1)[0][0]   # (Np*T*? ) -> first half = mu coords
    # mu corresponds to input coords in order; check current-frame block
    violations = 0
    # build Jacobian of the current-frame mu coords wrt inputs
    for o in range(cur0, D):
        g, = torch.autograd.grad(mu[o], x, retain_graph=True)
        g = g.reshape(-1).detach().numpy()
        for q in range(cur0, D):
            if deg[q] >= deg[o] and abs(g[q]) > 1e-9:
                violations += 1
    print(f"scene MADE (N'={Np}) forbidden-region violations: {violations}")
    assert violations == 0


def test_pedestrian_can_see_vehicles():
    """Vehicles are ordered first, so a pedestrian keypoint SHOULD be able to
    depend on vehicle keypoints (the interaction pathway)."""
    T, M, Kv = 2, 2, 6
    Np = PED_KP + M * Kv
    A = 2
    D = A * Np * T
    model = PartialAutoregressiveSceneFC(dim=D, n_current_kp=Np,
                                         hidden_dims=[D - 2], droppout=0.0).double()
    model.eval()
    x = torch.randn(1, T, Np, A, dtype=torch.float64, requires_grad=True)
    out = model(x)
    mu = torch.chunk(out, 2, dim=1)[0][0]
    cur0 = D - A * Np
    veh_block = slice(cur0, cur0 + A * M * Kv)        # current-frame vehicle coords
    ped_block = range(cur0 + A * M * Kv, D)           # current-frame pedestrian coords
    # gradient of a late pedestrian coord wrt current-frame vehicle coords
    o = D - 1                                          # last pedestrian coord
    g, = torch.autograd.grad(mu[o], x, retain_graph=True)
    g = g.reshape(-1).detach().numpy()
    assert np.abs(g[veh_block]).sum() > 1e-9, "pedestrian cannot see vehicles!"
    print("pedestrian-sees-vehicles pathway OK")


def test_scene_builder_layout():
    spec = ContextSpec(mode="vehicle", max_vehicles=2, kv=6)
    W, H = spec.img_wh
    T = 4
    Np = PED_KP + 2 * 6
    # pedestrian near image center
    ped_xy = np.full((T, PED_KP, 2), [[400.0, 240.0]], dtype=np.float32)
    ped_cf = np.ones((T, PED_KP), dtype=np.float32)
    # one car detection per frame (oriented 14-col row), normalised coords
    kp = oriented_box_keypoints(0.5, 0.5, 0.2, 0.1, 0.0).reshape(-1)
    row = np.concatenate([[2, 0.8], kp]).astype(np.float32)
    dets = [row[None] for _ in range(T)]
    fidx = np.arange(T)
    aug_xy, aug_cf = build_scene_skeleton(ped_xy, ped_cf, dets, fidx, spec)
    assert aug_xy.shape == (T, Np, 2) and aug_cf.shape == (T, Np)
    # pedestrian block is last 18 and equals the input pedestrian keypoints
    assert np.allclose(aug_xy[:, 2 * 6:], ped_xy)
    # first vehicle slot populated with conf 0.8, pixel coords (~0.5*W, 0.5*H center)
    assert aug_cf[0, 0] == np.float32(0.8)
    assert abs(aug_xy[0, 4, 0] - 0.5 * W) < 2 and abs(aug_xy[0, 4, 1] - 0.5 * H) < 2
    # second vehicle slot empty (only one detection) -> conf 0
    assert aug_cf[0, 6] == 0.0
    print("scene builder layout OK")


def test_scene_builder_ped_first_ordering():
    """The ped_first ablation must place the pedestrian block FIRST (indices
    0:18) and the vehicles AFTER (18:18+M*Kv) -- the mirror of the default,
    with identical content, only reordered."""
    W = ContextSpec().img_wh[0]
    H = ContextSpec().img_wh[1]
    T, M, Kv = 4, 2, 6
    Np = PED_KP + M * Kv
    ped_xy = np.full((T, PED_KP, 2), [[400.0, 240.0]], dtype=np.float32)
    ped_cf = np.ones((T, PED_KP), dtype=np.float32)
    kp = oriented_box_keypoints(0.5, 0.5, 0.2, 0.1, 0.0).reshape(-1)
    row = np.concatenate([[2, 0.8], kp]).astype(np.float32)
    dets = [row[None] for _ in range(T)]
    fidx = np.arange(T)

    default = ContextSpec(mode="vehicle", max_vehicles=M, kv=Kv)               # veh first
    pedfirst = ContextSpec(mode="vehicle", max_vehicles=M, kv=Kv, ped_first=True)
    dxy, dcf = build_scene_skeleton(ped_xy, ped_cf, dets, fidx, default)
    pxy, pcf = build_scene_skeleton(ped_xy, ped_cf, dets, fidx, pedfirst)
    assert pxy.shape == (T, Np, 2)
    # ped block now FIRST 18
    assert np.allclose(pxy[:, :PED_KP], ped_xy)
    assert pcf[0, 0] == np.float32(1.0)          # a pedestrian conf, not a vehicle
    # first vehicle now begins at index PED_KP, conf 0.8
    assert pcf[0, PED_KP] == np.float32(0.8)
    # same content, mirrored: default ped (last 18) == pedfirst ped (first 18)
    assert np.allclose(dxy[:, M * Kv:], pxy[:, :PED_KP])
    assert np.allclose(dxy[:, :M * Kv], pxy[:, PED_KP:])
    print("scene builder ped_first ordering OK")


def test_scene_builder_missing_cache():
    spec = ContextSpec(mode="vehicle", max_vehicles=2, kv=6)
    T = 3
    ped_xy = np.zeros((T, PED_KP, 2), np.float32)
    ped_cf = np.ones((T, PED_KP), np.float32)
    aug_xy, aug_cf = build_scene_skeleton(ped_xy, ped_cf, None, np.arange(T), spec)
    assert aug_xy.shape[1] == PED_KP + 12 and aug_cf[:, :12].sum() == 0
    print("scene builder missing-cache OK")


if __name__ == "__main__":
    test_scene_made_ar_no_future_leakage()
    test_pedestrian_can_see_vehicles()
    test_scene_builder_layout()
    test_scene_builder_ped_first_ordering()
    test_scene_builder_missing_cache()
    print("\nAll scene-model tests passed.")
