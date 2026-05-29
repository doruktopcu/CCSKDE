"""Scene-skeleton builder for the unified model.

Assembles one augmented skeleton per frame:

    Z_t = [ vehicle_1 (6 kp), ..., vehicle_M (6 kp), pedestrian (18 kp) ]   -> N' = 18 + 6M

Vehicles are ordered FIRST so that, under SeeKer's autoregressive order, the
pedestrian keypoints are predicted *given* the surrounding vehicles — the
pedestrian-vehicle interaction is captured by the conditioning. Everything is
emitted in PIXEL coordinates (x, y, conf); the dataset then applies SeeKer's
`normalize_pose` jointly over all N' keypoints, exactly as for a normal pose.

Missing vehicles (fewer than M present) are zero-filled with confidence 0, so
they neither move the normalisation much nor contribute to the anomaly score.
"""
from __future__ import annotations

import numpy as np

from .config import ContextSpec
from .vehicle import oriented_box_keypoints

PED_KP = 18


def _veh_keypoints_norm(row: np.ndarray, kv: int) -> tuple[np.ndarray, np.ndarray]:
    """(kv,2) vehicle keypoints (normalised [0,1]) + center, from a cache row."""
    if row.shape[0] >= 2 + kv * 2:                       # oriented schema
        kp = row[2:2 + kv * 2].reshape(kv, 2).astype(np.float32)
        return kp, kp[kv - 2]
    cx, cy, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
    return oriented_box_keypoints(cx, cy, w, h, 0.0), np.array([cx, cy], np.float32)


def build_scene_skeleton(
    ped_xy_px: np.ndarray,        # (T, 18, 2) pedestrian keypoints, PIXELS
    ped_conf: np.ndarray,         # (T, 18)
    dets_per_frame: list,         # per-video oriented detection cache (or None)
    frame_indices: np.ndarray,    # (T,) video frame index for each segment frame
    spec: ContextSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (aug_xy_px (T, N', 2), aug_conf (T, N')) with vehicles first."""
    T = ped_xy_px.shape[0]
    M, Kv = spec.max_vehicles, spec.kv
    W, H = spec.img_wh
    n_prime = PED_KP + M * Kv
    aug_xy = np.zeros((T, n_prime, 2), dtype=np.float32)
    aug_cf = np.zeros((T, n_prime), dtype=np.float32)

    # pedestrian block goes last
    ped0 = M * Kv
    aug_xy[:, ped0:] = ped_xy_px
    aug_cf[:, ped0:] = ped_conf

    if dets_per_frame is None:
        return aug_xy, aug_cf

    keep = set(spec.coco_ids)
    # pedestrian centroid per frame in normalised coords (for nearest selection)
    ped_norm = ped_xy_px.copy()
    ped_norm[..., 0] /= float(W)
    ped_norm[..., 1] /= float(H)
    for t in range(T):
        f = int(frame_indices[t])
        if f < 0 or f >= len(dets_per_frame):
            continue
        dets = dets_per_frame[f]
        if dets is None or len(dets) == 0:
            continue
        rows = [d for d in dets if int(d[0]) in keep]
        if not rows:
            continue
        cw = ped_conf[t]
        cen = ((ped_norm[t] * cw[:, None]).sum(0) / cw.sum()
               if cw.sum() > 0 else ped_norm[t].mean(0))
        kps_centers = [_veh_keypoints_norm(np.asarray(r), Kv) for r in rows]
        dists = [float(np.linalg.norm(c - cen)) for _, c in kps_centers]
        order = np.argsort(dists)[:M]
        for slot, idx in enumerate(order):
            kp_norm, _ = kps_centers[idx]
            kp_px = kp_norm.copy()
            kp_px[:, 0] *= float(W)
            kp_px[:, 1] *= float(H)
            base = slot * Kv
            aug_xy[t, base:base + Kv] = kp_px
            aug_cf[t, base:base + Kv] = float(rows[idx][1])   # detection conf
    return aug_xy, aug_cf
