"""Oriented vehicle-keypoint ("car skeleton") context — the CCSKDE uplift.

Motivation
----------
The original context vector collapsed each hazard to a per-class
``[1/d_min, count]`` proximity scalar (``build.py``). That throws away the
*geometry* of the hazard: its orientation, extent, which side of the
pedestrian it is on, and how it is moving. This module replaces that with a
compact **oriented keypoint set per vehicle** — "like a box, but better" —
expressed in pedestrian-relative, scale-normalised coordinates so it is
invariant to scene scale and camera viewpoint (report challenge iii).

Per vehicle we emit ``K_v`` 2D keypoints derived from an *oriented* box
(``cv2.minAreaRect`` over the YOLOv11-seg mask, which—unlike the axis-aligned
detection box—encodes heading):

    4 oriented corners + center + front-edge midpoint   (K_v = 6)

We keep the ``M`` nearest hazards to the pedestrian centroid and append, per
vehicle, an inverse-distance scalar and a normalised class id. Missing
vehicles are zero-padded, so the dimensionality is fixed at::

    dim = M * (K_v * 2 + 2)

All inputs are in normalised ``[0,1]`` image coordinates (see ``detect.py``);
outputs are ``(kp - ped_centroid) / ped_scale`` so the representation is
translation- and scale-invariant.

This module consumes either of two detection-cache schemas (one row per
detection), auto-detected by row width:

    "oriented" (14 cols): cls, conf, then 6 keypoints (x,y) precomputed in
        normalised [0,1] coords — 4 oriented corners + center + heading.
        The oriented corners are produced in *pixel* space by the seg
        extractor (``cv2.minAreaRect``/``boxPoints``) and then normalised,
        so the box orientation is geometrically faithful.

    "box" (6 cols): cls, conf, cx, cy, w, h — the original axis-aligned box
        cache. We synthesise an axis-aligned 6-keypoint "box pseudo-skeleton"
        (angle = 0). This is the zero-cost fallback / ablation that reuses the
        existing YOLO box caches with no re-extraction.
"""
from __future__ import annotations

import math

import numpy as np

from .config import ContextSpec

EPS = 1e-3
DIAG_NORM = math.sqrt(2.0)
PED_SCALE_FLOOR = 0.02  # ~2% of the diagonal; guards tiny/degenerate skeletons


def oriented_box_keypoints(cx: float, cy: float, w: float, h: float,
                           angle_rad: float) -> np.ndarray:
    """Return K_v=6 keypoints of an oriented box in absolute (normalised) coords.

    Order: 4 corners (CCW from top-left in the box's local frame), center,
    front-edge midpoint (the +x local axis — a stable heading proxy).
    """
    dx, dy = w / 2.0, h / 2.0
    local = np.array(
        [[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy],  # corners
         [0.0, 0.0],                                   # center
         [dx, 0.0]],                                   # front-edge midpoint (heading)
        dtype=np.float32,
    )
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return local @ rot.T + np.array([cx, cy], dtype=np.float32)


def _ped_scale(keypoints_xy: np.ndarray) -> float:
    """Pedestrian size = bbox diagonal of the skeleton (normalised coords)."""
    valid = keypoints_xy[(keypoints_xy != 0).any(axis=1)]
    if valid.shape[0] < 2:
        return PED_SCALE_FLOOR
    span = valid.max(axis=0) - valid.min(axis=0)
    return max(float(np.hypot(span[0], span[1])), PED_SCALE_FLOOR)


def vehicle_matrix_for_frame(
    detections: np.ndarray,   # (n_dets, 6 or 7): cls,conf,cx,cy,w,h[,angle]
    centroid: np.ndarray,     # (2,) pedestrian centroid, normalised [0,1]
    ped_scale: float,         # pedestrian size for scale normalisation
    spec: ContextSpec,
) -> np.ndarray:
    """Pedestrian-relative oriented-keypoint matrix for the M nearest hazards.

    Returns a flat vector of length ``spec.vehicle_dim``.
    """
    M, Kv = spec.max_vehicles, spec.kv
    per_veh = Kv * 2 + 2
    out = np.zeros(M * per_veh, dtype=np.float32)
    if detections.shape[0] == 0:
        return out

    keep = set(spec.coco_ids)
    rows = [d for d in detections if int(d[0]) in keep]
    if not rows:
        return out
    rows = np.stack(rows, axis=0)

    def _keypoints(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (Kv,2) keypoints and the (2,) box center for one detection."""
        if d.shape[0] >= 2 + Kv * 2:                 # "oriented" schema
            kp = d[2:2 + Kv * 2].reshape(Kv, 2).astype(np.float32)
            return kp, kp[Kv - 2]                    # center is keypoint index Kv-2
        cx, cy, w, h = float(d[2]), float(d[3]), float(d[4]), float(d[5])
        return oriented_box_keypoints(cx, cy, w, h, 0.0), np.array([cx, cy], np.float32)

    centers = np.stack([_keypoints(d)[1] for d in rows], axis=0)
    dists = np.linalg.norm(centers - centroid[None, :], axis=1) / DIAG_NORM
    order = np.argsort(dists)[:M]

    cls_to_slot = {cid: i for i, cid in enumerate(spec.coco_ids)}
    for slot, idx in enumerate(order):
        d = rows[idx]
        kp, _ = _keypoints(d)                                    # (Kv, 2) absolute
        rel = (kp - centroid[None, :]) / ped_scale              # (Kv, 2) relative
        base = slot * per_veh
        out[base:base + Kv * 2] = rel.reshape(-1)
        out[base + Kv * 2] = 1.0 / (dists[idx] + EPS)           # inverse distance
        out[base + Kv * 2 + 1] = cls_to_slot[int(d[0])] / max(len(spec.coco_ids) - 1, 1)
    return out


def vehicle_matrix_for_track(
    detections_per_frame: list[np.ndarray],
    centroids: np.ndarray,        # (T, 2) normalised
    ped_scales: np.ndarray,       # (T,)
    frame_indices: np.ndarray,    # (T,) int
    spec: ContextSpec,
) -> np.ndarray:
    """Return C in R^{T × vehicle_dim} for one pedestrian track."""
    T = centroids.shape[0]
    out = np.zeros((T, spec.vehicle_dim), dtype=np.float32)
    for i in range(T):
        f = int(frame_indices[i])
        if f < 0 or f >= len(detections_per_frame):
            continue
        out[i] = vehicle_matrix_for_frame(
            detections_per_frame[f], centroids[i], float(ped_scales[i]), spec
        )
    return out
