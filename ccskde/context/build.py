"""Build per-pedestrian, per-frame context vectors C_t from cached YOLO
detections and SeeKer pose tracks.

For each (pedestrian track p, frame t) we emit C_t ∈ R^{2K}:
    [1/d^min_c1, n^{≤r}_c1, 1/d^min_c2, n^{≤r}_c2, ...]
where d_{t,b} = ||(x_b, y_b) - centroid(p,t)|| / sqrt(H^2 + W^2). All
detection coordinates are already normalized (see detect.py), so the
diagonal normalization reduces to euclidean distance in [0,1]^2 divided by
sqrt(2). For numerical stability the inverse-min-distance is computed as
1/(d + EPS), and frames with no in-class detection get zeros for that class.
"""
from __future__ import annotations

import math

import numpy as np

from .config import ContextSpec

EPS = 1e-3
DIAG_NORM = math.sqrt(2.0)  # detections are in [0,1]^2 already


def _slot_lookup(spec: ContextSpec) -> dict[int, int]:
    """COCO class id -> slot index (0..K-1)."""
    return {cid: i for i, cid in enumerate(spec.coco_ids)}


def context_for_frame(
    detections: np.ndarray,    # (n_dets, 6): cls, conf, cx, cy, w, h (normalized)
    centroid: np.ndarray,      # shape (2,), normalized image coords
    spec: ContextSpec,
) -> np.ndarray:
    """Return C_t for one (pedestrian, frame) pair as a length-2K vector."""
    K = len(spec.classes)
    out = np.zeros(2 * K, dtype=np.float32)
    if detections.shape[0] == 0:
        return out
    slot = _slot_lookup(spec)
    cls_ids = detections[:, 0].astype(np.int32)
    centers = detections[:, 2:4]                       # (n_dets, 2)
    deltas = centers - centroid[None, :]
    dists = np.linalg.norm(deltas, axis=1) / DIAG_NORM  # ∈ [0,1]
    for cid, k in slot.items():
        m = cls_ids == cid
        if not m.any():
            continue
        d_class = dists[m]
        out[2 * k] = 1.0 / (d_class.min() + EPS)
        out[2 * k + 1] = float((d_class <= spec.near_radius_norm).sum())
    return out


def context_for_track(
    detections_per_frame: list[np.ndarray],   # length T_video
    centroids: np.ndarray,                    # (T_track, 2), normalized
    frame_indices: np.ndarray,                # (T_track,), int — index into video
    spec: ContextSpec,
) -> np.ndarray:
    """Return C ∈ R^{T_track × 2K} for one pedestrian track."""
    T_track = centroids.shape[0]
    out = np.zeros((T_track, spec.dim), dtype=np.float32)
    for i in range(T_track):
        f = int(frame_indices[i])
        if f < 0 or f >= len(detections_per_frame):
            continue
        out[i] = context_for_frame(detections_per_frame[f], centroids[i], spec)
    return out


def pose_centroid(keypoints_xy: np.ndarray, conf: np.ndarray | None = None) -> np.ndarray:
    """Confidence-weighted centroid of a single skeleton.

    keypoints_xy: (V, 2) in normalized [0,1]^2 image coords.
    conf:         (V,) confidences (optional). If None or all-zero, use mean.
    """
    if conf is None or float(conf.sum()) <= 0.0:
        return keypoints_xy.mean(axis=0).astype(np.float32)
    w = conf / conf.sum()
    return (keypoints_xy * w[:, None]).sum(axis=0).astype(np.float32)
