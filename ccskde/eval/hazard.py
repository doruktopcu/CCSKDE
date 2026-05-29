"""Hazard-subset evaluation — the analysis the original project never did.

The proposal's actual claim is that context conditioning helps detect
*pedestrian-vehicle hazard* events specifically. Generic ShanghaiTech AUROC
mixes those with motion-intrinsic anomalies (running, fighting, throwing) and
so cannot answer the question. Here we:

  1. label each test frame with a per-frame proximity = max over present
     pedestrians of 1/(d_min to nearest hazard vehicle), in [0,1] image space;
  2. report AUROC restricted to the *interaction regime* (proximity >= tau)
     and to {normal} U {anomalous AND near-vehicle} (vehicle-hazard detection);
  3. report the Spearman correlation between anomaly score and proximity over
     anomalous frames — does the score actually rise as a hazard approaches?

These functions are pure (numpy/scipy) so they unit-test without a GPU.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def hazard_subset_metrics(
    scores: np.ndarray,       # (F,) frame anomaly scores, all clips concatenated
    gt: np.ndarray,           # (F,) 0/1 frame labels
    proximity: np.ndarray,    # (F,) per-frame pedestrian-vehicle proximity
    tau: float,               # proximity threshold defining "near a vehicle"
) -> dict:
    """Return AUROC/AP on the full set, the interaction regime, and the
    vehicle-hazard detection set."""
    scores = np.asarray(scores, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.int32)
    proximity = np.asarray(proximity, dtype=np.float64)
    near = proximity >= tau

    def _safe(mask):
        m = np.asarray(mask, dtype=bool)
        y = gt[m]
        if m.sum() < 2 or y.min() == y.max():
            return dict(auroc=float("nan"), ap=float("nan"),
                        n=int(m.sum()), n_pos=int(y.sum()))
        s = scores[m]
        return dict(auroc=float(roc_auc_score(y, s)),
                    ap=float(average_precision_score(y, s)),
                    n=int(m.sum()), n_pos=int(y.sum()))

    # vehicle-hazard detection: keep all normal frames + only anomalies that
    # co-occur with a nearby vehicle (drop motion-intrinsic anomalies).
    veh_hazard = (gt == 0) | ((gt == 1) & near)

    return {
        "full": _safe(np.ones_like(gt, dtype=bool)),
        "interaction_regime": _safe(near),       # frames where a vehicle is near
        "vehicle_hazard": _safe(veh_hazard),     # normal + near-vehicle anomalies
        "tau": tau,
        "frac_near": float(near.mean()),
    }


def score_proximity_correlation(
    scores: np.ndarray,
    proximity: np.ndarray,
    gt: np.ndarray | None = None,
    anomalous_only: bool = True,
) -> dict:
    """Spearman & Pearson correlation between anomaly score and proximity.

    By default computed over anomalous frames (where a hazard, if present,
    should drive the score up)."""
    scores = np.asarray(scores, dtype=np.float64)
    proximity = np.asarray(proximity, dtype=np.float64)
    mask = np.ones_like(scores, dtype=bool)
    if anomalous_only and gt is not None:
        mask = np.asarray(gt, dtype=np.int32) == 1
    s, p = scores[mask], proximity[mask]
    if s.size < 3 or np.std(p) == 0 or np.std(s) == 0:
        return dict(spearman=float("nan"), pearson=float("nan"), n=int(mask.sum()))
    rho, _ = spearmanr(s, p)
    pear = float(np.corrcoef(s, p)[0, 1])
    return dict(spearman=float(rho), pearson=pear, n=int(mask.sum()))


def clip_frame_proximity(
    detections_per_frame: list,   # per-frame YOLO cache (box or oriented schema)
    ped_centroids_per_frame: dict,  # {frame_idx: list of (2,) ped centroids [0,1]}
    n_frames: int,
    coco_ids: tuple,
    eps: float = 1e-3,
) -> np.ndarray:
    """Per-frame pedestrian-vehicle proximity for one clip.

    proximity[f] = max over pedestrians present in frame f of 1/(d_min+eps),
    where d_min is the normalised distance to the nearest hazard vehicle. 0 if
    no pedestrian or no hazard in the frame. Length == n_frames (the GT length).
    """
    import math
    diag = math.sqrt(2.0)
    keep = set(coco_ids)
    out = np.zeros(n_frames, dtype=np.float64)
    for f in range(n_frames):
        if f >= len(detections_per_frame):
            break
        dets = detections_per_frame[f]
        peds = ped_centroids_per_frame.get(f, [])
        if dets is None or len(dets) == 0 or len(peds) == 0:
            continue
        centers = np.array([d[2:4] for d in dets if int(d[0]) in keep], dtype=np.float64)
        if centers.shape[0] == 0:
            continue
        best = 0.0
        for c in peds:
            d = np.linalg.norm(centers - np.asarray(c)[None, :], axis=1) / diag
            best = max(best, 1.0 / (float(d.min()) + eps))
        out[f] = best
    return out


def sweep_tau(scores, gt, proximity, taus) -> list[dict]:
    """Convenience: hazard_subset_metrics across several thresholds."""
    out = []
    for t in taus:
        m = hazard_subset_metrics(scores, gt, proximity, t)
        out.append(dict(tau=float(t),
                        interaction_auroc=m["interaction_regime"]["auroc"],
                        vehicle_hazard_auroc=m["vehicle_hazard"]["auroc"],
                        frac_near=m["frac_near"]))
    return out
