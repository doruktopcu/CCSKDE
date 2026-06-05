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


def balanced_hazard_auroc(scores, gt, proximity, tau, n_boot=1000, seed=0):
    """W8 fix: AUROC of normal vs anomalous *within near-vehicle frames*, but
    class-balanced (the raw near-vehicle subset is ~91% positive on ShanghaiTech,
    which makes plain AUROC unstable). We balance by subsampling to the minority
    count and bootstrap a 95% CI so the small negative set's uncertainty is
    explicit."""
    rng = np.random.default_rng(seed)
    near = np.asarray(proximity, dtype=np.float64) >= tau
    s = np.asarray(scores, dtype=np.float64)[near]
    y = np.asarray(gt, dtype=np.int32)[near]
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    k = int(min(len(pos), len(neg)))
    if k < 5:
        return dict(auroc=float("nan"), lo=float("nan"), hi=float("nan"), n_per_class=k)
    aucs = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, k, replace=True),
                              rng.choice(neg, k, replace=True)])
        yy, ss = y[idx], s[idx]
        if yy.min() != yy.max():
            aucs.append(roc_auc_score(yy, ss))
    aucs = np.asarray(aucs)
    return dict(auroc=float(aucs.mean()), lo=float(np.percentile(aucs, 2.5)),
                hi=float(np.percentile(aucs, 97.5)), n_per_class=k)


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


# =============================================================================
# Extra report metrics (all pure numpy/scipy/sklearn — GPU-free, unit-testable)
# =============================================================================

def equal_error_rate(scores: np.ndarray, gt: np.ndarray) -> dict:
    """Equal error rate: the operating point where FPR == FNR (1-TPR).

    A single operating-point-free summary that complements AUROC; lower is
    better. We also return the threshold and the AUROC for convenience."""
    from sklearn.metrics import roc_curve
    y = np.asarray(gt, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float64)
    if y.min() == y.max():
        return dict(eer=float("nan"), threshold=float("nan"), auroc=float("nan"))
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return dict(eer=float((fpr[i] + fnr[i]) / 2.0),
                threshold=float(thr[i]),
                auroc=float(roc_auc_score(y, s)))


def auroc_ap(scores: np.ndarray, gt: np.ndarray) -> dict:
    """AUROC + Average Precision (AUPRC). AP is the more honest headline under
    class imbalance (ShanghaiTech test is ~42.5% positive), so we report both."""
    y = np.asarray(gt, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float64)
    if y.min() == y.max():
        return dict(auroc=float("nan"), ap=float("nan"),
                    base_rate=float(y.mean()), n=int(y.size))
    return dict(auroc=float(roc_auc_score(y, s)),
                ap=float(average_precision_score(y, s)),
                base_rate=float(y.mean()), n=int(y.size))


def false_alarm_at_recall(scores, gt, recall: float = 0.90) -> dict:
    """Operationally meaningful for a hazard detector: the false-positive rate
    when the detector catches `recall` of the true anomalies."""
    from sklearn.metrics import roc_curve
    y = np.asarray(gt, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float64)
    if y.min() == y.max():
        return dict(fpr=float("nan"), threshold=float("nan"), recall=recall)
    fpr, tpr, thr = roc_curve(y, s)
    idx = np.where(tpr >= recall)[0]
    if idx.size == 0:
        return dict(fpr=1.0, threshold=float(thr[-1]), recall=recall)
    i = int(idx[0])
    return dict(fpr=float(fpr[i]), threshold=float(thr[i]), recall=recall)


# -------------------------------------------------- DeLong correlated-AUC test
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Mid-ranks (ties averaged), used by the fast DeLong estimator."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=np.float64)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
    """Fast DeLong (Sun & Xu, 2014). Returns (aucs, covariance) for k predictors
    sharing the same ground-truth ordering (positives first)."""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    pos = predictions_sorted_transposed[:, :m]
    neg = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m], dtype=np.float64)
    ty = np.empty([k, n], dtype=np.float64)
    tz = np.empty([k, m + n], dtype=np.float64)
    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, np.atleast_2d(delongcov)


def delong_roc_test(gt: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    """DeLong's test for two CORRELATED ROC AUCs (same frames, two models).

    Returns each AUC, the AUC difference, the z statistic and a two-sided
    p-value. This is the statistically correct way to say "model B's AUROC is
    significantly higher than model A's" on the same test set — stronger than an
    unpaired t-test over seed means, because it accounts for the shared samples."""
    import scipy.stats
    y = np.asarray(gt, dtype=np.int32)
    order = (-y).argsort(kind="mergesort")          # positives (label 1) first
    label_1_count = int(y.sum())
    preds = np.vstack((np.asarray(scores_a, dtype=np.float64),
                       np.asarray(scores_b, dtype=np.float64)))[:, order]
    aucs, cov = _fast_delong(preds, label_1_count)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = float(aucs[0] - aucs[1])
    if var <= 0:
        z = float("inf") if diff != 0 else 0.0
        p = 0.0 if diff != 0 else 1.0
    else:
        z = diff / np.sqrt(var)
        p = float(2 * scipy.stats.norm.sf(abs(z)))
    return dict(auc_a=float(aucs[0]), auc_b=float(aucs[1]), diff=diff,
                z=float(z), p_value=p)


def paired_bootstrap_auroc(gt, scores_a, scores_b, n_boot=2000, seed=0) -> dict:
    """Paired bootstrap of the AUROC difference (B - A) on the same frames.
    Resamples frame indices with replacement; reports the mean diff, a 95% CI,
    and the fraction of resamples where B>A (a bootstrap p-value)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(gt, dtype=np.int32)
    sa = np.asarray(scores_a, dtype=np.float64)
    sb = np.asarray(scores_b, dtype=np.float64)
    n = y.size
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        diffs.append(roc_auc_score(yy, sb[idx]) - roc_auc_score(yy, sa[idx]))
    diffs = np.asarray(diffs)
    return dict(mean_diff=float(diffs.mean()),
                lo=float(np.percentile(diffs, 2.5)),
                hi=float(np.percentile(diffs, 97.5)),
                p_b_gt_a=float((diffs <= 0).mean()),  # one-sided bootstrap p
                n_boot=int(diffs.size))


def clustered_bootstrap_auroc_diff(gt, scores_a, scores_b, frame_counts,
                                   n_boot=2000, seed=0) -> dict:
    """Clip-level (block) bootstrap of the AUROC difference (B - A).

    Frames within a ShanghaiTech clip are temporally correlated, so a per-frame
    bootstrap (and DeLong) understate the variance. Here we resample whole CLIPS
    with replacement -- the statistically honest unit -- giving a wider, more
    defensible CI and a clip-level bootstrap p-value. `frame_counts` are the
    per-clip frame counts in the same concatenation order as `scores`/`gt`."""
    rng = np.random.default_rng(seed)
    y = np.asarray(gt, dtype=np.int32)
    sa = np.asarray(scores_a, dtype=np.float64)
    sb = np.asarray(scores_b, dtype=np.float64)
    offsets = np.concatenate([[0], np.cumsum(frame_counts)]).astype(int)
    n_clips = len(frame_counts)
    clip_idx = [np.arange(offsets[i], offsets[i + 1]) for i in range(n_clips)]
    diffs = []
    for _ in range(n_boot):
        pick = rng.integers(0, n_clips, n_clips)
        idx = np.concatenate([clip_idx[c] for c in pick])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        diffs.append(roc_auc_score(yy, sb[idx]) - roc_auc_score(yy, sa[idx]))
    diffs = np.asarray(diffs)
    return dict(mean_diff=float(diffs.mean()),
                lo=float(np.percentile(diffs, 2.5)),
                hi=float(np.percentile(diffs, 97.5)),
                p_b_le_a=float((diffs <= 0).mean()),
                n_boot=int(diffs.size), n_clips=int(n_clips))


def per_scene_auroc(scores, gt, clip_keys, frame_counts) -> dict:
    """ShanghaiTech has 13 scenes; the clip key encodes the scene as its prefix
    (e.g. '01_0014' -> scene '01'). Reports per-scene AUROC plus the macro mean
    (unweighted over scenes) and micro (pooled) AUROC. SeeKer reports micro
    only, so the macro/per-scene breakdown is a genuine addition.

    `clip_keys` and `frame_counts` are aligned lists giving, in the SAME order
    as `scores`/`gt` were concatenated, each clip's key and its frame count."""
    scores = np.asarray(scores, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(frame_counts)])
    by_scene_s: dict[str, list] = {}
    by_scene_y: dict[str, list] = {}
    for i, key in enumerate(clip_keys):
        scene = str(key).split("_")[0]
        a, b = int(offsets[i]), int(offsets[i + 1])
        by_scene_s.setdefault(scene, []).append(scores[a:b])
        by_scene_y.setdefault(scene, []).append(gt[a:b])
    per = {}
    for scene in sorted(by_scene_s):
        s = np.concatenate(by_scene_s[scene])
        y = np.concatenate(by_scene_y[scene])
        per[scene] = (float(roc_auc_score(y, s)) if y.min() != y.max()
                      else float("nan"))
    vals = [v for v in per.values() if not np.isnan(v)]
    return dict(per_scene=per,
                macro=float(np.mean(vals)) if vals else float("nan"),
                micro=float(roc_auc_score(gt, scores)) if gt.min() != gt.max() else float("nan"))
