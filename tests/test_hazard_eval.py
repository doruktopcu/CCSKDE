"""Unit tests for the hazard-subset evaluation (ccskde.eval.hazard).

Run:  python tests/test_hazard_eval.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ccskde.eval.hazard import (
    hazard_subset_metrics,
    score_proximity_correlation,
    sweep_tau,
)


def test_perfect_hazard_separation():
    # 100 normal, 50 near-vehicle anomalies, 50 motion-only anomalies
    rng = np.random.default_rng(0)
    gt = np.array([0] * 100 + [1] * 50 + [1] * 50)
    proximity = np.concatenate([
        rng.uniform(0, 0.1, 100),       # normals: far
        rng.uniform(0.5, 1.0, 50),      # vehicle anomalies: near
        rng.uniform(0, 0.1, 50),        # motion anomalies: far
    ])
    # a scorer that perfectly flags ONLY vehicle anomalies (high), else low
    scores = np.concatenate([
        rng.uniform(0, 0.4, 100),       # normals low
        rng.uniform(0.6, 1.0, 50),      # vehicle anomalies high
        rng.uniform(0, 0.4, 50),        # motion anomalies low (missed)
    ])
    m = hazard_subset_metrics(scores, gt, proximity, tau=0.3)
    # On the vehicle-hazard subset (normals + near anomalies) it is perfect:
    assert m["vehicle_hazard"]["auroc"] > 0.99, m["vehicle_hazard"]
    # On the full set it is dragged down by the missed motion anomalies:
    assert m["full"]["auroc"] < m["vehicle_hazard"]["auroc"]
    # frac_near sanity (50 near-vehicle anomalies out of 200 frames)
    assert 0.23 < m["frac_near"] < 0.30
    print("PASS perfect_hazard_separation",
          f"(veh={m['vehicle_hazard']['auroc']:.3f} full={m['full']['auroc']:.3f})")


def test_score_proximity_correlation_positive():
    rng = np.random.default_rng(1)
    gt = np.array([1] * 200)
    proximity = rng.uniform(0, 1, 200)
    scores = proximity * 2 + rng.normal(0, 0.1, 200)   # score rises with proximity
    c = score_proximity_correlation(scores, proximity, gt, anomalous_only=True)
    assert c["spearman"] > 0.8, c
    print(f"PASS score_proximity_correlation (rho={c['spearman']:.3f})")


def test_degenerate_subset_is_nan_not_crash():
    gt = np.array([0, 0, 0, 0])           # no positives
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    proximity = np.array([0.0, 0.0, 0.9, 0.9])
    m = hazard_subset_metrics(scores, gt, proximity, tau=0.5)
    assert np.isnan(m["full"]["auroc"])   # single-class -> NaN, no crash
    print("PASS degenerate_subset_is_nan_not_crash")


def test_sweep_tau_monotone_frac():
    rng = np.random.default_rng(2)
    gt = (rng.uniform(size=300) > 0.7).astype(int)
    scores = rng.uniform(size=300)
    proximity = rng.uniform(size=300)
    rows = sweep_tau(scores, gt, proximity, [0.1, 0.5, 0.9])
    fracs = [r["frac_near"] for r in rows]
    assert fracs[0] >= fracs[1] >= fracs[2]   # higher tau -> fewer near frames
    print("PASS sweep_tau_monotone_frac", [round(f, 2) for f in fracs])


if __name__ == "__main__":
    test_perfect_hazard_separation()
    test_score_proximity_correlation_positive()
    test_degenerate_subset_is_nan_not_crash()
    test_sweep_tau_monotone_frac()
    print("\nAll hazard-eval tests passed.")
