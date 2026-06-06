"""Unit tests for the extra report metrics (GPU-free).

Validates the DeLong correlated-AUC test against sklearn's AUROC, plus the
sanity properties of EER, AUPRC, per-scene AUROC and the paired bootstrap.
Run:  set PYTHONPATH=%CD% && .venv\\Scripts\\python -m pytest tests/test_extra_metrics.py -q
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from ccskde.eval.hazard import (
    delong_roc_test, paired_bootstrap_auroc, equal_error_rate, auroc_ap,
    false_alarm_at_recall, per_scene_auroc,
)


def _data(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    a = rng.normal(0, 1, n) + 0.3 * y      # weak detector
    b = rng.normal(0, 1, n) + 0.9 * y      # strong detector
    return y, a, b


def test_delong_aucs_match_sklearn():
    y, a, b = _data()
    d = delong_roc_test(y, a, b)
    assert abs(d["auc_a"] - roc_auc_score(y, a)) < 1e-9
    assert abs(d["auc_b"] - roc_auc_score(y, b)) < 1e-9
    assert abs(d["diff"] - (d["auc_a"] - d["auc_b"])) < 1e-12


def test_delong_detects_significant_difference():
    y, a, b = _data()
    d = delong_roc_test(y, a, b)          # b clearly better -> a worse
    assert d["auc_b"] > d["auc_a"]
    assert d["p_value"] < 1e-6


def test_delong_identical_models_not_significant():
    y, a, _ = _data()
    d = delong_roc_test(y, a, a)
    assert abs(d["diff"]) < 1e-12
    assert d["p_value"] > 0.99


def test_paired_bootstrap_sign_and_ci():
    y, a, b = _data()
    pb = paired_bootstrap_auroc(y, a, b, n_boot=400, seed=1)
    assert pb["mean_diff"] > 0            # B - A > 0
    assert pb["lo"] > 0                   # CI excludes 0
    assert pb["p_b_gt_a"] < 0.05


def test_eer_in_unit_interval_and_auroc():
    y, _, b = _data()
    e = equal_error_rate(b, y)
    assert 0.0 <= e["eer"] <= 1.0
    assert abs(e["auroc"] - roc_auc_score(y, b)) < 1e-9


def test_auroc_ap_matches_sklearn():
    y, _, b = _data()
    m = auroc_ap(b, y)
    assert abs(m["auroc"] - roc_auc_score(y, b)) < 1e-9
    assert abs(m["ap"] - average_precision_score(y, b)) < 1e-9
    assert 0.0 <= m["base_rate"] <= 1.0


def test_false_alarm_at_recall_monotone():
    y, _, b = _data()
    # higher required recall must not yield a lower false-alarm rate
    f80 = false_alarm_at_recall(b, y, 0.80)["fpr"]
    f95 = false_alarm_at_recall(b, y, 0.95)["fpr"]
    assert f95 >= f80 - 1e-9


def test_per_scene_micro_matches_pooled():
    y, a, _ = _data(n=3000)
    keys = ["01_001", "01_002", "02_001"]
    fc = [1000, 1000, 1000]
    ps = per_scene_auroc(a, y, keys, fc)
    assert abs(ps["micro"] - roc_auc_score(y, a)) < 1e-9
    assert set(ps["per_scene"]) == {"01", "02"}
    assert not np.isnan(ps["macro"])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"\nAll {len(fns)} extra-metric tests passed.")
