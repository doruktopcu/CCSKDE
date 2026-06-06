"""Regression test for the C_t coordinate-frame fix.

Before the fix, the pedestrian centroid was taken from the zero-mean/unit-std
*normalized* pose, while YOLO boxes are in [0,1] image space — so C_t was
effectively pedestrian-independent. This test pins the corrected behaviour:
C_t must be a function of the pedestrian's image-space position.

Uses the shipped box cache under colab_results/ (no YOLO / GPU needed).
Run:  python tests/test_context_coords.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ccskde.context import ContextSpec
from ccskde.data.contextual_dataset import ContextualSkeletonSequenceDataset

CACHE = os.path.join(ROOT, "colab_results", "results", "context_cache", "test")
T, V = 24, 18


class _FakeBase:
    """Stand-in for SeeKer's SkeletonSequenceDataset with a static skeleton."""
    def __init__(self, cx_px, cy_px, start_frame):
        seg = np.zeros((3, T, V), dtype=np.float32)
        seg[0], seg[1], seg[2] = cx_px, cy_px, 1.0
        self.segs_data_np = seg[None]
        self.metadata = np.array([[1, 14, 0, start_frame]])
        self.num_samples = 1

    def __len__(self):
        return 1

    def __getitem__(self, i):
        return [np.zeros((3, T, V), dtype=np.float32), np.ones((T,), dtype=np.float32)]


def _ct_at(cx_px, cy_px, start_frame, spec):
    ds = ContextualSkeletonSequenceDataset(_FakeBase(cx_px, cy_px, start_frame), CACHE, spec=spec)
    return ds[0][2]


def test_ct_is_pedestrian_dependent():
    if not os.path.isdir(CACHE):
        print("SKIP (no shipped cache)")
        return
    spec = ContextSpec()
    W, H = spec.img_wh
    # find a detection frame in clip 01_0014
    cache = np.load(os.path.join(CACHE, "01_0014.npy"), allow_pickle=True)
    f = next(i for i, fr in enumerate(cache) if fr.shape[0] > 0)
    det = cache[f][0]
    dcx, dcy = det[2], det[3]

    near = _ct_at(dcx * W, dcy * H, f, spec)[0]            # pedestrian ON detection
    far = _ct_at((1 - dcx) * W, (1 - dcy) * H, f, spec)[0]  # opposite corner

    assert np.abs(near - far).sum() > 1.0, "C_t did not change with pedestrian pos"
    assert near.max() > far.max(), "near pedestrian must have larger 1/d_min"
    print(f"PASS ct_is_pedestrian_dependent (near max {near.max():.1f} > far max {far.max():.1f})")


def test_centroid_in_unit_square():
    # the corrected centroid uses raw px / (W,H); a skeleton at (600,95) px must
    # map to ~ (0.70, 0.20), NOT ~ origin (the old buggy normalized centroid)
    from ccskde.context.build import pose_centroid
    spec = ContextSpec()
    W, H = spec.img_wh
    xy = np.array([[600.0 / W, 95.0 / H]] * V, dtype=np.float32)
    c = pose_centroid(xy, np.ones(V, dtype=np.float32))
    assert abs(c[0] - 0.70) < 0.02 and abs(c[1] - 0.20) < 0.02, c
    print(f"PASS centroid_in_unit_square ({c[0]:.3f},{c[1]:.3f})")


if __name__ == "__main__":
    test_centroid_in_unit_square()
    test_ct_is_pedestrian_dependent()
    print("\nAll context-coordinate tests passed.")
