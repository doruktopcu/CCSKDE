"""Unit tests for the oriented vehicle-skeleton context (ccskde.context.vehicle).

Run directly:   python tests/test_vehicle_context.py
(Also discoverable by pytest.)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ccskde.context import ContextSpec
from ccskde.context.vehicle import (
    oriented_box_keypoints,
    vehicle_matrix_for_frame,
)


def test_oriented_box_axis_aligned():
    kp = oriented_box_keypoints(0.5, 0.5, 0.2, 0.1, 0.0)
    assert kp.shape == (6, 2)
    # center is keypoint index 4, heading index 5 (+x of center)
    assert np.allclose(kp[4], [0.5, 0.5])
    assert np.allclose(kp[5], [0.6, 0.5])           # center + w/2 along +x
    # corners span the box
    assert np.allclose(kp[:4].min(0), [0.4, 0.45])
    assert np.allclose(kp[:4].max(0), [0.6, 0.55])
    print("PASS oriented_box_axis_aligned")


def test_oriented_box_rotation_90deg():
    kp = oriented_box_keypoints(0.5, 0.5, 0.2, 0.1, np.pi / 2)
    # heading should now point +y (rotated 90 deg)
    assert np.allclose(kp[5], [0.5, 0.6], atol=1e-6)
    print("PASS oriented_box_rotation_90deg")


def _oriented_row(cls, cx, cy, w, h):
    kp = oriented_box_keypoints(cx, cy, w, h, 0.0).reshape(-1)
    return np.concatenate([[cls, 0.9], kp]).astype(np.float32)  # 14 cols


def test_pedestrian_relative_and_scale_invariant():
    spec = ContextSpec(mode="vehicle", max_vehicles=1, kv=6)
    car = _oriented_row(2, 0.6, 0.5, 0.2, 0.1)        # car right of center
    dets = car[None]

    ped = np.array([0.5, 0.5], np.float32)
    v1 = vehicle_matrix_for_frame(dets, ped, ped_scale=0.1, spec=spec)
    # First keypoint block is relative coords; car center (kp idx 4) relative:
    cen_rel = v1[8:10]                                 # idx4 -> offset 8,9
    assert np.allclose(cen_rel, [(0.6 - 0.5) / 0.1, 0.0], atol=1e-5)

    # Scale invariance: double the pedestrian size AND the offset -> same vector.
    car2 = _oriented_row(2, 0.7, 0.5, 0.4, 0.2)        # offset 0.2, size doubled
    v2 = vehicle_matrix_for_frame(car2[None], ped, ped_scale=0.2, spec=spec)
    assert np.allclose(v1[:12], v2[:12], atol=1e-5), (v1[:12], v2[:12])
    print("PASS pedestrian_relative_and_scale_invariant")


def test_pedestrian_dependence():
    spec = ContextSpec(mode="vehicle", max_vehicles=1, kv=6)
    car = _oriented_row(2, 0.6, 0.5, 0.2, 0.1)[None]
    a = vehicle_matrix_for_frame(car, np.array([0.5, 0.5], np.float32), 0.1, spec)
    b = vehicle_matrix_for_frame(car, np.array([0.2, 0.2], np.float32), 0.1, spec)
    assert not np.allclose(a, b)
    # inverse distance (idx 12) larger when nearer
    assert a[12] > b[12]
    print("PASS pedestrian_dependence")


def test_topM_selection_and_padding():
    spec = ContextSpec(mode="vehicle", max_vehicles=2, kv=6)
    dets = np.stack([
        _oriented_row(2, 0.55, 0.5, 0.1, 0.1),   # nearest
        _oriented_row(2, 0.70, 0.5, 0.1, 0.1),   # 2nd
        _oriented_row(2, 0.95, 0.5, 0.1, 0.1),   # farthest -> dropped
    ])
    ped = np.array([0.5, 0.5], np.float32)
    v = vehicle_matrix_for_frame(dets, ped, 0.1, spec)
    assert v.shape[0] == spec.vehicle_dim == 28
    # both vehicle slots populated (non-zero inverse-distance), none from 3rd car
    assert v[12] > 0 and v[26] > 0
    print("PASS topM_selection_and_padding")


def test_box_schema_fallback():
    spec = ContextSpec(mode="vehicle", max_vehicles=1, kv=6)
    box6 = np.array([2, 0.9, 0.6, 0.5, 0.2, 0.1], np.float32)  # cls,conf,cx,cy,w,h
    v = vehicle_matrix_for_frame(box6[None], np.array([0.5, 0.5], np.float32), 0.1, spec)
    assert v[12] > 0 and not np.allclose(v[:12], 0)
    print("PASS box_schema_fallback")


def test_empty_detections():
    spec = ContextSpec(mode="vehicle", max_vehicles=2, kv=6)
    v = vehicle_matrix_for_frame(np.zeros((0, 14), np.float32),
                                 np.array([0.5, 0.5], np.float32), 0.1, spec)
    assert v.shape[0] == 28 and np.allclose(v, 0)
    print("PASS empty_detections")


if __name__ == "__main__":
    test_oriented_box_axis_aligned()
    test_oriented_box_rotation_90deg()
    test_pedestrian_relative_and_scale_invariant()
    test_pedestrian_dependence()
    test_topM_selection_and_padding()
    test_box_schema_fallback()
    test_empty_detections()
    print("\nAll vehicle-context tests passed.")
