"""Validate the YOLO-pose JSON builder emits SeeKer's AlphaPose schema.

Run: python tests/test_pose_extract.py
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from extract_yolo_poses import build_pose_json   # noqa: E402


def _mock(n_frames=3, n_people=2):
    rng = np.random.default_rng(0)
    pf = {}
    for f in range(n_frames):
        people = []
        for pid in range(1, n_people + 1):
            kp = rng.uniform(0, 800, size=(17, 3))
            kp[:, 2] = rng.uniform(0.3, 0.95, size=17)   # confidences
            people.append((pid, kp, float(rng.uniform(1.0, 3.0))))
        pf[f] = people
    return pf


def test_schema_shape():
    out = build_pose_json(_mock())
    assert isinstance(out, dict)
    for pid, frames in out.items():
        assert isinstance(pid, str)
        for fk, rec in frames.items():
            assert len(fk) == 4 and fk.isdigit()           # zero-padded frame id
            assert set(rec) == {"keypoints", "scores"}
            assert len(rec["keypoints"]) == 51             # 17 x (x,y,c)
            assert isinstance(rec["scores"], float)
    print("PASS test_schema_shape")


def test_matches_shanghaitech_schema():
    files = glob.glob(os.path.join(ROOT, "data", "ShanghaiTech", "pose", "test",
                                   "*tracked_person.json"))
    if not files:
        print("SKIP (no ShanghaiTech poses on disk)"); return
    real = json.load(open(sorted(files)[0]))
    rp = next(iter(real.values())); rr = next(iter(rp.values()))
    out = build_pose_json(_mock())
    op = next(iter(out.values())); orr = next(iter(op.values()))
    assert set(orr) == set(rr), (set(orr), set(rr))        # same record fields
    assert len(orr["keypoints"]) == len(rr["keypoints"]) == 51
    # frame keys are zero-padded strings in both
    assert all(k.isdigit() for k in op) and all(k.isdigit() for k in rp)
    print("PASS test_matches_shanghaitech_schema")


def test_roundtrip_json():
    out = build_pose_json(_mock())
    s = json.dumps(out)
    back = json.loads(s)
    assert back == out
    print("PASS test_roundtrip_json")


if __name__ == "__main__":
    test_schema_shape()
    test_matches_shanghaitech_schema()
    test_roundtrip_json()
    print("\nAll pose-extract tests passed.")
