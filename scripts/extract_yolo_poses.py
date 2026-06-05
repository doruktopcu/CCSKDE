"""Extract pedestrian poses with YOLO11-pose in SeeKer's AlphaPose JSON format.

SeeKer (and our ccskde datasets) consume per-clip files named
`<clip>_alphapose_tracked_person.json` with the schema:

    { "<track_id>": { "<frame:04d>": {"keypoints": [x,y,c, ... 17 kpts = 51],
                                       "scores": <person box conf>}, ... }, ... }

keypoints are COCO-17 (x,y,confidence) in PIXEL coordinates; SeeKer converts 17->18
internally. YOLO11-pose emits exactly COCO-17 with per-keypoint confidence and,
with `.track()`, stable person ids -- a drop-in source when no AlphaPose release
exists (e.g. UBnormal, Street Scene).

The JSON assembly (`build_pose_json`) is a pure function and is unit-tested
(tests/test_pose_extract.py) against the real ShanghaiTech schema, so it is
correct independent of the YOLO run.

Usage (run once data is on disk; downloads a small YOLO11-pose weight on first use):
    set PYTHONPATH=%CD%
    .venv\\Scripts\\python scripts\\extract_yolo_poses.py ^
        --frames-root data\\UBnormal\\test\\frames --out-dir data\\UBnormal\\pose\\test
    # or from videos:
    .venv\\Scripts\\python scripts\\extract_yolo_poses.py ^
        --video-glob "data\\UBnormal\\videos\\*.mp4" --out-dir data\\UBnormal\\pose\\test
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path


def build_pose_json(per_frame: dict) -> dict:
    """per_frame: {frame_idx(int): [ (track_id(int), kpts(17,3) iterable, box_conf(float)) ]}
    -> SeeKer AlphaPose JSON dict {track_id: {frame:04d: {keypoints[51], scores}}}."""
    out: dict[str, dict] = {}
    for f in sorted(per_frame):
        for tid, kpts, conf in per_frame[f]:
            flat = [float(v) for kp in kpts for v in (kp[0], kp[1], kp[2])]
            if len(flat) != 51:
                raise ValueError(f"expected 17x3=51 keypoint values, got {len(flat)}")
            out.setdefault(str(int(tid)), {})[f"{int(f):04d}"] = {
                "keypoints": flat, "scores": float(conf),
            }
    return out


def _iter_clips(args):
    """Yield (clip_name, ordered_frame_source) where frame_source is a list of
    image paths (frames-root mode) or a single video path (video mode)."""
    if args.frames_root:
        for sub in sorted(os.listdir(args.frames_root)):
            d = os.path.join(args.frames_root, sub)
            if os.path.isdir(d):
                imgs = sorted(glob.glob(os.path.join(d, "*.jpg")) +
                              glob.glob(os.path.join(d, "*.png")))
                if imgs:
                    yield sub, ("frames", imgs)
    if args.video_glob:
        for v in sorted(glob.glob(args.video_glob)):
            yield Path(v).stem, ("video", v)


def _run_yolo(args):   # pragma: no cover  (requires ultralytics + weights + GPU)
    from ultralytics import YOLO
    import cv2
    model = YOLO(args.weights)
    os.makedirs(args.out_dir, exist_ok=True)
    clips = list(_iter_clips(args))
    print(f"[poses] {len(clips)} clips -> {args.out_dir} (weights={args.weights})")
    for clip, (kind, src) in clips:
        out_path = os.path.join(args.out_dir, f"{clip}_alphapose_tracked_person.json")
        if os.path.exists(out_path) and not args.force:
            print(f"  skip {clip} (exists)"); continue
        frames = src if kind == "frames" else _video_frames(src)
        per_frame: dict[int, list] = {}
        for fi, img in enumerate(frames):
            res = model.track(img, persist=True, classes=[0], conf=args.conf,
                              verbose=False)[0]
            if res.keypoints is None or res.boxes is None or res.boxes.id is None:
                continue
            kp = res.keypoints.data.cpu().numpy()          # (n,17,3)
            ids = res.boxes.id.cpu().numpy().astype(int)
            cf = res.boxes.conf.cpu().numpy()
            per_frame[fi] = [(int(ids[i]), kp[i], float(cf[i])) for i in range(len(ids))]
        with open(out_path, "w") as f:
            json.dump(build_pose_json(per_frame), f)
        print(f"  wrote {clip}: {len(per_frame)} frames")


def _video_frames(path):   # pragma: no cover
    import cv2
    cap = cv2.VideoCapture(path)
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        yield fr
    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", default=None,
                    help="dir containing one sub-dir of frames per clip")
    ap.add_argument("--video-glob", default=None, help="glob of per-clip videos")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--weights", default="yolo11l-pose.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if not (a.frames_root or a.video_glob):
        ap.error("provide --frames-root or --video-glob")
    _run_yolo(a)


if __name__ == "__main__":
    main()
