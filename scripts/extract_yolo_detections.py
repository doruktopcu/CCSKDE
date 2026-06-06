"""CLI: run YOLOv11 over ShanghaiTech clips and cache per-frame detections.

Test split — frames already extracted under data/ShanghaiTech/shanghaitech/
testing/frames/<scene>_<clip>/*.jpg.
Train split — only .avi videos exist under shanghaitech/training/videos/;
this script will read them with OpenCV (no separate ffmpeg pass needed).

Run, e.g.:
    .venv/bin/python scripts/extract_yolo_detections.py \\
        --split test --limit 1   # smoke-test on a single clip
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ccskde.context import ContextSpec
from ccskde.context.detect import (
    extract_clip,
    extract_clip_oriented,
    load_yolo,
    load_yolo_seg,
)


def _list_test_clips(root: Path) -> list[tuple[str, Path, None]]:
    base = root / "shanghaitech" / "testing" / "frames"
    return [(d.name, d, None) for d in sorted(base.iterdir()) if d.is_dir()]


def _list_train_clips(root: Path) -> list[tuple[str, None, Path]]:
    base = root / "shanghaitech" / "training" / "videos"
    return [(v.stem, None, v) for v in sorted(base.glob("*.avi"))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data/ShanghaiTech"))
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: data/ShanghaiTech/context/<split>/")
    ap.add_argument("--weights", default=None,
                    help="default: yolo11n.pt (box) or yolo11s-seg.pt (--oriented)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0,
                    help="if >0, only process the first N clips (smoke testing)")
    ap.add_argument("--oriented", action="store_true",
                    help="run a YOLO *seg* model and cache oriented car-skeleton "
                         "keypoints (14-col schema) instead of axis-aligned boxes")
    args = ap.parse_args()

    default_sub = "context_oriented" if args.oriented else "context"
    out_dir = args.out_dir or args.data_root / default_sub / args.split
    spec = ContextSpec()

    if args.oriented:
        model = load_yolo_seg(args.weights or "yolo11s-seg.pt")
        extract_fn = extract_clip_oriented
    else:
        model = load_yolo(args.weights or "yolo11n.pt")
        extract_fn = extract_clip

    clips = (_list_test_clips(args.data_root) if args.split == "test"
             else _list_train_clips(args.data_root))
    if args.limit > 0:
        clips = clips[: args.limit]

    print(f"[ccskde] {'ORIENTED ' if args.oriented else ''}{len(clips)} clips -> {out_dir}")
    for clip_id, frames_dir, video_path in clips:
        cache = out_dir / f"{clip_id}.npy"
        if cache.exists():
            print(f"  skip {clip_id} (cached)")
            continue
        path = extract_fn(
            clip_id=clip_id,
            frames_dir=frames_dir,
            video_path=video_path,
            out_dir=out_dir,
            model=model,
            spec=spec,
            conf_threshold=args.conf,
        )
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
