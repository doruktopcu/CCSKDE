"""YOLOv11 detection pass — runs offline, once per video, caches to .npy.

Output cache layout (one file per clip):
    <out_dir>/<scene>_<clip>.npy   — object array, length = num_frames.
    Each entry is an ndarray of shape (n_dets, 6): (cls, conf, cx, cy, w, h),
    in *normalized* image coordinates (cx,cy,w,h ∈ [0,1]).

Frames are read from a directory of jpgs (test split layout) or, if missing,
decoded from a video file with OpenCV (train split layout).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from .config import ContextSpec


def _iter_frames_from_dir(frames_dir: Path) -> Iterator[np.ndarray]:
    import cv2
    for jpg in sorted(frames_dir.glob("*.jpg")):
        img = cv2.imread(str(jpg))
        if img is None:
            raise RuntimeError(f"cv2.imread failed: {jpg}")
        yield img  # BGR uint8


def _iter_frames_from_video(video_path: Path) -> Iterator[np.ndarray]:
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame
    finally:
        cap.release()


def _extract_one(
    model,
    frames: Iterator[np.ndarray],
    spec: ContextSpec,
    conf_threshold: float,
) -> list[np.ndarray]:
    keep_ids = set(spec.coco_ids)
    per_frame: list[np.ndarray] = []
    for frame in frames:
        H, W = frame.shape[:2]
        # ultralytics YOLO accepts a BGR ndarray and returns a Results list.
        result = model.predict(frame, verbose=False, conf=conf_threshold)[0]
        if result.boxes is None or len(result.boxes) == 0:
            per_frame.append(np.zeros((0, 6), dtype=np.float32))
            continue
        cls = result.boxes.cls.cpu().numpy().astype(np.int32)
        conf = result.boxes.conf.cpu().numpy().astype(np.float32)
        # xywh in pixel coords; normalize.
        xywh = result.boxes.xywh.cpu().numpy().astype(np.float32)
        xywh[:, 0] /= W
        xywh[:, 2] /= W
        xywh[:, 1] /= H
        xywh[:, 3] /= H
        mask = np.array([c in keep_ids for c in cls], dtype=bool)
        if not mask.any():
            per_frame.append(np.zeros((0, 6), dtype=np.float32))
            continue
        rec = np.empty((mask.sum(), 6), dtype=np.float32)
        rec[:, 0] = cls[mask]
        rec[:, 1] = conf[mask]
        rec[:, 2:6] = xywh[mask]
        per_frame.append(rec)
    return per_frame


def extract_clip(
    clip_id: str,
    frames_dir: Path | None,
    video_path: Path | None,
    out_dir: Path,
    model,
    spec: ContextSpec,
    conf_threshold: float = 0.25,
) -> Path:
    """Run YOLO over one clip and dump the per-frame detection list to .npy.

    Pass exactly one of `frames_dir` or `video_path`.
    """
    if (frames_dir is None) == (video_path is None):
        raise ValueError("pass exactly one of frames_dir or video_path")
    if frames_dir is not None:
        frames = _iter_frames_from_dir(frames_dir)
    else:
        frames = _iter_frames_from_video(video_path)
    per_frame = _extract_one(model, frames, spec, conf_threshold)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clip_id}.npy"
    np.save(out_path, np.array(per_frame, dtype=object), allow_pickle=True)
    return out_path


def load_yolo(weights: str = "yolo11n.pt"):
    """Lazy-import wrapper so the module is importable without ultralytics."""
    from ultralytics import YOLO
    return YOLO(weights)


# ---------------------------------------------------------------------------
# Oriented "car skeleton" extraction (segmentation -> minAreaRect keypoints)
# ---------------------------------------------------------------------------
# Cache schema (one row per detection), all coords normalised to [0,1]:
#     cls, conf, x1,y1, x2,y2, x3,y3, x4,y4, cx,cy, hx,hy   (14 cols)
# i.e. 4 oriented corners + center + heading point. The oriented corners are
# computed in PIXEL space (cv2.minAreaRect / boxPoints) and then normalised, so
# orientation is geometrically faithful (unlike rotating in anisotropic
# normalised space). Consumed by ccskde.context.vehicle.
ORIENTED_COLS = 14


def _oriented_kps_from_points(pts_px: np.ndarray, W: int, H: int) -> np.ndarray:
    """pts_px: (n,2) polygon/box points in pixels -> 12 normalised coords (6 kp)."""
    import cv2
    rect = cv2.minAreaRect(pts_px.astype(np.float32))   # ((cx,cy),(w,h),angle)
    box = cv2.boxPoints(rect).astype(np.float32)        # (4,2) corners, pixels
    (cx, cy), (w, h), _ = rect
    center = np.array([cx, cy], dtype=np.float32)
    # heading = center + half the MAJOR axis (front/back sign is ambiguous;
    # velocity disambiguates downstream).
    e01, e12 = box[1] - box[0], box[2] - box[1]
    if np.linalg.norm(e01) >= np.linalg.norm(e12):
        heading = center + e01 / 2.0
    else:
        heading = center + e12 / 2.0
    kps = np.vstack([box, center, heading]).astype(np.float32)  # (6,2) pixels
    kps[:, 0] /= float(W)
    kps[:, 1] /= float(H)
    return kps.reshape(-1)                              # (12,)


def _extract_one_oriented(model, frames, spec: ContextSpec,
                          conf_threshold: float) -> list[np.ndarray]:
    import numpy as _np
    keep_ids = set(spec.coco_ids)
    per_frame: list[_np.ndarray] = []
    for frame in frames:
        H, W = frame.shape[:2]
        result = model.predict(frame, verbose=False, conf=conf_threshold)[0]
        boxes = result.boxes
        masks = result.masks
        if boxes is None or len(boxes) == 0:
            per_frame.append(_np.zeros((0, ORIENTED_COLS), dtype=_np.float32))
            continue
        cls = boxes.cls.cpu().numpy().astype(_np.int32)
        conf = boxes.conf.cpu().numpy().astype(_np.float32)
        xyxy = boxes.xyxy.cpu().numpy().astype(_np.float32)
        polys = masks.xy if masks is not None else None
        recs = []
        for i in range(len(cls)):
            if int(cls[i]) not in keep_ids:
                continue
            if polys is not None and i < len(polys) and len(polys[i]) >= 3:
                pts = _np.asarray(polys[i], dtype=_np.float32)
            else:  # no mask -> fall back to the axis-aligned box corners
                x1, y1, x2, y2 = xyxy[i]
                pts = _np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], _np.float32)
            kps = _oriented_kps_from_points(pts, W, H)
            recs.append(_np.concatenate([[float(cls[i]), float(conf[i])], kps]))
        if not recs:
            per_frame.append(_np.zeros((0, ORIENTED_COLS), dtype=_np.float32))
        else:
            per_frame.append(_np.stack(recs, axis=0).astype(_np.float32))
    return per_frame


def extract_clip_oriented(
    clip_id: str,
    frames_dir: Path | None,
    video_path: Path | None,
    out_dir: Path,
    model,
    spec: ContextSpec,
    conf_threshold: float = 0.25,
) -> Path:
    """Run a YOLO *seg* model over one clip; cache oriented car-skeleton rows."""
    if (frames_dir is None) == (video_path is None):
        raise ValueError("pass exactly one of frames_dir or video_path")
    frames = (_iter_frames_from_dir(frames_dir) if frames_dir is not None
              else _iter_frames_from_video(video_path))
    per_frame = _extract_one_oriented(model, frames, spec, conf_threshold)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clip_id}.npy"
    np.save(out_path, np.array(per_frame, dtype=object), allow_pickle=True)
    return out_path


def load_yolo_seg(weights: str = "yolo11n-seg.pt"):
    """Lazy-import wrapper for a YOLOv11 segmentation model."""
    from ultralytics import YOLO
    return YOLO(weights)
