"""Context-vector extraction (C_t) from object detections.

Two-stage pipeline:
  1. detect.extract_clip — run YOLOv11 on each frame of a clip, cache
     normalized boxes per frame as a .npy of object arrays.
  2. build.context_for_track — for each pedestrian track, combine cached
     detections with the track's per-frame skeleton centroid to emit a
     dense C_t ∈ R^{2K} time series.
"""
from .config import HAZARD_CLASSES, NEAR_RADIUS_NORM, ContextSpec
from .build import context_for_frame, context_for_track, pose_centroid

__all__ = [
    "HAZARD_CLASSES",
    "NEAR_RADIUS_NORM",
    "ContextSpec",
    "context_for_frame",
    "context_for_track",
    "pose_centroid",
]
