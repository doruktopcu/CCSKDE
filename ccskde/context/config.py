"""Static configuration for C_t construction.

Classes are COCO ids as emitted by YOLOv11's COCO-pretrained checkpoint.
Order here defines the slot order in C_t (2K dims: [1/d_min, count] per class).
"""
from dataclasses import dataclass

# COCO class ids for the five hazard classes named in the report.
HAZARD_CLASSES: dict[str, int] = {
    "bicycle":    1,
    "car":        2,
    "motorcycle": 3,
    "bus":        5,
    "truck":      7,
}

# Diagonal-normalized radius defining "near" for the count feature.
# 0.15 ≈ 15% of the image diagonal; tuned later on validation.
NEAR_RADIUS_NORM: float = 0.15

# ShanghaiTech Campus frames are uniformly 856x480 across all 13 scenes
# (verified on every test scene). YOLO detections in the cache are stored in
# normalized [0,1] image coordinates (see detect.py); to place the pedestrian
# skeleton centroid in the SAME frame we divide its raw pixel keypoints by
# (W, H). Override per-dataset if a non-uniform-resolution benchmark is added.
SHANGHAITECH_IMG_WH: tuple[int, int] = (856, 480)


@dataclass(frozen=True)
class ContextSpec:
    classes: tuple[str, ...] = tuple(HAZARD_CLASSES.keys())
    near_radius_norm: float = NEAR_RADIUS_NORM
    img_wh: tuple[int, int] = SHANGHAITECH_IMG_WH
    # Context representation:
    #   "proximity" — original per-class [1/d_min, count]  (dim = 2K)
    #   "vehicle"   — oriented car-skeleton matrix (see vehicle.py)
    mode: str = "proximity"
    max_vehicles: int = 2   # M nearest hazards kept in "vehicle" mode
    kv: int = 6             # keypoints per vehicle (4 corners + center + heading)

    @property
    def proximity_dim(self) -> int:
        return 2 * len(self.classes)

    @property
    def vehicle_dim(self) -> int:
        # per vehicle: K_v keypoints (x,y) + inverse-distance + class id
        return self.max_vehicles * (self.kv * 2 + 2)

    @property
    def dim(self) -> int:
        return self.vehicle_dim if self.mode == "vehicle" else self.proximity_dim

    @property
    def coco_ids(self) -> tuple[int, ...]:
        return tuple(HAZARD_CLASSES[c] for c in self.classes)
