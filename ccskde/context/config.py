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


@dataclass(frozen=True)
class ContextSpec:
    classes: tuple[str, ...] = tuple(HAZARD_CLASSES.keys())
    near_radius_norm: float = NEAR_RADIUS_NORM

    @property
    def dim(self) -> int:
        return 2 * len(self.classes)

    @property
    def coco_ids(self) -> tuple[int, ...]:
        return tuple(HAZARD_CLASSES[c] for c in self.classes)
