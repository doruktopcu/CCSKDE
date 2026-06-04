"""CCSKDE evaluation utilities: hazard-subset AUROC and score-vs-proximity."""
from .hazard import (
    balanced_hazard_auroc,
    clip_frame_proximity,
    hazard_subset_metrics,
    score_proximity_correlation,
    sweep_tau,
)

__all__ = [
    "hazard_subset_metrics",
    "score_proximity_correlation",
    "balanced_hazard_auroc",
    "clip_frame_proximity",
    "sweep_tau",
]
