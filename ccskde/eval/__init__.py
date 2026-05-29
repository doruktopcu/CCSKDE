"""CCSKDE evaluation utilities: hazard-subset AUROC and score-vs-proximity."""
from .hazard import (
    hazard_subset_metrics,
    score_proximity_correlation,
)

__all__ = ["hazard_subset_metrics", "score_proximity_correlation"]
