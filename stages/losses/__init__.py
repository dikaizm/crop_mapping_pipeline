"""Versioned loss functions for Stage 3 training.

    v1 — WeightedCrossEntropy   (baseline; used for Exp A / B / C)
    v2 — PhenologyAwareLoss     (NDVI-based dormancy weighting; used for Exp C_v2 / D_v2)

Usage:
    from crop_mapping_pipeline.stages.losses import build_loss_v1, build_loss_v2, PhenologyAwareLoss
"""

from crop_mapping_pipeline.stages.losses.v1 import build_loss_v1
from crop_mapping_pipeline.stages.losses.v2 import PhenologyAwareLoss, build_loss_v2

__all__ = [
    "build_loss_v1",
    "build_loss_v2",
    "PhenologyAwareLoss",
]
