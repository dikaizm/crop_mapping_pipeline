"""Versioned loss functions for Stage 3 training.

    v1 — WeightedCrossEntropy        (baseline; inverse-frequency weights)
    v2 — PhenologyAwareLoss          (NDVI dormancy weighting)
    v3 — FocalCE + FocalTversky      (Effective-Number weights;
                                      designed for spatial generalisation
                                      under class imbalance + prior shift)

Usage:
    from crop_mapping_pipeline.stages.losses import (
        build_loss_v1, build_loss_v2, build_loss_v3,
        PhenologyAwareLoss, FocalCEPlusFocalTversky,
    )
"""

from crop_mapping_pipeline.stages.losses.v1 import build_loss_v1
from crop_mapping_pipeline.stages.losses.v2 import PhenologyAwareLoss, build_loss_v2
from crop_mapping_pipeline.stages.losses.v3 import (
    FocalCEPlusFocalTversky, FocalCELoss, FocalTverskyLoss,
    effective_number_weights, build_loss_v3,
)

__all__ = [
    "build_loss_v1",
    "build_loss_v2",
    "build_loss_v3",
    "PhenologyAwareLoss",
    "FocalCEPlusFocalTversky",
    "FocalCELoss",
    "FocalTverskyLoss",
    "effective_number_weights",
]
