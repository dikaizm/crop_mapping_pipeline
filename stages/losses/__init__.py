"""Versioned loss functions for Stage 3 training.

    v1 — WeightedCrossEntropy             (baseline; static inverse-frequency weights)
    v2 — PhenologyAwareLoss               (NDVI dormancy weighting)
    v3 — FocalCE + FocalTversky           (Effective-Number weights; spatial generalisation)
    v4 — DynamicEffectiveClassBalanced    (per-batch Cui+2019 weights; RS-specific)
    v5 — RecallLoss                       (EMA per-class recall weighting; hard-class mining)

Usage:
    from crop_mapping_pipeline.stages.losses import (
        build_loss_v1, build_loss_v2, build_loss_v3,
        build_loss_v4, build_loss_v5,
    )
"""

from crop_mapping_pipeline.stages.losses.v1 import build_loss_v1
from crop_mapping_pipeline.stages.losses.v2 import PhenologyAwareLoss, build_loss_v2
from crop_mapping_pipeline.stages.losses.v3 import (
    FocalCEPlusFocalTversky, FocalCELoss, FocalTverskyLoss,
    effective_number_weights, build_loss_v3,
)
from crop_mapping_pipeline.stages.losses.v4 import (
    DynamicEffectiveClassBalancedLoss, build_loss_v4,
)
from crop_mapping_pipeline.stages.losses.v5 import RecallLoss, build_loss_v5

__all__ = [
    "build_loss_v1",
    "build_loss_v2",
    "build_loss_v3",
    "build_loss_v4",
    "build_loss_v5",
    "PhenologyAwareLoss",
    "FocalCEPlusFocalTversky",
    "FocalCELoss",
    "FocalTverskyLoss",
    "effective_number_weights",
    "DynamicEffectiveClassBalancedLoss",
    "RecallLoss",
]
