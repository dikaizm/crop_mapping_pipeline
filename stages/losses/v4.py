"""Loss v4 — Dynamic Effective Class Balanced Loss.

Extends Cui et al. 2019 effective-number weighting to be computed per batch
rather than from static global train-set counts. Per-batch pixel counts are
used to derive effective-number weights dynamically, so the loss adapts as
class distribution shifts within mini-batches.

Key difference from v3 (static weights):
- v3 weights computed once from full CDL raster before training
- v4 weights recomputed every forward pass from current batch pixel counts

This adapts to class distributions encountered during training and is less
sensitive to train/test prior shift.

Reference:
  Zhang et al. 2023 — "A Dynamic Effective Class Balanced Approach for
  Remote Sensing Imagery Semantic Segmentation of Imbalanced Data"
  Remote Sensing 15(7), MDPI. https://doi.org/10.3390/rs15071768

  Cui et al. 2019 — "Class-Balanced Loss Based on Effective Number of Samples"
  CVPR. https://arxiv.org/abs/1901.05555
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicEffectiveClassBalancedLoss(nn.Module):
    """Cross-entropy with per-batch effective-number class weights.

    For each forward pass:
      1. Count pixels per class n_c in the batch.
      2. Effective number: E_c = (1 - β^n_c) / (1 - β).
      3. Weight: w_c = 1 / E_c, normalised to mean = 1.
      4. Apply as class weights to cross-entropy.

    Classes absent in the batch receive weight = fallback_weight (default 2.0,
    ensuring unseen classes are still penalised when predicted).

    Args:
        num_classes:      Total number of classes (including background).
        beta:             Effective-number smoothing factor. 0.9999 works well
                          for batch-scale pixel counts (thousands per class).
        fallback_weight:  Weight assigned to classes with n_c = 0 in batch.
        ignore_index:     Label index to ignore (default -100).
    """

    def __init__(self, num_classes, beta=0.9999,
                 fallback_weight=2.0, ignore_index=-100):
        super().__init__()
        self.num_classes     = num_classes
        self.beta            = beta
        self.fallback_weight = fallback_weight
        self.ignore_index    = ignore_index

    def forward(self, logits, target):
        """
        Args:
            logits: (B, C, H, W) float
            target: (B, H, W)    long
        """
        C    = self.num_classes
        beta = self.beta

        # --- per-batch pixel counts per class ---
        flat   = target.view(-1)
        counts = torch.zeros(C, dtype=torch.float32, device=logits.device)
        for c in range(C):
            if c != self.ignore_index:
                counts[c] = (flat == c).sum().float()

        # --- effective-number weights ---
        # E_c = (1 - β^n_c) / (1 - β);  w_c = 1/E_c
        # For n_c=0: E_c→0 → assign fallback weight directly.
        one_minus_beta    = 1.0 - beta
        beta_pow          = torch.pow(torch.tensor(beta, device=logits.device), counts)
        eff_num           = (1.0 - beta_pow).clamp(min=1e-12) / one_minus_beta
        w                 = one_minus_beta / (1.0 - beta_pow).clamp(min=1e-12)

        absent            = counts == 0
        w[absent]         = self.fallback_weight

        # normalise to mean = 1 (excluding absent classes from normalisation)
        present_w         = w[~absent]
        if present_w.numel() > 0:
            w[~absent]    = w[~absent] / present_w.mean()

        return F.cross_entropy(logits, target,
                               weight=w, ignore_index=self.ignore_index)


def build_loss_v4(num_classes, beta=0.9999,
                  fallback_weight=2.0, ignore_index=-100):
    """Build DynamicEffectiveClassBalancedLoss.

    Args:
        num_classes:     Number of output classes.
        beta:            Effective-number beta (0.9999 for batch-pixel scale).
        fallback_weight: Weight for classes absent from batch.
        ignore_index:    Label to ignore.
    """
    return DynamicEffectiveClassBalancedLoss(
        num_classes=num_classes, beta=beta,
        fallback_weight=fallback_weight, ignore_index=ignore_index,
    )
