"""Loss v3 — Focal CE + Focal Tversky for spatial generalisation.

Designed for class imbalance AND class-prior shift between train and held-out
areas (raw_v6 test_b has ~0% Rice while train has 14.6%).

Components:
  - Class weighting (default: Median-Frequency Balancing — Eigen & Fergus 2015):
    w_c = median(f) / f_c. Moderate weighting; well-proven for satellite seg.
    Alternatives: 'invsqrt' (1/√f) and 'effnum' (Cui+2019, adaptive β).
  - Focal Cross-Entropy (Lin et al. 2017): (1-p_t)^γ modulator down-weights
    well-classified abundant pixels, focuses learning on hard examples.
  - Focal Tversky (Abraham & Khan 2018): region-level IoU loss biased
    toward recall (α>β); aligns with mIoU evaluation metric.

Compound: L = ce_weight · FocalCE + ft_weight · FocalTversky.

References:
  - Lin et al. 2017 — Focal Loss for Dense Object Detection.
  - Eigen & Fergus 2015 — Predicting Depth, Surface Normals & Semantic Labels.
  - Cui et al. 2019 — Class-Balanced Loss Based on Effective Number of Samples.
  - Abraham & Khan 2018 — A Novel Focal Tversky Loss for Lesion Segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def median_frequency_weights(class_counts):
    """Eigen & Fergus 2015. w_c = median(f) / f_c. Returns tensor (mean ≈ 1)."""
    counts = torch.as_tensor(class_counts, dtype=torch.float64)
    freq   = counts / (counts.sum() + 1e-12)
    med    = torch.median(freq)
    w      = med / (freq + 1e-12)
    return w.float()


def inverse_sqrt_freq_weights(class_counts):
    """w_c = 1/√f_c, normalised to mean=1. Less extreme than 1/f, more than uniform."""
    counts = torch.as_tensor(class_counts, dtype=torch.float64)
    freq   = counts / (counts.sum() + 1e-12)
    w      = 1.0 / torch.sqrt(freq + 1e-12)
    w      = w / w.mean()
    return w.float()


def effective_number_weights(class_counts, beta=None):
    """Cui et al. 2019. w_c = (1-β) / (1-β^n_c).

    If beta is None, auto-select β = 1 - 1/n_min so that β^n_min ≈ 1/e
    and weights span a meaningful range (avoids collapse at very large n).
    Returns tensor normalised to mean=1.
    """
    counts = torch.as_tensor(class_counts, dtype=torch.float64)
    if beta is None:
        n_min = counts[counts > 0].min().item()
        beta  = max(0.0, 1.0 - 1.0 / max(n_min, 1.0))
    eff = 1.0 - torch.pow(torch.tensor(beta, dtype=torch.float64), counts)
    w   = (1.0 - beta) / (eff + 1e-12)
    w   = w / w.mean()
    return w.float()


def build_class_weights(class_counts, mode="median_freq", beta=None):
    """Dispatch on mode: 'median_freq' (default), 'invsqrt', 'effnum'."""
    if mode == "median_freq":
        return median_frequency_weights(class_counts)
    if mode == "invsqrt":
        return inverse_sqrt_freq_weights(class_counts)
    if mode == "effnum":
        return effective_number_weights(class_counts, beta=beta)
    raise ValueError(f"Unknown class-weight mode: {mode}")


class FocalCELoss(nn.Module):
    """Multi-class focal CE with per-class alpha (1-D weight tensor)."""

    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100):
        super().__init__()
        self.alpha        = alpha
        self.gamma        = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        ce = F.cross_entropy(
            logits, target, weight=self.alpha,
            ignore_index=self.ignore_index, reduction="none",
        )
        with torch.no_grad():
            valid = target != self.ignore_index
            t_safe = target.clamp(min=0)
            p      = F.softmax(logits, dim=1)
            pt     = p.gather(1, t_safe.unsqueeze(1)).squeeze(1)
            pt     = pt.clamp(min=1e-7, max=1.0 - 1e-7)
        mod = (1.0 - pt) ** self.gamma
        loss = ce * mod
        if valid.any():
            return loss[valid].mean()
        return loss.mean()


class FocalTverskyLoss(nn.Module):
    """Region-level Tversky with focal exponent. Skips background by default.

    α weights FN, β weights FP — set α > β to favour recall on rare classes.
    γ < 1 focuses gradients on classes with low Tversky index.
    """

    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75,
                 ignore_background=True, smooth=1e-6):
        super().__init__()
        self.alpha             = alpha
        self.beta              = beta
        self.gamma             = gamma
        self.ignore_background = ignore_background
        self.smooth            = smooth

    def forward(self, logits, target):
        C = logits.shape[1]
        p  = F.softmax(logits, dim=1)                                # (B, C, H, W)
        oh = F.one_hot(target.clamp(min=0), num_classes=C)           # (B, H, W, C)
        oh = oh.permute(0, 3, 1, 2).float()                           # (B, C, H, W)

        classes = range(1, C) if self.ignore_background else range(C)
        losses = []
        for c in classes:
            pc = p[:, c]
            gc = oh[:, c]
            tp = (pc * gc).sum()
            fn = ((1 - pc) * gc).sum()
            fp = (pc * (1 - gc)).sum()
            t  = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
            losses.append((1.0 - t).clamp(min=1e-7) ** self.gamma)
        return torch.stack(losses).mean()


class FocalCEPlusFocalTversky(nn.Module):
    """Compound loss: λ_ce · FocalCE + λ_ft · FocalTversky."""

    def __init__(self, alpha=None, gamma_focal=2.0,
                 tv_alpha=0.7, tv_beta=0.3, tv_gamma=0.75,
                 ce_weight=0.6, ft_weight=0.4):
        super().__init__()
        self.focal_ce = FocalCELoss(alpha=alpha, gamma=gamma_focal)
        self.focal_tv = FocalTverskyLoss(alpha=tv_alpha, beta=tv_beta, gamma=tv_gamma)
        self.ce_w     = ce_weight
        self.ft_w     = ft_weight

    def forward(self, logits, target):
        return self.ce_w * self.focal_ce(logits, target) \
             + self.ft_w * self.focal_tv(logits, target)


def build_loss_v3(class_weights_tensor=None, class_counts=None,
                  weight_mode="median_freq", beta=None,
                  gamma_focal=2.0, tv_alpha=0.7, tv_beta=0.3, tv_gamma=0.75,
                  ce_weight=0.6, ft_weight=0.4):
    """Build v3 compound loss.

    Pass class_counts (preferred — applies build_class_weights with weight_mode)
    or a pre-computed class_weights_tensor (fallback).
    """
    if class_counts is not None:
        alpha = build_class_weights(class_counts, mode=weight_mode, beta=beta)
    else:
        alpha = class_weights_tensor.float() if class_weights_tensor is not None else None
    return FocalCEPlusFocalTversky(
        alpha=alpha, gamma_focal=gamma_focal,
        tv_alpha=tv_alpha, tv_beta=tv_beta, tv_gamma=tv_gamma,
        ce_weight=ce_weight, ft_weight=ft_weight,
    )
