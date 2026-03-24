"""Experiment registry for Stage 3 training.

Each experiment is declared as an ExperimentConfig entry.
To add a new experiment:
  1. Build its band indices in main() of train_segmentation.py
  2. Add one ExperimentConfig entry in build_registry() below

ExperimentConfig fields
-----------------------
key             Unique string identifier (matches --exp CLI value).
description     Human-readable description logged to MLflow.
band_indices    list[int] or dict{yr: (list[int], list[str])} — channel indices.
band_names      list[str] — channel names for the reference year (for logging).
default_loss    "v1" (WeightedCrossEntropy) or "v2" (PhenologyAwareLoss).
                CLI --loss-version overrides this for all experiments in a run.
extra_kw        Extra keyword arguments forwarded to run_experiment()
                (e.g. source MLflow run IDs for artifact linking).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from crop_mapping_pipeline.config import (
    MLFLOW_EXPERIMENT_TRAIN,
    MLFLOW_EXPERIMENT_TRAIN_V3,
)


@dataclass
class ExperimentConfig:
    key:              str
    description:      str
    band_indices:     Any                        # list[int] or dict{yr: (list, list)}
    band_names:       list                       # reference-year channel names
    default_loss:     str  = "v1"               # "v1" | "v2"
    mlflow_experiment: str = MLFLOW_EXPERIMENT_TRAIN
    extra_kw:         dict = field(default_factory=dict)


def build_registry(
    # ── Exp A ──────────────────────────────────────────────────────────────
    exp_A_idx,   exp_A_names,   july30_key,
    # ── Exp B ──────────────────────────────────────────────────────────────
    exp_B_idx,   exp_B_names,   phenol_map,
    # ── Exp C (Stage 2 CNN forward-selection) ──────────────────────────────
    exp_C_idx,   exp_C_names,
    resolved_stage2_run_id   = None,
    resolved_project_run_id  = None,
    # ── Exp C_v2 (Stage 2v2 two-phase CNN) ────────────────────────────────
    exp_C_v2_idx  = None, exp_C_v2_names  = None,
    resolved_stage2v3_run_id    = None,
    resolved_project_run_id_v2  = None,
    # ── Exp C_v2_rf (Stage 2v2 RF selector) ───────────────────────────────
    exp_C_v2_rf_idx = None, exp_C_v2_rf_names = None,
    # ── Exp C_v3 (Stage 2v3 incremental sweep, multi phase × k) ───────────
    exp_C_v3_variants = None,              # {(phase, k): (idx, names)}
    # ── Exp D (Stage 1v3 GSI direct, no Stage 2) ──────────────────────────
    exp_D_idx  = None, exp_D_names  = None,
    # ── Exp D_v2 (Stage 1v2 channel union) ────────────────────────────────
    exp_D_v2_idx = None, exp_D_v2_names = None,
    # ── Exp A_v2 (per-window single-date) ──────────────────────────────────
    exp_A_v2_variants = None,              # {label: (idx, names, date)}
) -> dict[str, ExperimentConfig]:
    """Build and return the experiment registry.

    Only experiments whose band indices are not None are registered.
    C_v3 variants are registered individually as 'C_v3_{phase}_k{k:02d}'.
    """

    reg: dict[str, ExperimentConfig] = {}

    # ── Baselines ───────────────────────────────────────────────────────────

    reg["A"] = ExperimentConfig(
        key         = "A",
        description = f"Single-date {july30_key}, 9ch — conventional baseline",
        band_indices= exp_A_idx,
        band_names  = exp_A_names,
        default_loss= "v1",
    )

    # ── Exp A_v2: one entry per phenological window ─────────────────────────
    for label, (idx, names, date) in (exp_A_v2_variants or {}).items():
        key = f"A_v2_{label}"
        reg[key] = ExperimentConfig(
            key         = key,
            description = f"Single-date {date} [{label}], 9ch — phenological window baseline",
            band_indices= idx,
            band_names  = names,
            default_loss= "v1",
        )

    reg["B"] = ExperimentConfig(
        key         = "B",
        description = f"4 phenological dates {list(phenol_map.values())}, {len(exp_B_idx)}ch — multi-temporal naive",
        band_indices= exp_B_idx,
        band_names  = exp_B_names,
        default_loss= "v1",
    )

    # ── Proposed method ─────────────────────────────────────────────────────

    if exp_C_idx is not None:
        n = len(exp_C_idx) if isinstance(exp_C_idx, list) else "per-year"
        reg["C"] = ExperimentConfig(
            key         = "C",
            description = f"Stage2 CNN forward-selection K*={n}ch — proposed method",
            band_indices= exp_C_idx,
            band_names  = exp_C_names,
            default_loss= "v1",
            extra_kw    = {
                "source_stage2_run_id":  resolved_stage2_run_id,
                "source_project_run_id": resolved_project_run_id,
            },
        )

    if exp_C_v2_idx is not None:
        n = len(exp_C_v2_idx) if isinstance(exp_C_v2_idx, list) else "per-year"
        reg["C_v2"] = ExperimentConfig(
            key         = "C_v2",
            description = f"Stage2v2 two-phase CNN K*_dates×K*_bands={n}ch",
            band_indices= exp_C_v2_idx,
            band_names  = exp_C_v2_names,
            default_loss= "v1",
            extra_kw    = {
                "source_stage2_run_id":  resolved_stage2v3_run_id,
                "source_project_run_id": resolved_project_run_id_v2,
            },
        )

    # ── Ablations ───────────────────────────────────────────────────────────

    if exp_C_v2_rf_idx is not None:
        reg["C_v2_rf"] = ExperimentConfig(
            key         = "C_v2_rf",
            description = f"Stage2v2 RF selector K*={len(exp_C_v2_rf_idx)}ch — ablation: RF vs CNN oracle",
            band_indices= exp_C_v2_rf_idx,
            band_names  = exp_C_v2_rf_names,
            default_loss= "v1",
        )

    if exp_D_idx is not None:
        reg["D"] = ExperimentConfig(
            key         = "D",
            description = f"Stage1v3 GSI top-K={len(exp_D_idx)}ch — ablation: no CNN validation",
            band_indices= exp_D_idx,
            band_names  = exp_D_names,
            default_loss= "v1",
        )

    if exp_D_v2_idx is not None:
        reg["D_v2"] = ExperimentConfig(
            key         = "D_v2",
            description = f"Stage1v2 candidate union={len(exp_D_v2_idx)}ch — legacy Stage1 baseline",
            band_indices= exp_D_v2_idx,
            band_names  = exp_D_v2_names,
            default_loss= "v1",
        )

    # ── C_v3: one entry per (phase, k) combination ─────────────────────────

    for (phase, k), (idx, names) in sorted((exp_C_v3_variants or {}).items()):
        key = f"C_v3_{phase}_k{k:02d}"
        reg[key] = ExperimentConfig(
            key               = key,
            description       = f"Stage2v3 {phase}_sweep k={k} {len(idx)}ch",
            band_indices      = idx,
            band_names        = names,
            default_loss      = "v1",
            mlflow_experiment = MLFLOW_EXPERIMENT_TRAIN_V3,
        )

    return reg


def expand_exp_keys(
    requested: list[str],
    registry:  dict[str, ExperimentConfig],
) -> list[str]:
    """Expand shorthand keys into concrete registry keys.

    'C_v3' expands to all registered 'C_v3_*' keys (sorted).
    All other keys are passed through as-is.
    """
    expanded = []
    for key in requested:
        if key == "A_v2":
            matched = sorted(k for k in registry if k.startswith("A_v2_"))
            if not matched:
                raise RuntimeError("No A_v2 variants registered.")
            expanded.extend(matched)
        elif key == "C_v3":
            matched = sorted(k for k in registry if k.startswith("C_v3_"))
            if not matched:
                raise RuntimeError(
                    "No C_v3 variants registered — did you pass --v3-phase and --v3-k?"
                )
            expanded.extend(matched)
        else:
            expanded.append(key)
    return expanded
