"""Experiment registry for Stage 3 training.

Six experiments:
  single_date_gsi     — peak NDVI date, GSI band selection
  single_date_rf      — peak NDVI date, RF band selection
  naive_multitemporal — 4 phenological dates, GSI band selection
  naive_mt_rf         — 4 phenological dates, RF band selection
  gsi                 — multi-temporal, GSI-direct top-K channels
  rf                  — multi-temporal, RF-importance top-K channels

To add an experiment:
  1. Build its band indices in main() of train_segmentation.py
  2. Add an ExperimentConfig entry in build_registry() below
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from crop_mapping_pipeline.config import (
    MLFLOW_EXPERIMENT_TRAIN_6CLASS,
)


@dataclass
class ExperimentConfig:
    key:               str
    description:       str
    band_indices:      Any           # list[int] or dict{yr: (list[int], list[str])}
    band_names:        list          # reference-year channel names
    default_loss:      str  = "v1"  # "v1" | "v2"
    mlflow_experiment: str  = MLFLOW_EXPERIMENT_TRAIN_6CLASS
    extra_kw:          dict = field(default_factory=dict)


def build_registry(
    single_date_idx      = None,  single_date_names      = None,  single_date_key = None,
    single_date_rf_idx   = None,  single_date_rf_names   = None,
    naive_mt_idx         = None,  naive_mt_names         = None,  phenol_map      = None,
    naive_mt_rf_idx      = None,  naive_mt_rf_names      = None,
    gsi_idx              = None,  gsi_names              = None,
    rf_idx               = None,  rf_names               = None,
) -> dict[str, ExperimentConfig]:
    """Build and return the experiment registry.

    Only experiments whose band indices are not None are registered.
    """
    reg: dict[str, ExperimentConfig] = {}

    if single_date_idx is not None:
        reg["single_date_gsi"] = ExperimentConfig(
            key         = "single_date_gsi",
            description = f"Single-date {single_date_key}, GSI bands — {len(single_date_idx)}ch",
            band_indices= single_date_idx,
            band_names  = single_date_names,
        )

    if single_date_rf_idx is not None:
        reg["single_date_rf"] = ExperimentConfig(
            key         = "single_date_rf",
            description = f"Single-date {single_date_key}, RF bands — {len(single_date_rf_idx)}ch",
            band_indices= single_date_rf_idx,
            band_names  = single_date_rf_names,
        )

    if naive_mt_idx is not None:
        reg["naive_mt_gsi"] = ExperimentConfig(
            key         = "naive_mt_gsi",
            description = f"4 phenological dates {list(phenol_map.values())}, GSI bands — {len(naive_mt_idx)}ch",
            band_indices= naive_mt_idx,
            band_names  = naive_mt_names,
        )

    if naive_mt_rf_idx is not None:
        reg["naive_mt_rf"] = ExperimentConfig(
            key         = "naive_mt_rf",
            description = f"4 phenological dates {list(phenol_map.values())}, RF bands — {len(naive_mt_rf_idx)}ch",
            band_indices= naive_mt_rf_idx,
            band_names  = naive_mt_rf_names,
        )

    if gsi_idx is not None:
        reg["gsi"] = ExperimentConfig(
            key         = "gsi",
            description = f"GSI-direct top-K, {len(gsi_idx)}ch — spectral-temporal selection",
            band_indices= gsi_idx,
            band_names  = gsi_names,
        )

    if rf_idx is not None:
        reg["rf"] = ExperimentConfig(
            key         = "rf",
            description = f"RF-direct top-K, {len(rf_idx)}ch — RF importance selection",
            band_indices= rf_idx,
            band_names  = rf_names,
        )

    return reg


def expand_exp_keys(
    requested: list[str],
    registry:  dict[str, ExperimentConfig],
) -> list[str]:
    """Pass-through: all keys are concrete (no shorthands needed)."""
    return list(requested)
