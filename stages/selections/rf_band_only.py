"""RF band-only selection — ranks bands on fixed domain dates (no date selection).

Trains a binary RandomForestClassifier per crop on the given domain channels,
extracts Gini importance, collapses to band-level ranking by averaging across
dates, and outputs top-K bands per crop.

Unlike rf_direct.py, this does NOT select dates — dates come from domain
knowledge (peak NDVI or phenological stages). Only bands are ranked via RF.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from crop_mapping_pipeline.config import (
    KEEP_CLASSES, CDL_CLASS_NAMES, S2_BAND_NAMES,
    RF_N_ESTIMATORS, RF_MAX_PIXELS,
)
from crop_mapping_pipeline.stages.selections._utils import build_channel_names, sample_pixels

log = logging.getLogger(__name__)


def run_rf_band_only(
    s2_paths: list[str],
    cdl_path: str,
    domain_channel_names: list[str],
    top_k: int = 9,
) -> dict[str, list[str]]:
    """Train per-crop binary RF on domain-date channels, rank bands by mean importance.

    For single-date experiments each band appears once → RF importance is band ranking.
    For multi-date experiments importance is averaged across dates per band.

    Args:
        s2_paths: reference-year S2 files.
        cdl_path: reference-year CDL raster path.
        domain_channel_names: channel names for the fixed domain dates
            (e.g. ["B2_20240730", "B3_20240730", ...]).
        top_k: bands per crop after ranking (default 9 = all VEGE_BANDS).

    Returns:
        {str(crop_id): [ranked_band_names, ...]} — ready for band_candidates_per_crop.
    """
    all_bandnames, _, _ = build_channel_names(s2_paths)

    valid_channels = [ch for ch in domain_channel_names if ch in all_bandnames]
    missing = set(domain_channel_names) - set(valid_channels)
    if missing:
        log.warning("rf_band_only: %d domain channels missing from S2 data", len(missing))
    if not valid_channels:
        raise ValueError("No valid domain channels found in S2 data")

    log.info("rf_band_only: %d domain channels, top_k=%d", len(valid_channels), top_k)

    df = sample_pixels(s2_paths, cdl_path, all_bandnames)

    band_candidates: dict[str, list[str]] = {}
    for crop_id in KEEP_CLASSES:
        crop_name = CDL_CLASS_NAMES[crop_id]

        y = (df["class_label"].values == crop_id).astype(int)
        x = df[valid_channels].values

        n_pos = y.sum()
        if n_pos < 10:
            log.warning("  %s: only %d positive samples — skipping", crop_name, n_pos)
            band_candidates[str(crop_id)] = []
            continue

        if len(y) > RF_MAX_PIXELS:
            rng = np.random.default_rng(crop_id)
            idx = rng.choice(len(y), RF_MAX_PIXELS, replace=False)
            x, y = x[idx], y[idx]

        col_medians = np.nanmedian(x, axis=0)
        nan_mask = np.isnan(x)
        x = np.where(nan_mask, col_medians, x)

        rf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            class_weight="balanced",
            n_jobs=-1,
            random_state=crop_id,
        )
        rf.fit(x, y)
        importance = pd.Series(rf.feature_importances_, index=valid_channels)

        band_imp: dict[str, list[float]] = {}
        for ch in valid_channels:
            band = ch.split("_")[0]
            if band in S2_BAND_NAMES:
                band_imp.setdefault(band, []).append(float(importance[ch]))

        band_avg = {b: float(np.mean(v)) for b, v in band_imp.items()}
        ranked = sorted(band_avg, key=band_avg.get, reverse=True)[:top_k]
        band_candidates[str(crop_id)] = ranked

        log.info("  %-20s: top-3 bands = %s", crop_name, ranked[:3])

    return band_candidates


def save_rf_band_json(
    band_candidates: dict[str, list[str]],
    json_path: Path,
):
    """Save band_candidates_per_crop to JSON (same key as gsi_candidates.json)."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "selector": "rf_band_only",
        "band_candidates_per_crop": band_candidates,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Saved rf_band_only → %s", json_path)
