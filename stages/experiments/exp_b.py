"""Exp B — 4 NDVI-based phenological dates × 9 vegetation bands = up to 36 channels."""

import sys
from pathlib import Path

import numpy as np
import rasterio

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT.parent))

from crop_mapping_pipeline.config import (
    S2_BAND_NAMES, N_BANDS_PER_DATE, VEGE_BANDS, KEEP_CLASSES,
)
from crop_mapping_pipeline.stages.experiments.exp_a import _mean_ndvi

import logging
log = logging.getLogger(__name__)

# Calendar fallback targets (mmdd) when NDVI unavailable
_CALENDAR_TARGETS = {"Dormant": "0115", "GreenUp": "0615", "Peak": "0715", "Senescence": "0912"}


def build_exp_B_indices(local_date_to_idx, local_band_to_idx,
                        s2_paths=None, cdl_path=None):
    """4 NDVI-based phenological dates × 9 vegetation bands = up to 36 channels.

    Selects: dormant (min NDVI), green-up (max NDVI rise), peak (max NDVI),
    senescence (max NDVI fall). Falls back to calendar heuristic if unavailable.
    """
    available_dates = sorted(local_date_to_idx.keys())
    phenol_map = {}

    if s2_paths and cdl_path:
        try:
            with rasterio.open(cdl_path) as src:
                cdl_arr = np.isin(src.read(1), KEEP_CLASSES).astype(np.uint8)

            ndvi_scores = {}
            for d in available_dates:
                fi = local_date_to_idx[d]
                ndvi, _ = _mean_ndvi(s2_paths[fi], cdl_arr)
                if ndvi is not None:
                    ndvi_scores[d] = ndvi

            if len(ndvi_scores) >= 4:
                valid_dates = sorted(ndvi_scores.keys())
                ndvis = np.array([ndvi_scores[d] for d in valid_dates])
                diffs = np.diff(ndvis)

                phenol_map["Dormant"]    = valid_dates[int(np.argmin(ndvis))]
                phenol_map["Peak"]       = valid_dates[int(np.argmax(ndvis))]
                phenol_map["GreenUp"]    = valid_dates[int(np.argmax(diffs)) + 1]
                phenol_map["Senescence"] = valid_dates[int(np.argmin(diffs)) + 1]

                log.info(f"Exp B: NDVI-selected dates={phenol_map}")
        except Exception as e:
            log.warning(f"Exp B: NDVI selection failed ({e}), falling back to calendar heuristic")

    if not phenol_map:
        for label, target_mmdd in _CALENDAR_TARGETS.items():
            target_doy = int(target_mmdd)
            phenol_map[label] = min(
                available_dates,
                key=lambda d: abs(int(d[4:]) - target_doy),
            )
        log.info(f"Exp B: calendar-heuristic dates={phenol_map}")

    exp_B_idx, exp_B_names = [], []
    for _label, d in phenol_map.items():
        off          = local_date_to_idx[d] * N_BANDS_PER_DATE
        exp_B_idx   += [off + S2_BAND_NAMES.index(b) for b in VEGE_BANDS]
        exp_B_names += [f"{b}_{d}" for b in VEGE_BANDS]

    seen, dedup_idx, dedup_names = set(), [], []
    for idx, name in zip(exp_B_idx, exp_B_names):
        if idx not in seen:
            seen.add(idx)
            dedup_idx.append(idx)
            dedup_names.append(name)

    log.info(f"Exp B: {len(dedup_idx)} channels")
    return dedup_idx, dedup_names, phenol_map
