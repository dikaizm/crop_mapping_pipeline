"""Exp A — Single date (peak growing season by NDVI) × 9 vegetation bands."""

import sys
from pathlib import Path

import numpy as np
import rasterio

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT.parent))

from crop_mapping_pipeline.config import (
    S2_BAND_NAMES, N_BANDS_PER_DATE, VEGE_BANDS, KEEP_CLASSES,
)

import logging
log = logging.getLogger(__name__)

# B4=Red, B8=NIR (0-based in S2_BAND_NAMES)
_B4_IDX = S2_BAND_NAMES.index("B4") + 1   # rasterio 1-based
_B8_IDX = S2_BAND_NAMES.index("B8") + 1


def _mean_ndvi(tif_path, cdl_arr, valid_thresh=0.80):
    """Return (mean_ndvi, valid_frac) over crop pixels. Returns (None, 0) on failure."""
    try:
        with rasterio.open(tif_path) as src:
            nodata = src.nodata if src.nodata is not None else -9999.0
            b4 = src.read(_B4_IDX).astype(np.float32)
            b8 = src.read(_B8_IDX).astype(np.float32)
        valid = (cdl_arr > 0) & (b4 != nodata) & (b8 != nodata) & np.isfinite(b4) & np.isfinite(b8)
        valid_frac = valid.sum() / max(cdl_arr.sum(), 1)
        if valid_frac < valid_thresh:
            return None, valid_frac
        denom = np.where((b8[valid] + b4[valid]) == 0, 1e-6, b8[valid] + b4[valid])
        return float(np.mean((b8[valid] - b4[valid]) / denom)), valid_frac
    except Exception:
        return None, 0.0


def build_exp_A_indices(local_date_to_idx, local_band_to_idx,
                        s2_paths=None, cdl_path=None):
    """Single date (peak NDVI over crop pixels) × 9 vegetation bands.

    Falls back to nearest Jul-14/Jul-29 date if NDVI computation is unavailable.
    """
    available_dates = sorted(local_date_to_idx.keys())

    best_date = None
    if s2_paths and cdl_path:
        try:
            with rasterio.open(cdl_path) as src:
                cdl_arr = np.isin(src.read(1), KEEP_CLASSES).astype(np.uint8)
            ndvi_scores = {}
            for d in available_dates:
                fi = local_date_to_idx[d]
                ndvi, vf = _mean_ndvi(s2_paths[fi], cdl_arr)
                if ndvi is not None:
                    ndvi_scores[d] = ndvi
            if ndvi_scores:
                best_date = max(ndvi_scores, key=ndvi_scores.get)
                log.info(f"Exp A: NDVI-selected date={best_date} (NDVI={ndvi_scores[best_date]:.4f})")
        except Exception as e:
            log.warning(f"Exp A: NDVI selection failed ({e}), falling back to Jul heuristic")

    if best_date is None:
        best_date = next(
            (k for k in available_dates if k[4:6] == "07" and k[6:8] in ("14", "29", "30")),
            available_dates[-1],
        )
        log.info(f"Exp A: heuristic date={best_date}")

    off   = local_date_to_idx[best_date] * N_BANDS_PER_DATE
    idx   = [off + S2_BAND_NAMES.index(b) for b in VEGE_BANDS]
    names = [f"{b}_{best_date}" for b in VEGE_BANDS]
    log.info(f"Exp A: {len(idx)} channels")
    return idx, names, best_date
