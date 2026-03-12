"""Exp A — Single date (Jul 30) × 9 vegetation bands."""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT.parent))

from crop_mapping_pipeline.config import S2_BAND_NAMES, N_BANDS_PER_DATE, VEGE_BANDS

import logging
log = logging.getLogger(__name__)


def build_exp_A_indices(local_date_to_idx, local_band_to_idx):
    """Single date (Jul 30) × 9 vegetation bands."""
    available_dates = sorted(local_date_to_idx.keys())
    july30_key = next(
        (k for k in available_dates if k[4:6] == "07" and k[6:8] in ("29", "30")),
        available_dates[-1],
    )
    off   = local_date_to_idx[july30_key] * N_BANDS_PER_DATE
    idx   = [off + S2_BAND_NAMES.index(b) for b in VEGE_BANDS]
    names = [f"{b}_{july30_key}" for b in VEGE_BANDS]
    log.info(f"Exp A: date={july30_key}, {len(idx)} channels")
    return idx, names, july30_key
