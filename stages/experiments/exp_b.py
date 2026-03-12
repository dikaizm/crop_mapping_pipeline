"""Exp B — 4 phenological dates × 9 vegetation bands = up to 36 channels."""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT.parent))

from crop_mapping_pipeline.config import S2_BAND_NAMES, N_BANDS_PER_DATE, VEGE_BANDS

import logging
log = logging.getLogger(__name__)


def build_exp_B_indices(local_date_to_idx, local_band_to_idx):
    """4 phenological dates × 9 vegetation bands = up to 36 channels.

    Dates are chosen as the acquisition nearest to the mid-point of each
    phenological season (Jan-15, Mar-15, Jul-15, Nov-15).
    """
    available_dates = sorted(local_date_to_idx.keys())

    phenol_targets = {"Jan": "0115", "Mar": "0315", "Jul": "0715", "Nov": "1115"}
    phenol_map     = {}
    for label, target_mmdd in phenol_targets.items():
        target_doy = int(target_mmdd)
        match = min(
            available_dates,
            key=lambda d: abs(int(d[4:]) - target_doy),
        )
        phenol_map[label] = match

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

    log.info(f"Exp B: dates={list(phenol_map.values())}, {len(dedup_idx)} channels")
    return dedup_idx, dedup_names, phenol_map
