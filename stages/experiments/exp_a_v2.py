"""Exp A_v2 — 4 individual phenological dates × 9 vegetation bands.

Each window is a separate experiment (one model per date), unlike Exp B which
combines all 4 dates into a single 36-channel model.

Windows (same targets as Exp B):
    Jan → nearest acquisition to Jan-15
    Mar → nearest acquisition to Mar-15
    Jul → nearest acquisition to Jul-15
    Nov → nearest acquisition to Nov-15
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT.parent))

from crop_mapping_pipeline.config import S2_BAND_NAMES, N_BANDS_PER_DATE, VEGE_BANDS

import logging
log = logging.getLogger(__name__)


PHENOL_TARGETS = {"Jan": "0115", "Mar": "0315", "Jul": "0715", "Nov": "1115"}


def build_exp_A_v2_indices(local_date_to_idx, local_band_to_idx):
    """Build per-window indices for Exp A_v2.

    Returns
    -------
    variants : dict[str, tuple[list[int], list[str], str]]
        {window_label: (idx_list, names_list, matched_date)}
    """
    available_dates = sorted(local_date_to_idx.keys())

    variants = {}
    for label, target_mmdd in PHENOL_TARGETS.items():
        target_doy = int(target_mmdd)
        matched_date = min(
            available_dates,
            key=lambda d: abs(int(d[4:]) - target_doy),
        )
        off   = local_date_to_idx[matched_date] * N_BANDS_PER_DATE
        idx   = [off + S2_BAND_NAMES.index(b) for b in VEGE_BANDS]
        names = [f"{b}_{matched_date}" for b in VEGE_BANDS]
        log.info(f"Exp A_v2 [{label}]: date={matched_date}, {len(idx)} channels")
        variants[label] = (idx, names, matched_date)

    return variants
