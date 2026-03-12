"""Exp C_v2_rf — Stage 2v2 RF importance selection (ablation: RF vs CNN oracle)."""

import json
import logging
from pathlib import Path
import sys

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT.parent))

from crop_mapping_pipeline.config import (
    S2_BAND_NAMES,
    STAGE3_EXP_C_V2_RF_JSON,
)

log = logging.getLogger(__name__)


def build_exp_C_v2_rf_indices(mmdd_to_date, local_band_to_idx):
    """
    Load STAGE3_EXP_C_V2_RF_JSON (written by feature_analysis_v2.py --selector rf).
    Same structure as Exp C_v2 but uses RF importance instead of CNN oracle.
    Returns (idx_list, names_list).
    """
    rf_json_path = STAGE3_EXP_C_V2_RF_JSON
    if not rf_json_path.exists():
        raise FileNotFoundError(
            f"Exp C_v2_rf input not found: {rf_json_path}\n"
            "Run Stage 2v2-RF first:  python stages/feature_analysis_v2.py --stage 2 --selector rf"
        )

    with open(rf_json_path) as f:
        payload = json.load(f)

    union_dates = payload.get("union_dates", [])
    union_bands = payload.get("union_bands", [])
    if not union_dates or not union_bands:
        raise ValueError("STAGE3_EXP_C_V2_RF_JSON is missing union_dates or union_bands")

    idx, names, skipped = [], [], 0
    for date_yyyymmdd in union_dates:
        mmdd       = date_yyyymmdd[4:]
        local_date = mmdd_to_date.get(mmdd)
        if local_date is None:
            skipped += 1
            continue
        for band in union_bands:
            local_name = f"{band}_{local_date}"
            i          = local_band_to_idx.get(local_name)
            if i is not None:
                idx.append(i)
                names.append(local_name)
            else:
                skipped += 1

    if not idx:
        raise ValueError(
            f"Exp C_v2_rf: no bands matched current S2 files from {rf_json_path.name}"
        )
    if skipped:
        log.warning(f"Exp C_v2_rf: {skipped} (date, band) pair(s) could not be matched")

    log.info(
        f"Exp C_v2_rf: {len(idx)} channels "
        f"({len(union_dates)} union dates × {len(union_bands)} union bands) "
        f"from {rf_json_path.name}"
    )
    return idx, names
