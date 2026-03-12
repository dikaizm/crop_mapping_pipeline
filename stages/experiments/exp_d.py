"""Exp D — Stage 1 GSI top-K direct (ablation: no Stage 2 CNN forward selection)."""

import json
import logging
from pathlib import Path
import sys

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT.parent))

from crop_mapping_pipeline.config import STAGE3_EXP_D_JSON

log = logging.getLogger(__name__)


def build_exp_D_indices(mmdd_to_date, local_band_to_idx):
    """
    Load STAGE3_EXP_D_JSON (written by feature_analysis_v2.py Stage 1v3).

    Exp D uses Stage 1 SI_global top-K dates × top-K bands directly, without
    any Stage 2 CNN forward selection.  Ablation vs Exp C_v2.

    Returns (idx_list, names_list).
    """
    d_json_path = STAGE3_EXP_D_JSON
    if not d_json_path.exists():
        raise FileNotFoundError(
            f"Exp D input not found: {d_json_path}\n"
            "Run Stage 1v3 first:  python stages/feature_analysis_v2.py --stage 1"
        )

    with open(d_json_path) as f:
        payload = json.load(f)

    union_dates = payload.get("union_dates", [])
    union_bands = payload.get("union_bands", [])
    if not union_dates or not union_bands:
        raise ValueError(f"STAGE3_EXP_D_JSON is missing union_dates or union_bands: {d_json_path}")

    exp_D_idx, exp_D_names = [], []
    skipped = 0

    for date_yyyymmdd in union_dates:
        mmdd       = date_yyyymmdd[4:]
        local_date = mmdd_to_date.get(mmdd)
        if local_date is None:
            skipped += 1
            continue
        for band in union_bands:
            local_name = f"{band}_{local_date}"
            idx        = local_band_to_idx.get(local_name)
            if idx is not None:
                exp_D_idx.append(idx)
                exp_D_names.append(local_name)
            else:
                skipped += 1

    if not exp_D_idx:
        raise ValueError(
            f"Exp D: no bands from {d_json_path.name} matched current processed S2 files.\n"
            "Check that S2 files for the same dates as Stage 1v3 are present."
        )
    if skipped:
        log.warning(f"Exp D: {skipped} (date, band) pair(s) could not be matched")

    log.info(
        f"Exp D: {len(exp_D_idx)} channels "
        f"({len(union_dates)} union dates × {len(union_bands)} union bands) "
        f"from {d_json_path.name}"
    )
    return exp_D_idx, exp_D_names
