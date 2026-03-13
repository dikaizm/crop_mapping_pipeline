"""Exp D_v2 — Stage 1 v2 per-crop top-K channel union (no Stage 2 validation)."""

import json
import logging
import re
from pathlib import Path
import sys

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT.parent))

from crop_mapping_pipeline.config import KEEP_CLASSES, STAGE1V2_CANDIDATES_JSON

log = logging.getLogger(__name__)


def build_exp_D_v2_indices(mmdd_to_date, local_band_to_idx):
    """
    Load STAGE1V2_CANDIDATES_JSON from the legacy feature_analysis.py Stage 1 run.

    The file stores per-crop ranked channel names such as ``B4_20220730``. For
    Stage 3 we take the ordered union across crops and map the 2022 dates to the
    current reference-year local band map by MMDD.

    Returns (idx_list, names_list).
    """
    candidates_path = STAGE1V2_CANDIDATES_JSON
    if not candidates_path.exists():
        raise FileNotFoundError(
            f"Exp D_v2 input not found: {candidates_path}\n"
            "Run legacy Stage 1 first: python stages/feature_analysis.py --stage 1"
        )

    with open(candidates_path) as f:
        payload = json.load(f)

    candidates_per_crop = payload.get("candidates_per_crop", {})
    if not candidates_per_crop:
        raise ValueError(
            f"{candidates_path} is missing candidates_per_crop. Re-run Stage 1 v2."
        )

    ordered_union = []
    seen = set()
    for crop_id in KEEP_CLASSES:
        for entry in candidates_per_crop.get(str(crop_id), []):
            if entry not in seen:
                seen.add(entry)
                ordered_union.append(entry)

    exp_idx, exp_names = [], []
    skipped = 0

    for entry in ordered_union:
        match = re.match(r"(.+)_(\d{4})(\d{2})(\d{2})$", entry)
        if not match:
            skipped += 1
            continue
        band = match.group(1)
        mmdd = match.group(3) + match.group(4)
        local_date = mmdd_to_date.get(mmdd)
        if local_date is None:
            skipped += 1
            continue
        local_name = f"{band}_{local_date}"
        idx = local_band_to_idx.get(local_name)
        if idx is None:
            skipped += 1
            continue
        exp_idx.append(idx)
        exp_names.append(local_name)

    if not exp_idx:
        raise ValueError(
            f"Exp D_v2: no channels from {candidates_path.name} matched current processed S2 files.\n"
            "Check that Stage 1 v2 output and local processed dates are compatible."
        )
    if skipped:
        log.warning(f"Exp D_v2: {skipped} candidate channel(s) could not be matched")

    log.info(
        f"Exp D_v2: {len(exp_idx)} channels from ordered union of "
        f"{sum(len(candidates_per_crop.get(str(crop_id), [])) for crop_id in KEEP_CLASSES)} "
        f"Stage 1 v2 per-crop candidates"
    )
    return exp_idx, exp_names
