"""Shared Stage 3 experiment loader for single-stage direct selectors (GSI-direct, RF-direct).

Reads a flat union_channels list from the selector JSON (individual channel names like
"B4_20220730") and maps each to the local channel index via MMDD matching — the same
mechanism used by exp_c_v2 but without the dates×bands cross-product assumption.
"""

import json
import logging
from pathlib import Path

from crop_mapping_pipeline.config import S2_BAND_NAMES
from crop_mapping_pipeline.stages.experiments.base import parse_date

log = logging.getLogger(__name__)


def build_direct_indices(
    json_path: Path,
    mmdd_to_date: dict[str, str],
    local_band_to_idx: dict[str, int],
    selector_name: str = "direct",
) -> tuple[list[int], list[str]]:
    """
    Load union_channels from a direct-selector JSON and map to local channel indices.

    Each channel in union_channels has the form "{band}_{YYYYMMDD}" (e.g., "B4_20220730").
    The date is matched by MMDD (month+day) against mmdd_to_date for the current year's files,
    so the selection transfers across years (2022 selection → 2023/2024 indices).

    Returns (idx_list, name_list).
    """
    if not json_path.exists():
        raise FileNotFoundError(
            f"{selector_name} JSON not found: {json_path}\n"
            f"Run feature selection first:  python feature_analysis_v2.py --stage select --selector {selector_name}"
        )

    with open(json_path) as f:
        payload = json.load(f)

    union_channels: list[str] = payload.get("union_channels", [])
    if not union_channels:
        raise ValueError(f"{json_path.name} has empty union_channels")

    idx_list:  list[int] = []
    name_list: list[str] = []
    skipped = 0

    for ch in union_channels:
        # ch format: "{band}_{YYYYMMDD}"
        parts = ch.rsplit("_", 1)
        if len(parts) != 2:
            log.warning(f"  {selector_name}: malformed channel name '{ch}' — skipped")
            skipped += 1
            continue

        band, date_yyyymmdd = parts
        if band not in S2_BAND_NAMES:
            log.warning(f"  {selector_name}: unknown band '{band}' — skipped")
            skipped += 1
            continue

        mmdd = date_yyyymmdd[4:]  # MMDD portion
        local_date = mmdd_to_date.get(mmdd)
        if local_date is None:
            log.debug(f"  {selector_name}: MMDD={mmdd} not in current year files — skipped")
            skipped += 1
            continue

        local_name = f"{band}_{local_date}"
        idx = local_band_to_idx.get(local_name)
        if idx is None:
            log.warning(f"  {selector_name}: channel '{local_name}' not in band map — skipped")
            skipped += 1
            continue

        idx_list.append(idx)
        name_list.append(local_name)

    if not idx_list:
        raise ValueError(
            f"{selector_name}: no channels from {json_path.name} matched current S2 files.\n"
            "Check that S2 files cover the same calendar periods as the selection year."
        )

    if skipped:
        log.warning(f"  {selector_name}: {skipped}/{len(union_channels)} channels skipped")

    log.info(
        f"  {selector_name}: {len(idx_list)} channels loaded "
        f"({skipped} skipped) from {json_path.name}"
    )
    return idx_list, name_list
