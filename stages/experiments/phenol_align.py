"""Phenological alignment for cross-year band index remapping.

Converts training-year band selections (expressed as date strings) to
phenologically equivalent dates in a target year using NDVI rank ordering.

NDVI rank is used solely as a phenological calendar proxy — rank 0 = lowest
NDVI (dormant/winter), rank N = highest NDVI (peak growing season). The actual
band selection (GSI/RF) is unchanged; only the date assignment is remapped.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import rasterio

from crop_mapping_pipeline.config import (
    KEEP_CLASSES, N_BANDS_PER_DATE, S2_BAND_NAMES,
)
from crop_mapping_pipeline.stages.experiments.exp_a import _mean_ndvi
from crop_mapping_pipeline.stages.experiments.base import parse_date

log = logging.getLogger(__name__)

_DATE_RE = re.compile(r"_(\d{8})$")


def _parse_date_from_band_name(band_name: str) -> str | None:
    """Extract YYYYMMDD from band name like 'B5_20230714'."""
    m = _DATE_RE.search(band_name)
    return m.group(1) if m else None


def _compute_ndvi_ranks(s2_paths: list[str], cdl_path: str) -> dict[str, int]:
    """Compute per-date NDVI rank for a year's S2 files.

    Returns {date_str: rank} where rank=0 is lowest NDVI (dormant)
    and rank=N is highest NDVI (peak growing season).
    Dates with <80% valid crop pixels are excluded from ranking but
    assigned the nearest valid rank.
    """
    with rasterio.open(cdl_path) as src:
        cdl_arr = np.isin(src.read(1), KEEP_CLASSES).astype(np.uint8)

    ndvi_scores: dict[str, float] = {}
    date_order: list[str] = []

    for p in sorted(s2_paths):
        d = parse_date(p)
        if d is None:
            continue
        date_order.append(d)
        ndvi, valid_frac = _mean_ndvi(p, cdl_arr)
        if ndvi is not None:
            ndvi_scores[d] = ndvi
        else:
            log.debug(f"  phenol_align: skipping {d} (valid={valid_frac:.2f})")

    if not ndvi_scores:
        log.warning("  phenol_align: no valid NDVI scores — falling back to date order")
        return {d: i for i, d in enumerate(date_order)}

    sorted_dates = sorted(ndvi_scores, key=ndvi_scores.get)
    ranks = {d: i for i, d in enumerate(sorted_dates)}

    # Assign excluded dates to nearest ranked neighbour by calendar position
    for d in date_order:
        if d not in ranks:
            nearest = min(sorted_dates, key=lambda x: abs(int(x) - int(d)))
            ranks[d] = ranks[nearest]
            log.debug(f"  phenol_align: {d} excluded → nearest rank neighbour {nearest}")

    log.info(
        f"  phenol_align: {len(ndvi_scores)} valid dates ranked. "
        f"dormant={sorted_dates[0]}  peak={sorted_dates[-1]}"
    )
    return ranks


def align_band_names_to_year(
    band_names: list[str],
    train_s2: list[str],
    test_s2: list[str],
    train_cdl: str,
    test_cdl: str,
) -> tuple[list[str], list[int]]:
    """Remap training-year band_names to phenologically equivalent test-year channels.

    Steps:
      1. Compute NDVI ranks for training year and test year independently.
      2. For each selected band: parse training date → training rank → test date → test channel.
      3. Return (remapped_band_names, test_band_indices).

    Band names format: '{SPECTRAL_BAND}_{YYYYMMDD}', e.g. 'B5_20230714'.
    Band indices are global: date_position * N_BANDS_PER_DATE + band_offset.
    """
    log.info("  phenol_align: computing NDVI ranks for training year …")
    train_ranks = _compute_ndvi_ranks(train_s2, train_cdl)

    log.info("  phenol_align: computing NDVI ranks for test year …")
    test_ranks = _compute_ndvi_ranks(test_s2, test_cdl)

    # Invert test ranks: rank → date
    rank_to_test_date: dict[int, str] = {}
    for d, r in test_ranks.items():
        if r not in rank_to_test_date:
            rank_to_test_date[r] = d

    # Build test year date → file index map
    test_date_to_fi: dict[str, int] = {}
    for fi, p in enumerate(sorted(test_s2)):
        d = parse_date(p)
        if d:
            test_date_to_fi[d] = fi

    remapped_names: list[str] = []
    remapped_indices: list[int] = []
    seen: set[int] = set()

    for band_name in band_names:
        train_date = _parse_date_from_band_name(band_name)
        spectral   = band_name.rsplit("_", 1)[0]   # e.g. 'B5'

        if train_date is None:
            log.warning(f"  phenol_align: cannot parse date from '{band_name}' — skipping")
            continue
        if train_date not in train_ranks:
            log.warning(f"  phenol_align: date {train_date} from '{band_name}' not in training year ranks — skipping (band_names may use wrong reference year)")
            continue

        rank = train_ranks[train_date]

        # Find test date with same rank; if missing, pick nearest rank
        if rank in rank_to_test_date:
            test_date = rank_to_test_date[rank]
        else:
            nearest_rank = min(rank_to_test_date, key=lambda r: abs(r - rank))
            test_date = rank_to_test_date[nearest_rank]
            log.debug(f"  phenol_align: rank {rank} missing in test → using rank {nearest_rank} ({test_date})")

        if test_date not in test_date_to_fi:
            log.warning(f"  phenol_align: test date {test_date} not found in test S2 files — skipping")
            continue

        fi = test_date_to_fi[test_date]
        if spectral not in S2_BAND_NAMES:
            log.warning(f"  phenol_align: unknown spectral band '{spectral}' — skipping")
            continue

        global_idx = fi * N_BANDS_PER_DATE + S2_BAND_NAMES.index(spectral)
        if global_idx in seen:
            continue
        seen.add(global_idx)

        remapped_names.append(f"{spectral}_{test_date}")
        remapped_indices.append(global_idx)

    log.info(
        f"  phenol_align: {len(band_names)} train channels → "
        f"{len(remapped_indices)} test channels remapped"
    )
    return remapped_names, remapped_indices
