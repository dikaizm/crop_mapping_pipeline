"""Naive multi-temporal experiments — 4 NDVI-based phenological dates × selected bands."""

import json
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


def _select_phenol_dates(local_date_to_idx, s2_paths=None, cdl_path=None):
    """Return phenol_map {label: date_str} using NDVI or calendar fallback."""
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

                log.info(f"naive_multitemporal: NDVI-selected dates={phenol_map}")
        except Exception as e:
            log.warning(f"naive_multitemporal: NDVI selection failed ({e}), falling back to calendar")

    if not phenol_map:
        for label, target_mmdd in _CALENDAR_TARGETS.items():
            target_doy = int(target_mmdd)
            phenol_map[label] = min(
                available_dates,
                key=lambda d: abs(int(d[4:]) - target_doy),
            )
        log.info(f"naive_multitemporal: calendar-heuristic dates={phenol_map}")

    return phenol_map


def _band_union_from_candidates(band_candidates: dict, top_k: int | None = None) -> list[str]:
    """Return union of top-K bands per crop from band_candidates_per_crop dict."""
    seen: set[str] = set()
    union_bands: list[str] = []
    for crop_id in KEEP_CLASSES:
        ranked = band_candidates.get(str(crop_id), [])
        k = top_k if top_k is not None else len(ranked)
        for band in ranked[:k]:
            if band not in seen and band in S2_BAND_NAMES:
                seen.add(band)
                union_bands.append(band)
    if not union_bands:
        raise ValueError("No valid bands found in band_candidates_per_crop")
    return union_bands


def build_naive_multitemporal_indices(local_date_to_idx, local_band_to_idx,
                                      s2_paths=None, cdl_path=None):
    """4 phenological dates × all 9 VEGE_BANDS = up to 36 channels."""
    phenol_map = _select_phenol_dates(local_date_to_idx, s2_paths=s2_paths, cdl_path=cdl_path)

    idx, names = [], []
    for _label, d in phenol_map.items():
        off    = local_date_to_idx[d] * N_BANDS_PER_DATE
        idx   += [off + S2_BAND_NAMES.index(b) for b in VEGE_BANDS]
        names += [f"{b}_{d}" for b in VEGE_BANDS]

    seen, dedup_idx, dedup_names = set(), [], []
    for i, name in zip(idx, names):
        if i not in seen:
            seen.add(i)
            dedup_idx.append(i)
            dedup_names.append(name)

    log.info(f"naive_multitemporal: {len(dedup_idx)} channels")
    return dedup_idx, dedup_names, phenol_map


def build_naive_multitemporal_selected_indices(
    local_date_to_idx,
    local_band_to_idx,
    s2_paths=None,
    cdl_path=None,
    top_k: int | None = 5,
    candidates_json: Path | None = None,
    force: bool = False,
):
    """4 phenological dates × GSI or RF top-K band union.

    When candidates_json is None: runs scoped GSI on only the 4 phenol date
    files and caches to gsi_naive_mt_candidates.json alongside the data.
    When candidates_json is provided (RF variant): loads that JSON directly.

    Parameters
    ----------
    top_k : int | None
        Bands per crop before union. None = use all ranked bands.
    candidates_json : Path | None
        Pre-computed candidates JSON (RF variant). None → compute GSI inline.
    force : bool
        Re-run scoped GSI even if cached JSON exists.
    """
    phenol_map = _select_phenol_dates(local_date_to_idx, s2_paths=s2_paths, cdl_path=cdl_path)

    if candidates_json is not None:
        json_path = Path(candidates_json)
        if not json_path.exists():
            raise FileNotFoundError(f"Candidates JSON not found: {json_path}")
        with open(json_path) as f:
            band_candidates = json.load(f)["band_candidates_per_crop"]
    else:
        if s2_paths is None or cdl_path is None:
            raise ValueError("s2_paths and cdl_path required for scoped GSI scoring")
        from crop_mapping_pipeline.stages.selections.band_scoring.gsi.v3 import compute_band_candidates
        phenol_files = [s2_paths[local_date_to_idx[d]] for d in phenol_map.values()]
        out_json     = Path(s2_paths[0]).parent / "gsi_naive_mt_candidates.json"
        band_candidates = compute_band_candidates(phenol_files, cdl_path, out_json=out_json, force=force)

    union_bands = _band_union_from_candidates(band_candidates, top_k=top_k)

    idx, names, skipped = [], [], 0
    for _label, d in phenol_map.items():
        for band in union_bands:
            local_name = f"{band}_{d}"
            i = local_band_to_idx.get(local_name)
            if i is not None:
                idx.append(i)
                names.append(local_name)
            else:
                skipped += 1

    seen, dedup_idx, dedup_names = set(), [], []
    for i, name in zip(idx, names):
        if i not in seen:
            seen.add(i)
            dedup_idx.append(i)
            dedup_names.append(name)

    if not dedup_idx:
        raise ValueError(
            "naive_multitemporal_selected: no bands matched. "
            "Check S2 files include the selected dates."
        )
    if skipped:
        log.warning(
            f"naive_multitemporal_selected: {skipped} (band, date) combos not in local band map"
        )

    log.info(
        f"naive_multitemporal_selected: top_k={top_k} per crop "
        f"→ {len(union_bands)} union bands × {len(phenol_map)} dates → {len(dedup_idx)} channels"
    )
    log.info(f"naive_multitemporal_selected bands: {union_bands}")
    return dedup_idx, dedup_names, phenol_map


# backwards-compat alias
build_exp_B_indices = build_naive_multitemporal_indices
