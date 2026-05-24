"""Shared pixel-sampling utilities for single-stage direct selectors."""

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from crop_mapping_pipeline.config import S2_BAND_NAMES, S2_NODATA, KEEP_CLASSES, SAMPLE_FRACTION


def build_channel_names(s2_paths: list[str]) -> tuple[list[str], list[str], dict[str, int]]:
    """Return (all_bandnames, all_dates, band_name_to_idx) for a list of S2 files."""
    all_bandnames: list[str] = []
    dates_seen: list[str] = []
    for path in s2_paths:
        fname = os.path.basename(path)
        m = re.search(r"_(\d{4}_\d{2}_\d{2})(_processed)?\.tif$", fname)
        date_str = m.group(1).replace("_", "") if m else fname[:8]
        if date_str not in dates_seen:
            dates_seen.append(date_str)
        all_bandnames.extend([f"{band}_{date_str}" for band in S2_BAND_NAMES])
    all_dates = sorted(dates_seen)
    band_name_to_idx = {name: idx for idx, name in enumerate(all_bandnames)}
    return all_bandnames, all_dates, band_name_to_idx


def sample_pixels(s2_paths: list[str], cdl_path: str,
                  bandnames: list[str]) -> pd.DataFrame:
    """Sample crop pixels from S2 files without loading all files into RAM at once.

    Strategy: read CDL once → determine valid pixel indices → for each S2 file
    read only the sampled rows. Peak RAM = 1 S2 file (11 bands × H × W × 4 bytes ≈ 1 GB)
    instead of all 25 files stacked (≈ 28 GB).
    """
    # Read CDL once to get valid pixel indices and labels
    with rasterio.open(cdl_path) as src:
        cdl = src.read(1).astype(np.int32)
        height, width = cdl.shape

    lbl_1d = cdl.flatten()
    del cdl

    valid_mask = np.isin(lbl_1d, KEEP_CLASSES)
    valid_indices = np.where(valid_mask)[0]   # flat pixel indices of crop pixels
    lbl_valid = lbl_1d[valid_mask]
    del lbl_1d

    # Draw sample indices once (same seed → reproducible)
    rng = np.random.default_rng(42)
    n = min(len(valid_indices), max(1000, int(len(valid_indices) * SAMPLE_FRACTION)))
    chosen = rng.choice(len(valid_indices), n, replace=False)
    sample_flat_idx = valid_indices[chosen]   # flat pixel positions to extract
    lbl_sample = lbl_valid[chosen]
    del valid_indices, lbl_valid

    # Pre-allocate output array: (n_samples, n_channels)
    n_channels = len(bandnames)
    data = np.full((n, n_channels), np.nan, dtype=np.float32)

    # Fill columns from each S2 file — one file at a time (11 bands × H × W × 4 bytes)
    col = 0
    for path in s2_paths:
        with rasterio.open(path) as src:
            n_file_bands = src.count
            arr = src.read().astype(np.float32)   # (11, H, W)

        arr[arr == S2_NODATA] = np.nan
        arr_2d = arr.reshape(n_file_bands, -1).T  # (H*W, 11)
        del arr

        data[:, col:col + n_file_bands] = arr_2d[sample_flat_idx]
        del arr_2d
        col += n_file_bands

    df = pd.DataFrame(data, columns=bandnames)
    df.insert(0, "class_label", lbl_sample.astype(int))
    return df


def save_selection(
    per_crop: dict[int, list[str]],
    json_path: Path,
    txt_path: Path,
    selector: str,
    top_k: int,
    meta: dict | None = None,
) -> list[str]:
    """Compute union of per-crop channels, save JSON + TXT, return union list."""
    seen: dict[str, None] = {}
    for channels in per_crop.values():
        for ch in channels:
            seen[ch] = None
    union: list[str] = list(seen.keys())

    from crop_mapping_pipeline.config import CDL_CLASS_NAMES
    from datetime import datetime

    payload = {
        "run_ts":       datetime.now().strftime("%Y%m%d-%H%M%S"),
        "selector":     selector,
        "top_k":        top_k,
        "n_union":      len(union),
        "per_crop":     {str(k): v for k, v in per_crop.items()},
        "union_channels": union,
        **(meta or {}),
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("\n".join(union) + "\n")

    return union
