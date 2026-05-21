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
        m = re.search(r"_(\d{4})_(\d{2})_(\d{2})_processed", fname)
        date_str = f"{m.group(1)}{m.group(2)}{m.group(3)}" if m else fname[:8]
        if date_str not in dates_seen:
            dates_seen.append(date_str)
        all_bandnames.extend([f"{band}_{date_str}" for band in S2_BAND_NAMES])
    all_dates = sorted(dates_seen)
    band_name_to_idx = {name: idx for idx, name in enumerate(all_bandnames)}
    return all_bandnames, all_dates, band_name_to_idx


def sample_pixels(s2_paths: list[str], cdl_path: str,
                  bandnames: list[str]) -> pd.DataFrame:
    """Stack S2 files, read CDL, sample crop pixels. Returns DataFrame(columns=[class_label]+bandnames)."""
    all_arrays = []
    for path in s2_paths:
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)
        arr[arr == S2_NODATA] = np.nan
        all_arrays.append(arr)

    stacked = np.concatenate(all_arrays, axis=0)
    _, height, width = stacked.shape

    with rasterio.open(cdl_path) as src:
        cdl = src.read(1).astype(np.int32)
    assert cdl.shape == (height, width), \
        f"CDL/S2 shape mismatch: {cdl.shape} vs ({height},{width})"

    img_2d = stacked.reshape(len(bandnames), -1).T
    lbl_1d = cdl.flatten()
    del stacked

    valid_mask = np.isin(lbl_1d, KEEP_CLASSES)
    img_valid = img_2d[valid_mask]
    lbl_valid = lbl_1d[valid_mask]

    rng = np.random.default_rng(42)
    n = min(len(lbl_valid), max(1000, int(len(lbl_valid) * SAMPLE_FRACTION)))
    idx = rng.choice(len(lbl_valid), n, replace=False)

    df = pd.DataFrame(img_valid[idx], columns=bandnames)
    df.insert(0, "class_label", lbl_valid[idx].astype(int))
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
