"""Shared pixel-sampling utilities for single-stage direct selectors."""

import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from crop_mapping_pipeline.config import S2_BAND_NAMES, S2_NODATA, KEEP_CLASSES, SAMPLE_FRACTION

log = logging.getLogger(__name__)


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
    percentile: float | None = None,
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
        "percentile":   percentile,
        "selection_mode": "percentile" if percentile is not None else "top_k",
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


def hardware_info() -> dict:
    """CPU/GPU/RAM identity for mlflow params — static per-machine, not a metric."""
    import platform

    cpu_name = platform.processor()
    if not cpu_name and platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        cpu_name = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass

    info = {"cpu_name": cpu_name or "unknown", "cpu_cores": os.cpu_count()}
    try:
        import psutil
        info["ram_total_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except ImportError:
        info["ram_total_gb"] = None

    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_name"]      = torch.cuda.get_device_name(0)
            info["gpu_count"]     = torch.cuda.device_count()
            info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        else:
            info["gpu_name"], info["gpu_count"], info["gpu_memory_gb"] = "none", 0, None
    except Exception:
        info["gpu_name"], info["gpu_count"], info["gpu_memory_gb"] = "unknown", None, None
    return info


def _metric_safe(name: str) -> str:
    """mlflow metric keys: keep alnum/_-./space — replace anything else."""
    return re.sub(r"[^0-9A-Za-z_\-./ ]", "_", name)


def log_selection_run(
    *,
    selector: str,
    run_name_prefix: str,
    per_crop: dict[int, list[str]],
    union: list[str],
    json_path: Path,
    params: dict,
    duration_s: float,
    threshold: float | None = None,
    extra_metrics: dict | None = None,
):
    """Log a band-selection run to MLflow: results, per-crop counts, runtime,
    machine identity, and (if available) live system metrics. Non-fatal on error."""
    import shutil
    import tempfile
    from datetime import datetime

    import mlflow
    from crop_mapping_pipeline.config import (
        MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_FEATURE, CDL_CLASS_NAMES,
    )

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_FEATURE)
        run_name = f"{run_name_prefix}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        try:
            run_ctx = mlflow.start_run(run_name=run_name, log_system_metrics=True)
        except TypeError:   # older mlflow without system-metrics kwarg
            run_ctx = mlflow.start_run(run_name=run_name)

        with run_ctx:
            hw = hardware_info()
            mlflow.log_params({
                **params,
                **{f"hw_{k}": v for k, v in hw.items()},
            })
            # Results
            mlflow.log_metric("n_union_channels", len(union))
            mlflow.log_metric("runtime_seconds", round(duration_s, 2))
            mlflow.log_metric("runtime_minutes", round(duration_s / 60.0, 3))
            if threshold is not None:
                mlflow.log_metric("pooled_threshold", float(threshold))
            # Per-crop selected counts
            for cid, chs in per_crop.items():
                cname = CDL_CLASS_NAMES.get(cid, str(cid))
                mlflow.log_metric(_metric_safe(f"n_sel_{cname}"), len(chs))
            for k, v in (extra_metrics or {}).items():
                if v is not None:
                    mlflow.log_metric(_metric_safe(k), float(v))
            # Full selection JSON (per-crop bands + union)
            with tempfile.TemporaryDirectory() as tmp:
                tmp_json = Path(tmp) / Path(json_path).name
                shutil.copy(json_path, tmp_json)
                mlflow.log_artifact(str(tmp_json))
    except Exception as e:
        log.warning(f"MLflow logging failed (non-fatal): {e}")
