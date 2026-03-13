import logging
import os
import pathlib
import re
import tempfile
from datetime import datetime
import json

import numpy as np
import pandas as pd
import rasterio

import mlflow

import crop_mapping_pipeline.stages.feature_analysis_v2 as fa2

log = logging.getLogger(__name__)


def run_stage1v3(s2_paths, cdl_path, data_dir=None):
    log.info("Stage 1v3: computing per-crop GSI and deriving date + band candidates...")

    fa2.mlflow_setup()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    stage1_run = mlflow.start_run(run_name=f"stage1v3_{ts}")

    all_bandnames = []
    all_dates_set = []
    for s2_path in s2_paths:
        fname = os.path.basename(s2_path)
        match = re.search(r"_(\d{4})_(\d{2})_(\d{2})_processed", fname)
        date_str = f"{match.group(1)}{match.group(2)}{match.group(3)}" if match else fname[:8]
        if date_str not in all_dates_set:
            all_dates_set.append(date_str)
        all_bandnames.extend([f"{band}_{date_str}" for band in fa2.S2_BAND_NAMES])

    all_dates = sorted(all_dates_set)
    band_name_to_idx = {name: idx for idx, name in enumerate(all_bandnames)}
    n_channels = len(all_bandnames)
    log.info(f"S2 files: {len(s2_paths)}  |  {n_channels} channels  |  {len(all_dates)} dates")

    log.info(f"Loading {len(s2_paths)} S2 files for Stage 1v3 pixel sampling...")
    all_arrays = []
    for s2_path in s2_paths:
        with rasterio.open(s2_path) as src:
            arr = src.read().astype(np.float32)
        arr[arr == fa2.S2_NODATA] = np.nan
        all_arrays.append(arr)

    stacked = np.concatenate(all_arrays, axis=0)
    _, height, width = stacked.shape
    log.info(f"Stacked S2: {n_channels} channels × {height} × {width} px")

    with rasterio.open(cdl_path) as src:
        cdl = src.read(1).astype(np.int32)
    assert cdl.shape == (height, width), f"CDL/S2 shape mismatch: {cdl.shape} vs ({height},{width})"

    img_2d = stacked.reshape(n_channels, -1).T
    lbl_1d = cdl.flatten()

    valid_mask = np.isin(lbl_1d, fa2.KEEP_CLASSES)
    img_valid = img_2d[valid_mask]
    lbl_valid = lbl_1d[valid_mask]

    log.info(
        f"Labeled crop pixels: {len(lbl_valid):,} "
        f"({100 * len(lbl_valid) / len(lbl_1d):.1f}% of {len(lbl_1d):,})"
    )

    rng = np.random.default_rng(42)
    n = min(len(lbl_valid), max(1000, int(len(lbl_valid) * fa2.SAMPLE_FRACTION)))
    idx = rng.choice(len(lbl_valid), n, replace=False)

    df = pd.DataFrame(img_valid[idx], columns=all_bandnames)
    df.insert(0, "class_label", lbl_valid[idx].astype(int))

    log.info(f"Sampled {len(df):,} pixels (SAMPLE_FRACTION={fa2.SAMPLE_FRACTION})")
    log.info(f"Classes in sample: {sorted(df['class_label'].unique())}")
    sample_values = df[all_bandnames].values
    nan_pixels = np.isnan(sample_values).any(axis=1).sum()
    nan_channels = np.isnan(sample_values).any(axis=0).sum()
    log.info(
        f"NaN in sample: {nan_pixels:,} pixels ({100 * nan_pixels / len(df):.1f}%) "
        f"have ≥1 NaN channel;  {nan_channels}/{n_channels} channels have ≥1 NaN pixel"
    )
    del stacked, img_2d

    log.info("Computing per-crop binary SI_global (one-vs-all)...")
    x_all = df[all_bandnames].values.astype(np.float32)
    y_all = df["class_label"].values

    gsi_dict = {}
    for crop_id in fa2.KEEP_CLASSES:
        crop_mask = y_all == crop_id
        rest_mask = np.isin(y_all, fa2.KEEP_CLASSES) & ~crop_mask
        if crop_mask.sum() < 10:
            log.warning(
                f"Crop {crop_id} ({fa2.CDL_CLASS_NAMES[crop_id]}) has only "
                f"{crop_mask.sum()} samples — using zeros"
            )
            gsi_dict[crop_id] = pd.Series(0.0, index=all_bandnames)
            continue
        x_crop = x_all[crop_mask]
        x_rest = x_all[rest_mask]
        mean_crop = np.nanmean(x_crop, axis=0)
        std_crop = np.nanstd(x_crop, axis=0)
        mean_rest = np.nanmean(x_rest, axis=0)
        si = np.abs(mean_crop - mean_rest) / (std_crop + 1e-9)
        gsi_dict[crop_id] = pd.Series(si.astype(np.float32), index=all_bandnames)

    gsi_df = pd.DataFrame(gsi_dict)
    gsi_mean_global = gsi_df.mean(axis=1).sort_values(ascending=False)
    values = gsi_df.values
    log.info(f"gsi_df shape: {gsi_df.shape}  (channels × crops)")
    log.info(
        f"  SI range:  min={np.nanmin(values):.4f}  p25={np.nanpercentile(values, 25):.4f}  "
        f"median={np.nanmedian(values):.4f}  p75={np.nanpercentile(values, 75):.4f}  "
        f"p95={np.nanpercentile(values, 95):.4f}  max={np.nanmax(values):.4f}"
    )
    log.info(f"  Top-K selection: TOP_DATES_PER_CROP={fa2.TOP_DATES_PER_CROP}  TOP_BANDS_PER_CROP={fa2.TOP_BANDS_PER_CROP}  (no threshold — always selects top-K)")

    date_candidates_per_crop = {}
    band_candidates_per_crop = {}

    for crop_id in fa2.KEEP_CLASSES:
        crop_key = str(crop_id)
        si_crop = gsi_df[crop_id] if crop_id in gsi_df.columns else gsi_mean_global
        if crop_id not in gsi_df.columns:
            log.warning(
                f"Crop {crop_id} ({fa2.CDL_CLASS_NAMES[crop_id]}) not in sample — "
                "falling back to global mean ranking"
            )

        date_si = {}
        for date in all_dates:
            band_keys = [f"{band}_{date}" for band in fa2.S2_BAND_NAMES if f"{band}_{date}" in si_crop.index]
            date_si[date] = float(si_crop[band_keys].mean()) if band_keys else 0.0
        sorted_dates = sorted(date_si.items(), key=lambda item: item[1], reverse=True)
        selected_dates = [date for date, _ in sorted_dates[: fa2.TOP_DATES_PER_CROP]]
        date_candidates_per_crop[crop_key] = selected_dates

        band_si = {}
        for band in fa2.S2_BAND_NAMES:
            band_keys = [f"{band}_{date}" for date in selected_dates if f"{band}_{date}" in si_crop.index]
            band_si[band] = float(si_crop[band_keys].mean()) if band_keys else 0.0
        sorted_bands = sorted(band_si.items(), key=lambda item: item[1], reverse=True)
        band_candidates_per_crop[crop_key] = [band for band, _ in sorted_bands[: fa2.TOP_BANDS_PER_CROP]]

        log.info(
            f"  {fa2.CDL_CLASS_NAMES[crop_id]:20s}: "
            f"top {len(selected_dates)} dates={selected_dates[:3]}...  "
            f"top {len(band_candidates_per_crop[crop_key])} bands={band_candidates_per_crop[crop_key][:3]}... "
            f"(scored within {len(selected_dates)}-date window)"
        )

    stage1_path = fa2.STAGE1V3_CANDIDATES_JSON
    if data_dir:
        stage1_path = pathlib.Path(data_dir) / "s2" / "2022" / "stage1v3_candidates.json"
    os.makedirs(os.path.dirname(stage1_path), exist_ok=True)

    run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    payload = {
        "run_ts": run_ts,
        "all_dates": all_dates,
        "date_candidates_per_crop": date_candidates_per_crop,
        "band_candidates_per_crop": band_candidates_per_crop,
    }
    with open(stage1_path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info(f"Stage 1v3 candidates saved: {stage1_path}")

    fa2.save_exp_d_bands(date_candidates_per_crop, band_candidates_per_crop, band_name_to_idx, data_dir=data_dir)

    mlflow.log_params(
        {
            "stage": "1v3_date_band_ranking",
            "version": "v3",
            "n_images": len(s2_paths),
            "n_dates": len(all_dates),
            "total_channels": n_channels,
            "sample_fraction": fa2.SAMPLE_FRACTION,
            "n_sampled": len(df),
            "top_dates_per_crop": fa2.TOP_DATES_PER_CROP,
            "top_bands_per_crop": fa2.TOP_BANDS_PER_CROP,
            "max_bands_per_crop": fa2.MAX_BANDS_PER_CROP,
            "keep_classes": str(fa2.KEEP_CLASSES),
        }
    )

    rows = []
    for crop_id in fa2.KEEP_CLASSES:
        crop_key = str(crop_id)
        for rank, date in enumerate(date_candidates_per_crop[crop_key], start=1):
            rows.append(
                {
                    "crop_id": crop_id,
                    "crop_name": fa2.CDL_CLASS_NAMES[crop_id],
                    "type": "date",
                    "rank": rank,
                    "value": date,
                }
            )
        for rank, band in enumerate(band_candidates_per_crop[crop_key], start=1):
            rows.append(
                {
                    "crop_id": crop_id,
                    "crop_name": fa2.CDL_CLASS_NAMES[crop_id],
                    "type": "band",
                    "rank": rank,
                    "value": band,
                }
            )

    with tempfile.TemporaryDirectory() as tmp:
        artifact_path = pathlib.Path(tmp) / "stage1v3_per_crop_candidates.csv"
        artifact_path.write_text(pd.DataFrame(rows).to_csv(index=False))
        mlflow.log_artifact(str(artifact_path))
    mlflow.log_artifact(str(stage1_path))
    if fa2.STAGE3_EXP_D_JSON.exists():
        mlflow.log_artifact(str(fa2.STAGE3_EXP_D_JSON))
    if fa2.STAGE3_EXP_D_BANDS.exists():
        mlflow.log_artifact(str(fa2.STAGE3_EXP_D_BANDS))

    heatmap_dir = fa2.FIGURES_DIR / "stage1v3_gsi"
    for heatmap_path in fa2.plot_gsi_heatmaps(gsi_df, all_dates, heatmap_dir):
        mlflow.log_artifact(str(heatmap_path))

    mlflow.end_run(status="FINISHED")
    log.info(f"Stage 1v3 MLflow run_id: {stage1_run.info.run_id}")

    return date_candidates_per_crop, band_candidates_per_crop, band_name_to_idx, all_dates
