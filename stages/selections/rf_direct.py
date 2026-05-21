"""RF-direct selector — single-stage, no CNN oracle, no Stage 1 prefilter.

Trains a binary RandomForestClassifier per crop on all (date × band) channels,
uses Gini importance to rank channels, selects top-K per crop, outputs union.

Pixel samples pooled from all training years for robust importance estimates.
"""

import logging
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from crop_mapping_pipeline.config import (
    KEEP_CLASSES, CDL_CLASS_NAMES,
    SELECT_TOP_K_PER_CROP, SELECT_RF_DIRECT_JSON, SELECT_RF_DIRECT_BANDS,
    RF_N_ESTIMATORS, RF_MAX_PIXELS,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_FEATURE,
)
from crop_mapping_pipeline.stages.selections._utils import (
    build_channel_names, sample_pixels, save_selection,
)

log = logging.getLogger(__name__)


def run_rf_direct(
    years_data: list[tuple[str, list[str], str]],
    top_k: int = SELECT_TOP_K_PER_CROP,
    data_dir: str | None = None,
) -> list[str]:
    """
    years_data: [(year, s2_paths, cdl_path), ...]
      Primary year (first) supplies channel names for output.
      All years contribute pixel samples — RF trained on pooled pixels.
      Extra-year channels are scored separately and averaged at band level
      to inform primary-year channel ranking.
    Returns union channel list.
    """
    log.info("RF-direct: scoring all channels via RF importance, no prefilter")
    log.info(f"  years={[yr for yr, _, _ in years_data]}  top_k={top_k}  n_trees={RF_N_ESTIMATORS}")

    primary_year, primary_s2, primary_cdl = years_data[0]
    primary_bandnames, _, _ = build_channel_names(primary_s2)
    n_channels = len(primary_bandnames)
    log.info(f"  Primary year {primary_year}: {n_channels} channels")

    # ── Sample pixels — primary year ─────────────────────────────────────────
    log.info(f"  Sampling {primary_year}...")
    df_primary = sample_pixels(primary_s2, primary_cdl, primary_bandnames)

    # ── Sample extra years for band-level RF importance averaging ─────────────
    extra_band_importance: list[dict[int, pd.Series]] = []
    for year, s2_paths, cdl_path in years_data[1:]:
        log.info(f"  Sampling extra year {year} for band importance averaging...")
        bandnames_yr, _, _ = build_channel_names(s2_paths)
        df_yr = sample_pixels(s2_paths, cdl_path, bandnames_yr)
        extra_band_importance.append((year, bandnames_yr, df_yr))

    # ── Per-crop RF on primary year + band-averaging across extra years ────────
    per_crop: dict[int, list[str]] = {}

    for crop_id in KEEP_CLASSES:
        crop_name = CDL_CLASS_NAMES[crop_id]

        # --- Primary year: full channel-level RF importance ---
        y = (df_primary["class_label"].values == crop_id).astype(int)
        x = df_primary[primary_bandnames].values

        # Cap pixel count
        if len(y) > RF_MAX_PIXELS:
            rng = np.random.default_rng(crop_id)
            idx = rng.choice(len(y), RF_MAX_PIXELS, replace=False)
            x, y = x[idx], y[idx]

        n_pos = y.sum()
        if n_pos < 10:
            log.warning(f"  {crop_name}: only {n_pos} positive samples — skipping RF, using zeros")
            per_crop[crop_id] = []
            continue

        # Handle NaN: replace with column median
        col_medians = np.nanmedian(x, axis=0)
        nan_mask = np.isnan(x)
        x = np.where(nan_mask, col_medians, x)

        rf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            class_weight="balanced",
            n_jobs=-1,
            random_state=crop_id,
        )
        rf.fit(x, y)
        importance_primary = pd.Series(rf.feature_importances_, index=primary_bandnames)

        # --- Extra years: band-level importance, average with primary ---
        if extra_band_importance:
            from crop_mapping_pipeline.config import S2_BAND_NAMES

            def _band_level(importance: pd.Series, bandnames: list[str]) -> pd.Series:
                band_imp = {}
                for band in S2_BAND_NAMES:
                    keys = [k for k in bandnames if k.startswith(f"{band}_")]
                    band_imp[band] = float(importance[keys].mean()) if keys else 0.0
                return pd.Series(band_imp)

            all_band_imps = [_band_level(importance_primary, primary_bandnames)]

            for _yr, bandnames_yr, df_yr in extra_band_importance:
                y_yr = (df_yr["class_label"].values == crop_id).astype(int)
                x_yr = df_yr[bandnames_yr].values
                if y_yr.sum() < 10:
                    continue
                if len(y_yr) > RF_MAX_PIXELS:
                    rng = np.random.default_rng(crop_id + 1000)
                    idx = rng.choice(len(y_yr), RF_MAX_PIXELS, replace=False)
                    x_yr, y_yr = x_yr[idx], y_yr[idx]
                col_med = np.nanmedian(x_yr, axis=0)
                x_yr = np.where(np.isnan(x_yr), col_med, x_yr)
                rf_yr = RandomForestClassifier(
                    n_estimators=RF_N_ESTIMATORS,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=crop_id,
                )
                rf_yr.fit(x_yr, y_yr)
                imp_yr = pd.Series(rf_yr.feature_importances_, index=bandnames_yr)
                all_band_imps.append(_band_level(imp_yr, bandnames_yr))

            avg_band_imp = pd.concat(all_band_imps, axis=1).mean(axis=1)
            primary_band_imp = _band_level(importance_primary, primary_bandnames)

            # Rescale primary channel importances by (avg_band / primary_band)
            adjusted = importance_primary.copy()
            for band in S2_BAND_NAMES:
                p = float(primary_band_imp.get(band, 1e-9))
                a = float(avg_band_imp.get(band, p))
                scale = a / (p + 1e-9)
                keys = [k for k in primary_bandnames if k.startswith(f"{band}_")]
                adjusted[keys] = adjusted[keys] * scale
        else:
            adjusted = importance_primary

        top_channels = adjusted.nlargest(top_k).index.tolist()
        per_crop[crop_id] = top_channels
        log.info(f"  {crop_name:20s}: top-3 = {top_channels[:3]}")

    # ── Save ──────────────────────────────────────────────────────────────────
    json_path = Path(data_dir) / "select_rf_direct.json" if data_dir else SELECT_RF_DIRECT_JSON
    txt_path  = Path(data_dir) / "select_rf_direct_bands.txt" if data_dir else SELECT_RF_DIRECT_BANDS

    union = save_selection(
        per_crop, json_path, txt_path,
        selector="rf_direct", top_k=top_k,
        meta={"years": [yr for yr, _, _ in years_data], "primary_year": primary_year,
              "n_primary_channels": n_channels, "rf_n_estimators": RF_N_ESTIMATORS},
    )
    log.info(f"RF-direct: {len(union)} union channels → {json_path}")

    # ── MLflow ────────────────────────────────────────────────────────────────
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_FEATURE)
        from datetime import datetime
        with mlflow.start_run(run_name=f"rf_direct_{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
            mlflow.log_params({
                "selector":      "rf_direct",
                "top_k":         top_k,
                "years":         str([yr for yr, _, _ in years_data]),
                "primary_year":  primary_year,
                "n_channels":    n_channels,
                "n_union":       len(union),
                "n_crops":       len(KEEP_CLASSES),
                "rf_n_estimators": RF_N_ESTIMATORS,
            })
            mlflow.log_metric("n_union_channels", len(union))
            with tempfile.TemporaryDirectory() as tmp:
                import shutil
                tmp_json = Path(tmp) / json_path.name
                shutil.copy(json_path, tmp_json)
                mlflow.log_artifact(str(tmp_json))
    except Exception as e:
        log.warning(f"MLflow logging failed (non-fatal): {e}")

    return union
