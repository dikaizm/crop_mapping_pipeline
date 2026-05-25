"""GSI-direct selector — single-stage, no CNN oracle, no Stage 1 prefilter.

Ranks all (date × band) channels by per-crop SI_global, selects top-K per crop,
outputs union for Stage 3.

Date candidates use primary year (2022) only — compatible with Stage 3 MMDD matching.
Band-level GSI is averaged across all training years for robustness.
"""

import logging
import tempfile
from datetime import datetime as _dt
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from crop_mapping_pipeline.config import (
    KEEP_CLASSES, CDL_CLASS_NAMES, S2_BAND_NAMES,
    SELECT_TOP_K_PER_CROP, SELECT_GSI_DIRECT_JSON, SELECT_GSI_DIRECT_BANDS,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_FEATURE,
)
from crop_mapping_pipeline.stages.selections._utils import (
    build_channel_names, sample_pixels, save_selection,
)

log = logging.getLogger(__name__)


def _gsi_per_crop(df: pd.DataFrame, bandnames: list[str]) -> dict[int, pd.Series]:
    """Per-crop binary SI_global for each channel. Returns {crop_id: Series(index=bandnames)}."""
    x_all = df[bandnames].values.astype(np.float32)
    y_all = df["class_label"].values

    gsi: dict[int, pd.Series] = {}
    for crop_id in KEEP_CLASSES:
        crop_mask = y_all == crop_id
        rest_mask = np.isin(y_all, KEEP_CLASSES) & ~crop_mask
        if crop_mask.sum() < 10:
            log.warning(f"  {CDL_CLASS_NAMES[crop_id]}: only {crop_mask.sum()} samples — zeros")
            gsi[crop_id] = pd.Series(0.0, index=bandnames)
            continue
        x_c = x_all[crop_mask]
        x_r = x_all[rest_mask]
        si = np.abs(np.nanmean(x_c, 0) - np.nanmean(x_r, 0)) / (np.nanstd(x_c, 0) + 1e-9)
        gsi[crop_id] = pd.Series(si.astype(np.float32), index=bandnames)
    return gsi


def run_gsi_direct(
    years_data: list[tuple[str, list[str], str]],
    top_k: int = SELECT_TOP_K_PER_CROP,
    data_dir: str | None = None,
    out_stem: str | None = None,
) -> list[str]:
    """
    years_data: [(year, s2_paths, cdl_path), ...]
      Primary year (first) supplies date strings; extra years contribute band-level GSI averaging.
    Returns union channel list.
    """
    log.info("GSI-direct: scoring all channels, no prefilter")
    log.info(f"  years={[yr for yr, _, _ in years_data]}  top_k={top_k}")

    # ── Per-year GSI ──────────────────────────────────────────────────────────
    primary_year, primary_s2, primary_cdl = years_data[0]
    primary_bandnames, primary_dates, _ = build_channel_names(primary_s2)

    # Primary year: full channel-level GSI (used for date ranking)
    log.info(f"  Sampling primary year {primary_year} ({len(primary_s2)} files)...")
    df_primary = sample_pixels(primary_s2, primary_cdl, primary_bandnames)
    gsi_primary = _gsi_per_crop(df_primary, primary_bandnames)

    def _doy(mmdd: str) -> int:
        return _dt.strptime(f"2000{mmdd}", "%Y%m%d").timetuple().tm_yday

    def _mmdd_level_gsi(si_series: pd.Series, bandnames: list[str]) -> dict[str, float]:
        """Collapse channel GSI to {band_MMDD: mean_SI}."""
        result: dict[str, list[float]] = {}
        for ch in bandnames:
            parts = ch.rsplit("_", 1)
            if len(parts) != 2:
                continue
            band, date8 = parts[0], parts[1]
            mmdd = date8[4:]
            key  = f"{band}_{mmdd}"
            result.setdefault(key, []).append(float(si_series[ch]))
        return {k: float(np.mean(v)) for k, v in result.items()}

    # Extra years: full channel GSI (date strings differ — stored with bandnames for MMDD matching)
    extra_gsi: list[tuple[str, list[str], dict[int, pd.Series]]] = []
    for year, s2_paths, cdl_path in years_data[1:]:
        log.info(f"  Sampling extra year {year} ({len(s2_paths)} files) for MMDD GSI...")
        bandnames_yr, _, _ = build_channel_names(s2_paths)
        df_yr = sample_pixels(s2_paths, cdl_path, bandnames_yr)
        gsi_yr = _gsi_per_crop(df_yr, bandnames_yr)
        extra_gsi.append((year, bandnames_yr, gsi_yr))

    # ── Select top-K per crop ─────────────────────────────────────────────────
    per_crop: dict[int, list[str]] = {}

    for crop_id in KEEP_CLASSES:
        si_primary = gsi_primary[crop_id]

        if extra_gsi:
            # Primary year MMDD-level GSI
            primary_mmdd = _mmdd_level_gsi(si_primary, primary_bandnames)

            # For each primary channel, average SI across years using nearest-MMDD match
            adjusted = si_primary.copy()
            for ch in primary_bandnames:
                parts = ch.rsplit("_", 1)
                if len(parts) != 2:
                    continue
                band, date8 = parts[0], parts[1]
                mmdd_p = date8[4:]
                doy_p  = _doy(mmdd_p)

                imps = [primary_mmdd.get(f"{band}_{mmdd_p}", float(si_primary[ch]))]
                for _yr, bandnames_yr, gsi_yr in extra_gsi:
                    yr_mmdd = _mmdd_level_gsi(gsi_yr[crop_id], bandnames_yr)
                    cands = [k for k in yr_mmdd if k.startswith(f"{band}_")]
                    if not cands:
                        continue
                    nearest = min(cands, key=lambda k: abs(_doy(k.split("_")[1]) - doy_p))
                    imps.append(yr_mmdd[nearest])

                adjusted[ch] = float(np.mean(imps))
        else:
            adjusted = si_primary

        # Drop NaN channels before ranking
        adjusted = adjusted.fillna(0.0)
        top_channels = adjusted.nlargest(top_k).index.tolist()
        per_crop[crop_id] = top_channels

        log.info(
            f"  {CDL_CLASS_NAMES[crop_id]:20s}: top-3 = {top_channels[:3]}"
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    stem      = out_stem or f"select_gsi_direct_k{top_k}"
    base_dir  = Path(data_dir) if data_dir else SELECT_GSI_DIRECT_JSON.parent
    json_path = base_dir / f"{stem}.json"
    txt_path  = base_dir / f"{stem}_bands.txt"

    union = save_selection(
        per_crop, json_path, txt_path,
        selector="gsi_direct", top_k=top_k,
        meta={"years": [yr for yr, _, _ in years_data], "primary_year": primary_year,
              "n_primary_channels": len(primary_bandnames)},
    )
    log.info(f"GSI-direct: {len(union)} union channels → {json_path}")

    # ── MLflow ────────────────────────────────────────────────────────────────
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_FEATURE)
        from datetime import datetime
        with mlflow.start_run(run_name=f"gsi_direct_{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
            mlflow.log_params({
                "selector":      "gsi_direct",
                "top_k":         top_k,
                "years":         str([yr for yr, _, _ in years_data]),
                "primary_year":  primary_year,
                "n_channels":    len(primary_bandnames),
                "n_union":       len(union),
                "n_crops":       len(KEEP_CLASSES),
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
