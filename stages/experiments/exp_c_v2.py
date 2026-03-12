"""Exp C_v2 — Stage 2v2 CNN forward selection (two-phase: date × band per crop)."""

import json
import logging
from pathlib import Path
import sys
import os

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT.parent))

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
import mlflow

from crop_mapping_pipeline.config import (
    S2_BAND_NAMES, N_BANDS_PER_DATE,
    PROCESSED_DIR, MLFLOW_TRACKING_URI,
    STAGE3_EXP_C_V2_JSON,
)
from crop_mapping_pipeline.stages.experiments.base import parse_date

log = logging.getLogger(__name__)


def build_exp_C_v2_indices(mmdd_to_date, local_band_to_idx, stage2v3_run_id=None):
    """
    Load STAGE3_EXP_C_V2_JSON (written by feature_analysis_v2.py).

    Reads union_dates and union_bands from the Stage 2v2 output and maps each
    (date, band) pair to a flat local channel index.
    Falls back to downloading from MLflow if not found locally and run_id given.

    Returns (idx_list, names_list, resolved_run_id).
    """
    v2_json_path             = STAGE3_EXP_C_V2_JSON
    resolved_stage2v3_run_id = stage2v3_run_id

    if not v2_json_path.exists():
        if stage2v3_run_id:
            log.info(
                f"STAGE3_EXP_C_V2_JSON not found locally — fetching from MLflow "
                f"run_id={stage2v3_run_id}"
            )
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            try:
                local_path = mlflow.artifacts.download_artifacts(
                    run_id=stage2v3_run_id,
                    artifact_path="stage3_exp_c_v2.json",
                    dst_path=str(PROCESSED_DIR),
                )
                v2_json_path = Path(local_path)
                log.info(f"Downloaded stage3_exp_c_v2.json → {local_path}")
            except Exception as e:
                raise RuntimeError(
                    f"Could not download stage3_exp_c_v2.json from MLflow run "
                    f"{stage2v3_run_id}: {e}"
                )
        else:
            raise FileNotFoundError(
                f"STAGE3_EXP_C_V2_JSON not found: {v2_json_path}\n"
                "Run Stage 2v2 first:  python feature_analysis_v2.py --stage 2\n"
                "Or pass --stage2v3-run-id to fetch from MLflow."
            )

    with open(v2_json_path) as f:
        payload = json.load(f)

    union_dates = payload.get("union_dates", [])
    union_bands = payload.get("union_bands", [])
    if not union_dates or not union_bands:
        raise ValueError(
            f"STAGE3_EXP_C_V2_JSON is missing union_dates or union_bands: {v2_json_path}"
        )

    exp_C_v2_idx, exp_C_v2_names = [], []
    skipped = 0

    for date_yyyymmdd in union_dates:
        mmdd       = date_yyyymmdd[4:]
        local_date = mmdd_to_date.get(mmdd)
        if local_date is None:
            log.warning(f"Exp C v2: date MMDD={mmdd} (from {date_yyyymmdd}) not in mmdd_to_date — skipped")
            skipped += 1
            continue
        for band in union_bands:
            if band not in S2_BAND_NAMES:
                log.warning(f"Exp C v2: band '{band}' not in S2_BAND_NAMES — skipped")
                skipped += 1
                continue
            local_name = f"{band}_{local_date}"
            idx        = local_band_to_idx.get(local_name)
            if idx is not None:
                exp_C_v2_idx.append(idx)
                exp_C_v2_names.append(local_name)
            else:
                skipped += 1

    if not exp_C_v2_idx:
        raise ValueError(
            f"Exp C v2: no bands from {v2_json_path.name} matched current processed S2 files.\n"
            "Check that S2 files for the same dates as Stage 2v2 are present."
        )
    if skipped:
        log.warning(f"Exp C v2: {skipped} (date, band) pair(s) could not be matched")

    log.info(
        f"Exp C v2: {len(exp_C_v2_idx)} channels "
        f"({len(union_dates)} union dates × {len(union_bands)} union bands) "
        f"from {v2_json_path.name}"
    )
    if resolved_stage2v3_run_id:
        log.info(f"Exp C v2 source Stage 2v2 run_id: {resolved_stage2v3_run_id}")
    return exp_C_v2_idx, exp_C_v2_names, resolved_stage2v3_run_id


def build_exp_C_v2_indices_projected(s2_processed, project_run_id=None):
    """
    Load stage3_exp_c_v2_bands_projected.json and compute per-year band indices.
    Returns dict {yr: (idx_list, names_list)} or (None, None) if unavailable.
    """
    p = PROCESSED_DIR / "stage3_exp_c_v2_bands_projected.json"
    resolved_project_run_id = project_run_id

    if not p.exists():
        log.info(
            "stage3_exp_c_v2_bands_projected.json not found locally — "
            + (f"fetching from MLflow run_id={project_run_id}"
               if project_run_id else "no project_run_id provided, skipping MLflow fetch")
        )
        if project_run_id:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            try:
                local_path = mlflow.artifacts.download_artifacts(
                    run_id=project_run_id,
                    artifact_path="stage3_exp_c_v2_bands_projected.json",
                    dst_path=str(PROCESSED_DIR),
                )
                p = Path(local_path)
                log.info(f"Downloaded stage3_exp_c_v2_bands_projected.json → {local_path}")
            except Exception as e:
                log.warning(f"Could not fetch projected v2 bands: {e}")
                return None, None
        else:
            return None, None

    with open(p) as f:
        projected = json.load(f)

    by_year = {}
    for path in s2_processed:
        yr = Path(path).name.split("_")[1]
        by_year.setdefault(yr, []).append(path)

    result = {}
    for yr, band_names in projected.items():
        yr_files = sorted(by_year.get(yr, []))
        if not yr_files:
            log.warning(f"Exp C v2 projected: no S2 files for year {yr} — skipping")
            continue

        yr_band_to_idx = {}
        for file_idx, path in enumerate(yr_files):
            d = parse_date(path)
            if d:
                for band_pos, b in enumerate(S2_BAND_NAMES):
                    yr_band_to_idx[f"{b}_{d}"] = file_idx * N_BANDS_PER_DATE + band_pos

        idx, names, skipped = [], [], 0
        for band_name in band_names:
            i = yr_band_to_idx.get(band_name)
            if i is not None:
                idx.append(i)
                names.append(band_name)
            else:
                skipped += 1

        if skipped:
            log.warning(f"Exp C v2 projected {yr}: {skipped} band(s) could not be matched")
        log.info(f"Exp C v2 projected {yr}: {len(idx)} channels")
        result[yr] = (idx, names)

    if resolved_project_run_id:
        log.info(f"Exp C v2 source project run_id: {resolved_project_run_id}")
    return (result if result else None), resolved_project_run_id
