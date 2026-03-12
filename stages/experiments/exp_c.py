"""Exp C (v1) — Stage 2v1 CNN forward selection (legacy single-phase)."""

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
    PROCESSED_DIR, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_FEATURE,
    STAGE3_EXP_C_BANDS, STAGE3_EXP_C_BANDS_PROJECTED,
)
from crop_mapping_pipeline.stages.experiments.base import parse_date

log = logging.getLogger(__name__)


def _fetch_exp_c_bands_from_mlflow(run_id=None):
    """
    Download stage3_exp_c_bands.txt from a Stage 2 MLflow artifact.
    If run_id is None, auto-selects the latest Stage 2 run.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    if run_id is None:
        exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT_FEATURE)
        if exp is None:
            raise RuntimeError(
                f"MLflow experiment '{MLFLOW_EXPERIMENT_FEATURE}' not found. "
                "Run Stage 2 first."
            )
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="attributes.run_name LIKE 'stage2v2_binary_fwd_%'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not runs:
            raise RuntimeError(
                f"No Stage 2 runs found in MLflow experiment '{MLFLOW_EXPERIMENT_FEATURE}'. "
                "Run Stage 2 first or pass --stage2-run-id explicitly."
            )
        run_id = runs[0].info.run_id
        log.info(f"Auto-selected Stage 2 run: {runs[0].info.run_name} (run_id={run_id})")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="stage3_exp_c_bands.txt",
        dst_path=str(PROCESSED_DIR),
    )
    log.info(f"Downloaded stage3_exp_c_bands.txt → {local_path}")
    return Path(local_path), run_id


def _fetch_projected_from_mlflow(run_id=None):
    """
    Download stage3_exp_c_bands_projected.json from a project run MLflow artifact.
    If run_id is None, auto-selects the latest project run.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    if run_id is None:
        exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT_FEATURE)
        if exp is None:
            raise RuntimeError(f"MLflow experiment '{MLFLOW_EXPERIMENT_FEATURE}' not found.")
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="attributes.run_name LIKE 'stage2v2_project_%'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not runs:
            raise RuntimeError(
                f"No project runs found in '{MLFLOW_EXPERIMENT_FEATURE}'. "
                "Run:  python feature_analysis.py --stage project"
            )
        run_id = runs[0].info.run_id
        log.info(f"Auto-selected project run: {runs[0].info.run_name} (run_id={run_id})")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="stage3_exp_c_bands_projected.json",
        dst_path=str(PROCESSED_DIR),
    )
    log.info(f"Downloaded stage3_exp_c_bands_projected.json → {local_path}")
    return Path(local_path), run_id


def build_exp_C_indices(mmdd_to_date, local_band_to_idx, stage2_run_id=None):
    """
    Load stage3_exp_c_bands.txt (written by feature_analysis.py / v2 notebook).
    Each line: '<BAND>_<YYYYMMDD>' or '<BAND>_<MMDD>'.
    Maps MMDD to local reference-year date, then to flat local index.
    Falls back to downloading from MLflow if not found locally.
    """
    bands_path = STAGE3_EXP_C_BANDS
    resolved_stage2_run_id = stage2_run_id
    if not bands_path.exists():
        log.info(
            "stage3_exp_c_bands.txt not found locally — fetching from MLflow "
            f"({'run_id=' + stage2_run_id if stage2_run_id else 'latest Stage 2 run'})"
        )
        bands_path, resolved_stage2_run_id = _fetch_exp_c_bands_from_mlflow(run_id=stage2_run_id)

    with open(bands_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    exp_C_idx, exp_C_names = [], []
    skipped = 0
    for band_name in lines:
        try:
            b, ds = band_name.rsplit("_", 1)
        except ValueError:
            skipped += 1
            continue
        mmdd       = ds[4:] if len(ds) == 8 else ds
        local_date = mmdd_to_date.get(mmdd)
        if local_date and b in S2_BAND_NAMES:
            local_name = f"{b}_{local_date}"
            idx        = local_band_to_idx.get(local_name)
            if idx is not None:
                exp_C_idx.append(idx)
                exp_C_names.append(local_name)
            else:
                skipped += 1
        else:
            skipped += 1

    if not exp_C_idx:
        raise ValueError(
            f"Exp C: no bands from {bands_path.name} matched current processed S2 files.\n"
            "Check that S2 files for the same dates as Stage 2 are present."
        )
    if skipped:
        log.warning(f"Exp C: {skipped} band(s) from {bands_path.name} could not be matched")

    log.info(f"Exp C: {len(exp_C_idx)} channels from {bands_path.name}")
    if resolved_stage2_run_id:
        log.info(f"Exp C source Stage 2 run_id: {resolved_stage2_run_id}")
    return exp_C_idx, exp_C_names, resolved_stage2_run_id


def build_exp_C_indices_projected(s2_processed, project_run_id=None):
    """
    Load stage3_exp_c_bands_projected.json and compute per-year band indices.
    Returns dict {yr: (idx_list, names_list)} or (None, None) if unavailable.
    """
    p = STAGE3_EXP_C_BANDS_PROJECTED
    resolved_project_run_id = project_run_id

    if not p.exists():
        log.info(
            "stage3_exp_c_bands_projected.json not found locally — fetching from MLflow "
            f"({'run_id=' + project_run_id if project_run_id else 'latest project run'})"
        )
        try:
            p, resolved_project_run_id = _fetch_projected_from_mlflow(run_id=project_run_id)
        except RuntimeError as e:
            log.warning(f"Could not fetch projected bands: {e}")
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
            log.warning(f"Exp C projected: no S2 files for year {yr} — skipping")
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
            log.warning(f"Exp C projected {yr}: {skipped} band(s) could not be matched")
        log.info(f"Exp C projected {yr}: {len(idx)} channels")
        result[yr] = (idx, names)

    if resolved_project_run_id:
        log.info(f"Exp C source project run_id: {resolved_project_run_id}")
    return (result if result else None), resolved_project_run_id
