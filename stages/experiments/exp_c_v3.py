"""Exp C_v3 — Stage 2v3 incremental top-K sweep channel loader.

Reads a band-list file produced by Stage 2v3:
  stage3_exp_c_v3_band_sweep_k{k:02d}_bands.txt   (Phase A)
  stage3_exp_c_v3_date_sweep_k{k:02d}_bands.txt   (Phase B)

Each file is a newline-separated list of "{band}_{YYYYMMDD}" channel names
representing the union across all crops for that sweep variant.

Falls back to downloading from MLflow if the file is absent locally.
"""

import logging
from pathlib import Path
import sys
import os

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT.parent))

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
import mlflow

from crop_mapping_pipeline.config import (
    S2_BAND_NAMES,
    PROCESSED_DIR,
    MLFLOW_TRACKING_URI,
)

log = logging.getLogger(__name__)

# MLflow run ID of the Stage 2v3 sweep run (used for fallback download)
STAGE2V3_SWEEP_RUN_ID = "b3dbb46372e147fd997c85ea8ce5b9d0"


def _bands_file_name(phase: str, k: int) -> str:
    return f"stage3_exp_c_v3_{phase}_sweep_k{k:02d}_bands.txt"


def _resolve_bands_file(phase: str, k: int, mlflow_run_id: str | None) -> Path:
    """Return a local path to the bands file, downloading from MLflow if needed."""
    filename = _bands_file_name(phase, k)
    local_path = PROCESSED_DIR / filename

    if local_path.exists():
        return local_path

    run_id = mlflow_run_id or STAGE2V3_SWEEP_RUN_ID
    if not run_id:
        raise FileNotFoundError(
            f"{filename} not found locally at {local_path}\n"
            "Run Stage 2v3 first:  python feature_analysis_v2.py --stage 2v3\n"
            "Or pass --stage2v3-sweep-run-id to fetch from MLflow."
        )

    log.info(f"{filename} not found locally — downloading from MLflow run_id={run_id}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=filename,
            dst_path=str(PROCESSED_DIR),
        )
        log.info(f"Downloaded {filename} → {downloaded}")
        return Path(downloaded)
    except Exception as e:
        raise RuntimeError(
            f"Could not download {filename} from MLflow run {run_id}: {e}"
        ) from e


def build_exp_C_v3_indices(
    phase: str,
    k: int,
    mmdd_to_date: dict,
    local_band_to_idx: dict,
    mlflow_run_id: str | None = None,
) -> tuple[list[int], list[str]]:
    """
    Build channel indices for Exp C_v3 (Stage 2v3 sweep variant).

    Parameters
    ----------
    phase           : "band" for Phase A sweep, "date" for Phase B sweep
    k               : sweep level (1-indexed)
    mmdd_to_date    : {"MMDD": local_date_key} from train_segmentation
    local_band_to_idx : {"{band}_{date}": channel_idx} for the loaded S2 files
    mlflow_run_id   : optional MLflow run_id to download artifact if missing

    Returns
    -------
    (idx_list, names_list)
    """
    if phase not in ("band", "date"):
        raise ValueError(f"phase must be 'band' or 'date', got '{phase}'")

    bands_file = _resolve_bands_file(phase, k, mlflow_run_id)

    raw_channels = [
        line.strip()
        for line in bands_file.read_text().splitlines()
        if line.strip()
    ]

    if not raw_channels:
        raise ValueError(f"Exp C_v3: {bands_file.name} is empty")

    idx_list: list[int] = []
    names_list: list[str] = []
    skipped = 0

    for ch in raw_channels:
        # ch format: "{band}_{YYYYMMDD}"
        parts = ch.rsplit("_", 1)
        if len(parts) != 2:
            log.warning(f"Exp C_v3: cannot parse channel '{ch}' — skipping")
            skipped += 1
            continue

        band, date_yyyymmdd = parts
        if band not in S2_BAND_NAMES:
            log.warning(f"Exp C_v3: band '{band}' not in S2_BAND_NAMES — skipping")
            skipped += 1
            continue

        mmdd = date_yyyymmdd[4:]          # YYYYMMDD → MMDD
        local_date = mmdd_to_date.get(mmdd)
        if local_date is None:
            log.warning(f"Exp C_v3: MMDD={mmdd} (from {date_yyyymmdd}) not in loaded S2 files — skipping")
            skipped += 1
            continue

        local_name = f"{band}_{local_date}"
        idx = local_band_to_idx.get(local_name)
        if idx is not None:
            idx_list.append(idx)
            names_list.append(local_name)
        else:
            skipped += 1

    if not idx_list:
        raise ValueError(
            f"Exp C_v3 (phase={phase}, k={k}): no channels from {bands_file.name} "
            "matched the loaded S2 files."
        )
    if skipped:
        log.warning(f"Exp C_v3 (phase={phase}, k={k}): {skipped} channel(s) skipped")

    log.info(
        f"Exp C_v3 phase={phase} k={k}: {len(idx_list)} channels "
        f"from {bands_file.name}"
    )
    return idx_list, names_list
