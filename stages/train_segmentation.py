"""
Stage 3 — Full Model Validation (Exp A / B / C).

Three experiment configurations × 2 architectures = 6 training runs.

| Config | Input                     | Channels | Purpose                   |
|--------|---------------------------|----------|---------------------------|
| Exp A  | Single-date (Jul 30)      | 9        | Conventional baseline     |
| Exp B  | 4 phenological dates      | 36       | Multi-temporal naive      |
| Exp C  | Stage 2 forward-selection | K*       | Proposed method           |

Usage:
    python scripts/train_segmentation.py                 # run all 6 experiments
    python scripts/train_segmentation.py --exp A         # only Exp A (both archs)
    python scripts/train_segmentation.py --exp C --arch segformer
    python scripts/train_segmentation.py --force         # re-run even if ckpt exists
    python scripts/train_segmentation.py --data-dir /mnt/data
"""

import os
import re
import sys
import time
import argparse
import logging
from glob import glob
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, ConcatDataset
import rasterio

os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
import mlflow

_ROOT = Path(__file__).parent.parent   # crop_mapping_pipeline/
sys.path.insert(0, str(_ROOT.parent))

from crop_mapping_pipeline.config import (
    S2_PROCESSED_DIR, CDL_BY_YEAR, MODELS_DIR, FIGURES_DIR, LOGS_DIR,
    PROCESSED_DIR,
    S2_BAND_NAMES, N_BANDS_PER_DATE, VEGE_BANDS,
    KEEP_CLASSES, CLASS_REMAP, NUM_CLASSES, CDL_CLASS_NAMES,
    REMAP_LUT, S2_NODATA,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_TRAIN, MLFLOW_EXPERIMENT_FEATURE,
    TRAIN_YEARS, TEST_YEAR,
    PATCH_SIZE, STRIDE, MIN_VALID_FRAC, BATCH_SIZE, MAX_EPOCHS, EARLY_STOP,
    VAL_FRAC, SEED, ARCH_CFG,
    STAGE3_EXP_C_BANDS, STAGE3_EXP_C_BANDS_PROJECTED,
)
from geoai.geoai.train import RasterPatchDataset, train_semantic_one_epoch
from geoai.geoai.utils.device import get_device
from crop_mapping_pipeline.models import DeepLabV3PlusCBAM, build_segformer

log = logging.getLogger(__name__)
DEVICE = get_device()


def _device_label() -> str:
    if torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"
    if torch.backends.mps.is_available():
        return "mps (Apple Silicon)"
    return "cpu"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _s2_for_year(s2_processed, yr):
    return sorted([p for p in s2_processed if Path(p).name.split("_")[1] == yr])


def _filter_s2_by_band_indices(s2_paths, band_indices, n_bands_per_file=N_BANDS_PER_DATE):
    """Return (filtered_paths, remapped_indices) keeping only TIF files that
    contribute at least one channel in band_indices, with indices remapped to
    their positions in the reduced stack.

    Example: 25 files × 11 bands = 275 channels.  Exp A selects bands [157..165]
    (file 14 only) → returns [s2_paths[14]], remapped to [0..8].
    """
    if band_indices is None:
        return s2_paths, None
    # Which file indices (0-based) are needed?
    needed_file_idxs = sorted({gi // n_bands_per_file for gi in band_indices
                                if gi // n_bands_per_file < len(s2_paths)})
    filtered_paths = [s2_paths[i] for i in needed_file_idxs]
    # Build global-index → new-stacked-index map for every band in kept files
    new_idx_map = {}
    stacked = 0
    for fi in needed_file_idxs:
        for local in range(n_bands_per_file):
            new_idx_map[fi * n_bands_per_file + local] = stacked
            stacked += 1
    remapped = [new_idx_map[gi] for gi in band_indices]
    return filtered_paths, remapped


def parse_date(path):
    m = re.search(r"_(\d{4})_(\d{2})_(\d{2})_processed", Path(path).name)
    return f"{m.group(1)}{m.group(2)}{m.group(3)}" if m else None


def build_local_band_map(s2_processed):
    """
    Return (local_band_names, local_band_to_idx, local_date_to_idx, mmdd_to_date)
    based on the year with the most processed files (used as reference).
    """
    by_year = {}
    for p in s2_processed:
        yr = Path(p).name.split("_")[1]
        by_year.setdefault(yr, []).append(p)

    ref_yr    = max(by_year, key=lambda y: len(by_year[y]))
    ref_files = sorted(by_year[ref_yr])

    local_band_names  = []
    local_date_to_idx = {}
    for i, p in enumerate(ref_files):
        d = parse_date(p)
        local_date_to_idx[d] = i
        local_band_names.extend([f"{b}_{d}" for b in S2_BAND_NAMES])

    local_band_to_idx = {n: i for i, n in enumerate(local_band_names)}
    available_dates   = sorted(local_date_to_idx.keys())
    mmdd_to_date      = {d[4:]: d for d in available_dates}

    log.info(
        f"Reference year={ref_yr}, {len(ref_files)} dates, "
        f"{len(local_band_names)} local channels"
    )
    return local_band_names, local_band_to_idx, local_date_to_idx, mmdd_to_date


def build_exp_A_indices(local_date_to_idx, local_band_to_idx):
    """Single date (Jul 30) × 9 vegetation bands."""
    available_dates = sorted(local_date_to_idx.keys())
    july30_key = next(
        (k for k in available_dates if k[4:6] == "07" and k[6:8] in ("29", "30")),
        available_dates[-1],
    )
    off  = local_date_to_idx[july30_key] * N_BANDS_PER_DATE
    idx  = [off + S2_BAND_NAMES.index(b) for b in VEGE_BANDS]
    names = [f"{b}_{july30_key}" for b in VEGE_BANDS]
    log.info(f"Exp A: date={july30_key}, {len(idx)} channels")
    return idx, names, july30_key


def build_exp_B_indices(local_date_to_idx, local_band_to_idx):
    """4 phenological dates × 9 vegetation bands = up to 36 channels.

    Dates are chosen as the acquisition nearest to the mid-point of each
    phenological season (Jan-15, Mar-15, Jul-15, Nov-15).
    """
    available_dates = sorted(local_date_to_idx.keys())

    # Target: mid-point of each season as day-of-year
    phenol_targets = {"Jan": "0115", "Mar": "0315", "Jul": "0715", "Nov": "1115"}
    phenol_map     = {}
    for label, target_mmdd in phenol_targets.items():
        target_doy = int(target_mmdd)   # treat MMDD as comparable integer
        match = min(
            available_dates,
            key=lambda d: abs(int(d[4:]) - target_doy),
        )
        phenol_map[label] = match

    exp_B_idx, exp_B_names = [], []
    for _label, d in phenol_map.items():
        off          = local_date_to_idx[d] * N_BANDS_PER_DATE
        exp_B_idx   += [off + S2_BAND_NAMES.index(b) for b in VEGE_BANDS]
        exp_B_names += [f"{b}_{d}" for b in VEGE_BANDS]

    # deduplicate
    seen, dedup_idx, dedup_names = set(), [], []
    for idx, name in zip(exp_B_idx, exp_B_names):
        if idx not in seen:
            seen.add(idx)
            dedup_idx.append(idx)
            dedup_names.append(name)

    log.info(f"Exp B: dates={list(phenol_map.values())}, {len(dedup_idx)} channels")
    return dedup_idx, dedup_names, phenol_map


def _fetch_exp_c_bands_from_mlflow(run_id=None):
    """
    Download stage3_exp_c_bands.txt from a Stage 2 MLflow artifact.

    If run_id is None, searches cropmap_feature_analysis_s2 for the latest
    finished Stage 2 parent run (name matches 'stage2v2_binary_fwd_*').
    Saves the file to PROCESSED_DIR and returns its Path.
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


def build_exp_C_indices(mmdd_to_date, local_band_to_idx, stage2_run_id=None):
    """
    Load stage3_exp_c_bands.txt (written by feature_analysis.py / v2 notebook).
    Each line: '<BAND>_<YYYYMMDD>' or '<BAND>_<MMDD>'.
    Maps MMDD to local reference-year date, then to flat local index.

    If the file is not found locally, it is downloaded from the MLflow artifact
    of the latest Stage 2 run (or the run specified by stage2_run_id).
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
        mmdd       = ds[4:] if len(ds) == 8 else ds    # YYYYMMDD or MMDD
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


def _fetch_projected_from_mlflow(run_id=None):
    """
    Download stage3_exp_c_bands_projected.json from a project run MLflow artifact.

    If run_id is None, searches cropmap_feature_analysis_s2 for the latest
    run named 'stage2v2_project_*'. Saves to PROCESSED_DIR and returns its Path.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    if run_id is None:
        exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT_FEATURE)
        if exp is None:
            raise RuntimeError(
                f"MLflow experiment '{MLFLOW_EXPERIMENT_FEATURE}' not found."
            )
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


def build_exp_C_indices_projected(s2_processed, project_run_id=None):
    """
    Load stage3_exp_c_bands_projected.json and compute per-year band indices.

    Returns dict:  {yr: (idx_list, names_list)}
    If the projected file is missing, falls back to downloading it from MLflow
    or raises an error with instructions.
    """
    import json as _json

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
            return None, None   # caller falls back to single-ref-year behaviour

    with open(p) as f:
        projected = _json.load(f)   # {"2022": ["B4_20220730", ...], "2023": [...], ...}

    # Build per-year file lists and their flat channel maps
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

        # Build flat channel map for this year: "B4_20230801" → int index
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


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(cdl_path=None):
    """Inverse-frequency weights from 2022 CDL (reference year)."""
    ref_cdl = cdl_path or CDL_BY_YEAR[TRAIN_YEARS[0]]
    with rasterio.open(ref_cdl) as src:
        cdl_arr = src.read(1).astype(np.int32)

    class_counts      = np.zeros(NUM_CLASSES, dtype=np.float64)
    class_counts[0]   = (cdl_arr == 0).sum()
    for cdl_id, model_id in CLASS_REMAP.items():
        class_counts[model_id] += (cdl_arr == cdl_id).sum()

    freq          = class_counts / (class_counts.sum() + 1e-9)
    weights       = 1.0 / (freq + 1e-9)
    weights      /= weights.sum()
    return torch.tensor(weights, dtype=torch.float32)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_miou(logits, labels, num_classes):
    preds  = logits.argmax(dim=1).view(-1).cpu().numpy()
    labels = labels.view(-1).cpu().numpy()
    ious   = []
    for cls in range(1, num_classes):
        p = (preds == cls)
        l = (labels == cls)
        inter = (p & l).sum()
        union = (p | l).sum()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def compute_per_class_iou(logits, labels, num_classes):
    preds  = logits.argmax(dim=1).view(-1).numpy()
    labels = labels.view(-1).numpy()
    ious   = {}
    for cls in range(1, num_classes):
        p = (preds == cls)
        l = (labels == cls)
        inter = (p & l).sum()
        union = (p | l).sum()
        ious[cls] = float(inter / union) if union > 0 else float("nan")
    return ious


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits      = model(imgs)
        total_loss += criterion(logits, masks).item()
        all_logits.append(logits.cpu())
        all_labels.append(masks.cpu())
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    preds      = all_logits.argmax(dim=1)
    oa         = (preds == all_labels).float().mean().item()
    miou       = compute_miou(all_logits, all_labels, num_classes)
    per_class  = compute_per_class_iou(all_logits, all_labels, num_classes)
    return {"loss": total_loss / len(loader), "miou": miou, "oa": oa, "per_class_iou": per_class}


@torch.no_grad()
def evaluate_test_set(model, loader, num_classes, device):
    model.eval()
    all_logits, all_labels = [], []
    for imgs, masks in loader:
        logits = model(imgs.to(device))
        all_logits.append(logits.cpu())
        all_labels.append(masks.cpu())
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    preds      = all_logits.argmax(dim=1)
    return {
        "miou":          compute_miou(all_logits, all_labels, num_classes),
        "oa":            (preds == all_labels).float().mean().item(),
        "per_class_iou": compute_per_class_iou(all_logits, all_labels, num_classes),
        "preds":         preds,
        "labels":        all_labels,
    }


# ── Model builder ─────────────────────────────────────────────────────────────

def build_model(arch, in_channels, num_classes):
    cfg = ARCH_CFG[arch]
    if arch == "deeplabv3plus_cbam":
        model = DeepLabV3PlusCBAM(
            encoder_name=cfg["encoder"],
            encoder_weights="imagenet",
            in_channels=in_channels,
            num_classes=num_classes,
        )
    elif arch == "segformer":
        model = build_segformer(
            encoder_name=cfg["encoder"],
            encoder_weights="imagenet",
            in_channels=in_channels,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    n = sum(p.numel() for p in model.parameters())
    log.info(f"  {arch} ({cfg['encoder']}): {n:,} params")
    return model.to(DEVICE)


# ── Confusion matrix ──────────────────────────────────────────────────────────

def _plot_confusion_matrix(preds, labels, save_path):
    """
    Normalized (row-wise) confusion matrix over all NUM_CLASSES classes.
    Rows = ground truth, columns = predicted.
    """
    p = preds.view(-1).numpy()
    l = labels.view(-1).numpy()

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t, pred in zip(l, p):
        if 0 <= t < NUM_CLASSES and 0 <= pred < NUM_CLASSES:
            cm[t, pred] += 1

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.divide(cm.astype(float), row_sums,
                         out=np.zeros_like(cm, dtype=float), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CLASS_LABELS, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Ground Truth", fontsize=11)
    ax.set_title("Confusion Matrix (row-normalized)", fontsize=12, fontweight="bold")

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            v = cm_norm[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if v > 0.5 else "black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {save_path}")


# ── Main experiment runner ────────────────────────────────────────────────────

def run_experiment(
    exp_name,
    arch,
    band_indices,           # list[int]  OR  dict{yr: (list[int], list[str])}
    band_names_list,        # list[str]  (reference year; used for logging/metadata)
    description,
    s2_processed,
    class_weights_tensor,
    force=False,
    skip_viz=False,
    source_stage2_run_id=None,
    source_project_run_id=None,
):
    """
    band_indices can be:
      - list[int]           → same indices applied to every year (Exp A / B, or
                              Exp C without projected file)
      - dict{yr: (idx, names)} → per-year indices from build_exp_C_indices_projected()
    """
    cfg        = ARCH_CFG[arch]
    exp_dir    = MODELS_DIR / exp_name
    best_ckpt  = exp_dir / "best_model.pth"
    last_ckpt  = exp_dir / "last_model.pth"
    exp_dir.mkdir(parents=True, exist_ok=True)

    if not force and best_ckpt.exists():
        log.info(f"Checkpoint exists — skipping {exp_name}  (use --force to re-run)")
        return None

    per_year = isinstance(band_indices, dict)

    def _yr_idx(yr):
        """Return (idx_list, names_list) for a given year."""
        if per_year:
            if yr in band_indices:
                return band_indices[yr]
            # fallback: use the first available year's indices
            fallback_yr = next(iter(band_indices))
            log.warning(
                f"Exp C projected: year {yr} not in projected map — "
                f"falling back to {fallback_yr} indices"
            )
            return band_indices[fallback_yr]
        return band_indices, band_names_list

    in_channels = len(_yr_idx(TRAIN_YEARS[0])[0])
    log.info(f"\n{'='*65}")
    log.info(f" {exp_name}")
    log.info(f"  arch={arch}  in_channels={in_channels}  per_year_indices={per_year}")
    log.info(f"  {description}")
    log.info(f"{'='*65}\n")

    # ── Year-based dataset split ──────────────────────────────────────────────
    train_year_datasets = []
    for yr in TRAIN_YEARS:
        yr_s2  = _s2_for_year(s2_processed, yr)
        yr_cdl = CDL_BY_YEAR[yr]
        if not yr_s2 or not yr_cdl.exists():
            log.warning(f"Skipping train year {yr}: {'no S2' if not yr_s2 else 'CDL missing'}")
            continue
        yr_idx, _ = _yr_idx(yr)
        yr_s2_filtered, yr_idx_local = _filter_s2_by_band_indices(yr_s2, yr_idx)
        ds = RasterPatchDataset(
            s2_paths=yr_s2_filtered, cdl_path=str(yr_cdl),
            patch_size=PATCH_SIZE, stride=STRIDE,
            keep_classes=KEEP_CLASSES, remap_lut=REMAP_LUT,
            min_valid_frac=MIN_VALID_FRAC, band_indices=yr_idx_local,
        )
        train_year_datasets.append(ds)
        log.info(f"  [{yr}] {len(ds):,} patches  ({len(yr_idx)} channels, {len(yr_s2_filtered)}/{len(yr_s2)} files)")

    assert train_year_datasets, "No training data for any TRAIN_YEAR"
    train_val_ds = ConcatDataset(train_year_datasets)

    n_total = len(train_val_ds)
    n_val   = max(1, int(VAL_FRAC * n_total))
    n_train = n_total - n_val
    gen     = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(train_val_ds, [n_train, n_val], generator=gen)

    test_s2  = _s2_for_year(s2_processed, TEST_YEAR)
    test_cdl = CDL_BY_YEAR[TEST_YEAR]
    assert test_s2 and test_cdl.exists(), f"Test year {TEST_YEAR} data missing"
    test_idx, _ = _yr_idx(TEST_YEAR)
    test_s2_filtered, test_idx_local = _filter_s2_by_band_indices(test_s2, test_idx)
    test_ds = RasterPatchDataset(
        s2_paths=test_s2_filtered, cdl_path=str(test_cdl),
        patch_size=PATCH_SIZE, stride=STRIDE,
        keep_classes=KEEP_CLASSES, remap_lut=REMAP_LUT,
        min_valid_frac=MIN_VALID_FRAC, band_indices=test_idx_local,
    )

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    log.info(f"  Patches: {n_train:,} train / {n_val:,} val / {len(test_ds):,} test ({TEST_YEAR})")

    # ── Model + optimiser + scheduler + loss ──────────────────────────────────
    model     = build_model(arch, in_channels, NUM_CLASSES)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=MAX_EPOCHS, power=0.9
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(DEVICE))

    # ── MLflow run ────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_TRAIN)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    with mlflow.start_run(run_name=f"{exp_name}_{timestamp}") as run:
        mlflow.log_params({
            "experiment":     exp_name,
            "architecture":   arch,
            "encoder":        cfg["encoder"],
            "in_channels":    in_channels,
            "num_classes":    NUM_CLASSES,
            "patch_size":     PATCH_SIZE,
            "batch_size":     BATCH_SIZE,
            "max_epochs":     MAX_EPOCHS,
            "early_stopping": EARLY_STOP,
            "learning_rate":  cfg["lr"],
            "weight_decay":   cfg["weight_decay"],
            "optimizer":      "AdamW",
            "lr_scheduler":   "PolynomialLR(power=0.9)",
            "loss":           "WeightedCrossEntropy",
            "train_years":    str(TRAIN_YEARS),
            "test_year":      TEST_YEAR,
            "train_patches":  n_train,
            "val_patches":    n_val,
            "test_patches":   len(test_ds),
            "description":    description,
            "keep_classes":   str(KEEP_CLASSES),
        })
        mlflow.set_tag("band_names", str(band_names_list))
        mlflow.set_tag("n_bands",    str(in_channels))
        if source_stage2_run_id:
            mlflow.set_tag("source_stage2_run_id", source_stage2_run_id)
        if source_project_run_id:
            mlflow.set_tag("source_project_run_id", source_project_run_id)

        # ── Training loop ─────────────────────────────────────────────────────
        best_miou  = 0.0
        no_improve = 0
        history    = []
        t_start    = time.time()

        for epoch in range(MAX_EPOCHS):
            t_ep = time.time()

            train_loss = train_semantic_one_epoch(
                model, optimizer, train_dl, DEVICE, epoch, criterion,
                print_freq=len(train_dl), verbose=False,
            )
            val_m = validate_one_epoch(model, val_dl, criterion, DEVICE, NUM_CLASSES)
            scheduler.step()

            ep_t = time.time() - t_ep
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss":   val_m["loss"],
                "val_miou":   val_m["miou"],
                "val_oa":     val_m["oa"],
                "lr":         scheduler.get_last_lr()[0],
            }, step=epoch)

            history.append({
                "epoch":      epoch + 1,
                "train_loss": round(train_loss,       4),
                "val_loss":   round(val_m["loss"],    4),
                "val_miou":   round(val_m["miou"],    4),
                "val_oa":     round(val_m["oa"],      4),
                "epoch_t_s":  round(ep_t,              1),
            })

            if val_m["miou"] > best_miou:
                best_miou  = val_m["miou"]
                no_improve = 0
                torch.save({
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "best_miou":        best_miou,
                    "band_indices":     band_indices,
                    "band_names":       band_names_list,
                    "in_channels":      in_channels,
                    "num_classes":      NUM_CLASSES,
                    "architecture":     arch,
                }, best_ckpt)
            else:
                no_improve += 1

            total_min = (time.time() - t_start) / 60
            log.info(
                f"  Ep {epoch+1:3d}/{MAX_EPOCHS} "
                f"loss={train_loss:.4f} val={val_m['loss']:.4f} "
                f"mIoU={val_m['miou']:.4f} best={best_miou:.4f} "
                f"patience={no_improve}/{EARLY_STOP} "
                f"{ep_t:.0f}s  {total_min:.1f}min"
            )
            _cls_parts = []
            for cls_id, iou in val_m["per_class_iou"].items():
                cdl_id    = KEEP_CLASSES[cls_id - 1]
                short     = CDL_CLASS_NAMES.get(cdl_id, f"cls{cls_id}").replace(" ", "")
                iou_str   = f"{iou:.3f}" if not np.isnan(iou) else "  nan"
                _cls_parts.append(f"{short}={iou_str}")
            log.info("         " + "  ".join(_cls_parts))

            # Save last checkpoint every epoch (overwrites previous)
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state":  optimizer.state_dict(),
                "val_miou":         val_m["miou"],
                "band_indices":     band_indices,
                "band_names":       band_names_list,
                "in_channels":      in_channels,
                "num_classes":      NUM_CLASSES,
                "architecture":     arch,
            }, last_ckpt)

            if no_improve >= EARLY_STOP:
                log.info(f"  Early stopping at epoch {epoch + 1}")
                break

        # ── Test evaluation ───────────────────────────────────────────────────
        ckpt = torch.load(best_ckpt, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        test_r = evaluate_test_set(model, test_dl, NUM_CLASSES, DEVICE)

        mlflow.log_metrics({
            "best_val_miou": best_miou,
            "test_miou":     test_r["miou"],
            "test_oa":       test_r["oa"],
            "total_epochs":  len(history),
        })
        for cls_id, iou in test_r["per_class_iou"].items():
            if not np.isnan(iou):
                cdl_id = KEEP_CLASSES[cls_id - 1]
                name   = CDL_CLASS_NAMES.get(cdl_id, f"cls{cls_id}")
                mlflow.log_metric(
                    f"test_iou_{name.lower().replace('/', '_').replace(' ', '_')}",
                    iou,
                )

        # ── Artifacts ─────────────────────────────────────────────────────────

        # Training history CSV
        hist_df  = pd.DataFrame(history)
        hist_csv = exp_dir / "training_history.csv"
        hist_df.to_csv(hist_csv, index=False)

        # Training curve PNG
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(hist_df["epoch"], hist_df["train_loss"], "--", label="Train")
        ax1.plot(hist_df["epoch"], hist_df["val_loss"],         label="Val")
        ax1.set(xlabel="Epoch", ylabel="Loss", title=f"{exp_name} — Loss")
        ax1.legend(); ax1.grid(True)
        ax2.plot(hist_df["epoch"], hist_df["val_miou"], color="green", label="Val mIoU")
        ax2.axhline(best_miou, linestyle="--", color="gray", label=f"Best={best_miou:.4f}")
        ax2.set(xlabel="Epoch", ylabel="mIoU", title=f"{exp_name} — mIoU")
        ax2.legend(); ax2.grid(True)
        plt.tight_layout()
        curve_path = exp_dir / "training_curve.png"
        plt.savefig(curve_path, dpi=150)
        plt.close()

        # Per-class IoU CSV
        iou_rows = []
        for cls_id, iou in test_r["per_class_iou"].items():
            cdl_id = KEEP_CLASSES[cls_id - 1]
            iou_rows.append({
                "class_id":   cls_id,
                "cdl_id":     cdl_id,
                "class_name": CDL_CLASS_NAMES.get(cdl_id, f"cls{cls_id}"),
                "iou":        round(iou, 4) if not np.isnan(iou) else float("nan"),
            })
        iou_csv = exp_dir / "test_per_class_iou.csv"
        pd.DataFrame(iou_rows).to_csv(iou_csv, index=False)

        # Confusion matrix PNG
        cm_path = exp_dir / "confusion_matrix.png"
        _plot_confusion_matrix(test_r["preds"], test_r["labels"], str(cm_path))

        # Segmentation map PNG (full-tile inference)
        seg_path = None
        if not skip_viz:
            log.info(f"  Running full-image inference for {exp_name}...")
            gt_map, _    = load_gt_remap(str(test_cdl))
            pred_map, _  = run_full_inference(
                model, test_s2_filtered, test_idx_local, patch_size=PATCH_SIZE, stride=PATCH_SIZE
            )
            seg_path = exp_dir / "test_segmentation_map.png"
            save_segmentation_map(
                pred_map, gt_map,
                title=f"{exp_name} — Test Segmentation ({TEST_YEAR})",
                save_path=str(seg_path),
            )
            del pred_map, gt_map

        mlflow.log_artifact(str(best_ckpt))
        mlflow.log_artifact(str(last_ckpt))
        mlflow.log_artifact(str(hist_csv))
        mlflow.log_artifact(str(curve_path))
        mlflow.log_artifact(str(iou_csv))
        mlflow.log_artifact(str(cm_path))
        if seg_path is not None:
            mlflow.log_artifact(str(seg_path))

        run_id = run.info.run_id

    summary = {
        "exp_name":      exp_name,
        "arch":          arch,
        "in_channels":   in_channels,
        "best_val_miou": round(best_miou,      4),
        "test_miou":     round(test_r["miou"], 4),
        "test_oa":       round(test_r["oa"],   4),
        "total_epochs":  len(history),
        "run_id":        run_id,
        "ckpt":          str(best_ckpt),
    }
    log.info(
        f"\n✅ {exp_name}  val_mIoU={best_miou:.4f}  "
        f"test_mIoU={test_r['miou']:.4f}  run={run_id}"
    )
    return summary


# ── Full-image inference & visualization ─────────────────────────────────────

CROP_COLORS = [
    "#000000",  # 0  background
    "#1E90FF",  # 1  Rice
    "#FFD700",  # 2  Sunflower
    "#8B4513",  # 3  Winter Wheat
    "#98FB98",  # 4  Alfalfa
    "#A9A9A9",  # 5  Other Hay
    "#FF6347",  # 6  Tomatoes
    "#800080",  # 7  Grapes
    "#FF8C00",  # 8  Almonds
    "#228B22",  # 9  Walnuts
    "#9370DB",  # 10 Plums
]
CLASS_LABELS = ["Background"] + [CDL_CLASS_NAMES[c] for c in KEEP_CLASSES]
SEG_CMAP     = ListedColormap(CROP_COLORS)
SEG_NORM     = BoundaryNorm(boundaries=range(NUM_CLASSES + 1), ncolors=NUM_CLASSES)


def run_full_inference(model, s2_paths, band_indices, patch_size=256, stride=256):
    """Tiled inference — reads one window at a time, never loads full rasters."""
    with rasterio.open(s2_paths[0]) as src:
        H, W    = src.height, src.width
        profile = dict(src.profile)

    srcs     = [rasterio.open(p) for p in s2_paths]
    pred_map = np.zeros((H, W), dtype=np.uint8)
    n_rows   = (H + stride - 1) // stride
    n_cols   = (W + stride - 1) // stride
    total    = n_rows * n_cols
    K        = len(band_indices)

    model.eval()
    done = 0
    try:
        with torch.no_grad():
            for y in range(0, H, stride):
                for x in range(0, W, stride):
                    ph  = min(patch_size, H - y)
                    pw  = min(patch_size, W - x)
                    win = rasterio.windows.Window(x, y, pw, ph)

                    # Read only this window from each file
                    bands = []
                    for src in srcs:
                        try:
                            arr = src.read(window=win).astype(np.float32)
                        except Exception:
                            arr = np.zeros((src.count, ph, pw), dtype=np.float32)
                        arr[arr == S2_NODATA] = 0.0
                        bands.append(arr)

                    patch = np.concatenate(bands, axis=0)[band_indices]  # (K, ph, pw)

                    # Per-channel min-max normalisation (matches RasterPatchDataset)
                    for ch in range(K):
                        lo, hi = float(patch[ch].min()), float(patch[ch].max())
                        if hi > lo:
                            patch[ch] = (patch[ch] - lo) / (hi - lo)
                        else:
                            patch[ch] = 0.0

                    # Pad to patch_size if at border
                    if ph < patch_size or pw < patch_size:
                        padded = np.zeros((K, patch_size, patch_size), dtype=np.float32)
                        padded[:, :ph, :pw] = patch
                        patch = padded

                    t   = torch.from_numpy(patch).unsqueeze(0).to(DEVICE)
                    out = model(t).argmax(dim=1).squeeze().cpu().numpy()
                    pred_map[y:y + ph, x:x + pw] = out[:ph, :pw]
                    done += 1
                    if done % 200 == 0 or done == total:
                        log.info(f"  {done}/{total} tiles")
    finally:
        for src in srcs:
            src.close()

    return pred_map, profile


def load_gt_remap(cdl_path):
    with rasterio.open(cdl_path) as src:
        cdl     = src.read(1).astype(np.int32)
        profile = dict(src.profile)
    gt = REMAP_LUT[np.clip(cdl, 0, 255)]
    return gt.astype(np.uint8), profile


def save_segmentation_map(pred_map, gt_map, title, save_path, downsample=4):
    pred_ds = pred_map[::downsample, ::downsample]
    gt_ds   = gt_map[::downsample, ::downsample]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(gt_ds,   cmap=SEG_CMAP, norm=SEG_NORM, interpolation="nearest")
    axes[0].set_title("Ground Truth (CDL)", fontsize=12, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(pred_ds, cmap=SEG_CMAP, norm=SEG_NORM, interpolation="nearest")
    axes[1].set_title("Prediction",         fontsize=12, fontweight="bold")
    axes[1].axis("off")

    patches = [mpatches.Patch(color=CROP_COLORS[i], label=CLASS_LABELS[i])
               for i in range(1, NUM_CLASSES)]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.01), frameon=True)
    plt.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    exps=None,
    archs=None,
    force=False,
    data_dir=None,
    skip_viz=False,
    stage2_run_id=None,
    project_run_id=None,
):
    # Override data directories
    # Use `global` so all module-level functions pick up the new paths at call time.
    if data_dir:
        global S2_PROCESSED_DIR, CDL_BY_YEAR, MODELS_DIR, FIGURES_DIR
        data_dir = Path(data_dir)
        S2_PROCESSED_DIR = data_dir / "s2"
        CDL_BY_YEAR      = {
            yr: data_dir / "cdl" / f"cdl_{yr}_study_area_filtered.tif"
            for yr in ["2022", "2023", "2024"]
        }
        MODELS_DIR  = data_dir / "models"
        FIGURES_DIR = data_dir / "figures"
        log.info(f"Data dir overridden to {data_dir}")

    s2_processed = sorted(glob(str(S2_PROCESSED_DIR / "*" / "*_processed.tif")))
    if not s2_processed:
        raise FileNotFoundError(f"No processed S2 files in {S2_PROCESSED_DIR}")

    # ── Validate TIF files — fail fast on corrupt/truncated downloads ──────
    corrupt = []
    for path in s2_processed:
        try:
            with rasterio.open(path) as src:
                src.read(1, window=rasterio.windows.Window(0, 0, min(256, src.width), min(256, src.height)))
        except Exception as e:
            corrupt.append((path, str(e)))
    if corrupt:
        log.error(f"Found {len(corrupt)} corrupt S2 file(s) — re-process before training:")
        for p, err in corrupt:
            log.error(f"  {p}  ({err})")
        raise RuntimeError(
            f"{len(corrupt)} corrupt S2 file(s) detected (likely truncated during processing). "
            "Delete the bad files and re-run:  python stages/process_data.py --years <year>"
        )
    log.info(f"All {len(s2_processed)} S2 files validated OK")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Build local band map (reference year) ──────────────────────────────
    (local_band_names, local_band_to_idx,
     local_date_to_idx, mmdd_to_date) = build_local_band_map(s2_processed)

    # ── Build experiment channel sets ─────────────────────────────────────
    exp_A_idx, exp_A_names, july30_key = build_exp_A_indices(
        local_date_to_idx, local_band_to_idx
    )
    exp_B_idx, exp_B_names, phenol_map = build_exp_B_indices(
        local_date_to_idx, local_band_to_idx
    )
    # Exp C: prefer per-year projected indices; fall back to single-ref-year
    exp_C_projected, resolved_project_run_id = build_exp_C_indices_projected(
        s2_processed, project_run_id=project_run_id
    )
    resolved_stage2_run_id = None
    if exp_C_projected:
        exp_C_idx   = exp_C_projected          # dict {yr: (idx, names)}
        exp_C_names = exp_C_projected[TRAIN_YEARS[0]][1]   # ref names for logging
        log.info("Exp C: using per-year projected band indices")
    else:
        exp_C_idx, exp_C_names, resolved_stage2_run_id = build_exp_C_indices(
            mmdd_to_date, local_band_to_idx, stage2_run_id=stage2_run_id
        )
        log.info("Exp C: using single reference-year band indices (no projected file)")

    # ── Class weights ──────────────────────────────────────────────────────
    cw_tensor = compute_class_weights()
    log.info("Class weights computed")

    # ── Experiment plan ────────────────────────────────────────────────────
    all_archs = list(ARCH_CFG.keys())
    run_exps  = exps  or ["A", "B", "C"]
    run_archs = archs or all_archs

    plan = []
    for exp_key in run_exps:
        if exp_key == "A":
            idx, names = exp_A_idx, exp_A_names
            desc_fn = lambda a: f"Single-date {july30_key}, 9ch, {a}"
        elif exp_key == "B":
            idx, names = exp_B_idx, exp_B_names
            desc_fn = lambda a, _p=phenol_map: f"4 phenol dates {list(_p.values())}, {len(exp_B_idx)}ch, {a}"
        elif exp_key == "C":
            idx, names = exp_C_idx, exp_C_names
            desc_fn = lambda a, _i=exp_C_idx: f"Stage2 K*={len(_i) if _i else 0}ch, {a}"
        else:
            log.warning(f"Unknown experiment key: {exp_key} — skipping")
            continue

        if idx is None:
            raise RuntimeError(
                f"Exp {exp_key}: band indices are None — Stage 2 must be run before training."
            )

        for arch in run_archs:
            plan.append((exp_key, arch, idx, names, desc_fn(arch)))

    log.info(f"Planned {len(plan)} run(s): {[(e, a) for e, a, *_ in plan]}")

    # ── Run experiments ────────────────────────────────────────────────────
    all_results = []
    for exp_key, arch, band_idx, band_names, description in plan:
        exp_name = f"exp_{exp_key}_{arch}"
        extra_kw = {}
        if exp_key == "C":
            extra_kw["source_stage2_run_id"]  = resolved_stage2_run_id
            extra_kw["source_project_run_id"] = resolved_project_run_id
        result   = run_experiment(
            exp_name=exp_name,
            arch=arch,
            band_indices=band_idx,
            band_names_list=band_names,
            description=description,
            s2_processed=s2_processed,
            class_weights_tensor=cw_tensor,
            force=force,
            skip_viz=skip_viz,
            **extra_kw,
        )
        if result is not None:
            all_results.append(result)

    # ── Summary table ──────────────────────────────────────────────────────
    if all_results:
        summary_df  = pd.DataFrame(all_results).sort_values("test_miou", ascending=False)
        summary_csv = MODELS_DIR / "experiment_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        log.info("\n=== Experiment Summary ===")
        log.info("\n" + summary_df[[
            "exp_name", "arch", "in_channels",
            "best_val_miou", "test_miou", "test_oa", "total_epochs",
        ]].to_string(index=False))
        log.info(f"Saved: {summary_csv}")

    log.info("All experiments done — segmentation maps, confusion matrices, and IoU CSVs logged to MLflow.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3 — Train segmentation models")
    parser.add_argument(
        "--exp", nargs="+", choices=["A", "B", "C"],
        default=["A", "B", "C"],
        help="Which experiments to run (default: A B C)",
    )
    parser.add_argument(
        "--arch", nargs="+", choices=list(ARCH_CFG.keys()),
        default=None,
        help="Which architectures to run (default: all)",
    )
    parser.add_argument("--force",    action="store_true", help="Re-run even if checkpoint exists")
    parser.add_argument("--skip-viz", action="store_true", help="Skip full-image visualization")
    parser.add_argument("--data-dir", default=None, help="Override data/processed directory")
    parser.add_argument(
        "--stage2-run-id", default=None,
        help=(
            "MLflow run ID of the Stage 2 parent run to fetch stage3_exp_c_bands.txt from. "
            "If omitted and the file is absent locally, the latest Stage 2 run is used."
        ),
    )
    parser.add_argument(
        "--project-run-id", default=None,
        help=(
            "MLflow run ID of the stage2v2_project run to fetch "
            "stage3_exp_c_bands_projected.json from. "
            "If omitted and the file is absent locally, the latest project run is used."
        ),
    )
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    # Suppress GDAL tile-decode noise (LZW/ZIP errors on legacy files).
    # Filters on a Logger only apply at that logger — not on propagation —
    # so we must filter on each Handler after basicConfig creates them.
    class _SuppressGDALFilter(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            return "GDAL signalled an error" not in msg and "IReadBlock failed" not in msg

    _gdal_filter = _SuppressGDALFilter()
    # Also silence rasterio._err directly (covers worker processes via fork)
    logging.getLogger("rasterio._err").setLevel(logging.ERROR)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                LOGS_DIR / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            ),
        ],
    )
    for _h in logging.root.handlers:
        _h.addFilter(_gdal_filter)

    log.info(f"Device: {_device_label()}  PyTorch: {torch.__version__}")

    main(
        exps=args.exp,
        archs=args.arch,
        force=args.force,
        data_dir=args.data_dir,
        skip_viz=args.skip_viz,
        stage2_run_id=args.stage2_run_id,
        project_run_id=args.project_run_id,
    )
