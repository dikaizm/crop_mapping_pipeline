"""
Stage 1v3 + 2v2 — Feature Analysis v2 (Date × Band Selection)

v2 decouples temporal and spectral selection into two sequential phases:

Stage 1v3: Per-crop SI_global ranking split into:
  - Date ranking: mean SI across all bands per acquisition date → top TOP_DATES_PER_CROP dates
  - Band ranking: mean SI across all dates per spectral band  → top TOP_BANDS_PER_CROP bands

Stage 2v2 (per crop):
  Phase A — Date selection: CNN forward selection over date candidates (all VEGE_BANDS at each date).
    Accept date_d if IoU(class1) improves by >= S2_DATE_DELTA; stop after S2_DATE_NO_IMPROVE
    consecutive rejections or S2_MAX_DATES dates selected.

  Phase B — Band selection: given the fixed selected_dates from Phase A, CNN forward selection
    over band candidates. Accept band_b if IoU(class1) improves by >= S2_BAND_DELTA; stop after
    S2_BAND_NO_IMPROVE consecutive rejections or S2_MAX_BANDS_V2 bands selected.

Outputs:
    data/processed/s2/2022/stage1v3_candidates.json    — per-crop date + band candidate lists
    data/processed/stage2v3_per_crop_results.json      — per-crop Phase A + B results
    data/processed/stage3_exp_c_v2.json                — union dates + bands + summary
    data/processed/stage3_exp_c_v2_bands.txt           — flat band list (B4_20220730, one per line)

Usage:
    python feature_analysis_v2.py                          # run both stages
    python feature_analysis_v2.py --stage 1                # Stage 1v3 only (run locally, CPU)
    python feature_analysis_v2.py --stage 2                # Stage 2v2 only (run on GPU server)
    python feature_analysis_v2.py --stage project          # project 2022 selections to other years
    python feature_analysis_v2.py --force                  # re-run even if outputs exist
    python feature_analysis_v2.py --data-dir /data/processed
"""

import os
import re
import sys
import json
import time
import logging
import argparse
import tempfile
import pathlib
from glob import glob
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

_ROOT = pathlib.Path(__file__).parent.parent   # crop_mapping_pipeline/
sys.path.insert(0, str(_ROOT.parent))

os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
import mlflow

import crop_mapping_pipeline.utils.band_selection as bs
from crop_mapping_pipeline.config import (
    S2_PROCESSED_DIR, CDL_BY_YEAR, PROCESSED_DIR, FIGURES_DIR, LOGS_DIR,
    S2_BAND_NAMES, S2_NODATA, KEEP_CLASSES, CLASS_REMAP, NUM_CLASSES, CDL_CLASS_NAMES,
    REMAP_LUT, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_FEATURE,
    SAMPLE_FRACTION, VEGE_BANDS,
    S2_ENCODER, S2_PATCH_SIZE, S2_STRIDE, S2_MIN_VALID,
    S2_EPOCHS, S2_PATIENCE, S2_BATCH_SIZE,
    TRAIN_YEARS, TEST_YEAR,
    TOP_DATES_PER_CROP, TOP_BANDS_PER_CROP,
    MAX_DATES_PER_CROP, MAX_BANDS_PER_CROP,
    S2_DATE_DELTA, S2_DATE_NO_IMPROVE, S2_MAX_DATES,
    S2_BAND_DELTA, S2_BAND_NO_IMPROVE, S2_MAX_BANDS_V2,
    STAGE1V3_CANDIDATES_JSON, STAGE2V3_PER_CROP_JSON,
    STAGE3_EXP_C_V2_JSON, STAGE3_EXP_C_V2_BANDS,
    STAGE3_EXP_D_JSON, STAGE3_EXP_D_BANDS,
    STAGE2V3_RF_PER_CROP_JSON, STAGE3_EXP_C_V2_RF_JSON, STAGE3_EXP_C_V2_RF_BANDS,
    RF_N_ESTIMATORS, RF_MAX_PIXELS, RF_IMPORTANCE_THRESH,
)

log = logging.getLogger(__name__)

STAGE3_EXP_C_V2_BANDS_PROJECTED = PROCESSED_DIR / "stage3_exp_c_v2_bands_projected.json"


# ── Device ─────────────────────────────────────────────────────────────────────

def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _device_label() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return f"cuda ({name})"
    if torch.backends.mps.is_available():
        return "mps (Apple Silicon)"
    return "cpu"

DEVICE = _get_device()


# ── RasterPatchDataset (binary-oracle variant) ─────────────────────────────────

class RasterPatchDataset(Dataset):
    """
    On-the-fly S2/CDL patch pairs.

    Parameters
    ----------
    remap_lut : ndarray (256,) int64, optional
        CDL ID → model class. Defaults to global multiclass REMAP_LUT.
        Pass a binary LUT {crop_id→1, else→0} for the per-crop oracle.
    target_class_id : int, optional
        If set, only include patches containing ≥1 pixel of this CDL class.
        Required for the binary oracle (ensures positive examples in every patch).
    """

    def __init__(self, s2_paths, cdl_path, patch_size, stride,
                 min_valid_frac=0.3, band_indices=None,
                 remap_lut=None, target_class_id=None):
        self.s2_paths     = s2_paths
        self.patch_size   = patch_size
        self.band_indices = band_indices
        self.remap_lut    = remap_lut if remap_lut is not None else REMAP_LUT

        with rasterio.open(cdl_path) as src:
            self._cdl   = src.read(1).astype(np.int32)
            self.height = src.height
            self.width  = src.width

        # num_workers=0 required — rasterio handles cannot be pickled
        self._s2_srcs = [rasterio.open(p) for p in s2_paths]

        ps = patch_size
        self.patches = [
            (r, c)
            for r in range(0, self.height - ps + 1, stride)
            for c in range(0, self.width  - ps + 1, stride)
            if (
                np.isin(self._cdl[r:r+ps, c:c+ps], KEEP_CLASSES).mean() >= min_valid_frac
                and (
                    target_class_id is None
                    or (self._cdl[r:r+ps, c:c+ps] == target_class_id).any()
                )
            )
        ]
        _tgt = (f", require class {target_class_id} ({CDL_CLASS_NAMES.get(target_class_id, '')})"
                if target_class_id is not None else "")
        log.info(f"  RasterPatchDataset: {len(self.patches)} patches "
                 f"(patch={ps}px, stride={stride}px, min_valid={min_valid_frac}{_tgt})")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        r, c = self.patches[idx]
        ps   = self.patch_size
        win  = rasterio.windows.Window(c, r, ps, ps)

        arrays = [src.read(window=win).astype(np.float32) for src in self._s2_srcs]
        img    = np.concatenate(arrays, axis=0)

        if self.band_indices is not None:
            img = img[self.band_indices]

        img[img == S2_NODATA] = 0.0
        for ch in range(img.shape[0]):
            mn, mx = img[ch].min(), img[ch].max()
            img[ch] = (img[ch] - mn) / (mx - mn + 1e-9)

        cdl_patch = self._cdl[r:r+ps, c:c+ps]
        mask      = self.remap_lut[np.clip(cdl_patch, 0, 255)]
        return torch.from_numpy(img), torch.from_numpy(mask.astype(np.int64))

    def __del__(self):
        for src in getattr(self, "_s2_srcs", []):
            try:
                src.close()
            except Exception:
                pass


def _preload_patches(dataset: "RasterPatchDataset") -> TensorDataset:
    """
    Eagerly load all patches from a RasterPatchDataset into a TensorDataset.

    RasterPatchDataset holds open rasterio file handles that cannot be pickled,
    so DataLoader must use num_workers=0, starving the GPU.  By loading all patches
    into RAM once we get a picklable TensorDataset that supports num_workers>0
    and pin_memory=True, keeping the GPU fed.
    """
    n = len(dataset)
    t0 = time.time()
    log.info(f"  Pre-loading {n} patches into RAM...")
    imgs_list, masks_list = [], []
    for i in range(n):
        img, mask = dataset[i]
        imgs_list.append(img)
        masks_list.append(mask)
    imgs_t  = torch.stack(imgs_list)
    masks_t = torch.stack(masks_list)
    elapsed = time.time() - t0
    mem_mb  = (imgs_t.nbytes + masks_t.nbytes) / 1e6
    log.info(f"  Pre-load done: {n} patches  {mem_mb:.1f} MB  ({elapsed:.1f}s)")
    return TensorDataset(imgs_t, masks_t)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(s2_year: str = "2022", stage: int = 1):
    """
    Load S2 file paths and band names for a given year.

    stage=1: stacks all rasters into RAM for GSI pixel sampling (~29 GB peak).
    stage=2: skips stacking — only derives band names from filenames (~2 GB).
             Stage 2 uses RasterPatchDataset which reads patches on-the-fly.

    Returns (df, all_bandnames, n_channels, s2_files, cdl_path).
    df and n_channels are None when stage=2.
    """
    s2_files = sorted([
        p for p in glob(f"{S2_PROCESSED_DIR}/{s2_year}/*_processed.tif")
    ])
    assert s2_files, f"No processed S2 files for year {s2_year} in {S2_PROCESSED_DIR}"

    cdl_path = str(CDL_BY_YEAR[s2_year])
    assert os.path.exists(cdl_path), f"CDL not found: {cdl_path}"

    # Derive band names from filenames — no I/O needed
    all_bandnames = []
    for s2_path in s2_files:
        fname    = os.path.basename(s2_path)
        m        = re.search(r"_(\d{4})_(\d{2})_(\d{2})_processed", fname)
        date_str = f"{m.group(1)}{m.group(2)}{m.group(3)}" if m else fname[:8]
        all_bandnames.extend([f"{b}_{date_str}" for b in S2_BAND_NAMES])

    n_channels = len(all_bandnames)
    log.info(f"S2 files: {len(s2_files)} ({s2_year})  |  {n_channels} channels")

    if stage == 2:
        log.info("Stage 2 mode: skipping raster stacking (patches read on-the-fly)")
        return None, all_bandnames, n_channels, s2_files, cdl_path

    # Stage 1: stack all rasters into RAM for pixel sampling
    log.info(f"Loading {len(s2_files)} S2 files ({s2_year})...")
    all_arrays = []
    for s2_path in s2_files:
        with rasterio.open(s2_path) as src:
            arr = src.read().astype(np.float32)
        arr[arr == S2_NODATA] = np.nan
        all_arrays.append(arr)

    stacked           = np.concatenate(all_arrays, axis=0)
    _, H, W           = stacked.shape
    log.info(f"Stacked S2: {n_channels} channels × {H} × {W} px")

    with rasterio.open(cdl_path) as src:
        cdl = src.read(1).astype(np.int32)
    assert cdl.shape == (H, W), f"CDL/S2 shape mismatch: {cdl.shape} vs ({H},{W})"

    img_2d = stacked.reshape(n_channels, -1).T
    lbl_1d = cdl.flatten()

    valid_mask = np.isin(lbl_1d, KEEP_CLASSES)
    img_valid  = img_2d[valid_mask]
    lbl_valid  = lbl_1d[valid_mask]

    log.info(f"Labeled crop pixels: {len(lbl_valid):,} "
             f"({100*len(lbl_valid)/len(lbl_1d):.1f}% of {len(lbl_1d):,})")

    rng = np.random.default_rng(42)
    n   = min(len(lbl_valid), max(1000, int(len(lbl_valid) * SAMPLE_FRACTION)))
    idx = rng.choice(len(lbl_valid), n, replace=False)

    df = pd.DataFrame(img_valid[idx], columns=all_bandnames)
    df.insert(0, "class_label", lbl_valid[idx].astype(int))

    log.info(f"Sampled {len(df):,} pixels (SAMPLE_FRACTION={SAMPLE_FRACTION})")
    log.info(f"Classes in sample: {sorted(df['class_label'].unique())}")

    del stacked, img_2d
    return df, all_bandnames, n_channels, s2_files, cdl_path


# ── Stage 2v2 helper: U-Net builder & metrics ──────────────────────────────────

def _build_unet(in_channels: int) -> nn.Module:
    return smp.Unet(
        encoder_name=S2_ENCODER,
        encoder_weights=None,
        in_channels=in_channels,
        classes=2,
    ).to(DEVICE)


def _compute_iou_class1(preds: torch.Tensor, labels: torch.Tensor) -> float:
    p     = (preds.view(-1)  == 1).cpu().numpy()
    l     = (labels.view(-1) == 1).cpu().numpy()
    inter = (p & l).sum()
    union = (p | l).sum()
    return float(inter / union) if union > 0 else 0.0


def _train_eval_binary(band_indices: list, crop_id: int, binary_lut: np.ndarray,
                       s2_files: list, cdl_path: str,
                       band_label: str = "",
                       preloaded_tensors: TensorDataset = None) -> float:
    """Train a binary U-Net oracle for crop_id vs. rest. Returns best IoU(class 1)."""
    if preloaded_tensors is None:
        dataset = RasterPatchDataset(
            s2_paths=s2_files, cdl_path=cdl_path,
            patch_size=S2_PATCH_SIZE, stride=S2_STRIDE,
            min_valid_frac=S2_MIN_VALID, band_indices=band_indices,
            remap_lut=binary_lut, target_class_id=crop_id,
        )
        if len(dataset) < 4:
            log.warning(f"    Only {len(dataset)} patches for crop {crop_id} — returning 0.0")
            return 0.0
        tensor_ds = _preload_patches(dataset)
    else:
        # Preloaded tensors are (imgs, masks) with ALL bands; slice band_indices
        imgs_full, masks = preloaded_tensors.tensors
        imgs_sliced      = imgs_full[:, band_indices, :, :]
        tensor_ds        = TensorDataset(imgs_sliced, masks)

    if len(tensor_ds) < 4:
        log.warning(f"    Only {len(tensor_ds)} patches for crop {crop_id} — returning 0.0")
        return 0.0

    n_val   = max(1, int(0.2 * len(tensor_ds)))
    n_train = len(tensor_ds) - n_val
    train_ds, val_ds = random_split(
        tensor_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    use_pin  = DEVICE.startswith("cuda")
    n_workers = min(4, os.cpu_count() or 1)
    train_dl  = DataLoader(train_ds, batch_size=S2_BATCH_SIZE, shuffle=True,
                           num_workers=n_workers, pin_memory=use_pin,
                           persistent_workers=n_workers > 0)
    val_dl    = DataLoader(val_ds,   batch_size=S2_BATCH_SIZE, shuffle=False,
                           num_workers=n_workers, pin_memory=use_pin,
                           persistent_workers=n_workers > 0)
    model     = _build_unet(len(band_indices))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()   # no ignore_index: binary, class 0 is informative

    log.info(f"    U-Net  in_ch={len(band_indices)}  "
             f"train={n_train} patches  val={n_val} patches  "
             f"max_epochs={S2_EPOCHS}  patience={S2_PATIENCE}  "
             f"workers={n_workers}  pin_memory={use_pin}"
             + (f"  [{band_label}]" if band_label else ""))

    best_iou, no_improve = 0.0, 0

    for epoch in range(S2_EPOCHS):
        model.train()
        train_loss, n_batches = 0.0, 0
        for imgs, masks in train_dl:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches  += 1

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, masks in val_dl:
                preds = model(imgs.to(DEVICE)).argmax(dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(masks)

        iou       = _compute_iou_class1(torch.cat(all_preds), torch.cat(all_labels))
        avg_loss  = train_loss / max(n_batches, 1)
        improved  = iou > best_iou + 1e-4
        marker    = " *" if improved else ""
        log.info(f"    epoch {epoch+1:>2}/{S2_EPOCHS}  loss={avg_loss:.4f}  "
                 f"IoU(c1)={iou:.4f}{marker}  "
                 f"[best={best_iou:.4f}  no_improve={no_improve}/{S2_PATIENCE}]")

        if improved:
            best_iou, no_improve = iou, 0
        else:
            no_improve += 1
            if no_improve >= S2_PATIENCE:
                log.info(f"    Early stop at epoch {epoch+1} (patience={S2_PATIENCE})")
                break

    return best_iou


# ── v2 index helpers ──────────────────────────────────────────────────────────

def _dates_to_band_indices(selected_dates, band_name_to_idx, vege_bands=None):
    """Return flat channel indices for selected_dates × all vege_bands."""
    if vege_bands is None:
        vege_bands = VEGE_BANDS
    idx = []
    for d in selected_dates:
        for b in vege_bands:
            key = f"{b}_{d}"
            if key in band_name_to_idx:
                idx.append(band_name_to_idx[key])
    return idx


def _dates_bands_to_indices(selected_dates, selected_bands, band_name_to_idx):
    """Return flat channel indices for selected_dates × selected_bands."""
    idx = []
    for d in selected_dates:
        for b in selected_bands:
            key = f"{b}_{d}"
            if key in band_name_to_idx:
                idx.append(band_name_to_idx[key])
    return idx


# ── Stage 1v3: Per-crop date + band ranking ────────────────────────────────────

def _fmt_date(d: str) -> str:
    """'20220715' → 'Jul 15'"""
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    try:
        m = int(d[4:6]) - 1
        day = int(d[6:8])
        return f"{months[m]} {day}"
    except Exception:
        return d


def _plot_gsi_heatmaps(gsi_df: pd.DataFrame, all_dates: list, save_dir: pathlib.Path) -> list:
    """
    Generate per-crop SI_global heatmaps (date × band) and a combined grid figure.

    gsi_df: DataFrame with index = "B4_20220730" channel names, columns = CDL class IDs.
    Returns list of saved file paths.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # Compute global vmax from 95th percentile across all crops — consistent scale
    all_vals = []
    matrices = {}
    for crop_id in KEEP_CLASSES:
        si_col = gsi_df[crop_id] if crop_id in gsi_df.columns else pd.Series(dtype=float)
        mat = np.zeros((len(all_dates), len(S2_BAND_NAMES)), dtype=np.float32)
        for di, date in enumerate(all_dates):
            for bi, band in enumerate(S2_BAND_NAMES):
                key = f"{band}_{date}"
                if key in si_col.index:
                    mat[di, bi] = si_col[key]
        matrices[crop_id] = mat
        all_vals.extend(mat.flatten().tolist())
    global_vmax = float(np.nanpercentile(all_vals, 95)) if all_vals else 1.0
    global_vmax = max(global_vmax, 1e-3)
    log.info(f"  GSI heatmap global_vmax (95th pct): {global_vmax:.4f}")

    n_crops  = len(KEEP_CLASSES)
    n_cols   = 4
    n_rows   = (n_crops + n_cols - 1) // n_cols
    fig_grid, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(n_cols * 5, n_rows * 4.5),
                                  constrained_layout=True)
    axes_flat = axes.flatten() if n_crops > 1 else [axes]

    for ax_idx, crop_id in enumerate(KEEP_CLASSES):
        crop_name = CDL_CLASS_NAMES.get(crop_id, f"cls{crop_id}")
        ax        = axes_flat[ax_idx]
        matrix    = matrices[crop_id]

        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
                       vmin=0, vmax=global_vmax)
        ax.set_xticks(range(len(S2_BAND_NAMES)))
        ax.set_xticklabels(S2_BAND_NAMES, fontsize=8)
        ax.set_yticks(range(len(all_dates)))
        ax.set_yticklabels([_fmt_date(d) for d in all_dates], fontsize=7)
        ax.set_title(crop_name, fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

        # Save individual crop heatmap
        fig_single, ax_s = plt.subplots(figsize=(6, 5))
        im_s = ax_s.imshow(matrix, aspect="auto", cmap="YlOrRd",
                           vmin=0, vmax=global_vmax)
        ax_s.set_xticks(range(len(S2_BAND_NAMES)))
        ax_s.set_xticklabels(S2_BAND_NAMES, fontsize=9)
        ax_s.set_yticks(range(len(all_dates)))
        ax_s.set_yticklabels([_fmt_date(d) for d in all_dates], fontsize=8)
        ax_s.set_title(f"SI_global — {crop_name}", fontsize=11)
        ax_s.set_xlabel("Spectral Band")
        ax_s.set_ylabel("Acquisition Date")
        plt.colorbar(im_s, ax=ax_s, label="SI_global")
        plt.tight_layout()
        out = save_dir / f"stage1v3_gsi_{crop_name.lower().replace(' ', '_')}.png"
        fig_single.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig_single)
        saved.append(out)

    # Hide unused grid axes
    for ax in axes_flat[n_crops:]:
        ax.set_visible(False)

    fig_grid.suptitle("SI_global Heatmaps — Date × Band per Crop", fontsize=13)
    grid_path = save_dir / "stage1v3_gsi_heatmaps_all.png"
    fig_grid.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig_grid)
    saved.append(grid_path)

    log.info(f"  Saved {len(saved)} GSI heatmap(s) to {save_dir}")
    return saved


def _plot_selection_table(results_per_crop: dict, save_path: pathlib.Path) -> None:
    """
    Save a per-crop selection summary table (PNG + CSV):
      Crop | Key Period (dates) | Selected Bands | IoU (bands)
    """
    rows = []
    for crop_id in KEEP_CLASSES:
        r = results_per_crop.get(crop_id, {})
        dates_fmt = ", ".join(_fmt_date(d) for d in r.get("dates", []))
        bands_str = ", ".join(r.get("bands", []))
        rows.append({
            "Crop":           CDL_CLASS_NAMES.get(crop_id, f"cls{crop_id}"),
            "Key Period":     dates_fmt or "—",
            "Selected Bands": bands_str or "—",
            "IoU (bands)":    f"{r.get('best_iou_after_bands', 0.0):.4f}",
        })
    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = save_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    log.info(f"  Saved selection table CSV: {csv_path}")

    # Save PNG table
    fig_h = 0.45 * (len(rows) + 1.5)
    fig, ax = plt.subplots(figsize=(13, max(fig_h, 3)))
    ax.axis("off")
    col_widths = [0.12, 0.38, 0.32, 0.12]
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    # Header styling
    for j in range(len(df.columns)):
        tbl[(0, j)].set_facecolor("#2d6a2d")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    # Alternating row colours
    for i in range(1, len(rows) + 1):
        fc = "#f0f7f0" if i % 2 == 0 else "white"
        for j in range(len(df.columns)):
            tbl[(i, j)].set_facecolor(fc)
    ax.set_title("Stage 2v2 Per-Crop Feature Selection", fontsize=12,
                 fontweight="bold", pad=10)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved selection table PNG: {save_path}")


def run_stage1v3(s2_paths, cdl_path, data_dir=None):
    """
    Compute per-crop SI_global and split into date candidates + band candidates.

    For each crop:
      - Date ranking: mean SI across all bands at each acquisition date → top TOP_DATES_PER_CROP
      - Band ranking: mean SI across all dates for each spectral band   → top TOP_BANDS_PER_CROP

    Saves STAGE1V3_CANDIDATES_JSON and logs to MLflow.

    Returns (date_candidates_per_crop, band_candidates_per_crop, band_name_to_idx, all_dates)
      - dict keys are str(crop_id)
      - date candidates: list of "YYYYMMDD" strings
      - band candidates: list of band names like "B4"
    """
    log.info("Stage 1v3: computing per-crop GSI and deriving date + band candidates...")

    # ── MLflow run starts here so file loading time is tracked ────────────────
    _mlflow_setup()
    ts         = datetime.now().strftime("%Y%m%d-%H%M%S")
    stage1_run = mlflow.start_run(run_name=f"stage1v3_{ts}")

    # ── Build band_name_to_idx ────────────────────────────────────────────────
    all_bandnames = []
    all_dates_set = []
    for s2_path in s2_paths:
        fname    = os.path.basename(s2_path)
        m        = re.search(r"_(\d{4})_(\d{2})_(\d{2})_processed", fname)
        date_str = f"{m.group(1)}{m.group(2)}{m.group(3)}" if m else fname[:8]
        if date_str not in all_dates_set:
            all_dates_set.append(date_str)
        all_bandnames.extend([f"{b}_{date_str}" for b in S2_BAND_NAMES])

    all_dates         = sorted(all_dates_set)
    band_name_to_idx  = {name: i for i, name in enumerate(all_bandnames)}
    n_channels        = len(all_bandnames)
    log.info(f"S2 files: {len(s2_paths)}  |  {n_channels} channels  |  {len(all_dates)} dates")

    # ── Load and stack all S2 rasters for pixel sampling ─────────────────────
    log.info(f"Loading {len(s2_paths)} S2 files for Stage 1v3 pixel sampling...")
    all_arrays = []
    for s2_path in s2_paths:
        with rasterio.open(s2_path) as src:
            arr = src.read().astype(np.float32)
        arr[arr == S2_NODATA] = np.nan
        all_arrays.append(arr)

    stacked  = np.concatenate(all_arrays, axis=0)
    _, H, W  = stacked.shape
    log.info(f"Stacked S2: {n_channels} channels × {H} × {W} px")

    with rasterio.open(cdl_path) as src:
        cdl = src.read(1).astype(np.int32)
    assert cdl.shape == (H, W), f"CDL/S2 shape mismatch: {cdl.shape} vs ({H},{W})"

    img_2d = stacked.reshape(n_channels, -1).T
    lbl_1d = cdl.flatten()

    valid_mask = np.isin(lbl_1d, KEEP_CLASSES)
    img_valid  = img_2d[valid_mask]
    lbl_valid  = lbl_1d[valid_mask]

    log.info(f"Labeled crop pixels: {len(lbl_valid):,} "
             f"({100*len(lbl_valid)/len(lbl_1d):.1f}% of {len(lbl_1d):,})")

    rng = np.random.default_rng(42)
    n   = min(len(lbl_valid), max(1000, int(len(lbl_valid) * SAMPLE_FRACTION)))
    idx = rng.choice(len(lbl_valid), n, replace=False)

    df = pd.DataFrame(img_valid[idx], columns=all_bandnames)
    df.insert(0, "class_label", lbl_valid[idx].astype(int))

    log.info(f"Sampled {len(df):,} pixels (SAMPLE_FRACTION={SAMPLE_FRACTION})")
    log.info(f"Classes in sample: {sorted(df['class_label'].unique())}")
    _X_sample = df[all_bandnames].values
    _nan_px   = np.isnan(_X_sample).any(axis=1).sum()
    _nan_ch   = np.isnan(_X_sample).any(axis=0).sum()
    log.info(f"NaN in sample: {_nan_px:,} pixels ({100*_nan_px/len(df):.1f}%) "
             f"have ≥1 NaN channel;  {_nan_ch}/{n_channels} channels have ≥1 NaN pixel")
    del stacked, img_2d

    # ── Compute per-crop binary SI (one-vs-all) ───────────────────────────────
    # For each crop: SI[channel] = |mean_crop - mean_rest| / (std_crop + std_rest)
    # "rest" = all other labeled crop pixels pooled together.
    # One-vs-all avoids pairwise averaging that dilutes the signal when a crop is
    # similar to just one or two other crops.  No 1.96 factor — values stay in
    # natural z-score units (0 = total overlap, 1 = means separated by 1 std).
    log.info("Computing per-crop binary SI_global (one-vs-all)...")
    X_all = df[all_bandnames].values.astype(np.float32)
    y_all = df["class_label"].values

    gsi_dict = {}
    for crop_id in KEEP_CLASSES:
        crop_mask = (y_all == crop_id)
        rest_mask = np.isin(y_all, KEEP_CLASSES) & ~crop_mask
        if crop_mask.sum() < 10:
            log.warning(f"Crop {crop_id} ({CDL_CLASS_NAMES[crop_id]}) has only "
                        f"{crop_mask.sum()} samples — using zeros")
            gsi_dict[crop_id] = pd.Series(0.0, index=all_bandnames)
            continue
        X_c = X_all[crop_mask]
        X_r = X_all[rest_mask]
        # Use nanmean/nanstd so NaN pixels (NoData / cloud) don't corrupt the per-channel stats.
        mean_c = np.nanmean(X_c, axis=0)
        std_c  = np.nanstd(X_c, axis=0)
        mean_r = np.nanmean(X_r, axis=0)
        # Normalise by crop's own std only.
        # Using pooled std_r inflates the denominator because "rest" contains 9 different
        # crop types whose means span a wide range, making std_r >> std_c and suppressing SI.
        si = np.abs(mean_c - mean_r) / (std_c + 1e-9)
        gsi_dict[crop_id] = pd.Series(si.astype(np.float32), index=all_bandnames)

    # gsi_df: index = channel names ("B4_20220730"), columns = crop_id
    gsi_df          = pd.DataFrame(gsi_dict)
    gsi_mean_global = gsi_df.mean(axis=1).sort_values(ascending=False)
    _v = gsi_df.values
    log.info(f"gsi_df shape: {gsi_df.shape}  (channels × crops)")
    log.info(f"  SI range:  min={np.nanmin(_v):.4f}  p25={np.nanpercentile(_v,25):.4f}  "
             f"median={np.nanmedian(_v):.4f}  p75={np.nanpercentile(_v,75):.4f}  "
             f"p95={np.nanpercentile(_v,95):.4f}  max={np.nanmax(_v):.4f}")
    log.info(f"  Top-K selection: TOP_DATES_PER_CROP={TOP_DATES_PER_CROP}  "
             f"TOP_BANDS_PER_CROP={TOP_BANDS_PER_CROP}  (no threshold — always selects top-K)")

    # ── Per-crop date and band ranking ────────────────────────────────────────
    date_candidates_per_crop: dict[str, list] = {}
    band_candidates_per_crop: dict[str, list] = {}

    for crop_id in KEEP_CLASSES:
        crop_key = str(crop_id)
        if crop_id in gsi_df.columns:
            si_crop = gsi_df[crop_id]
        else:
            log.warning(
                f"Crop {crop_id} ({CDL_CLASS_NAMES[crop_id]}) not in sample — "
                "falling back to global mean ranking"
            )
            si_crop = gsi_mean_global

        # Step 1 — Date ranking: mean SI across all S2_BAND_NAMES at each date.
        # Identifies the key phenological window where the crop is most separable.
        # Always keeps top-K — no threshold, so results are never empty.
        date_si: dict[str, float] = {}
        for date in all_dates:
            band_keys = [f"{b}_{date}" for b in S2_BAND_NAMES if f"{b}_{date}" in si_crop.index]
            date_si[date] = float(si_crop[band_keys].mean()) if band_keys else 0.0
        sorted_dates   = sorted(date_si.items(), key=lambda x: x[1], reverse=True)
        selected_dates = [d for d, _ in sorted_dates[:TOP_DATES_PER_CROP]]
        date_candidates_per_crop[crop_key] = selected_dates

        # Step 2 — Band ranking: mean SI across SELECTED DATES ONLY.
        # Finds the most informative bands within the key phenological window.
        # Always keeps top-K — no threshold.
        band_si: dict[str, float] = {}
        for band in S2_BAND_NAMES:
            band_keys = [f"{band}_{d}" for d in selected_dates if f"{band}_{d}" in si_crop.index]
            band_si[band] = float(si_crop[band_keys].mean()) if band_keys else 0.0
        sorted_bands = sorted(band_si.items(), key=lambda x: x[1], reverse=True)
        band_candidates_per_crop[crop_key] = [b for b, _ in sorted_bands[:TOP_BANDS_PER_CROP]]

        log.info(
            f"  {CDL_CLASS_NAMES[crop_id]:20s}: "
            f"top {len(selected_dates)} dates={selected_dates[:3]}...  "
            f"top {len(band_candidates_per_crop[crop_key])} bands="
            f"{band_candidates_per_crop[crop_key][:3]}... "
            f"(scored within {len(selected_dates)}-date window)"
        )

    # ── Save handoff file ─────────────────────────────────────────────────────
    _stage1v3_path = STAGE1V3_CANDIDATES_JSON
    if data_dir:
        _stage1v3_path = pathlib.Path(data_dir) / "s2" / "2022" / "stage1v3_candidates.json"
    os.makedirs(os.path.dirname(_stage1v3_path), exist_ok=True)

    run_ts  = datetime.now().strftime("%Y%m%d-%H%M%S")
    payload = {
        "run_ts":                    run_ts,
        "all_dates":                 all_dates,
        "date_candidates_per_crop":  date_candidates_per_crop,
        "band_candidates_per_crop":  band_candidates_per_crop,
    }
    with open(_stage1v3_path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info(f"Stage 1v3 candidates saved: {_stage1v3_path}")

    # ── Write Exp D output (Stage 1 top-K without Stage 2 CNN) ────────────────
    _save_exp_d_bands(date_candidates_per_crop, band_candidates_per_crop,
                      band_name_to_idx, data_dir=data_dir)

    # ── MLflow params + artifacts ─────────────────────────────────────────────
    mlflow.log_params({
        "stage":              "1v3_date_band_ranking",
        "version":            "v3",
        "n_images":           len(s2_paths),
        "n_dates":            len(all_dates),
        "total_channels":     n_channels,
        "sample_fraction":    SAMPLE_FRACTION,
        "n_sampled":           len(df),
        "top_dates_per_crop":  TOP_DATES_PER_CROP,
        "top_bands_per_crop":  TOP_BANDS_PER_CROP,
        "max_bands_per_crop": MAX_BANDS_PER_CROP,
        "keep_classes":       str(KEEP_CLASSES),
    })

    rows = []
    for crop_id in KEEP_CLASSES:
        crop_key = str(crop_id)
        for rank, date in enumerate(date_candidates_per_crop[crop_key]):
            rows.append({
                "crop_id":   crop_id, "crop_name": CDL_CLASS_NAMES[crop_id],
                "type":      "date", "rank": rank + 1, "value": date,
            })
        for rank, band in enumerate(band_candidates_per_crop[crop_key]):
            rows.append({
                "crop_id":   crop_id, "crop_name": CDL_CLASS_NAMES[crop_id],
                "type":      "band", "rank": rank + 1, "value": band,
            })

    with tempfile.TemporaryDirectory() as tmp:
        p = pathlib.Path(tmp) / "stage1v3_per_crop_candidates.csv"
        p.write_text(pd.DataFrame(rows).to_csv(index=False))
        mlflow.log_artifact(str(p))
    mlflow.log_artifact(str(_stage1v3_path))
    if STAGE3_EXP_D_JSON.exists():
        mlflow.log_artifact(str(STAGE3_EXP_D_JSON))
    if STAGE3_EXP_D_BANDS.exists():
        mlflow.log_artifact(str(STAGE3_EXP_D_BANDS))

    # ── GSI heatmaps ─────────────────────────────────────────────────────────
    heatmap_dir = FIGURES_DIR / "stage1v3_gsi"
    heatmap_files = _plot_gsi_heatmaps(gsi_df, all_dates, heatmap_dir)
    for hf in heatmap_files:
        mlflow.log_artifact(str(hf))

    mlflow.end_run(status="FINISHED")
    log.info(f"Stage 1v3 MLflow run_id: {stage1_run.info.run_id}")

    return date_candidates_per_crop, band_candidates_per_crop, band_name_to_idx, all_dates


# ── Stage 2v2: Per-crop date + band forward selection ─────────────────────────

def run_stage2v2(s2_paths, cdl_path, date_candidates_per_crop, band_candidates_per_crop,
                 band_name_to_idx, all_dates, data_dir=None):
    """
    Per-crop binary CNN forward selection — Phase A (dates) then Phase B (bands).

    MLflow structure:
        stage2v3_{ts}                         — parent: hyperparams + summary
        └── stage2v3_{crop_name}_{ts}         — child per crop: Phase A + B history

    Returns results_per_crop = {crop_id: {"dates": [...], "bands": [...], ...}}.
    """
    # Attach a file handler so all log output is also written to disk
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    run_ts      = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path    = PROCESSED_DIR / f"stage2v2_run_{run_ts}.log"
    _fh         = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(_fh)

    _mlflow_setup()
    parent_run = mlflow.start_run(run_name=f"stage2v3_{run_ts}")
    mlflow.log_params({
        "stage":               "2v2_date_band_fwd_selection",
        "version":             "v2",
        "encoder":             S2_ENCODER,
        "patch_size":          S2_PATCH_SIZE,
        "stride":              S2_STRIDE,
        "min_valid":           S2_MIN_VALID,
        "epochs":              S2_EPOCHS,
        "patience":            S2_PATIENCE,
        "date_delta":          S2_DATE_DELTA,
        "date_no_improve":     S2_DATE_NO_IMPROVE,
        "max_dates":           S2_MAX_DATES,
        "band_delta":          S2_BAND_DELTA,
        "band_no_improve":     S2_BAND_NO_IMPROVE,
        "max_bands":           S2_MAX_BANDS_V2,
        "top_dates_per_crop":  TOP_DATES_PER_CROP,
        "top_bands_per_crop":  TOP_BANDS_PER_CROP,
        "max_bands_per_crop":  MAX_BANDS_PER_CROP,
        "device":              DEVICE,
        "n_crops":             len(KEEP_CLASSES),
    })

    # Pre-load ALL patches into TensorDataset with ALL bands before per-crop loop.
    # This is the same GPU optimization as the existing Stage 2: rasterio handles
    # cannot be pickled, forcing num_workers=0 if we read on-the-fly.
    log.info("Pre-loading all patches into RAM (all bands)...")
    full_dataset = RasterPatchDataset(
        s2_paths=s2_paths, cdl_path=cdl_path,
        patch_size=S2_PATCH_SIZE, stride=S2_STRIDE,
        min_valid_frac=S2_MIN_VALID, band_indices=None,  # all bands
        remap_lut=None,   # multiclass for pre-loading shapes; binary slicing done per crop
        target_class_id=None,
    )
    preloaded_all = _preload_patches(full_dataset)
    imgs_all, masks_all = preloaded_all.tensors
    log.info(f"Pre-loaded: imgs={tuple(imgs_all.shape)}  masks={tuple(masks_all.shape)}")

    results_per_crop: dict[int, dict] = {}
    n_crops = len(KEEP_CLASSES)

    log.info(f"\nStage 2v2 — per-crop binary CNN forward selection (date + band phases)")
    log.info(f"  Crops: {n_crops}  |  δ_date={S2_DATE_DELTA}  δ_band={S2_BAND_DELTA}  "
             f"max_dates={S2_MAX_DATES}  max_bands={S2_MAX_BANDS_V2}")
    log.info(f"  Epochs={S2_EPOCHS}  patience={S2_PATIENCE}  "
             f"batch={S2_BATCH_SIZE}  patch={S2_PATCH_SIZE}px  stride={S2_STRIDE}px")

    try:
        for crop_idx, crop_id in enumerate(KEEP_CLASSES, 1):
            crop_name   = CDL_CLASS_NAMES[crop_id]
            crop_key    = str(crop_id)
            date_cands  = date_candidates_per_crop[crop_key]
            band_cands  = band_candidates_per_crop[crop_key]

            log.info(f"\n{'='*60}")
            log.info(f"[{crop_idx}/{n_crops}] Crop: {crop_name} (CDL id={crop_id}) "
                     f"— {len(date_cands)} date candidates, {len(band_cands)} band candidates")

            binary_lut          = np.zeros(256, dtype=np.int64)
            binary_lut[crop_id] = 1

            # Build per-crop binary mask TensorDataset from pre-loaded patches
            # Remap the pre-loaded multiclass masks to binary for this crop
            binary_masks = (masks_all == crop_id).long()
            crop_has_class = binary_masks.sum(dim=(1, 2)) > 0
            crop_imgs  = imgs_all[crop_has_class]
            crop_masks = binary_masks[crop_has_class]
            crop_tensor_ds = TensorDataset(crop_imgs, crop_masks)

            log.info(f"  Crop patches (has class {crop_id}): {len(crop_tensor_ds)}")
            if len(crop_tensor_ds) < 4:
                log.warning(f"  Too few patches for crop {crop_id} — skipping")
                results_per_crop[crop_id] = {
                    "dates": date_cands[:1] if date_cands else [],
                    "bands": band_cands[:1] if band_cands else [],
                    "k_dates": 1, "k_bands": 1,
                    "best_iou_after_dates": 0.0, "best_iou_after_bands": 0.0,
                    "fallback_dates": True, "fallback_bands": True,
                }
                continue

            # ── Nested run per crop ───────────────────────────────────────────
            with mlflow.start_run(
                run_name=f"stage2v3_{crop_name.replace('/', '-')}_{run_ts}",
                nested=True,
            ) as crop_run:
                mlflow.log_params({
                    "crop_id":          crop_id,
                    "crop_name":        crop_name,
                    "n_date_candidates": len(date_cands),
                    "n_band_candidates": len(band_cands),
                })

                # ── Phase A: Date selection ───────────────────────────────────
                log.info(f"\n  === Phase A: Date selection for {crop_name} ===")
                selected_dates, best_iou_dates, no_improve_dates = [], 0.0, 0

                for step, date in enumerate(date_cands):
                    if len(selected_dates) >= S2_MAX_DATES:
                        log.info(f"  max_dates={S2_MAX_DATES} reached — stopping Phase A")
                        break
                    if no_improve_dates >= S2_DATE_NO_IMPROVE:
                        log.info(f"  {S2_DATE_NO_IMPROVE} consecutive date rejections — stopping Phase A")
                        break

                    trial_dates  = selected_dates + [date]
                    trial_idx    = _dates_to_band_indices(trial_dates, band_name_to_idx)
                    if not trial_idx:
                        log.warning(f"  date={date}: no valid band indices — skipping")
                        no_improve_dates += 1
                        continue

                    log.info(f"\n  --- Phase A step {step+1}/{len(date_cands)}  "
                             f"trial date: {date}  (selected: {len(selected_dates)}) ---")
                    t0      = time.time()

                    # Slice band indices from crop_tensor_ds
                    crop_imgs_sliced = crop_imgs[:, trial_idx, :, :]
                    trial_ds         = TensorDataset(crop_imgs_sliced, crop_masks)
                    n_val   = max(1, int(0.2 * len(trial_ds)))
                    n_train = len(trial_ds) - n_val
                    train_ds, val_ds = random_split(
                        trial_ds, [n_train, n_val],
                        generator=torch.Generator().manual_seed(42),
                    )
                    use_pin   = DEVICE.startswith("cuda")
                    n_workers = min(4, os.cpu_count() or 1)
                    train_dl  = DataLoader(train_ds, batch_size=S2_BATCH_SIZE, shuffle=True,
                                           num_workers=n_workers, pin_memory=use_pin,
                                           persistent_workers=n_workers > 0)
                    val_dl    = DataLoader(val_ds, batch_size=S2_BATCH_SIZE, shuffle=False,
                                           num_workers=n_workers, pin_memory=use_pin,
                                           persistent_workers=n_workers > 0)
                    model     = _build_unet(len(trial_idx))
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                    criterion = nn.CrossEntropyLoss()

                    best_epoch_iou, no_improve_ep = 0.0, 0
                    for epoch in range(S2_EPOCHS):
                        model.train()
                        train_loss, n_batches = 0.0, 0
                        for imgs_b, masks_b in train_dl:
                            imgs_b, masks_b = imgs_b.to(DEVICE), masks_b.to(DEVICE)
                            optimizer.zero_grad()
                            loss = criterion(model(imgs_b), masks_b)
                            loss.backward()
                            optimizer.step()
                            train_loss += loss.item()
                            n_batches  += 1
                        model.eval()
                        all_preds, all_labels = [], []
                        with torch.no_grad():
                            for imgs_b, masks_b in val_dl:
                                preds = model(imgs_b.to(DEVICE)).argmax(dim=1)
                                all_preds.append(preds.cpu())
                                all_labels.append(masks_b)
                        iou      = _compute_iou_class1(torch.cat(all_preds), torch.cat(all_labels))
                        improved = iou > best_epoch_iou + 1e-4
                        if improved:
                            best_epoch_iou, no_improve_ep = iou, 0
                        else:
                            no_improve_ep += 1
                            if no_improve_ep >= S2_PATIENCE:
                                break

                    iou      = best_epoch_iou
                    elapsed  = time.time() - t0
                    gain     = iou - best_iou_dates
                    accepted = gain >= S2_DATE_DELTA

                    if accepted:
                        selected_dates = selected_dates + [date]
                        best_iou_dates = iou
                        no_improve_dates = 0
                    else:
                        no_improve_dates += 1

                    mlflow.log_metrics({
                        "phaseA_iou":      iou,
                        "phaseA_gain":     gain,
                        "phaseA_accepted": int(accepted),
                        "phaseA_n_dates":  len(selected_dates),
                    }, step=step)

                    tag = "OK" if accepted else "--"
                    log.info(f"  [{tag}] +{date}  IoU={iou:.4f} gain={gain:+.4f} ({elapsed:.0f}s)")

                # Fallback: if nothing selected, take top-1 date
                fallback_dates = False
                if not selected_dates and date_cands:
                    selected_dates = [date_cands[0]]
                    fallback_dates = True
                    log.warning(f"  K_dates=0 for {crop_name} — fallback to top-1 date: {date_cands[0]}")
                    mlflow.set_tag("phaseA_fallback_date", date_cands[0])

                log.info(f"\n  Phase A done: K_dates={len(selected_dates)}  "
                         f"best_iou={best_iou_dates:.4f}  dates={selected_dates}")
                mlflow.log_metrics({"phaseA_final_iou": best_iou_dates,
                                    "phaseA_k_dates":   len(selected_dates)})
                mlflow.set_tag("phaseA_selected_dates", str(selected_dates))

                # ── Phase B: Band selection ───────────────────────────────────
                log.info(f"\n  === Phase B: Band selection for {crop_name} "
                         f"(fixed dates={selected_dates}) ===")
                selected_bands, best_iou_bands, no_improve_bands = [], 0.0, 0

                for step, band in enumerate(band_cands):
                    if len(selected_bands) >= S2_MAX_BANDS_V2:
                        log.info(f"  max_bands={S2_MAX_BANDS_V2} reached — stopping Phase B")
                        break
                    if no_improve_bands >= S2_BAND_NO_IMPROVE:
                        log.info(f"  {S2_BAND_NO_IMPROVE} consecutive band rejections — stopping Phase B")
                        break

                    trial_bands = selected_bands + [band]
                    trial_idx   = _dates_bands_to_indices(selected_dates, trial_bands, band_name_to_idx)
                    if not trial_idx:
                        log.warning(f"  band={band}: no valid indices — skipping")
                        no_improve_bands += 1
                        continue

                    log.info(f"\n  --- Phase B step {step+1}/{len(band_cands)}  "
                             f"trial band: {band}  (selected: {len(selected_bands)}) ---")
                    t0      = time.time()

                    crop_imgs_sliced = crop_imgs[:, trial_idx, :, :]
                    trial_ds         = TensorDataset(crop_imgs_sliced, crop_masks)
                    n_val   = max(1, int(0.2 * len(trial_ds)))
                    n_train = len(trial_ds) - n_val
                    train_ds, val_ds = random_split(
                        trial_ds, [n_train, n_val],
                        generator=torch.Generator().manual_seed(42),
                    )
                    use_pin   = DEVICE.startswith("cuda")
                    n_workers = min(4, os.cpu_count() or 1)
                    train_dl  = DataLoader(train_ds, batch_size=S2_BATCH_SIZE, shuffle=True,
                                           num_workers=n_workers, pin_memory=use_pin,
                                           persistent_workers=n_workers > 0)
                    val_dl    = DataLoader(val_ds, batch_size=S2_BATCH_SIZE, shuffle=False,
                                           num_workers=n_workers, pin_memory=use_pin,
                                           persistent_workers=n_workers > 0)
                    model     = _build_unet(len(trial_idx))
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                    criterion = nn.CrossEntropyLoss()

                    best_epoch_iou, no_improve_ep = 0.0, 0
                    for epoch in range(S2_EPOCHS):
                        model.train()
                        train_loss, n_batches = 0.0, 0
                        for imgs_b, masks_b in train_dl:
                            imgs_b, masks_b = imgs_b.to(DEVICE), masks_b.to(DEVICE)
                            optimizer.zero_grad()
                            loss = criterion(model(imgs_b), masks_b)
                            loss.backward()
                            optimizer.step()
                            train_loss += loss.item()
                            n_batches  += 1
                        model.eval()
                        all_preds, all_labels = [], []
                        with torch.no_grad():
                            for imgs_b, masks_b in val_dl:
                                preds = model(imgs_b.to(DEVICE)).argmax(dim=1)
                                all_preds.append(preds.cpu())
                                all_labels.append(masks_b)
                        iou      = _compute_iou_class1(torch.cat(all_preds), torch.cat(all_labels))
                        improved = iou > best_epoch_iou + 1e-4
                        if improved:
                            best_epoch_iou, no_improve_ep = iou, 0
                        else:
                            no_improve_ep += 1
                            if no_improve_ep >= S2_PATIENCE:
                                break

                    iou      = best_epoch_iou
                    elapsed  = time.time() - t0
                    gain     = iou - best_iou_bands
                    accepted = gain >= S2_BAND_DELTA

                    if accepted:
                        selected_bands = selected_bands + [band]
                        best_iou_bands = iou
                        no_improve_bands = 0
                    else:
                        no_improve_bands += 1

                    mlflow.log_metrics({
                        "phaseB_iou":      iou,
                        "phaseB_gain":     gain,
                        "phaseB_accepted": int(accepted),
                        "phaseB_n_bands":  len(selected_bands),
                    }, step=step)

                    tag = "OK" if accepted else "--"
                    log.info(f"  [{tag}] +{band}  IoU={iou:.4f} gain={gain:+.4f} ({elapsed:.0f}s)")

                # Fallback: if nothing selected, take top-1 band
                fallback_bands = False
                if not selected_bands and band_cands:
                    selected_bands = [band_cands[0]]
                    fallback_bands = True
                    log.warning(f"  K_bands=0 for {crop_name} — fallback to top-1 band: {band_cands[0]}")
                    mlflow.set_tag("phaseB_fallback_band", band_cands[0])

                log.info(f"\n  Phase B done: K_bands={len(selected_bands)}  "
                         f"best_iou={best_iou_bands:.4f}  bands={selected_bands}")
                mlflow.log_metrics({"phaseB_final_iou": best_iou_bands,
                                    "phaseB_k_bands":   len(selected_bands)})
                mlflow.set_tag("phaseB_selected_bands", str(selected_bands))

                result = {
                    "dates":                 selected_dates,
                    "bands":                 selected_bands,
                    "k_dates":               len(selected_dates),
                    "k_bands":               len(selected_bands),
                    "best_iou_after_dates":  round(best_iou_dates, 4),
                    "best_iou_after_bands":  round(best_iou_bands, 4),
                    "fallback_dates":        fallback_dates,
                    "fallback_bands":        fallback_bands,
                    "mlflow_run_id":         crop_run.info.run_id,
                }
                results_per_crop[crop_id] = result

            log.info(f"\n  -> [{crop_idx}/{n_crops}] {crop_name}: "
                     f"K_dates={len(selected_dates)}  K_bands={len(selected_bands)}  "
                     f"IoU_dates={best_iou_dates:.4f}  IoU_bands={best_iou_bands:.4f}")

            # Mirror summary to parent run
            mlflow.log_metrics({
                f"crop_{crop_id}_k_dates":    len(selected_dates),
                f"crop_{crop_id}_k_bands":    len(selected_bands),
                f"crop_{crop_id}_iou_dates":  best_iou_dates,
                f"crop_{crop_id}_iou_bands":  best_iou_bands,
            })

        _save_results_v2(results_per_crop, band_name_to_idx,
                         per_crop_json=STAGE2V3_PER_CROP_JSON,
                         exp_json=STAGE3_EXP_C_V2_JSON,
                         exp_bands=STAGE3_EXP_C_V2_BANDS)
        mlflow.log_artifact(str(STAGE2V3_PER_CROP_JSON))
        mlflow.log_artifact(str(STAGE3_EXP_C_V2_JSON))
        mlflow.log_artifact(str(STAGE3_EXP_C_V2_BANDS))
        log.info(f"Stage 2v2 parent run_id: {parent_run.info.run_id}")
        logging.getLogger().removeHandler(_fh)
        _fh.flush()
        _fh.close()
        mlflow.log_artifact(str(log_path))
        mlflow.end_run(status="FINISHED")

    except Exception as e:
        logging.getLogger().removeHandler(_fh)
        _fh.flush()
        _fh.close()
        mlflow.log_artifact(str(log_path))
        mlflow.end_run(status="FAILED")
        raise e

    return results_per_crop


# ── Stage 2v2-RF: Per-crop date + band selection via Random Forest ─────────────

def run_stage2v2_rf(s2_paths, cdl_path, date_candidates_per_crop, band_candidates_per_crop,
                    band_name_to_idx, all_dates, data_dir=None):
    """
    Per-crop binary RF feature selection — Phase A (dates) then Phase B (bands).

    Replaces the iterative CNN oracle with a single Random Forest fit per phase.
    RF feature importance is used to rank and threshold-select dates/bands.

    Phase A: train RF with features = [date × VEGE_BANDS] for all date candidates.
             Group importance by date (mean over its VEGE_BANDS channels).
             Keep dates with relative importance >= RF_IMPORTANCE_THRESH × max_date_importance.

    Phase B: given selected_dates, train RF with features = [selected_dates × band] for
             all band candidates.
             Group importance by band (mean over its date channels).
             Keep bands with relative importance >= RF_IMPORTANCE_THRESH × max_band_importance.

    MLflow structure:
        stage2v3_rf_{ts}                    — parent run
        └── stage2v3_rf_{crop_name}_{ts}   — child per crop

    Returns results_per_crop = {crop_id: {"dates": [...], "bands": [...], ...}}.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    run_ts   = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = PROCESSED_DIR / f"stage2v2_rf_run_{run_ts}.log"
    _fh      = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(_fh)

    _mlflow_setup()
    parent_run = mlflow.start_run(run_name=f"stage2v3_rf_{run_ts}")
    mlflow.log_params({
        "stage":               "2v2_rf_date_band_selection",
        "selector":            "rf",
        "rf_n_estimators":     RF_N_ESTIMATORS,
        "rf_max_pixels":       RF_MAX_PIXELS,
        "rf_importance_thresh": RF_IMPORTANCE_THRESH,
        "top_dates_per_crop":  TOP_DATES_PER_CROP,
        "top_bands_per_crop":  TOP_BANDS_PER_CROP,
        "max_bands_per_crop":  MAX_BANDS_PER_CROP,
        "n_crops":             len(KEEP_CLASSES),
    })

    # Pre-load all patches (all bands) into RAM — same as CNN version
    log.info("Pre-loading all patches into RAM (all bands)...")
    full_dataset = RasterPatchDataset(
        s2_paths=s2_paths, cdl_path=cdl_path,
        patch_size=S2_PATCH_SIZE, stride=S2_STRIDE,
        min_valid_frac=S2_MIN_VALID, band_indices=None,
        remap_lut=None, target_class_id=None,
    )
    preloaded_all = _preload_patches(full_dataset)
    imgs_all, masks_all = preloaded_all.tensors   # (N, C, H, W), (N, H, W)
    N, C, H, W = imgs_all.shape
    log.info(f"Pre-loaded: imgs={tuple(imgs_all.shape)}  masks={tuple(masks_all.shape)}")

    # Flatten patches → pixel matrix once (shared across all crops)
    # Shape: (N*H*W, C) and (N*H*W,)
    pixels_X = imgs_all.permute(0, 2, 3, 1).reshape(-1, C).numpy().astype(np.float32)
    pixels_y = masks_all.reshape(-1).numpy().astype(np.int32)
    log.info(f"Flattened pixel matrix: {pixels_X.shape}  labels: {pixels_y.shape}")

    results_per_crop: dict[int, dict] = {}
    n_crops = len(KEEP_CLASSES)

    log.info(f"\nStage 2v2-RF — per-crop binary RF feature selection (date + band phases)")
    log.info(f"  n_estimators={RF_N_ESTIMATORS}  max_pixels={RF_MAX_PIXELS}  "
             f"importance_thresh={RF_IMPORTANCE_THRESH}")

    try:
        for crop_idx, crop_id in enumerate(KEEP_CLASSES, 1):
            crop_name  = CDL_CLASS_NAMES[crop_id]
            crop_key   = str(crop_id)
            date_cands = date_candidates_per_crop[crop_key]
            band_cands = band_candidates_per_crop[crop_key]

            log.info(f"\n{'='*60}")
            log.info(f"[{crop_idx}/{n_crops}] Crop: {crop_name} (CDL id={crop_id}) "
                     f"— {len(date_cands)} date candidates, {len(band_cands)} band candidates")

            # Build binary pixel arrays for this crop
            crop_px_mask = (pixels_y == crop_id)
            rest_px_mask = np.isin(pixels_y, KEEP_CLASSES) & ~crop_px_mask
            n_crop = crop_px_mask.sum()
            n_rest = rest_px_mask.sum()
            log.info(f"  Crop pixels: {n_crop:,}  rest pixels: {n_rest:,}")

            if n_crop < 50:
                log.warning(f"  Too few crop pixels ({n_crop}) — skipping")
                results_per_crop[crop_id] = {
                    "dates": date_cands[:1] if date_cands else [],
                    "bands": band_cands[:1] if band_cands else [],
                    "k_dates": 1, "k_bands": 1,
                    "best_iou_after_dates": 0.0, "best_iou_after_bands": 0.0,
                    "fallback_dates": True, "fallback_bands": True,
                }
                continue

            # Sample pixels to stay within RF_MAX_PIXELS budget
            rng = np.random.default_rng(42)
            n_sample_each = min(RF_MAX_PIXELS // 2, n_crop, n_rest)
            crop_idx_arr  = rng.choice(np.where(crop_px_mask)[0], n_sample_each, replace=False)
            rest_idx_arr  = rng.choice(np.where(rest_px_mask)[0], n_sample_each, replace=False)
            sample_idx    = np.concatenate([crop_idx_arr, rest_idx_arr])
            X_sample      = pixels_X[sample_idx]   # (2*n_sample_each, C)
            y_binary      = np.concatenate([
                np.ones(n_sample_each, dtype=np.int32),
                np.zeros(n_sample_each, dtype=np.int32),
            ])
            log.info(f"  RF sample: {len(X_sample):,} pixels ({n_sample_each:,} crop + {n_sample_each:,} rest)")

            with mlflow.start_run(
                run_name=f"stage2v3_rf_{crop_name.replace('/', '-')}_{run_ts}",
                nested=True,
            ) as crop_run:
                mlflow.log_params({
                    "crop_id":           crop_id,
                    "crop_name":         crop_name,
                    "n_date_candidates": len(date_cands),
                    "n_band_candidates": len(band_cands),
                    "n_crop_pixels":     int(n_crop),
                    "n_sample_pixels":   len(X_sample),
                })

                # ── Phase A: Date selection via RF importance ─────────────────
                log.info(f"\n  === Phase A: Date selection for {crop_name} ===")
                selected_dates = []
                fallback_dates = False

                if date_cands:
                    # Build feature matrix: cols = [VEGE_BANDS × each date candidate]
                    phA_cols, phA_feat_names = [], []
                    for date in date_cands:
                        for band in VEGE_BANDS:
                            key = f"{band}_{date}"
                            if key in band_name_to_idx:
                                phA_cols.append(band_name_to_idx[key])
                                phA_feat_names.append(key)

                    if phA_cols:
                        X_phA = X_sample[:, phA_cols]
                        rf_A  = RandomForestClassifier(
                            n_estimators=RF_N_ESTIMATORS, n_jobs=-1,
                            random_state=42, class_weight="balanced",
                        )
                        rf_A.fit(X_phA, y_binary)
                        imp_A = rf_A.feature_importances_   # per feature

                        # Group importance by date (mean over its VEGE_BANDS features)
                        date_importance = {}
                        for feat_i, feat_name in enumerate(phA_feat_names):
                            date = feat_name.split("_", 1)[1]   # "B4_20220730" → "20220730"
                            date_importance.setdefault(date, []).append(imp_A[feat_i])
                        date_mean_imp = {d: float(np.mean(v)) for d, v in date_importance.items()}

                        max_imp = max(date_mean_imp.values()) if date_mean_imp else 1e-9
                        thresh  = RF_IMPORTANCE_THRESH * max_imp
                        sorted_dates_imp = sorted(date_mean_imp.items(), key=lambda x: x[1], reverse=True)
                        selected_dates = [d for d, imp in sorted_dates_imp
                                          if imp >= thresh][:MAX_DATES_PER_CROP]

                        log.info(f"  Date importances: " +
                                 "  ".join(f"{d}={v:.4f}" for d, v in sorted_dates_imp[:5]))
                        mlflow.log_metrics({f"phaseA_imp_{d}": v for d, v in date_mean_imp.items()})
                    else:
                        log.warning(f"  Phase A: no valid features for {crop_name}")

                if not selected_dates and date_cands:
                    selected_dates = [date_cands[0]]
                    fallback_dates = True
                    log.warning(f"  K_dates=0 for {crop_name} — fallback to top-1: {date_cands[0]}")
                    mlflow.set_tag("phaseA_fallback_date", date_cands[0])

                log.info(f"\n  Phase A done: K_dates={len(selected_dates)}  dates={selected_dates}")
                mlflow.log_metric("phaseA_k_dates", len(selected_dates))
                mlflow.set_tag("phaseA_selected_dates", str(selected_dates))

                # ── Phase B: Band selection via RF importance ─────────────────
                log.info(f"\n  === Phase B: Band selection for {crop_name} "
                         f"(fixed dates={selected_dates}) ===")
                selected_bands = []
                fallback_bands = False

                if band_cands and selected_dates:
                    # Build feature matrix: cols = [selected_dates × each band candidate]
                    phB_cols, phB_feat_names = [], []
                    for band in band_cands:
                        for date in selected_dates:
                            key = f"{band}_{date}"
                            if key in band_name_to_idx:
                                phB_cols.append(band_name_to_idx[key])
                                phB_feat_names.append(key)

                    if phB_cols:
                        X_phB = X_sample[:, phB_cols]
                        rf_B  = RandomForestClassifier(
                            n_estimators=RF_N_ESTIMATORS, n_jobs=-1,
                            random_state=42, class_weight="balanced",
                        )
                        rf_B.fit(X_phB, y_binary)
                        imp_B = rf_B.feature_importances_

                        # Group importance by band (mean over its date channels)
                        band_importance = {}
                        for feat_i, feat_name in enumerate(phB_feat_names):
                            band = feat_name.split("_")[0]   # "B4_20220730" → "B4"
                            band_importance.setdefault(band, []).append(imp_B[feat_i])
                        band_mean_imp = {b: float(np.mean(v)) for b, v in band_importance.items()}

                        max_imp = max(band_mean_imp.values()) if band_mean_imp else 1e-9
                        thresh  = RF_IMPORTANCE_THRESH * max_imp
                        sorted_bands_imp = sorted(band_mean_imp.items(), key=lambda x: x[1], reverse=True)
                        selected_bands = [b for b, imp in sorted_bands_imp
                                          if imp >= thresh][:MAX_BANDS_PER_CROP]

                        log.info(f"  Band importances: " +
                                 "  ".join(f"{b}={v:.4f}" for b, v in sorted_bands_imp))
                        mlflow.log_metrics({f"phaseB_imp_{b}": v for b, v in band_mean_imp.items()})
                    else:
                        log.warning(f"  Phase B: no valid features for {crop_name}")

                if not selected_bands and band_cands:
                    selected_bands = [band_cands[0]]
                    fallback_bands = True
                    log.warning(f"  K_bands=0 for {crop_name} — fallback to top-1: {band_cands[0]}")
                    mlflow.set_tag("phaseB_fallback_band", band_cands[0])

                log.info(f"\n  Phase B done: K_bands={len(selected_bands)}  bands={selected_bands}")
                mlflow.log_metric("phaseB_k_bands", len(selected_bands))
                mlflow.set_tag("phaseB_selected_bands", str(selected_bands))

                result = {
                    "dates":                selected_dates,
                    "bands":                selected_bands,
                    "k_dates":              len(selected_dates),
                    "k_bands":              len(selected_bands),
                    "best_iou_after_dates": 0.0,   # RF has no IoU — placeholder
                    "best_iou_after_bands": 0.0,
                    "fallback_dates":       fallback_dates,
                    "fallback_bands":       fallback_bands,
                    "mlflow_run_id":        crop_run.info.run_id,
                }
                results_per_crop[crop_id] = result

            log.info(f"\n  -> [{crop_idx}/{n_crops}] {crop_name}: "
                     f"K_dates={len(selected_dates)}  K_bands={len(selected_bands)}")
            mlflow.log_metrics({
                f"crop_{crop_id}_k_dates": len(selected_dates),
                f"crop_{crop_id}_k_bands": len(selected_bands),
            })

        _save_results_v2(results_per_crop, band_name_to_idx,
                         per_crop_json=STAGE2V3_RF_PER_CROP_JSON,
                         exp_json=STAGE3_EXP_C_V2_RF_JSON,
                         exp_bands=STAGE3_EXP_C_V2_RF_BANDS)
        mlflow.log_artifact(str(STAGE2V3_RF_PER_CROP_JSON))
        mlflow.log_artifact(str(STAGE3_EXP_C_V2_RF_JSON))
        mlflow.log_artifact(str(STAGE3_EXP_C_V2_RF_BANDS))
        log.info(f"Stage 2v2-RF parent run_id: {parent_run.info.run_id}")
        logging.getLogger().removeHandler(_fh)
        _fh.flush(); _fh.close()
        mlflow.log_artifact(str(log_path))
        mlflow.end_run(status="FINISHED")

    except Exception as e:
        logging.getLogger().removeHandler(_fh)
        _fh.flush(); _fh.close()
        mlflow.log_artifact(str(log_path))
        mlflow.end_run(status="FAILED")
        raise e

    return results_per_crop


# ── Save results ──────────────────────────────────────────────────────────────

def _save_results_v2(results_per_crop: dict, band_name_to_idx: dict,
                     per_crop_json=None, exp_json=None, exp_bands=None) -> None:
    """
    Save per-crop results and union band list for Stage 3.

    Defaults to CNN output paths (STAGE2V3_PER_CROP_JSON etc.).
    Pass explicit path overrides for the RF variant.
    """
    _per_crop_json = per_crop_json or STAGE2V3_PER_CROP_JSON
    _exp_json      = exp_json      or STAGE3_EXP_C_V2_JSON
    _exp_bands     = exp_bands     or STAGE3_EXP_C_V2_BANDS
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Build union_dates: sorted set of all selected dates across crops
    union_dates_set = set()
    for crop_id in KEEP_CLASSES:
        r = results_per_crop.get(crop_id, {})
        union_dates_set.update(r.get("dates", []))
    union_dates = sorted(union_dates_set)

    # Build union_bands: maintain order by first appearance across KEEP_CLASSES order
    seen_bands, union_bands = set(), []
    for crop_id in KEEP_CLASSES:
        r = results_per_crop.get(crop_id, {})
        for band in r.get("bands", []):
            if band not in seen_bands:
                seen_bands.add(band)
                union_bands.append(band)

    total_channels = len(union_dates) * len(union_bands)

    # Per-crop summary
    per_crop_summary = {}
    for crop_id in KEEP_CLASSES:
        r = results_per_crop.get(crop_id, {})
        per_crop_summary[str(crop_id)] = {
            "crop_name":            CDL_CLASS_NAMES[crop_id],
            "dates":                r.get("dates", []),
            "bands":                r.get("bands", []),
            "k_dates":              r.get("k_dates", 0),
            "k_bands":              r.get("k_bands", 0),
            "best_iou_after_dates": r.get("best_iou_after_dates", 0.0),
            "best_iou_after_bands": r.get("best_iou_after_bands", 0.0),
            "fallback_dates":       r.get("fallback_dates", False),
            "fallback_bands":       r.get("fallback_bands", False),
            "mlflow_run_id":        r.get("mlflow_run_id", ""),
        }

    # Save per-crop JSON
    with open(_per_crop_json, "w") as f:
        json.dump(per_crop_summary, f, indent=2)
    log.info(f"Saved: {_per_crop_json}")

    # Save exp summary JSON
    exp_payload = {
        "union_dates":     union_dates,
        "union_bands":     union_bands,
        "total_channels":  total_channels,
        "per_crop":        per_crop_summary,
    }
    with open(_exp_json, "w") as f:
        json.dump(exp_payload, f, indent=2)
    log.info(f"Saved: {_exp_json}")

    # Save flat band list — one entry per (date, band) in union_dates × union_bands
    band_lines = []
    for date in union_dates:
        for band in union_bands:
            key = f"{band}_{date}"
            if key in band_name_to_idx:
                band_lines.append(key)
    with open(_exp_bands, "w") as f:
        f.write("\n".join(band_lines))
    log.info(f"Saved: {_exp_bands}  ({len(band_lines)} channel entries)")

    # Summary
    log.info("\n=== Per-Crop Stage 2v2 Results ===")
    for crop_id in KEEP_CLASSES:
        s = per_crop_summary[str(crop_id)]
        log.info(f"  {s['crop_name']:20s}  K_dates={s['k_dates']}  K_bands={s['k_bands']}  "
                 f"IoU_dates={s['best_iou_after_dates']:.4f}  IoU_bands={s['best_iou_after_bands']:.4f}")
    log.info(f"\nUnion: {len(union_dates)} dates × {len(union_bands)} bands = "
             f"{total_channels} total channels (before filtering by availability)")
    log.info(f"Band list entries: {len(band_lines)}")
    log.info(f"union_dates: {union_dates}")
    log.info(f"union_bands: {union_bands}")

    # ── Selection summary table (PNG + CSV) ──────────────────────────────────
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    table_path = FIGURES_DIR / "stage2v2_selection_table.png"
    _plot_selection_table(results_per_crop, table_path)

    # Log to active MLflow run (called from within parent run context)
    try:
        mlflow.log_metrics({
            "n_union_dates":   len(union_dates),
            "n_union_bands":   len(union_bands),
            "total_channels":  total_channels,
        })
        mlflow.log_artifact(str(table_path))
        mlflow.log_artifact(str(table_path.with_suffix(".csv")))
    except Exception:
        pass


# ── Exp D: Stage 1 top-K output (no Stage 2 CNN) ─────────────────────────────

def _save_exp_d_bands(date_candidates_per_crop: dict, band_candidates_per_crop: dict,
                      band_name_to_idx: dict, data_dir=None) -> None:
    """
    Write union of Stage 1 top-K dates × top-K bands as input for Exp D.

    Exp D uses SI_global-ranked features directly from Stage 1, without any
    CNN forward selection (Stage 2).  This is an ablation vs Exp C (Stage 2).

    Writes:
      STAGE3_EXP_D_JSON    — {union_dates, union_bands, total_channels, per_crop}
      STAGE3_EXP_D_BANDS   — flat txt, one "B4_20220730" per (union_date, union_band) pair
    """
    _d_json  = STAGE3_EXP_D_JSON
    _d_bands = STAGE3_EXP_D_BANDS
    if data_dir:
        _d_json  = pathlib.Path(data_dir) / "stage3_exp_d.json"
        _d_bands = pathlib.Path(data_dir) / "stage3_exp_d_bands.txt"

    # Build union_dates (maintain first-appearance order across KEEP_CLASSES order)
    seen_dates, union_dates = set(), []
    for crop_id in KEEP_CLASSES:
        for date in date_candidates_per_crop.get(str(crop_id), []):
            if date not in seen_dates:
                seen_dates.add(date)
                union_dates.append(date)

    # Build union_bands (maintain first-appearance order across KEEP_CLASSES order)
    seen_bands, union_bands = set(), []
    for crop_id in KEEP_CLASSES:
        for band in band_candidates_per_crop.get(str(crop_id), []):
            if band not in seen_bands:
                seen_bands.add(band)
                union_bands.append(band)

    total_channels = len(union_dates) * len(union_bands)

    per_crop = {}
    for crop_id in KEEP_CLASSES:
        crop_key = str(crop_id)
        per_crop[crop_key] = {
            "crop_name":  CDL_CLASS_NAMES[crop_id],
            "top_dates":  date_candidates_per_crop.get(crop_key, []),
            "top_bands":  band_candidates_per_crop.get(crop_key, []),
        }

    payload = {
        "union_dates":    union_dates,
        "union_bands":    union_bands,
        "total_channels": total_channels,
        "per_crop":       per_crop,
    }
    os.makedirs(os.path.dirname(_d_json) if os.path.dirname(str(_d_json)) else ".", exist_ok=True)
    with open(_d_json, "w") as f:
        json.dump(payload, f, indent=2)
    log.info(f"Exp D JSON saved: {_d_json}")

    # Flat band list: one "B4_20220730" entry per (date, band) pair present in band_name_to_idx
    band_lines = []
    for date in union_dates:
        for band in union_bands:
            key = f"{band}_{date}"
            if key in band_name_to_idx:
                band_lines.append(key)
    with open(_d_bands, "w") as f:
        f.write("\n".join(band_lines))
    log.info(
        f"Exp D bands saved: {_d_bands}  "
        f"({len(union_dates)} dates × {len(union_bands)} bands = "
        f"{total_channels} theoretical, {len(band_lines)} available channels)"
    )


# ── Stage project v2 ──────────────────────────────────────────────────────────

def run_project_v2() -> None:
    """
    Project Stage 2v2 band selections (2022 dates) to nearest equivalent dates
    in every training and test year.

    Reads STAGE3_EXP_C_V2_BANDS  — lines like 'B4_20220730'
    Writes stage3_exp_c_v2_bands_projected.json:
        {
          "2022": ["B4_20220730", "B8_20220730", ...],
          "2023": ["B4_20230801", "B8_20230801", ...],
          "2024": ["B4_20240730", "B8_20240730", ...]
        }

    This file is consumed by train_segmentation.py to compute per-year band
    indices so that each year's RasterPatchDataset selects the seasonally
    correct channels, even when acquisition dates differ across years.
    """
    from datetime import date as _date

    if not STAGE3_EXP_C_V2_BANDS.exists():
        raise FileNotFoundError(
            f"Stage 2v2 output not found: {STAGE3_EXP_C_V2_BANDS}\n"
            "Run Stage 2v2 first:  python feature_analysis_v2.py --stage 2"
        )

    with open(STAGE3_EXP_C_V2_BANDS) as f:
        selected_bands = [l.strip() for l in f if l.strip()]

    if not selected_bands:
        raise ValueError(f"{STAGE3_EXP_C_V2_BANDS} is empty — re-run Stage 2v2.")

    # Parse band name + MMDD from each entry  e.g. "B4_20220730" → ("B4", "0730")
    band_mmdd = []
    for entry in selected_bands:
        m = re.match(r"(.+)_(\d{4})(\d{2})(\d{2})$", entry)
        if m:
            band_mmdd.append((m.group(1), m.group(3) + m.group(4)))
        else:
            log.warning(f"run_project_v2: cannot parse band entry '{entry}' — skipped")

    log.info(f"Loaded {len(band_mmdd)} selected bands from {STAGE3_EXP_C_V2_BANDS.name}")

    all_years = list(dict.fromkeys(list(TRAIN_YEARS) + [TEST_YEAR]))
    projected = {}

    for yr in all_years:
        yr_files = sorted(glob(str(S2_PROCESSED_DIR / yr / "*_processed.tif")))
        if not yr_files:
            log.warning(f"  {yr}: no S2 files found — skipping")
            continue

        # Build list of available YYYYMMDD dates for this year
        yr_dates = []
        for p in yr_files:
            m = re.search(r"_(\d{4})_(\d{2})_(\d{2})_processed", pathlib.Path(p).name)
            if m:
                yr_dates.append(f"{m.group(1)}{m.group(2)}{m.group(3)}")
        yr_dates = sorted(set(yr_dates))

        # For each selected band's MMDD, find nearest acquisition date in this year
        yr_bands = []
        far_matches = []
        for band, mmdd in band_mmdd:
            month, day = int(mmdd[:2]), int(mmdd[2:])
            try:
                target = _date(int(yr), month, day)
            except ValueError:
                # e.g. Feb 29 in non-leap year — shift to Feb 28
                target = _date(int(yr), month, min(day, 28))
            target_doy = target.timetuple().tm_yday

            best_date, best_dist = None, 999
            for yyyymmdd in yr_dates:
                d = _date(int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8]))
                dist = abs(d.timetuple().tm_yday - target_doy)
                if dist < best_dist:
                    best_dist, best_date = dist, yyyymmdd

            yr_bands.append(f"{band}_{best_date}")
            if best_dist > 15:
                far_matches.append((band, mmdd, best_date, best_dist))

        projected[yr] = yr_bands

        if far_matches:
            log.warning(f"  {yr}: {len(far_matches)} band(s) matched with gap > 15 days:")
            for band, mmdd, matched, dist in far_matches:
                log.warning(f"    {band}_{mmdd} → {matched} (gap={dist}d)")
        log.info(f"  {yr}: {len(yr_bands)} bands projected  ({len(yr_dates)} available dates)")

    if not projected:
        raise RuntimeError("No years projected — check S2_PROCESSED_DIR paths.")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(STAGE3_EXP_C_V2_BANDS_PROJECTED, "w") as f:
        json.dump(projected, f, indent=2)
    log.info(f"Saved: {STAGE3_EXP_C_V2_BANDS_PROJECTED}")

    # Log as MLflow artifact
    _mlflow_setup()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Identify the Stage 2v2 run that produced stage3_exp_c_v2_bands.txt
    source_stage2v2_run_id   = None
    source_stage2v2_run_name = None
    try:
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT_FEATURE)
        if exp:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string="attributes.run_name LIKE 'stage2v3_%'",
                order_by=["attributes.start_time DESC"],
                max_results=1,
            )
            if runs:
                source_stage2v2_run_id   = runs[0].info.run_id
                source_stage2v2_run_name = runs[0].info.run_name
                log.info(
                    f"Source Stage 2v2 run: {source_stage2v2_run_name} "
                    f"(run_id={source_stage2v2_run_id})"
                )
    except Exception as e:
        log.warning(f"Could not look up source Stage 2v2 run: {e}")

    with mlflow.start_run(run_name=f"stage2v3_project_{ts}"):
        mlflow.set_tag("stage", "project_v2")
        for yr, bands in projected.items():
            mlflow.log_param(f"n_bands_{yr}", len(bands))
            mlflow.set_tag(f"bands_{yr}", str(bands))
        if source_stage2v2_run_id:
            mlflow.set_tag("source_stage2v2_run_id",   source_stage2v2_run_id)
            mlflow.set_tag("source_stage2v2_run_name", source_stage2v2_run_name)
        mlflow.log_artifact(str(STAGE3_EXP_C_V2_BANDS_PROJECTED))
        mlflow.log_artifact(str(STAGE3_EXP_C_V2_BANDS))
    log.info("MLflow artifact logged.")


# ── MLflow helpers ─────────────────────────────────────────────────────────────

def _mlflow_setup() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_FEATURE)
    if mlflow.active_run():
        log.warning(f"Closing stale MLflow run: {mlflow.active_run().info.run_id}")
        mlflow.end_run(status="FAILED")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(force: bool = False, data_dir: str = None, stage: str = "all",
         selector: str = "cnn") -> None:
    # Override data paths if requested
    if data_dir:
        global S2_PROCESSED_DIR, CDL_BY_YEAR, PROCESSED_DIR, FIGURES_DIR, \
               STAGE1V3_CANDIDATES_JSON, STAGE2V3_PER_CROP_JSON, \
               STAGE3_EXP_C_V2_JSON, STAGE3_EXP_C_V2_BANDS, STAGE3_EXP_C_V2_BANDS_PROJECTED, \
               STAGE3_EXP_D_JSON, STAGE3_EXP_D_BANDS, \
               STAGE2V3_RF_PER_CROP_JSON, STAGE3_EXP_C_V2_RF_JSON, STAGE3_EXP_C_V2_RF_BANDS
        processed                        = pathlib.Path(data_dir)
        PROCESSED_DIR                    = processed
        S2_PROCESSED_DIR                 = processed / "s2"
        CDL_BY_YEAR                      = {
            yr: processed / "cdl" / f"cdl_{yr}_study_area_filtered.tif"
            for yr in ["2022", "2023", "2024"]
        }
        STAGE1V3_CANDIDATES_JSON         = processed / "s2" / "2022" / "stage1v3_candidates.json"
        STAGE2V3_PER_CROP_JSON           = processed / "stage2v3_per_crop_results.json"
        STAGE3_EXP_C_V2_JSON             = processed / "stage3_exp_c_v2.json"
        STAGE3_EXP_C_V2_BANDS            = processed / "stage3_exp_c_v2_bands.txt"
        STAGE3_EXP_C_V2_BANDS_PROJECTED  = processed / "stage3_exp_c_v2_bands_projected.json"
        STAGE3_EXP_D_JSON                = processed / "stage3_exp_d.json"
        STAGE3_EXP_D_BANDS               = processed / "stage3_exp_d_bands.txt"
        STAGE2V3_RF_PER_CROP_JSON        = processed / "stage2v3_rf_per_crop_results.json"
        STAGE3_EXP_C_V2_RF_JSON         = processed / "stage3_exp_c_v2_rf.json"
        STAGE3_EXP_C_V2_RF_BANDS        = processed / "stage3_exp_c_v2_rf_bands.txt"
        log.info(f"Data dir overridden to {processed}")

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── Stage 1v3 ─────────────────────────────────────────────────────────────
    if stage in ("1", "all"):
        if not force and os.path.exists(STAGE1V3_CANDIDATES_JSON):
            log.info(f"Stage 1v3 output already exists: {STAGE1V3_CANDIDATES_JSON}")
            log.info("Use --force to re-run.")
        else:
            log.info(f"Device: {_device_label()}")
            s2_year = TRAIN_YEARS[0]
            s2_files = sorted(glob(str(S2_PROCESSED_DIR / s2_year / "*_processed.tif")))
            assert s2_files, f"No S2 files for year {s2_year} in {S2_PROCESSED_DIR}"
            cdl_path = str(CDL_BY_YEAR[s2_year])
            assert os.path.exists(cdl_path), f"CDL not found: {cdl_path}"

            run_stage1v3(s2_files, cdl_path, data_dir=data_dir)
            log.info("Stage 1v3 complete.")

        if stage == "1":
            return

    # ── Stage 2v2 ─────────────────────────────────────────────────────────────
    if stage in ("2", "all"):
        _output_check = STAGE3_EXP_C_V2_RF_BANDS if selector == "rf" else STAGE3_EXP_C_V2_BANDS
        if not force and os.path.exists(_output_check):
            log.info(f"Stage 2v2-{selector.upper()} output already exists: {_output_check}")
            log.info("Use --force to re-run.")
            if stage == "2":
                return
        else:
            # Load Stage 1v3 candidates from handoff file
            if not os.path.exists(STAGE1V3_CANDIDATES_JSON):
                raise FileNotFoundError(
                    f"Stage 1v3 candidates not found: {STAGE1V3_CANDIDATES_JSON}\n"
                    "Run Stage 1v3 first:  python feature_analysis_v2.py --stage 1"
                )
            with open(STAGE1V3_CANDIDATES_JSON) as f:
                payload = json.load(f)
            date_candidates_per_crop = payload["date_candidates_per_crop"]
            band_candidates_per_crop = payload["band_candidates_per_crop"]
            all_dates                = payload["all_dates"]
            log.info(f"Loaded Stage 1v3 candidates from {STAGE1V3_CANDIDATES_JSON}")

            # Build band_name_to_idx from 2022 S2 files
            s2_year  = TRAIN_YEARS[0]
            s2_files = sorted(glob(str(S2_PROCESSED_DIR / s2_year / "*_processed.tif")))
            assert s2_files, f"No S2 files for year {s2_year} in {S2_PROCESSED_DIR}"
            cdl_path = str(CDL_BY_YEAR[s2_year])
            assert os.path.exists(cdl_path), f"CDL not found: {cdl_path}"

            all_bandnames = []
            for s2_path in s2_files:
                fname    = os.path.basename(s2_path)
                m        = re.search(r"_(\d{4})_(\d{2})_(\d{2})_processed", fname)
                date_str = f"{m.group(1)}{m.group(2)}{m.group(3)}" if m else fname[:8]
                all_bandnames.extend([f"{b}_{date_str}" for b in S2_BAND_NAMES])
            band_name_to_idx = {name: i for i, name in enumerate(all_bandnames)}

            log.info(f"Device: {_device_label()}  selector={selector}")
            _stage2_fn = run_stage2v2_rf if selector == "rf" else run_stage2v2
            _stage2_fn(
                s2_paths=s2_files, cdl_path=cdl_path,
                date_candidates_per_crop=date_candidates_per_crop,
                band_candidates_per_crop=band_candidates_per_crop,
                band_name_to_idx=band_name_to_idx,
                all_dates=all_dates,
                data_dir=data_dir,
            )
            log.info(f"Stage 2v2-{selector.upper()} complete.")

        if stage == "2":
            return

    # ── Stage project ─────────────────────────────────────────────────────────
    if stage == "project":
        if not force and STAGE3_EXP_C_V2_BANDS_PROJECTED.exists():
            log.info(f"Projected bands already exist: {STAGE3_EXP_C_V2_BANDS_PROJECTED}")
            log.info("Use --force to re-run.")
            return
        run_project_v2()
        log.info("Band projection complete.")
        return

    log.info("Feature analysis v2 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature analysis v2: Stage 1v3 + Stage 2v2")
    parser.add_argument(
        "--stage",
        choices=["1", "2", "all", "project"],
        default="all",
        help=(
            "Which stage to run: "
            "1 (CPU, Stage 1v3 date+band ranking), "
            "2 (GPU, Stage 2v2 date+band forward selection), "
            "all (1+2), "
            "project (map 2022 selections to 2023/2024 nearest dates)"
        ),
    )
    parser.add_argument(
        "--selector", choices=["cnn", "rf"], default="cnn",
        help="Stage 2 feature selector: 'cnn' (iterative CNN oracle, default) or "
             "'rf' (Random Forest importance, faster — outputs *_rf_* files)",
    )
    parser.add_argument("--force",    action="store_true", help="Re-run even if outputs exist")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override processed data directory")
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                LOGS_DIR / f"feature_analysis_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            ),
        ],
    )

    main(force=args.force, data_dir=args.data_dir, stage=args.stage, selector=args.selector)
