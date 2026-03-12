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
    S2_DATE_DELTA, S2_DATE_NO_IMPROVE, S2_MAX_DATES,
    S2_BAND_DELTA, S2_BAND_NO_IMPROVE, S2_MAX_BANDS_V2,
    STAGE1V3_CANDIDATES_JSON, STAGE2V3_PER_CROP_JSON,
    STAGE3_EXP_C_V2_JSON, STAGE3_EXP_C_V2_BANDS,
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
    del stacked, img_2d

    # ── Compute SI_global ─────────────────────────────────────────────────────
    log.info("Computing GSI (SI_global) per band × class...")
    gsi_df          = bs.calculate_gsi(df, "class_label")
    gsi_mean_global = gsi_df.mean(axis=1).sort_values(ascending=False)
    log.info(f"gsi_df shape: {gsi_df.shape}  (bands × classes)")

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

        # Date ranking: for each date, compute mean SI across all S2_BAND_NAMES at that date
        date_si: dict[str, float] = {}
        for date in all_dates:
            band_keys = [f"{b}_{date}" for b in S2_BAND_NAMES if f"{b}_{date}" in si_crop.index]
            if band_keys:
                date_si[date] = float(si_crop[band_keys].mean())
            else:
                date_si[date] = 0.0
        sorted_dates = sorted(date_si.items(), key=lambda x: x[1], reverse=True)
        date_candidates_per_crop[crop_key] = [d for d, _ in sorted_dates[:TOP_DATES_PER_CROP]]

        # Band ranking: for each band in S2_BAND_NAMES, compute mean SI across all dates
        band_si: dict[str, float] = {}
        for band in S2_BAND_NAMES:
            band_keys = [f"{band}_{d}" for d in all_dates if f"{band}_{d}" in si_crop.index]
            if band_keys:
                band_si[band] = float(si_crop[band_keys].mean())
            else:
                band_si[band] = 0.0
        sorted_bands = sorted(band_si.items(), key=lambda x: x[1], reverse=True)
        band_candidates_per_crop[crop_key] = [b for b, _ in sorted_bands[:TOP_BANDS_PER_CROP]]

        log.info(f"  {CDL_CLASS_NAMES[crop_id]:20s}: "
                 f"top dates={date_candidates_per_crop[crop_key][:3]}... "
                 f"top bands={band_candidates_per_crop[crop_key][:3]}...")

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

    # ── MLflow ────────────────────────────────────────────────────────────────
    _mlflow_setup()
    ts         = datetime.now().strftime("%Y%m%d-%H%M%S")
    stage1_run = mlflow.start_run(run_name=f"stage1v3_{ts}")

    mlflow.log_params({
        "stage":              "1v3_date_band_ranking",
        "version":            "v3",
        "n_images":           len(s2_paths),
        "n_dates":            len(all_dates),
        "total_channels":     n_channels,
        "sample_fraction":    SAMPLE_FRACTION,
        "n_sampled":          len(df),
        "top_dates_per_crop": TOP_DATES_PER_CROP,
        "top_bands_per_crop": TOP_BANDS_PER_CROP,
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

        _save_results_v2(results_per_crop, band_name_to_idx)
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


# ── Save results ──────────────────────────────────────────────────────────────

def _save_results_v2(results_per_crop: dict, band_name_to_idx: dict) -> None:
    """
    Save per-crop results and union band list for Stage 3 Exp C v2.

    Writes:
      STAGE2V3_PER_CROP_JSON   — per-crop detail (dates, bands, IoU, etc.)
      STAGE3_EXP_C_V2_JSON     — union_dates, union_bands, total_channels, per-crop summary
      STAGE3_EXP_C_V2_BANDS    — flat txt, one "B4_20220730" entry per (union_date, union_band) pair
    """
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

    # Save STAGE2V3_PER_CROP_JSON
    with open(STAGE2V3_PER_CROP_JSON, "w") as f:
        json.dump(per_crop_summary, f, indent=2)
    log.info(f"Saved: {STAGE2V3_PER_CROP_JSON}")

    # Save STAGE3_EXP_C_V2_JSON
    exp_c_v2_payload = {
        "union_dates":     union_dates,
        "union_bands":     union_bands,
        "total_channels":  total_channels,
        "per_crop":        per_crop_summary,
    }
    with open(STAGE3_EXP_C_V2_JSON, "w") as f:
        json.dump(exp_c_v2_payload, f, indent=2)
    log.info(f"Saved: {STAGE3_EXP_C_V2_JSON}")

    # Save STAGE3_EXP_C_V2_BANDS — one entry per (date, band) in union_dates × union_bands
    band_lines = []
    for date in union_dates:
        for band in union_bands:
            key = f"{band}_{date}"
            if key in band_name_to_idx:
                band_lines.append(key)
    with open(STAGE3_EXP_C_V2_BANDS, "w") as f:
        f.write("\n".join(band_lines))
    log.info(f"Saved: {STAGE3_EXP_C_V2_BANDS}  ({len(band_lines)} channel entries)")

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

    # Log to active MLflow run (called from within parent run context)
    try:
        mlflow.log_metrics({
            "n_union_dates":   len(union_dates),
            "n_union_bands":   len(union_bands),
            "total_channels":  total_channels,
        })
    except Exception:
        pass


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

def main(force: bool = False, data_dir: str = None, stage: str = "all") -> None:
    # Override data paths if requested
    if data_dir:
        global S2_PROCESSED_DIR, CDL_BY_YEAR, PROCESSED_DIR, FIGURES_DIR, \
               STAGE1V3_CANDIDATES_JSON, STAGE2V3_PER_CROP_JSON, \
               STAGE3_EXP_C_V2_JSON, STAGE3_EXP_C_V2_BANDS, STAGE3_EXP_C_V2_BANDS_PROJECTED
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
        if not force and os.path.exists(STAGE3_EXP_C_V2_BANDS):
            log.info(f"Stage 2v2 output already exists: {STAGE3_EXP_C_V2_BANDS}")
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

            log.info(f"Device: {_device_label()}")
            run_stage2v2(
                s2_paths=s2_files, cdl_path=cdl_path,
                date_candidates_per_crop=date_candidates_per_crop,
                band_candidates_per_crop=band_candidates_per_crop,
                band_name_to_idx=band_name_to_idx,
                all_dates=all_dates,
                data_dir=data_dir,
            )
            log.info("Stage 2v2 complete.")

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

    main(force=args.force, data_dir=args.data_dir, stage=args.stage)
