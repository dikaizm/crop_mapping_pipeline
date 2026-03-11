"""
Stage 1 + 2 — ASTFS-Inspired Per-Crop Band Selection (v2)

Stage 1: Per-crop SIglobal ranking (Yin et al. 2020, RS 12(1):162)
Stage 2: Per-crop binary CNN forward selection (crop_i vs. rest, metric = IoU class 1)

Outputs:
    data/processed/stage1v2_candidates.json       — Stage 1 per-crop candidate lists (handoff file)
    data/processed/stage2v2_per_crop_results.csv  — per-crop results table
    data/processed/stage3_exp_c_bands.txt         — union of selections (Stage 3 Exp C input)

Usage:
    python feature_analysis.py                          # run both stages
    python feature_analysis.py --stage 1                # Stage 1 only (run locally, CPU)
    python feature_analysis.py --stage 2                # Stage 2 only (run on GPU server)
    python feature_analysis.py --force                  # re-run even if outputs exist
    python feature_analysis.py --data-dir /data/processed
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
    S2_PROCESSED_DIR, CDL_BY_YEAR, PROCESSED_DIR, FIGURES_DIR,
    S2_BAND_NAMES, S2_NODATA, KEEP_CLASSES, CLASS_REMAP, NUM_CLASSES, CDL_CLASS_NAMES,
    REMAP_LUT, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_FEATURE,
    SAMPLE_FRACTION, TOP_K_PER_CROP,
    S2_ENCODER, S2_PATCH_SIZE, S2_STRIDE, S2_MIN_VALID,
    S2_EPOCHS, S2_PATIENCE, S2_DELTA, S2_NO_IMPROVE, S2_MAX_BANDS, S2_BATCH_SIZE,
    STAGE2_RESULTS_CSV, STAGE3_EXP_C_BANDS, STAGE3_EXP_C_BANDS_PROJECTED,
    TRAIN_YEARS, TEST_YEAR,
)

# Handoff file written by Stage 1, read by Stage 2 when run separately
STAGE1_CANDIDATES_JSON = PROCESSED_DIR / "s2" / "2022" / "stage1v2_candidates.json"

log = logging.getLogger(__name__)


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


# ── Stage 1: Per-crop SIglobal ─────────────────────────────────────────────────

def run_stage1(df: pd.DataFrame, all_bandnames: list, n_channels: int, s2_files: list):
    """
    Compute per-crop SIglobal, build candidate lists, log to MLflow.
    Returns (candidates_per_crop, run_timestamp).
    """
    log.info("Stage 1: computing per-crop GSI (SIglobal)...")
    gsi_df          = bs.calculate_gsi(df, "class_label")
    gsi_mean_global = gsi_df.mean(axis=1).sort_values(ascending=False)
    log.info(f"gsi_df shape: {gsi_df.shape}  (bands × classes)")

    candidates_per_crop: dict[int, list] = {}
    for crop_id in KEEP_CLASSES:
        if crop_id in gsi_df.columns:
            si_crop = gsi_df[crop_id].sort_values(ascending=False)
            candidates_per_crop[crop_id] = si_crop.head(TOP_K_PER_CROP).index.tolist()
        else:
            log.warning(
                f"Crop {crop_id} ({CDL_CLASS_NAMES[crop_id]}) not in sample — "
                "falling back to global mean ranking"
            )
            candidates_per_crop[crop_id] = gsi_mean_global.head(TOP_K_PER_CROP).index.tolist()
        log.info(f"  {CDL_CLASS_NAMES[crop_id]:20s}: {len(candidates_per_crop[crop_id])} candidates")

    # ── MLflow ────────────────────────────────────────────────────────────────
    _mlflow_setup()
    run_ts     = datetime.now().strftime("%Y%m%d-%H%M%S")
    stage1_run = mlflow.start_run(run_name=f"stage1v2_gsi_percrop_{run_ts}")

    mlflow.log_params({
        "stage":           "1_per_crop_siglobal",
        "version":         "v2",
        "n_images":        len(s2_files),
        "total_channels":  n_channels,
        "sample_fraction": SAMPLE_FRACTION,
        "n_sampled":       len(df),
        "top_k_per_crop":  TOP_K_PER_CROP,
        "keep_classes":    str(KEEP_CLASSES),
    })

    rows = []
    for crop_id in KEEP_CLASSES:
        si_vals = gsi_df[crop_id] if crop_id in gsi_df.columns else pd.Series(dtype=float)
        mlflow.set_tag(f"stage1_candidates_{crop_id}", str(candidates_per_crop[crop_id]))
        for rank, band in enumerate(candidates_per_crop[crop_id]):
            rows.append({
                "crop_id":   crop_id,
                "crop_name": CDL_CLASS_NAMES[crop_id],
                "rank":      rank + 1,
                "band":      band,
                "si_global": float(si_vals.get(band, 0.0)),
            })

    with tempfile.TemporaryDirectory() as tmp:
        p = pathlib.Path(tmp) / "stage1v2_per_crop_candidates.csv"
        p.write_text(pd.DataFrame(rows).to_csv(index=False))
        mlflow.log_artifact(str(p))

    mlflow.end_run(status="FINISHED")
    log.info(f"Stage 1 MLflow run_id: {stage1_run.info.run_id}")

    # Save candidates as handoff file so Stage 2 can run independently
    os.makedirs(os.path.dirname(STAGE1_CANDIDATES_JSON), exist_ok=True)
    payload = {
        "run_ts":              run_ts,
        "candidates_per_crop": {str(k): v for k, v in candidates_per_crop.items()},
    }
    with open(STAGE1_CANDIDATES_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    log.info(f"Stage 1 candidates saved: {STAGE1_CANDIDATES_JSON}")

    return candidates_per_crop, run_ts


# ── Stage 2: Per-crop binary forward selection ─────────────────────────────────

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
                       band_label: str = "") -> float:
    """Train a binary U-Net oracle for crop_id vs. rest. Returns best IoU(class 1)."""
    dataset = RasterPatchDataset(
        s2_paths=s2_files, cdl_path=cdl_path,
        patch_size=S2_PATCH_SIZE, stride=S2_STRIDE,
        min_valid_frac=S2_MIN_VALID, band_indices=band_indices,
        remap_lut=binary_lut, target_class_id=crop_id,
    )

    if len(dataset) < 4:
        log.warning(f"    Only {len(dataset)} patches for crop {crop_id} — returning 0.0")
        return 0.0

    # Pre-load all patches into RAM so DataLoader can use multiple workers + pin_memory.
    # RasterPatchDataset holds open rasterio handles (not picklable), forcing num_workers=0
    # which starves the GPU.  TensorDataset is fully picklable.
    tensor_ds = _preload_patches(dataset)

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


def run_stage2(candidates_per_crop: dict, all_bandnames: list,
               s2_files: list, cdl_path: str, run_ts: str) -> dict:
    """
    Per-crop binary CNN forward selection.

    MLflow structure:
        stage2v2_binary_fwd_{ts}          — parent: hyperparams + summary
        └── stage2v2_{crop_name}_{ts}     — child per crop: IoU curve, K*, accepted bands

    Returns selected_per_crop = {crop_id: [band_name, ...]}.
    """
    band_index = {name: i for i, name in enumerate(all_bandnames)}

    # Attach a file handler so all log output is also written to disk
    log_path    = PROCESSED_DIR / f"stage2_run_{run_ts}.log"
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    _fh         = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(_fh)

    _mlflow_setup()
    parent_run = mlflow.start_run(run_name=f"stage2v2_binary_fwd_{run_ts}")
    mlflow.log_params({
        "stage":          "2_per_crop_binary_forward",
        "version":        "v2",
        "encoder":        S2_ENCODER,
        "patch_size":     S2_PATCH_SIZE,
        "stride":         S2_STRIDE,
        "min_valid":      S2_MIN_VALID,
        "epochs":         S2_EPOCHS,
        "patience":       S2_PATIENCE,
        "delta":          S2_DELTA,
        "no_improve":     S2_NO_IMPROVE,
        "max_bands":      S2_MAX_BANDS,
        "top_k_per_crop": TOP_K_PER_CROP,
        "device":         DEVICE,
        "n_crops":        len(KEEP_CLASSES),
    })

    selected_per_crop: dict[int, list] = {}
    history_per_crop:  dict[int, list] = {}
    crop_run_ids:      dict[int, str]  = {}

    n_crops     = len(KEEP_CLASSES)
    total_cands = sum(len(v) for v in candidates_per_crop.values())
    log.info(f"\nStage 2 — per-crop binary CNN forward selection")
    log.info(f"  Crops: {n_crops}  |  Total candidates: {total_cands}  "
             f"|  δ={S2_DELTA}  max_bands={S2_MAX_BANDS}  no_improve={S2_NO_IMPROVE}")
    log.info(f"  Epochs={S2_EPOCHS}  patience={S2_PATIENCE}  "
             f"batch={S2_BATCH_SIZE}  patch={S2_PATCH_SIZE}px  stride={S2_STRIDE}px")

    try:
        for crop_idx, crop_id in enumerate(KEEP_CLASSES, 1):
            crop_name  = CDL_CLASS_NAMES[crop_id]
            candidates = candidates_per_crop[crop_id]

            log.info(f"\n{'='*60}")
            log.info(f"[{crop_idx}/{n_crops}] Crop: {crop_name} (CDL id={crop_id}) "
                     f"— {len(candidates)} candidates")

            binary_lut          = np.zeros(256, dtype=np.int64)
            binary_lut[crop_id] = 1

            selected, prev_iou, no_improve_cnt, history = [], 0.0, 0, []

            # ── Nested run per crop ───────────────────────────────────────────
            with mlflow.start_run(
                run_name=f"stage2v2_{crop_name.replace('/', '-')}_{run_ts}",
                nested=True,
            ) as crop_run:
                mlflow.log_params({
                    "crop_id":         crop_id,
                    "crop_name":       crop_name,
                    "n_candidates":    len(candidates),
                })

                for step, band in enumerate(candidates):
                    if len(selected) >= S2_MAX_BANDS:
                        log.info(f"  max_bands={S2_MAX_BANDS} reached — stopping")
                        break
                    if no_improve_cnt >= S2_NO_IMPROVE:
                        log.info(f"  {S2_NO_IMPROVE} consecutive rejections — stopping")
                        break

                    trial_bands   = selected + [band]
                    trial_indices = [band_index[b] for b in trial_bands]
                    log.info(f"\n  --- step {step+1}/{len(candidates)}  "
                             f"trial band: {band}  "
                             f"(selected so far: {len(selected)}) ---")
                    t0      = time.time()
                    iou     = _train_eval_binary(trial_indices, crop_id, binary_lut,
                                                 s2_files, cdl_path, band_label=band)
                    elapsed = time.time() - t0
                    gain          = iou - prev_iou
                    accepted      = gain >= S2_DELTA

                    if accepted:
                        selected = selected + [band]
                        prev_iou = iou
                        no_improve_cnt = 0
                    else:
                        no_improve_cnt += 1

                    history.append({
                        "crop_id":    crop_id,   "crop_name": crop_name,
                        "step":       step,       "band":      band,
                        "n_bands":    len(selected),
                        "iou_class1": round(iou,  4), "gain":  round(gain, 4),
                        "accepted":   accepted,        "elapsed_s": round(elapsed),
                    })

                    # step = band rank within this crop → clean IoU curve per crop
                    mlflow.log_metrics({
                        "iou_class1": iou,
                        "gain":       gain,
                        "accepted":   int(accepted),
                        "n_selected": len(selected),
                    }, step=step)

                    tag = "✅" if accepted else "❌"
                    log.info(f"  {tag} +{band:<22} IoU={iou:.4f} gain={gain:+.4f} ({elapsed:.0f}s)")

                # ── Option B: top-1 fallback if nothing was selected ─────────
                if not selected and candidates:
                    fallback = candidates[0]
                    selected = [fallback]
                    log.warning(
                        f"  K*=0 for {crop_name} — fallback to top-1 GSI band: {fallback}"
                    )
                    mlflow.set_tag("fallback_band", fallback)

                mlflow.log_metrics({"final_iou": prev_iou, "k_star": len(selected)})
                mlflow.set_tag("selected_bands", str(selected))
                mlflow.set_tag("stop_reason",
                    "max_bands"  if len(selected) >= S2_MAX_BANDS
                    else "no_improve" if no_improve_cnt >= S2_NO_IMPROVE
                    else "exhausted"
                )

                crop_run_ids[crop_id] = crop_run.info.run_id

            selected_per_crop[crop_id] = selected
            history_per_crop[crop_id]  = history
            log.info(f"\n  → [{crop_idx}/{n_crops}] {crop_name}: "
                     f"K*={len(selected)}  final IoU(c1)={prev_iou:.4f}")
            if selected:
                log.info(f"     Selected bands: {selected}")

            # Mirror summary to parent so it's visible without drilling into children
            mlflow.log_metrics({
                f"crop_{crop_id}_final_iou": prev_iou,
                f"crop_{crop_id}_k_star":    len(selected),
            })

        _save_results(selected_per_crop, history_per_crop, crop_run_ids)
        mlflow.log_artifact(str(STAGE2_RESULTS_CSV))
        mlflow.log_artifact(str(STAGE3_EXP_C_BANDS))
        log.info(f"Stage 2 parent run_id: {parent_run.info.run_id}")
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

    return selected_per_crop


def _save_results(selected_per_crop: dict, history_per_crop: dict,
                  crop_run_ids: dict) -> None:
    """Save stage2v2_per_crop_results.csv and stage3_exp_c_bands.txt."""
    rows = []
    for crop_id in KEEP_CLASSES:
        sel       = selected_per_crop.get(crop_id, [])
        hist      = history_per_crop.get(crop_id, [])
        final_iou = max((r["iou_class1"] for r in hist if r["accepted"]), default=0.0)

        dates    = sorted(set(
            re.search(r"_(\d{8})$", b).group(1)
            for b in sel if re.search(r"_(\d{8})$", b)
        ))
        spectral = sorted(set(re.sub(r"_\d{8}$", "", b) for b in sel))

        rows.append({
            "crop_id":        crop_id,
            "crop_name":      CDL_CLASS_NAMES[crop_id],
            "k_star":         len(sel),
            "key_dates":      ", ".join(f"{d[4:6]}/{d[6:8]}" for d in dates),
            "key_bands":      ", ".join(spectral),
            "selected_bands": str(sel),
            "final_iou_c1":   round(final_iou, 4),
            "mlflow_run_id":  crop_run_ids.get(crop_id, ""),
        })

    os.makedirs(os.path.dirname(STAGE2_RESULTS_CSV), exist_ok=True)
    pd.DataFrame(rows).to_csv(STAGE2_RESULTS_CSV, index=False)
    log.info(f"Saved: {STAGE2_RESULTS_CSV}")

    # Union in first-occurrence order across crops
    seen, union = set(), []
    for crop_id in KEEP_CLASSES:
        for band in selected_per_crop.get(crop_id, []):
            if band not in seen:
                seen.add(band)
                union.append(band)

    with open(STAGE3_EXP_C_BANDS, "w") as f:
        f.write("\n".join(union))
    log.info(f"Saved: {STAGE3_EXP_C_BANDS}  ({len(union)} unique bands)")

    # Summary table
    df = pd.DataFrame(rows)
    log.info("\n=== Per-Crop Stage 2 Results ===")
    log.info("\n" + df[["crop_name", "k_star", "key_dates", "key_bands", "final_iou_c1"]].to_string(index=False))
    log.info(f"\nStage 3 Exp C — union: {len(union)} bands: {union}")


# ── Stage project ──────────────────────────────────────────────────────────────

def run_project() -> None:
    """
    Project Stage 2 band selections (2022 dates) to nearest equivalent dates
    in every training and test year.

    Reads stage3_exp_c_bands.txt  — lines like 'B4_20220730'
    Writes stage3_exp_c_bands_projected.json:
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

    if not STAGE3_EXP_C_BANDS.exists():
        raise FileNotFoundError(
            f"Stage 2 output not found: {STAGE3_EXP_C_BANDS}\n"
            "Run Stage 2 first:  python feature_analysis.py --stage 2"
        )

    with open(STAGE3_EXP_C_BANDS) as f:
        selected_bands = [l.strip() for l in f if l.strip()]

    if not selected_bands:
        raise ValueError(f"{STAGE3_EXP_C_BANDS} is empty — re-run Stage 2.")

    # Parse band name + MMDD from each entry  e.g. "B4_20220730" → ("B4", "0730")
    band_mmdd = []
    for entry in selected_bands:
        m = re.match(r"(.+)_(\d{4})(\d{2})(\d{2})$", entry)
        if m:
            band_mmdd.append((m.group(1), m.group(3) + m.group(4)))
        else:
            log.warning(f"run_project: cannot parse band entry '{entry}' — skipped")

    log.info(f"Loaded {len(band_mmdd)} selected bands from {STAGE3_EXP_C_BANDS.name}")

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

    os.makedirs(os.path.dirname(STAGE3_EXP_C_BANDS_PROJECTED), exist_ok=True)
    with open(STAGE3_EXP_C_BANDS_PROJECTED, "w") as f:
        json.dump(projected, f, indent=2)
    log.info(f"Saved: {STAGE3_EXP_C_BANDS_PROJECTED}")

    # Log as MLflow artifact
    _mlflow_setup()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Identify the Stage 2 run that produced stage3_exp_c_bands.txt
    source_stage2_run_id = None
    source_stage2_run_name = None
    try:
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT_FEATURE)
        if exp:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string="attributes.run_name LIKE 'stage2v2_binary_fwd_%'",
                order_by=["attributes.start_time DESC"],
                max_results=1,
            )
            if runs:
                source_stage2_run_id   = runs[0].info.run_id
                source_stage2_run_name = runs[0].info.run_name
                log.info(
                    f"Source Stage 2 run: {source_stage2_run_name} "
                    f"(run_id={source_stage2_run_id})"
                )
    except Exception as e:
        log.warning(f"Could not look up source Stage 2 run: {e}")

    with mlflow.start_run(run_name=f"stage2v2_project_{ts}"):
        mlflow.set_tag("stage", "project")
        for yr, bands in projected.items():
            mlflow.log_param(f"n_bands_{yr}", len(bands))
            mlflow.set_tag(f"bands_{yr}", str(bands))
        if source_stage2_run_id:
            mlflow.set_tag("source_stage2_run_id",   source_stage2_run_id)
            mlflow.set_tag("source_stage2_run_name", source_stage2_run_name)
        mlflow.log_artifact(str(STAGE3_EXP_C_BANDS_PROJECTED))
        mlflow.log_artifact(str(STAGE3_EXP_C_BANDS))
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
         delta: float = None) -> None:
    # Override data paths if requested
    # Use `global` so all module-level functions pick up the new paths at call time.
    if data_dir:
        global S2_PROCESSED_DIR, CDL_BY_YEAR, PROCESSED_DIR, FIGURES_DIR, \
               STAGE2_RESULTS_CSV, STAGE3_EXP_C_BANDS, STAGE1_CANDIDATES_JSON
        processed               = pathlib.Path(data_dir)
        PROCESSED_DIR           = processed
        S2_PROCESSED_DIR        = processed / "s2"
        CDL_BY_YEAR             = {
            yr: processed / "cdl" / f"cdl_{yr}_study_area_filtered.tif"
            for yr in ["2022", "2023", "2024"]
        }
        STAGE2_RESULTS_CSV      = processed / "stage2v2_per_crop_results.csv"
        STAGE3_EXP_C_BANDS      = processed / "stage3_exp_c_bands.txt"
        STAGE1_CANDIDATES_JSON  = processed / "s2" / "2022" / "stage1v2_candidates.json"
        log.info(f"Data dir overridden to {processed}")

    if delta is not None:
        global S2_DELTA
        S2_DELTA = delta
        log.info(f"S2_DELTA overridden to {S2_DELTA}")

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    if stage in ("1", "all"):
        if not force and os.path.exists(STAGE1_CANDIDATES_JSON):
            log.info(f"Stage 1 output already exists: {STAGE1_CANDIDATES_JSON}")
            log.info("Use --force to re-run.")
        else:
            log.info(f"Device: {_device_label()}")
            df, all_bandnames, n_channels, s2_files, cdl_path = load_data(s2_year="2022")
            run_stage1(df, all_bandnames, n_channels, s2_files)
            log.info("Stage 1 complete.")

        if stage == "1":
            return

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    if stage in ("2", "all"):
        if not force and os.path.exists(STAGE3_EXP_C_BANDS):
            log.info(f"Stage 2 output already exists: {STAGE3_EXP_C_BANDS}")
            log.info("Use --force to re-run.")
            if stage == "2":
                return
        else:
            # Load Stage 1 candidates from handoff file
            if not os.path.exists(STAGE1_CANDIDATES_JSON):
                raise FileNotFoundError(
                    f"Stage 1 candidates not found: {STAGE1_CANDIDATES_JSON}\n"
                    "Run Stage 1 first:  python feature_analysis.py --stage 1"
                )
            with open(STAGE1_CANDIDATES_JSON) as f:
                payload = json.load(f)
            candidates_per_crop = {int(k): v for k, v in payload["candidates_per_crop"].items()}
            run_ts              = payload["run_ts"]
            log.info(f"Loaded Stage 1 candidates from {STAGE1_CANDIDATES_JSON}  (run_ts={run_ts})")

            log.info(f"Device: {_device_label()}")
            _, all_bandnames, _, s2_files, cdl_path = load_data(s2_year="2022", stage=2)
            run_stage2(candidates_per_crop, all_bandnames, s2_files, cdl_path, run_ts)
            log.info("Stage 2 complete.")

        if stage == "2":
            return

    # ── Stage project ──────────────────────────────────────────────────────────
    if stage == "project":
        if not force and STAGE3_EXP_C_BANDS_PROJECTED.exists():
            log.info(f"Projected bands already exist: {STAGE3_EXP_C_BANDS_PROJECTED}")
            log.info("Use --force to re-run.")
            return
        run_project()
        log.info("Band projection complete.")
        return

    log.info("Feature analysis complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature analysis v2: Stage 1 + Stage 2")
    parser.add_argument(
        "--stage",
        choices=["1", "2", "all", "project"],
        default="all",
        help=(
            "Which stage to run: "
            "1 (CPU, run locally), "
            "2 (GPU, run on server), "
            "all (1+2), "
            "project (map 2022 selections to 2023/2024 nearest dates)"
        ),
    )
    parser.add_argument("--force",    action="store_true", help="Re-run even if outputs exist")
    parser.add_argument("--data-dir", type=str,   default=None,
                        help="Override processed data directory")
    parser.add_argument("--delta",    type=float, default=None,
                        help="Override S2_DELTA (min IoU gain to accept a band, default: 0.005)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )
    main(force=args.force, data_dir=args.data_dir, stage=args.stage, delta=args.delta)
