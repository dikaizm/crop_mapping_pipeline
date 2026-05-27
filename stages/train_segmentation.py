"""
Stage 3 — Full Model Validation.

Six experiment configurations × 2 architectures = up to 12 training runs.

| Config             | Dates               | Band selection | Purpose                      |
|--------------------|---------------------|----------------|------------------------------|
| single_date_gsi    | peak NDVI           | GSI            | Domain temporal + GSI bands  |
| single_date_rf     | peak NDVI           | RF             | Domain temporal + RF bands   |
| naive_mt_gsi       | 4 phenological      | GSI            | Multi-temporal + GSI bands   |
| naive_mt_rf        | 4 phenological      | RF             | Multi-temporal + RF bands    |
| gsi                | GSI-direct          | GSI-direct     | GSI spectral-temporal        |
| rf                 | RF-direct           | RF-direct      | RF spectral-temporal         |

Usage:
    python stages/train_segmentation.py                       # run all 6 experiments
    python stages/train_segmentation.py --exp single_date     # only single-date baseline
    python stages/train_segmentation.py --exp gsi --arch segformer
    python stages/train_segmentation.py --force               # re-run even if ckpt exists
    python stages/train_segmentation.py --data-dir /mnt/data
"""

import os
import re
import sys
import time
import json
import hashlib
import argparse
import logging
from glob import glob
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, ConcatDataset, WeightedRandomSampler
import rasterio

os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
# Cache HuggingFace model weights persistently so they are not re-downloaded each run
os.environ.setdefault("HF_HOME", str(Path(__file__).parent.parent / ".hf_cache"))
import mlflow

_ROOT = Path(__file__).parent.parent   # crop_mapping_pipeline/
sys.path.insert(0, str(_ROOT.parent))

from crop_mapping_pipeline.utils.mlflow_utils import patch_artifact_logging
patch_artifact_logging()

from crop_mapping_pipeline.config import (
    S2_TRAIN_DIR, S2_PROCESSED_DIR, CDL_BY_YEAR, CDL_TRAIN, MODELS_DIR, FIGURES_DIR, LOGS_DIR,
    PROCESSED_DIR, PRELOAD_CACHE_DIR,
    S2_BAND_NAMES, N_BANDS_PER_DATE, VEGE_BANDS,
    KEEP_CLASSES, CLASS_REMAP, NUM_CLASSES, CDL_CLASS_NAMES,
    REMAP_LUT, S2_NODATA,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_FEATURE,
    TRAIN_YEARS, TEST_YEAR, SPATIAL_TEST_AREAS,
    PATCH_SIZE, STRIDE, MIN_VALID_FRAC, BATCH_SIZE, MAX_EPOCHS, EARLY_STOP, EARLY_STOP_DELTA,
    VAL_FRAC, SEED, ARCH_CFG,
    GDRIVE_OAUTH_TOKEN, GDRIVE_MODELS_FOLDER_ID,
)
from geoai.geoai.train import RasterPatchDataset, train_semantic_one_epoch
from crop_mapping_pipeline.stages.losses import build_loss_v1, build_loss_v2, PhenologyAwareLoss
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
    # Flat train/ dir — all files belong to the single training year
    return sorted(s2_processed)


def _valid_global_indices(s2_paths, band_indices, n_bands_per_file=N_BANDS_PER_DATE):
    """Return the subset of band_indices that are in range for s2_paths."""
    if band_indices is None:
        return set()
    needed = sorted({gi // n_bands_per_file for gi in band_indices
                     if gi // n_bands_per_file < len(s2_paths)})
    new_idx_map = set()
    for fi in needed:
        for local in range(n_bands_per_file):
            new_idx_map.add(fi * n_bands_per_file + local)
    return set(gi for gi in band_indices if gi in new_idx_map)


def _filter_s2_by_band_indices(s2_paths, band_indices, n_bands_per_file=N_BANDS_PER_DATE):
    """Return (filtered_paths, remapped_indices) keeping only TIF files that
    contribute at least one channel in band_indices, with indices remapped to
    their positions in the reduced stack.

    Example: 25 files × 11 bands = 275 channels.  single_date selects bands [157..165]
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
    skipped = [gi for gi in band_indices if gi not in new_idx_map]
    if skipped:
        log.warning("  Dropping %d channel(s) from excluded/empty S2 files: %s",
                    len(skipped), skipped)
    remapped = [new_idx_map[gi] for gi in band_indices if gi in new_idx_map]
    return filtered_paths, remapped


from crop_mapping_pipeline.stages.experiments import (
    parse_date,
    build_local_band_map,
    build_single_date_indices,
    build_single_date_selected_indices,
    build_naive_multitemporal_indices,
    build_naive_multitemporal_selected_indices,
    build_registry,
    expand_exp_keys,
)
from crop_mapping_pipeline.stages.experiments.exp_select_direct import build_direct_indices
from crop_mapping_pipeline.stages.selections.rf_band_only import run_rf_band_only, save_rf_band_json
from crop_mapping_pipeline.config import (
    SELECT_GSI_DIRECT_JSON,
    SELECT_RF_DIRECT_JSON,
    GSI_CANDIDATES_JSON,
    PROCESSED_DIR,
)


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(cdl_path=None):
    """Inverse-frequency weights from CDL (train area). Caches result alongside CDL."""
    ref_cdl   = Path(cdl_path) if cdl_path else CDL_TRAIN
    cache_key = {"cdl": str(ref_cdl), "keep_classes": KEEP_CLASSES, "num_classes": NUM_CLASSES}
    cache_h   = hashlib.sha256(json.dumps(cache_key, sort_keys=True).encode()).hexdigest()[:12]
    cache_path = ref_cdl.parent / f"class_weights_{cache_h}.json"

    if cache_path.exists():
        try:
            with open(cache_path) as f:
                w = json.load(f)["weights"]
            log.info(f"Class weights cache hit → {cache_path.name}")
            return torch.tensor(w, dtype=torch.float32)
        except Exception:
            pass

    with rasterio.open(ref_cdl) as src:
        cdl_arr = src.read(1).astype(np.int32)

    class_counts      = np.zeros(NUM_CLASSES, dtype=np.float64)
    class_counts[0]   = (cdl_arr == 0).sum()
    for cdl_id, model_id in CLASS_REMAP.items():
        class_counts[model_id] += (cdl_arr == cdl_id).sum()

    freq    = class_counts / (class_counts.sum() + 1e-9)
    weights = 1.0 / (freq + 1e-9)
    weights /= weights.sum()

    with open(cache_path, "w") as f:
        json.dump({"weights": weights.tolist(), "class_counts": class_counts.tolist()}, f)
    log.info(f"Class weights cached → {cache_path.name}")

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
        imgs        = torch.nan_to_num(imgs, nan=0.0, posinf=1.0, neginf=0.0)
        logits      = model(imgs)
        
        # Check if criterion is phenology-aware
        if isinstance(criterion, PhenologyAwareLoss):
            loss = criterion(logits, masks, imgs)
        else:
            loss = criterion(logits, masks)
            
        total_loss += loss.item()
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
    model._n_params = n
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


# ── Class-weighted patch sampler ──────────────────────────────────────────────

def _patch_weights(datasets: list) -> np.ndarray:
    """
    Compute a weight per patch across a list of RasterPatchDataset objects.
    Weight = sum over classes of (patch_pixel_count[c] / global_pixel_count[c]).
    Rare-class patches get higher weight → balanced mini-batches.
    Uses the in-memory _cdl array — no S2 I/O.
    """
    ps = datasets[0].patch_size

    # Pass 1: global class pixel counts
    global_counts: dict[int, int] = {}
    for ds in datasets:
        cdl = ds._cdl
        remap = ds._remap_lut
        for r, c in ds.patches:
            patch_cdl = cdl[r:r + ps, c:c + ps]
            remapped  = remap[np.clip(patch_cdl, 0, 255)]
            for cls_id in np.unique(remapped):
                if cls_id == 0:
                    continue
                global_counts[int(cls_id)] = global_counts.get(int(cls_id), 0) + int((remapped == cls_id).sum())

    if not global_counts:
        # Fallback: uniform weights
        return np.ones(sum(len(ds.patches) for ds in datasets), dtype=np.float32)

    # Pass 2: per-patch weight
    weights = []
    for ds in datasets:
        cdl   = ds._cdl
        remap = ds._remap_lut
        for r, c in ds.patches:
            patch_cdl = cdl[r:r + ps, c:c + ps]
            remapped  = remap[np.clip(patch_cdl, 0, 255)]
            w = 0.0
            for cls_id in np.unique(remapped):
                if cls_id == 0:
                    continue
                cnt = int((remapped == cls_id).sum())
                w  += cnt / global_counts[int(cls_id)]
            weights.append(w if w > 0 else 1e-6)

    return np.array(weights, dtype=np.float64)


# ── Augmentation wrapper ───────────────────────────────────────────────────────

class AugmentedSubset(torch.utils.data.Dataset):
    """Wraps a Subset and applies random H/V flips + 90° rotations to (img, mask)."""

    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, mask = self.subset[idx]   # img: (C,H,W) float, mask: (H,W) long
        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            img  = torch.flip(img,  [-1])
            mask = torch.flip(mask, [-1])
        # Random vertical flip
        if torch.rand(1).item() > 0.5:
            img  = torch.flip(img,  [-2])
            mask = torch.flip(mask, [-2])
        # Random 90° rotation (k ∈ {0,1,2,3})
        k = torch.randint(0, 4, (1,)).item()
        if k:
            img  = torch.rot90(img,  k, [-2, -1])
            mask = torch.rot90(mask, k, [-2, -1])
        return img, mask


# ── In-memory dataset cache ───────────────────────────────────────────────────

class PreloadedDataset(torch.utils.data.Dataset):
    """Builds a persistent disk cache of all patches; loads imgs via memory-map.

    Reads each TIF file once in full (parallel threads) instead of per-patch
    window reads → ~30–60s instead of 15+ min for large datasets.
    Cache key covers s2_paths/cdl_path/bands/patch_size.

    Imgs stored as float16 .npy → loaded with mmap_mode='r' so the OS pages in
    only what each minibatch needs.  Peak RAM = model + batch, not full dataset.
    Masks stored as int64 .pt (typically <1 GB, always in RAM).
    """

    def __init__(self, dataset, desc="preload", cache_dir=None, n_threads=None):
        imgs_path, masks_path = self._cache_paths(dataset, cache_dir) if cache_dir else (None, None)

        if imgs_path and imgs_path.exists() and masks_path and masks_path.exists():
            log.info(f"  [{desc}] Cache hit → mmap {imgs_path.name}")
            t0 = time.time()
            self._imgs  = np.load(str(imgs_path), mmap_mode="r")   # memory-mapped, not in RAM
            self._masks = torch.load(masks_path, map_location="cpu", weights_only=True)
            gb_disk = imgs_path.stat().st_size / 1e9
            log.info(f"  [{desc}] mmap ready in {time.time()-t0:.1f}s ({gb_disk:.2f} GB on disk)")
            return

        log.info(f"  [{desc}] Cache miss → preloading from {len(dataset._s2_srcs)} TIF files …")
        t0 = time.time()

        n  = len(dataset)
        ps = dataset.patch_size
        band_indices  = dataset.band_indices
        n_ch_per_file = [src.count for src in dataset._s2_srcs]
        ch_offsets    = np.cumsum([0] + n_ch_per_file).tolist()
        n_ch          = len(band_indices) if band_indices is not None else ch_offsets[-1]

        # file_extraction[fi] = [(output_col, local_band_idx_1based), ...]
        file_extraction: dict = {}
        targets = band_indices if band_indices is not None else list(range(ch_offsets[-1]))
        for out_pos, gi in enumerate(targets):
            for fi in range(len(n_ch_per_file)):
                if ch_offsets[fi] <= gi < ch_offsets[fi + 1]:
                    file_extraction.setdefault(fi, []).append((out_pos, gi - ch_offsets[fi] + 1))
                    break

        patches = dataset.patches
        nodata  = dataset.nodata

        # Allocate buf as a disk-backed float16 memmap — never occupies RAM regardless
        # of channel count. 70ch × 1800 patches × 256² × float32 ≈ 33 GB; float16
        # memmap keeps peak RAM to ~O(one TIF file) during the fill loop.
        _buf_path = (imgs_path.with_suffix(".tmp.npy") if imgs_path
                     else Path(PRELOAD_CACHE_DIR) / f"_tmp_{os.getpid()}.npy")
        _buf_path.parent.mkdir(parents=True, exist_ok=True)
        buf = np.lib.format.open_memmap(
            str(_buf_path), mode="w+", dtype=np.float16, shape=(n, n_ch, ps, ps)
        )
        gb_alloc = buf.nbytes / 1e9
        log.info(f"  [{desc}] Buf: {n}×{n_ch}×{ps}×{ps} float16 = {gb_alloc:.1f} GB on disk")

        def _read_one_file(fi):
            extractions = file_extraction[fi]
            local_idxs  = [e[1] for e in extractions]
            out_cols    = [e[0] for e in extractions]
            try:
                with rasterio.open(dataset.s2_paths[fi]) as src:
                    arr = src.read(indexes=local_idxs).astype(np.float32)
                arr[arr == nodata]      = 0.0
                arr[~np.isfinite(arr)]  = 0.0
                return fi, arr, out_cols
            except Exception as e:
                log.warning(f"  [{desc}] read failed file {fi}: {e}")
                return fi, None, out_cols

        # Single-threaded write to memmap — concurrent writes to overlapping patches
        # cause data races; read threads are fine, write serialised via main thread.
        _n_threads = n_threads or min(len(file_extraction), os.cpu_count() or 8)
        log.info(f"  [{desc}] Using {_n_threads} read threads for {len(file_extraction)} files")
        with ThreadPoolExecutor(max_workers=_n_threads) as pool:
            for fi, arr, out_cols in pool.map(_read_one_file, list(file_extraction.keys())):
                if arr is None:
                    continue
                for ci, out_pos in enumerate(out_cols):
                    band_plane = arr[ci]
                    for pi, (r, c) in enumerate(patches):
                        buf[pi, out_pos, :, :] = band_plane[r:r+ps, c:c+ps]
                del arr

        # Per-patch normalisation in float32 chunks to avoid float16 precision loss
        # on raw DN values (0–10000 range). Process CHUNK_PATCHES at a time → bounded RAM.
        CHUNK = 128
        for start in range(0, n, CHUNK):
            end   = min(start + CHUNK, n)
            chunk = buf[start:end].astype(np.float32)          # (chunk, C, H, W) in RAM
            lo    = chunk.min(axis=(-2, -1), keepdims=True)
            hi    = chunk.max(axis=(-2, -1), keepdims=True)
            rng   = np.where(hi > lo, hi - lo, 1.0)
            chunk -= lo
            chunk /= rng
            buf[start:end] = chunk.astype(np.float16)          # write back to disk
        buf.flush()

        masks = [
            torch.from_numpy(
                dataset._remap_lut[np.clip(dataset._cdl[r:r+ps, c:c+ps], 0, 255)].astype(np.int64)
            )
            for r, c in patches
        ]
        self._masks = torch.stack(masks)

        elapsed = time.time() - t0
        log.info(f"  [{desc}] Preloaded in {elapsed:.1f}s — {gb_alloc:.1f} GB float16 on disk")

        if imgs_path:
            _buf_path.rename(imgs_path)
            torch.save(self._masks, masks_path)
            log.info(f"  [{desc}] Cached → {imgs_path.name} + {masks_path.name}")
            self._imgs = np.load(str(imgs_path), mmap_mode="r")
        else:
            self._imgs = np.array(buf)   # no cache dir: load into RAM
            _buf_path.unlink(missing_ok=True)

    @staticmethod
    def _cache_paths(dataset, cache_dir):
        key = {
            "s2":             [str(p) for p in dataset.s2_paths],
            "cdl":            str(dataset.cdl_path),
            "ps":             dataset.patch_size,
            "bands":          list(dataset.band_indices) if dataset.band_indices is not None else None,
            "stride":         getattr(dataset, "stride", None),
            "min_valid_frac": getattr(dataset, "min_valid_frac", None),
            "n_patches":      len(dataset.patches),  # changes with KEEP_CLASSES → prevents stale cache
        }
        h = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
        base = Path(cache_dir) / f"preload_{h}"
        return base.with_suffix(".npy"), base.with_name(base.name + "_masks.pt")

    def __len__(self):
        return len(self._masks)

    def __getitem__(self, idx):
        # np array (memmap or plain) → float32 tensor; .copy() required for mmap slices
        img = torch.tensor(self._imgs[idx], dtype=torch.float32)
        return img, self._masks[idx]


# ── Spatial test area evaluation ─────────────────────────────────────────────

def _evaluate_spatial_area(
    model,
    area: dict,
    band_names: list,
    exp_name: str,
    exp_dir: Path,
    skip_viz: bool = False,
) -> "dict | None":
    """Evaluate model on one held-out spatial test area.

    area: {"name": str, "s2_dir": Path, "cdl": Path}
    band_names: channel names from experiment (e.g. ["B4_20240730", ...]).
    Returns evaluate_test_set result dict, or None if area data missing.
    """
    import glob as _glob

    area_name = area["name"]
    s2_dir    = Path(area["s2_dir"])
    cdl_path  = Path(area["cdl"])

    area_s2 = sorted(f for f in _glob.glob(str(s2_dir / "*.tif")) if not Path(f).name.startswith("._"))
    if not area_s2:
        log.warning(f"  Spatial test {area_name}: no S2 files in {s2_dir} — skipping")
        return None
    if not cdl_path.exists():
        log.warning(f"  Spatial test {area_name}: CDL not found at {cdl_path} — skipping")
        return None

    log.info(f"  Spatial test [{area_name}]: {len(area_s2)} S2 files, CDL={cdl_path.name}")

    _, area_band_to_idx, _, _ = build_local_band_map(area_s2)

    area_global_indices = []
    skipped_bands = []
    for bname in band_names:
        idx = area_band_to_idx.get(bname)
        if idx is not None:
            area_global_indices.append(idx)
        else:
            skipped_bands.append(bname)

    if skipped_bands:
        log.warning(f"  Spatial test {area_name}: {len(skipped_bands)} band(s) not found in area files (date mismatch?): {skipped_bands[:3]}...")
    if not area_global_indices:
        log.error(f"  Spatial test {area_name}: no matching bands — skipping")
        return None

    area_s2_filtered, area_idx_local = _filter_s2_by_band_indices(area_s2, area_global_indices)

    area_ds = RasterPatchDataset(
        s2_paths=area_s2_filtered, cdl_path=str(cdl_path),
        patch_size=PATCH_SIZE, stride=STRIDE,
        keep_classes=KEEP_CLASSES, remap_lut=REMAP_LUT,
        min_valid_frac=MIN_VALID_FRAC, band_indices=area_idx_local,
    )
    area_dl = DataLoader(area_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    area_r = evaluate_test_set(model, area_dl, NUM_CLASSES, DEVICE)
    log.info(f"  [{area_name}] mIoU={area_r['miou']:.4f}  OA={area_r['oa']:.4f}")
    log.info(f"  {'Class':<20} {'IoU':>7}")
    for cls_id, iou in area_r["per_class_iou"].items():
        cdl_id = KEEP_CLASSES[cls_id - 1]
        name   = CDL_CLASS_NAMES.get(cdl_id, f"cls{cls_id}")
        log.info(f"  {name:<20} {iou:.4f}" if not np.isnan(iou) else f"  {name:<20}     nan")

    # MLflow metrics prefixed with area name
    mlflow.log_metrics({
        f"{area_name}_miou": area_r["miou"],
        f"{area_name}_oa":   area_r["oa"],
    })
    for cls_id, iou in area_r["per_class_iou"].items():
        if not np.isnan(iou):
            cdl_id = KEEP_CLASSES[cls_id - 1]
            cname  = CDL_CLASS_NAMES.get(cdl_id, f"cls{cls_id}")
            mlflow.log_metric(
                f"{area_name}_iou_{cname.lower().replace('/', '_').replace(' ', '_')}",
                iou,
            )

    # Per-class IoU CSV
    iou_rows = [
        {
            "class_id":   cls_id,
            "cdl_id":     KEEP_CLASSES[cls_id - 1],
            "class_name": CDL_CLASS_NAMES.get(KEEP_CLASSES[cls_id - 1], f"cls{cls_id}"),
            "iou":        round(iou, 4) if not np.isnan(iou) else float("nan"),
        }
        for cls_id, iou in area_r["per_class_iou"].items()
    ]
    iou_csv = exp_dir / f"{area_name}_per_class_iou.csv"
    pd.DataFrame(iou_rows).to_csv(iou_csv, index=False)
    mlflow.log_artifact(str(iou_csv))

    # Confusion matrix
    cm_path = exp_dir / f"{area_name}_confusion_matrix.png"
    _plot_confusion_matrix(area_r["preds"], area_r["labels"], str(cm_path))
    mlflow.log_artifact(str(cm_path))

    # Segmentation map
    if not skip_viz:
        gt_map, _   = load_gt_remap(str(cdl_path))
        pred_map, _ = run_full_inference(
            model, area_s2_filtered, area_idx_local,
            patch_size=PATCH_SIZE, stride=PATCH_SIZE,
        )
        seg_path = exp_dir / f"{area_name}_segmentation_map.png"
        save_segmentation_map(
            pred_map, gt_map,
            title=f"{exp_name} — {area_name}",
            save_path=str(seg_path),
        )
        mlflow.log_artifact(str(seg_path))
        del pred_map, gt_map

    return area_r


# ── Main experiment runner ────────────────────────────────────────────────────

def run_experiment(
    exp_name,
    arch,
    band_indices,           # list[int]  OR  dict{yr: (list[int], list[str])}
    band_names_list,        # list[str]  (reference year; used for logging/metadata)
    description,
    s2_processed,
    class_weights_tensor,
    loss_version="v1",      # "v1" = WeightedCrossEntropy | "v2" = PhenologyAwareLoss
    force=False,
    skip_viz=False,
):
    """band_indices: list[int] same for all years, or dict{yr: (idx, names)} per-year."""
    cfg           = ARCH_CFG[arch]
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_dir       = MODELS_DIR / f"{exp_name}_{run_timestamp}"
    best_ckpt     = exp_dir / "best_model.pth"
    last_ckpt     = exp_dir / "last_model.pth"
    exp_dir.mkdir(parents=True, exist_ok=True)

    if not force and best_ckpt.exists():
        log.info(f"Checkpoint exists — skipping {exp_name}  (use --force to re-run)")
        return None

    # Per-run log file — captured from start of training; uploaded as MLflow artifact at end
    run_log_path    = exp_dir / f"{exp_name}_train.log"
    run_log_handler = logging.FileHandler(run_log_path, mode="w")
    run_log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(run_log_handler)

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

    # Pre-pass: find globally consistent band indices available in ALL years.
    # Prevents channel-count mismatch when some years lack a file (e.g. excluded empty date).
    if not per_year:
        base_idx = band_indices  # same list applied to every year
        all_years = list(TRAIN_YEARS) + [TEST_YEAR]
        valid_sets = []
        for yr in all_years:
            yr_s2_all = _s2_for_year(s2_processed, yr)
            valid_sets.append(_valid_global_indices(yr_s2_all, base_idx))
        consistent = sorted(set.intersection(*valid_sets))
        dropped = len(base_idx) - len(consistent)
        if dropped:
            log.warning(
                f"  Dropping {dropped} channel(s) not available in all years "
                f"({', '.join(all_years)}) — keeping {len(consistent)} consistent channels"
            )
        consistent_set  = set(consistent)
        band_names_list = [name for gi, name in zip(base_idx, band_names_list) if gi in consistent_set]
        band_indices    = consistent

    in_channels = len(_yr_idx(TRAIN_YEARS[0])[0])
    log.info(f"\n{'='*65}")
    log.info(f" {exp_name}")
    log.info(f"  arch={arch}  in_channels={in_channels}  per_year_indices={per_year}")
    log.info(f"  {description}")
    log.info(f"{'='*65}\n")

    # ── Year-based dataset split ──────────────────────────────────────────────
    train_year_datasets_raw = []   # RasterPatchDataset — for _patch_weights (needs _cdl etc.)
    train_year_datasets     = []   # PreloadedDataset  — for DataLoader
    for yr in TRAIN_YEARS:
        yr_s2  = _s2_for_year(s2_processed, yr)
        yr_cdl = CDL_TRAIN
        if not yr_s2 or not yr_cdl.exists():
            log.warning(f"Skipping train year {yr}: {'no S2' if not yr_s2 else 'CDL missing'}")
            continue
        yr_idx, _ = _yr_idx(yr)
        yr_s2_filtered, yr_idx_local = _filter_s2_by_band_indices(yr_s2, yr_idx)
        ds_raw = RasterPatchDataset(
            s2_paths=yr_s2_filtered, cdl_path=str(yr_cdl),
            patch_size=PATCH_SIZE, stride=STRIDE,
            keep_classes=KEEP_CLASSES, remap_lut=REMAP_LUT,
            min_valid_frac=MIN_VALID_FRAC, band_indices=yr_idx_local,
        )
        log.info(f"  [{yr}] {len(ds_raw):,} patches  ({len(yr_idx)} channels, {len(yr_s2_filtered)}/{len(yr_s2)} files)")
        train_year_datasets_raw.append(ds_raw)
        train_year_datasets.append(PreloadedDataset(ds_raw, desc=yr, cache_dir=PRELOAD_CACHE_DIR))

    assert train_year_datasets, "No training data for any TRAIN_YEAR"
    train_val_ds = ConcatDataset(train_year_datasets)

    gen = torch.Generator().manual_seed(SEED)

    # 2-way split: train/val from main area; test via held-out SPATIAL_TEST_AREAS
    n_total = len(train_val_ds)
    n_val   = max(1, int(VAL_FRAC * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(train_val_ds, [n_train, n_val], generator=gen)
    test_ds          = val_ds   # unused placeholder for DataLoader construction
    test_s2_filtered = None
    test_idx_local   = None

    # Class-weighted sampler: rare-class patches sampled more frequently
    log.info("  Computing patch weights for class-balanced sampling...")
    all_weights = _patch_weights(train_year_datasets_raw)
    train_weights = all_weights[train_ds.indices]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(train_weights).double(),
        num_samples=n_train,
        replacement=True,
    )
    aug_train_ds = AugmentedSubset(train_ds)
    train_dl = DataLoader(aug_train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,       batch_size=BATCH_SIZE, shuffle=False,   num_workers=4, pin_memory=True)
    test_dl  = DataLoader(test_ds,      batch_size=BATCH_SIZE, shuffle=False,   num_workers=4, pin_memory=True)
    log.info(f"  Patches: {n_train:,} train (augmented) / {n_val:,} val [spatial test via test_a/test_b]")

    # ── Model + optimiser + scheduler + loss ──────────────────────────────────
    model     = build_model(arch, in_channels, NUM_CLASSES)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=MAX_EPOCHS, power=0.9
    )

    # ── Loss function (versioned) ──────────────────────────────────────────
    if loss_version == "v2":
        criterion, red_idx, nir_idx = build_loss_v2(
            class_weights_tensor.to(DEVICE), band_names_list
        )
        log.info(
            f"  Loss v2 — PhenologyAwareLoss "
            f"(Red={band_names_list[red_idx]}, NIR={band_names_list[nir_idx]})"
        )
    else:
        criterion = build_loss_v1(class_weights_tensor.to(DEVICE))
        log.info("  Loss v1 — WeightedCrossEntropy")

    # ── MLflow run (child — nested under parent created in main()) ────────────

    with mlflow.start_run(run_name=exp_name, nested=True) as run:
        mlflow.log_params({
            "experiment":     exp_name,
            "architecture":   arch,
            "encoder":        cfg["encoder"],
            "in_channels":    in_channels,
            "num_classes":    NUM_CLASSES,
            "patch_size":     PATCH_SIZE,
            "stride":         STRIDE,
            "batch_size":     BATCH_SIZE,
            "max_epochs":     MAX_EPOCHS,
            "early_stopping": EARLY_STOP,
            "learning_rate":  cfg["lr"],
            "weight_decay":   cfg["weight_decay"],
            "optimizer":      "AdamW",
            "lr_scheduler":   "PolynomialLR(power=0.9)",
            "loss":           f"loss_{loss_version}",
            "train_years":    str(TRAIN_YEARS),
            "test_year":      TEST_YEAR,
            "train_patches":  n_train,
            "val_patches":    n_val,
            "test_patches":   len(test_ds),
            "description":    description,
            "keep_classes":   str(KEEP_CLASSES),
            "model_params":   getattr(model, "_n_params", None),
        })
        mlflow.set_tag("band_names", str(band_names_list))
        mlflow.set_tag("n_bands",    str(in_channels))


        # ── Training loop ─────────────────────────────────────────────────────
        best_miou              = 0.0
        best_val_per_class_iou = {}
        no_improve             = 0
        history                = []
        t_start    = time.time()

        for epoch in range(MAX_EPOCHS):
            t_ep = time.time()

            model.train()
            train_loss_acc, n_batches = 0.0, 0
            _logged_vram = epoch > 0   # log VRAM once on first batch of epoch 0
            for imgs, masks in train_dl:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                imgs        = torch.nan_to_num(imgs, nan=0.0, posinf=1.0, neginf=0.0)
                optimizer.zero_grad()
                logits = model(imgs)

                if isinstance(criterion, PhenologyAwareLoss):
                    loss = criterion(logits, masks, imgs)
                else:
                    loss = criterion(logits, masks)

                loss.backward()
                optimizer.step()
                train_loss_acc += loss.item()
                n_batches += 1

                if not _logged_vram and torch.cuda.is_available():
                    alloc  = torch.cuda.memory_allocated()  / 1024**3
                    reserv = torch.cuda.memory_reserved()   / 1024**3
                    log.info(f"  [VRAM] allocated={alloc:.2f} GB  reserved={reserv:.2f} GB")
                    _logged_vram = True

            train_loss = train_loss_acc / n_batches
            val_m = validate_one_epoch(model, val_dl, criterion, DEVICE, NUM_CLASSES)
            scheduler.step()

            ep_t = time.time() - t_ep
            per_cls_metrics = {}
            for cls_id, iou in val_m["per_class_iou"].items():
                if not np.isnan(iou):
                    cdl_id = KEEP_CLASSES[cls_id - 1]
                    name   = CDL_CLASS_NAMES.get(cdl_id, f"cls{cls_id}")
                    key    = f"val_iou_{name.lower().replace('/', '_').replace(' ', '_')}"
                    per_cls_metrics[key] = iou
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss":   val_m["loss"],
                "val_miou":   val_m["miou"],
                "val_oa":     val_m["oa"],
                "lr":         scheduler.get_last_lr()[0],
                **per_cls_metrics,
            }, step=epoch)

            history.append({
                "epoch":      epoch + 1,
                "train_loss": round(train_loss,       4),
                "val_loss":   round(val_m["loss"],    4),
                "val_miou":   round(val_m["miou"],    4),
                "val_oa":     round(val_m["oa"],      4),
                "epoch_t_s":  round(ep_t,              1),
            })

            if val_m["miou"] > best_miou + EARLY_STOP_DELTA:
                best_miou              = val_m["miou"]
                best_val_per_class_iou = val_m["per_class_iou"]
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
                f"mIoU={val_m['miou']:.4f} OA={val_m['oa']:.4f} best={best_miou:.4f} "
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

        # Test done via SPATIAL_TEST_AREAS — no patch-level test set
        test_r = {"miou": float("nan"), "oa": float("nan"), "per_class_iou": {}}

        mlflow.log_metrics({
            "best_val_miou": best_miou,
            "test_miou":     test_r["miou"],
            "test_oa":       test_r["oa"],
            "total_epochs":  len(history),
        })
        for cls_id, iou in best_val_per_class_iou.items():
            if not np.isnan(iou):
                cdl_id = KEEP_CLASSES[cls_id - 1]
                name   = CDL_CLASS_NAMES.get(cdl_id, f"cls{cls_id}")
                mlflow.log_metric(
                    f"best_val_iou_{name.lower().replace('/', '_').replace(' ', '_')}",
                    iou,
                )
        for cls_id, iou in test_r["per_class_iou"].items():
            if not np.isnan(iou):
                cdl_id = KEEP_CLASSES[cls_id - 1]
                name   = CDL_CLASS_NAMES.get(cdl_id, f"cls{cls_id}")
                mlflow.log_metric(
                    f"test_iou_{name.lower().replace('/', '_').replace(' ', '_')}",
                    iou,
                )

        # ── Log per-class IoU table to console ───────────────────────────────
        log.info(f"  Test results  mIoU={test_r['miou']:.4f}  OA={test_r['oa']:.4f}")
        log.info(f"  {'Class':<20} {'CDL ID':>6}  {'IoU':>7}")
        log.info(f"  {'-'*38}")
        for cls_id, iou in test_r["per_class_iou"].items():
            cdl_id = KEEP_CLASSES[cls_id - 1]
            name   = CDL_CLASS_NAMES.get(cdl_id, f"cls{cls_id}")
            iou_s  = f"{iou:.4f}" if not np.isnan(iou) else "    nan"
            log.info(f"  {name:<20} {cdl_id:>6}  {iou_s:>7}")
        log.info(f"  {'-'*38}")
        log.info(f"  {'mIoU':<20} {'':>6}  {test_r['miou']:>7.4f}")

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

        # Confusion matrix PNG (skipped when no patch-level test set)
        cm_path = exp_dir / "confusion_matrix.png"
        if "preds" in test_r and "labels" in test_r:
            _plot_confusion_matrix(test_r["preds"], test_r["labels"], str(cm_path))

        # Segmentation map PNG (full-tile inference — skipped in single-year patch-split mode)
        seg_path = None
        if not skip_viz and test_s2_filtered is not None:
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

        # ── Spatial test area evaluation ──────────────────────────────────────
        spatial_results = {}
        log.info(f"  Running spatial test on {len(SPATIAL_TEST_AREAS)} held-out area(s)...")
        for area in SPATIAL_TEST_AREAS:
            area_r = _evaluate_spatial_area(
                model, area, exp_cfg.band_names, exp_name, exp_dir, skip_viz=skip_viz,
            )
            if area_r is not None:
                spatial_results[area["name"]] = area_r

        gdrive_links = upload_models_to_gdrive(
            run_name=f"{exp_name}_{run_timestamp}",
            model_files=[best_ckpt, last_ckpt],
        )
        for fname, link in gdrive_links.items():
            mlflow.set_tag(f"gdrive_{fname}", link)
        mlflow.log_artifact(str(hist_csv))
        mlflow.log_artifact(str(curve_path))
        mlflow.log_artifact(str(iou_csv))
        if cm_path.exists():
            mlflow.log_artifact(str(cm_path))
        if seg_path is not None:
            mlflow.log_artifact(str(seg_path))

        # Training log
        run_log_handler.flush()
        mlflow.log_artifact(str(run_log_path))

        run_id = run.info.run_id

    run_log_handler.close()
    log.removeHandler(run_log_handler)

    summary = {
        "exp_name":      exp_name,
        "arch":          arch,
        "in_channels":   in_channels,
        "best_val_miou": round(best_miou,      4),
        "test_miou":     round(test_r["miou"], 4) if not np.isnan(test_r["miou"]) else float("nan"),
        "test_oa":       round(test_r["oa"],   4) if not np.isnan(test_r["oa"])   else float("nan"),
        "total_epochs":  len(history),
        "run_id":        run_id,
        "ckpt":          str(best_ckpt),
    }
    for aname, ar in spatial_results.items():
        summary[f"{aname}_miou"] = round(ar["miou"], 4)
        summary[f"{aname}_oa"]   = round(ar["oa"],   4)

    spatial_str = "  ".join(
        f"{n}={r['miou']:.4f}" for n, r in spatial_results.items()
    ) if spatial_results else f"test_mIoU={test_r['miou']:.4f}"
    log.info(f"\n✅ {exp_name}  val_mIoU={best_miou:.4f}  {spatial_str}  run={run_id}")
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


def _build_drive_service():
    """Authenticate GDrive API v3 using the OAuth token."""
    import pickle
    from googleapiclient.discovery import build
    from google.auth.transport.requests import Request

    if not GDRIVE_OAUTH_TOKEN.exists():
        raise FileNotFoundError(
            f"OAuth token not found: {GDRIVE_OAUTH_TOKEN}\n"
            "Generate it locally with:\n"
            "  python stages/batch_process_v2.py --auth\n"
            "Then copy to the server via scp."
        )
    with open(GDRIVE_OAUTH_TOKEN, "rb") as f:
        creds = pickle.load(f)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(GDRIVE_OAUTH_TOKEN, "wb") as f:
            pickle.dump(creds, f)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _get_or_create_folder(service, name, parent_id):
    """Return GDrive folder ID for `name` under `parent_id`, creating it if needed."""
    query  = (f"name='{name}' and '{parent_id}' in parents "
              f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
    result = service.files().list(q=query, fields="files(id)").execute()
    if result.get("files"):
        return result["files"][0]["id"]
    meta   = {"name": name, "mimeType": "application/vnd.google-apps.folder",
              "parents": [parent_id]}
    folder = service.files().create(body=meta, fields="id").execute()
    return folder["id"]


def _upload_file_gdrive(service, local_path, folder_id):
    """Upload a single file to a GDrive folder (resumable). Skips if already exists."""
    from googleapiclient.http import MediaFileUpload

    fname  = os.path.basename(local_path)
    query  = f"name='{fname}' and '{folder_id}' in parents and trashed=false"
    result = service.files().list(q=query, fields="files(id)").execute()
    if result.get("files"):
        log.info(f"  GDrive: already exists — {fname}")
        return result["files"][0]["id"]

    size  = os.path.getsize(local_path)
    log.info(f"  GDrive: uploading {fname}  ({size/1e6:.0f} MB)")
    media = MediaFileUpload(local_path, mimetype="application/octet-stream", resumable=True)
    meta  = {"name": fname, "parents": [folder_id]}
    req   = service.files().create(body=meta, media_body=media, fields="id")
    resp  = None
    while resp is None:
        status, resp = req.next_chunk()
        if status:
            log.info(f"    {int(status.progress() * 100)}%")
    log.info(f"  GDrive: uploaded {fname}  (id={resp['id']})")
    return resp["id"]


def upload_models_to_gdrive(run_name, model_files):
    """
    Upload model checkpoint files to GDrive under:
      <GDRIVE_MODELS_FOLDER_ID>/runs/<run_name>/

    Creates the `runs/` and `<run_name>/` folders if they don't exist.
    Returns dict {filename: gdrive_view_link} for MLflow tag logging.
    """
    try:
        service   = _build_drive_service()
        runs_id   = _get_or_create_folder(service, "runs", GDRIVE_MODELS_FOLDER_ID)
        run_id    = _get_or_create_folder(service, run_name, runs_id)
        links = {}
        for path in model_files:
            file_id = _upload_file_gdrive(service, str(path), run_id)
            links[os.path.basename(path)] = f"https://drive.google.com/file/d/{file_id}/view"
        log.info(f"  GDrive upload complete for {run_name}")
        return links
    except Exception as e:
        log.warning(f"  GDrive upload failed ({e}) — models kept locally only")
        return {}


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
    loss_version="v1",
    force=False,
    data_dir=None,
    skip_viz=False,
    top_k=None,
    batch_size=None,
):
    global BATCH_SIZE
    if batch_size:
        BATCH_SIZE = batch_size
        log.info(f"Batch size overridden: {BATCH_SIZE}")

    # Override data directories
    # Use `global` so all module-level functions pick up the new paths at call time.
    if data_dir:
        global S2_TRAIN_DIR, S2_PROCESSED_DIR, CDL_BY_YEAR, CDL_TRAIN, MODELS_DIR, FIGURES_DIR, SPATIAL_TEST_AREAS
        data_dir = Path(data_dir)
        S2_TRAIN_DIR     = data_dir / "s2" / "train"
        S2_PROCESSED_DIR = S2_TRAIN_DIR
        CDL_TRAIN        = data_dir / "cdl" / "cdl_train.tif"
        CDL_BY_YEAR      = {"2024": CDL_TRAIN}
        MODELS_DIR       = data_dir / "models"
        FIGURES_DIR      = data_dir / "figures"
        SPATIAL_TEST_AREAS = [
            {"name": "test_a", "s2_dir": data_dir / "s2" / "test_a", "cdl": data_dir / "cdl" / "cdl_test_a.tif"},
            {"name": "test_b", "s2_dir": data_dir / "s2" / "test_b", "cdl": data_dir / "cdl" / "cdl_test_b.tif"},
        ]
        log.info(f"Data dir overridden to {data_dir}")

    s2_processed = sorted(
        glob(str(S2_TRAIN_DIR / "*_processed.tif")) +
        glob(str(S2_TRAIN_DIR / "S2H_*.tif"))
    )
    seen = set()
    s2_processed = [p for p in s2_processed if not (p in seen or seen.add(p))
                    and not Path(p).name.startswith("._")]
    if not s2_processed:
        raise FileNotFoundError(f"No processed S2 files in {S2_TRAIN_DIR}")

    # ── Validate TIF files — drop corrupt and empty-data files ────────────
    # Cache result to s2_validation_cache.json; invalidated when file set changes.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    MIN_VALID_FRAC_FILE = 0.01
    VALIDATION_WIN      = 512

    _val_cache_path = S2_TRAIN_DIR / "s2_validation_cache.json"
    _val_cache_key  = sorted(Path(p).name for p in s2_processed)

    def _load_validation_cache():
        if not _val_cache_path.exists():
            return None
        try:
            with open(_val_cache_path) as f:
                c = json.load(f)
            if c.get("files_key") == _val_cache_key:
                return c
        except Exception:
            pass
        return None

    _cached = _load_validation_cache()
    if _cached:
        log.info(f"S2 validation cache hit ({len(_cached['valid'])} valid files)")
        valid_s2  = [p for p in s2_processed if Path(p).name in set(_cached["valid"])]
        _corrupt_names = set(_cached.get("corrupt", []))
        _nodata_names  = {r[0] for r in _cached.get("no_data", [])}
        corrupt  = [(p, "") for p in s2_processed if Path(p).name in _corrupt_names]
        no_data  = [(p, r[1]) for p in s2_processed
                    for r in _cached.get("no_data", []) if r[0] == Path(p).name]
    else:
        corrupt  = []
        no_data  = []
        valid_s2 = []

        def _check_file(path):
            try:
                with rasterio.open(path) as src:
                    h, w      = src.height, src.width
                    n_bands   = src.count
                    sz        = min(VALIDATION_WIN, w // 4, h // 4)
                    valid_px, total_px = 0, 0
                    check_bands = sorted({1, n_bands // 2, n_bands})
                    for band in check_bands:
                        for gy in range(3):
                            for gx in range(3):
                                ox = int((gx + 0.5) * w / 3) - sz // 2
                                oy = int((gy + 0.5) * h / 3) - sz // 2
                                ox = max(0, min(ox, w - sz))
                                oy = max(0, min(oy, h - sz))
                                win  = rasterio.windows.Window(ox, oy, sz, sz)
                                data = src.read(band, window=win).astype(np.float32)
                                ok   = (data != S2_NODATA) & np.isfinite(data)
                                valid_px += ok.sum()
                                total_px += ok.size
                return path, valid_px / total_px, None
            except Exception as e:
                return path, 0.0, str(e)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_check_file, p): p for p in s2_processed}
            for fut in as_completed(futures):
                path, frac, err = fut.result()
                if err:
                    corrupt.append((path, err))
                elif frac < MIN_VALID_FRAC_FILE:
                    no_data.append((path, frac))
                else:
                    valid_s2.append(path)

        valid_s2.sort()

        try:
            with open(_val_cache_path, "w") as f:
                json.dump({
                    "files_key": _val_cache_key,
                    "valid":     [Path(p).name for p in valid_s2],
                    "corrupt":   [Path(p).name for p, _ in corrupt],
                    "no_data":   [[Path(p).name, frac] for p, frac in no_data],
                }, f)
            log.info(f"S2 validation cached → {_val_cache_path.name}")
        except Exception as e:
            log.warning(f"Could not write validation cache: {e}")

    if corrupt:
        log.error(f"Found {len(corrupt)} corrupt S2 file(s) — re-download before training:")
        for p, err in corrupt:
            log.error(f"  {Path(p).name}  ({err})")
        raise RuntimeError(
            f"{len(corrupt)} corrupt S2 file(s) detected. "
            "Re-download:  python stages/fetch_data_v2.py --processed --years <year> --overwrite"
        )
    if no_data:
        log.warning(f"Excluding {len(no_data)} file(s) with <{MIN_VALID_FRAC_FILE*100:.0f}% valid pixels (no capture):")
        for p, frac in no_data:
            log.warning(f"  {Path(p).name}  ({frac*100:.2f}% valid)")
    s2_processed = valid_s2
    log.info(f"{len(s2_processed)} S2 files valid for training ({len(no_data)} empty excluded)")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Build local band map (reference year) ──────────────────────────────
    (local_band_names, local_band_to_idx,
     local_date_to_idx, mmdd_to_date) = build_local_band_map(s2_processed)

    # ── Build experiment channel sets ─────────────────────────────────────
    _ref_year_s2  = _s2_for_year(s2_processed, TRAIN_YEARS[0])
    _ref_year_cdl = CDL_TRAIN

    _base_dir = Path(data_dir) if data_dir else PROCESSED_DIR

    # ── Base domain channels (all 9 VEGE_BANDS, no band selection) ─────────
    needs_sd  = not exps or "single_date_gsi" in exps or "single_date_rf" in exps
    needs_nmt = not exps or "naive_mt_gsi" in exps or "naive_mt_rf" in exps

    sd_base_idx = sd_base_names = sd_date_key = None
    nmt_base_idx = nmt_base_names = phenol_map_base = None

    if needs_sd:
        sd_base = build_single_date_indices(
            local_date_to_idx, local_band_to_idx,
            s2_paths=_ref_year_s2, cdl_path=str(_ref_year_cdl),
        )
        sd_base_idx, sd_base_names, sd_date_key = sd_base

    if needs_nmt:
        nmt_base = build_naive_multitemporal_indices(
            local_date_to_idx, local_band_to_idx,
            s2_paths=_ref_year_s2, cdl_path=str(_ref_year_cdl),
        )
        nmt_base_idx, nmt_base_names, phenol_map_base = nmt_base

    # ── single_date_gsi (GSI — scoped to peak date only) ─────────────────
    single_date_idx = single_date_names = single_date_key = None
    if not exps or "single_date_gsi" in exps:
        single_date_idx, single_date_names, single_date_key = build_single_date_selected_indices(
            local_date_to_idx, local_band_to_idx,
            s2_paths=_ref_year_s2, cdl_path=str(_ref_year_cdl),
            top_k=top_k, force=force,
            best_date=sd_date_key,   # reuse peak date from build_single_date_indices
        )

    # ── single_date (RF — scoped to peak date only) ───────────────────────
    single_date_rf_idx = single_date_rf_names = None
    if not exps or "single_date_rf" in exps:
        rf_sd_json = _base_dir / "rf_band_single_date.json"
        # sd_date_key guaranteed set when needs_sd is True (single_date_rf implies it)
        sd_peak_file = _ref_year_s2[local_date_to_idx[sd_date_key]]
        if not force and rf_sd_json.exists():
            log.info(f"rf_band single_date: cached → {rf_sd_json.name}")
        else:
            save_rf_band_json(
                run_rf_band_only([sd_peak_file], str(_ref_year_cdl), sd_base_names),
                rf_sd_json,
            )
        single_date_rf_idx, single_date_rf_names, _ = build_single_date_selected_indices(
            local_date_to_idx, local_band_to_idx,
            s2_paths=_ref_year_s2, cdl_path=str(_ref_year_cdl),
            candidates_json=rf_sd_json, top_k=top_k,
            best_date=sd_date_key,
        )

    # ── naive_mt_gsi (GSI — scoped to 4 phenol dates only) ──────────────
    naive_mt_idx = naive_mt_names = phenol_map = None
    if not exps or "naive_mt_gsi" in exps:
        naive_mt_idx, naive_mt_names, phenol_map = build_naive_multitemporal_selected_indices(
            local_date_to_idx, local_band_to_idx,
            s2_paths=_ref_year_s2, cdl_path=str(_ref_year_cdl),
            top_k=top_k, force=force,
        )

    # ── naive_mt_rf (RF — scoped to 4 phenol dates only) ────────────────
    naive_mt_rf_idx = naive_mt_rf_names = None
    if not exps or "naive_mt_rf" in exps:
        rf_nmt_json = _base_dir / "rf_band_naive_mt.json"
        # phenol_map_base guaranteed set when needs_nmt is True
        nmt_phenol_files = [_ref_year_s2[local_date_to_idx[d]] for d in phenol_map_base.values()]
        if not force and rf_nmt_json.exists():
            log.info(f"rf_band naive_mt: cached → {rf_nmt_json.name}")
        else:
            save_rf_band_json(
                run_rf_band_only(nmt_phenol_files, str(_ref_year_cdl), nmt_base_names),
                rf_nmt_json,
            )
        naive_mt_rf_idx, naive_mt_rf_names, _ = build_naive_multitemporal_selected_indices(
            local_date_to_idx, local_band_to_idx,
            s2_paths=_ref_year_s2, cdl_path=str(_ref_year_cdl),
            candidates_json=rf_nmt_json, top_k=top_k,
            phenol_map=phenol_map_base,
        )

    def _find_direct_json(selector: str) -> Path:
        """Return JSON path for a direct selector; falls back to largest k if exact k missing."""
        base = Path(data_dir) if data_dir else SELECT_GSI_DIRECT_JSON.parent
        if top_k:
            exact = base / f"select_{selector}_k{top_k}.json"
            if exact.exists():
                return exact
            candidates = sorted(base.glob(f"select_{selector}_k*.json"))
            if candidates:
                log.info(f"  {selector}: k={top_k} JSON not found, using {candidates[-1].name} with subset_k")
                return candidates[-1]
        return base / f"select_{selector}_k{SELECT_TOP_K_PER_CROP}.json"

    gsi_idx = gsi_names = None
    if not exps or "gsi" in exps:
        gsi_json = _find_direct_json("gsi_direct")
        gsi_idx, gsi_names = build_direct_indices(
            gsi_json, mmdd_to_date, local_band_to_idx,
            selector_name="gsi", subset_k=top_k,
        )
        log.info(f"gsi (k={top_k or 'all'}): {len(gsi_idx)} channels")

    rf_idx = rf_names = None
    if not exps or "rf" in exps:
        rf_json = _find_direct_json("rf_direct")
        rf_idx, rf_names = build_direct_indices(
            rf_json, mmdd_to_date, local_band_to_idx,
            selector_name="rf", subset_k=top_k,
        )
        log.info(f"rf (k={top_k or 'all'}): {len(rf_idx)} channels")

    # ── Class weights ──────────────────────────────────────────────────────
    cw_tensor = compute_class_weights()
    log.info("Class weights computed")

    # ── Build experiment registry & plan ───────────────────────────────────
    all_archs = list(ARCH_CFG.keys())
    run_exps  = exps  or ["single_date_gsi", "single_date_rf", "naive_mt_gsi", "naive_mt_rf", "gsi", "rf"]
    run_archs = archs or all_archs

    registry = build_registry(
        single_date_idx=single_date_idx,           single_date_names=single_date_names,           single_date_key=sd_date_key,
        single_date_rf_idx=single_date_rf_idx,     single_date_rf_names=single_date_rf_names,
        naive_mt_idx=naive_mt_idx,                 naive_mt_names=naive_mt_names,                 phenol_map=phenol_map_base,
        naive_mt_rf_idx=naive_mt_rf_idx,           naive_mt_rf_names=naive_mt_rf_names,
        gsi_idx=gsi_idx,     gsi_names=gsi_names,
        rf_idx=rf_idx,       rf_names=rf_names,
    )

    expanded_exps = expand_exp_keys(run_exps, registry)
    log.info(f"Selected experiments: {expanded_exps}")

    plan = []
    for exp_key in expanded_exps:
        cfg = registry.get(exp_key)
        if cfg is None:
            log.warning(f"Experiment '{exp_key}' not in registry — skipping")
            continue
        if cfg.band_indices is None:
            raise RuntimeError(
                f"Exp {exp_key}: band indices are None — required feature-selection output is missing."
            )
        for arch in run_archs:
            plan.append((exp_key, arch, cfg.band_indices, cfg.band_names,
                         f"{cfg.description}, {arch}", cfg.extra_kw))

    log.info(f"Planned {len(plan)} run(s): {[(e, a) for e, a, *_ in plan]}")

    # ── MLflow setup ────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # ── Run experiments — one top-level run per exp_key, nested run per arch ─
    all_results = []
    exp_groups: dict = {}
    for exp_key, arch, band_idx, band_names, description, extra_kw in plan:
        exp_groups.setdefault(exp_key, []).append((arch, band_idx, band_names, description, extra_kw))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for exp_key, arch_runs in exp_groups.items():
        cfg_entry = registry[exp_key]
        mlflow.set_experiment(cfg_entry.mlflow_experiment)
        n_ch = len(arch_runs[0][1]) if arch_runs[0][1] else 0
        parent_run_name = f"exp_{exp_key}_k{top_k}_{timestamp}" if top_k else f"exp_{exp_key}_{timestamp}"
        with mlflow.start_run(run_name=parent_run_name) as parent_run:
            mlflow.log_params({
                "experiment":   f"exp_{exp_key}",
                "n_channels":   n_ch,
                "train_years":  str(TRAIN_YEARS),
                "test_year":    TEST_YEAR,
                "description":  cfg_entry.description,
                "loss_version": loss_version,
                **({"top_k": top_k} if top_k else {}),
            })
            log.info(f"Parent MLflow run: {parent_run_name}  (id={parent_run.info.run_id})")
            for arch, band_idx, band_names, description, extra_kw in arch_runs:
                exp_name = f"exp_{exp_key}_k{top_k}_{arch}" if top_k else f"exp_{exp_key}_{arch}"
                result = run_experiment(
                    exp_name=exp_name,
                    arch=arch,
                    band_indices=band_idx,
                    band_names_list=band_names,
                    description=description,
                    s2_processed=s2_processed,
                    class_weights_tensor=cw_tensor,
                    loss_version=loss_version,
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


def _upload_existing_models(filter_exps=None, filter_archs=None):
    """Upload best_model.pth + last_model.pth for all existing run dirs.

    Scans MODELS_DIR for subdirectories that contain at least one of the two
    checkpoint files and uploads them to GDrive under runs/<run_dir_name>/.

    filter_exps  — optional list of exp shorthand keys (e.g. ["C_v3", "A_v2"]).
                   Run dir must contain any of the keys as a substring.
    filter_archs — optional list of arch names to further filter.
    """
    import re as _re

    def _matches(run_dir_name):
        if filter_exps:
            if not any(
                _re.search(r"(?i)" + _re.escape(e.lower()), run_dir_name.lower())
                for e in filter_exps
            ):
                return False
        if filter_archs:
            if not any(arch.lower() in run_dir_name.lower() for arch in filter_archs):
                return False
        return True

    candidates = sorted(MODELS_DIR.iterdir()) if MODELS_DIR.exists() else []
    run_dirs = [
        d for d in candidates
        if d.is_dir() and _matches(d.name)
        and (
            (d / "best_model.pth").exists()
            or (d / "last_model.pth").exists()
        )
    ]

    if not run_dirs:
        log.warning("No matching run dirs with model checkpoints found under %s", MODELS_DIR)
        return

    log.info("Uploading models for %d run(s)…", len(run_dirs))
    for run_dir in run_dirs:
        model_files = [
            f for f in [run_dir / "best_model.pth", run_dir / "last_model.pth"]
            if f.exists()
        ]
        log.info("  %s: %s", run_dir.name, [f.name for f in model_files])
        links = upload_models_to_gdrive(run_name=run_dir.name, model_files=model_files)
        if links:
            for fname, link in links.items():
                log.info("    %s → %s", fname, link)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train segmentation models for band selection comparison")
    parser.add_argument(
        "--exp", nargs="+",
        choices=["single_date_gsi", "single_date_rf", "naive_mt_gsi", "naive_mt_rf", "gsi", "rf"],
        default=["single_date_gsi", "single_date_rf", "naive_mt_gsi", "naive_mt_rf", "gsi", "rf"],
        help=(
            "Experiments to run (default: all six). "
            "single_date_gsi=peak NDVI + GSI bands, single_date_rf=peak NDVI + RF bands, "
            "naive_mt_gsi=4 phenol dates + GSI bands, naive_mt_rf=4 phenol dates + RF bands, "
            "gsi=GSI-direct, rf=RF-direct."
        ),
    )
    parser.add_argument(
        "--arch", nargs="+", choices=list(ARCH_CFG.keys()),
        default=None,
        help="Which architectures to run (default: all)",
    )
    parser.add_argument(
        "--loss-version", choices=["v1", "v2"], default="v1",
        help="Loss function version: v1=WeightedCrossEntropy (default), v2=PhenologyAwareLoss",
    )
    parser.add_argument("--force",    action="store_true", help="Re-run even if checkpoint exists")
    parser.add_argument("--skip-viz", action="store_true", help="Skip full-image visualization")
    parser.add_argument("--data-dir", default=None, help="Override data/processed directory")
    parser.add_argument("--shutdown", action="store_true", help="Stop the RunPod pod after training")
    parser.add_argument(
        "--upload-existing", action="store_true",
        help=(
            "Upload best_model.pth and last_model.pth for all existing run dirs under "
            "MODELS_DIR to Google Drive without re-training. "
            "Optionally filter with --exp / --arch."
        ),
    )
    parser.add_argument(
        "--top-k", type=int, nargs="+", default=None, metavar="K",
        help="Top-K value(s) to sweep (loads select_gsi/rf_direct_k{K}.json per k). E.g. --top-k 5 10 15 20 30",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, metavar="N",
        help=f"Override BATCH_SIZE from config (default: {BATCH_SIZE}).",
    )
    parser.add_argument(
        "--eval-only", metavar="CKPT_PATH",
        help="Skip training — load checkpoint and run spatial test evaluation only.",
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

    if args.upload_existing:
        _upload_existing_models(filter_exps=args.exp, filter_archs=args.arch)
        sys.exit(0)

    if args.eval_only:
        ckpt_path = Path(args.eval_only)
        if not ckpt_path.exists():
            log.error(f"Checkpoint not found: {ckpt_path}")
            sys.exit(1)
        if args.data_dir:
            global S2_TRAIN_DIR, S2_PROCESSED_DIR, CDL_TRAIN, MODELS_DIR, FIGURES_DIR, SPATIAL_TEST_AREAS
            _dd = Path(args.data_dir)
            S2_TRAIN_DIR     = _dd / "s2" / "train"
            S2_PROCESSED_DIR = S2_TRAIN_DIR
            CDL_TRAIN        = _dd / "cdl" / "cdl_train.tif"
            MODELS_DIR       = _dd / "models"
            FIGURES_DIR      = _dd / "figures"
            SPATIAL_TEST_AREAS = [
                {"name": "test_a", "s2_dir": _dd / "s2" / "test_a", "cdl": _dd / "cdl" / "cdl_test_a.tif"},
                {"name": "test_b", "s2_dir": _dd / "s2" / "test_b", "cdl": _dd / "cdl" / "cdl_test_b.tif"},
            ]
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        arch = ckpt.get("architecture", (args.arch or ["segformer"])[0])
        in_ch = ckpt["in_channels"]
        band_names = ckpt.get("band_names", [])
        exp_name = ckpt_path.parent.name
        exp_dir  = ckpt_path.parent
        model = build_model(arch, in_ch, NUM_CLASSES)
        model.load_state_dict(ckpt["model_state_dict"])
        log.info(f"Loaded checkpoint: {ckpt_path}  arch={arch}  in_ch={in_ch}")
        with mlflow.start_run(run_name=f"{exp_name}_eval"):
            mlflow.set_tag("eval_only", "true")
            mlflow.set_tag("checkpoint", str(ckpt_path))
            for area in SPATIAL_TEST_AREAS:
                _evaluate_spatial_area(model, area, band_names, exp_name, exp_dir)
        sys.exit(0)

    top_k_list = args.top_k or [None]
    for k in top_k_list:
        if k is not None:
            log.info(f"{'='*65}")
            log.info(f"  Top-K sweep: k={k}")
            log.info(f"{'='*65}")
        main(
            exps=args.exp,
            archs=args.arch,
            loss_version=args.loss_version,
            force=args.force,
            data_dir=args.data_dir,
            skip_viz=args.skip_viz,
            top_k=k,
            batch_size=args.batch_size,
        )

    if args.shutdown:
        import urllib.request, urllib.error, json as _json, time as _time
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent / ".env")
        pod_id  = os.environ.get("RUNPOD_POD_ID")
        api_key = os.environ.get("RUNPOD_API_KEY")
        delay   = 5   # minutes
        if pod_id and api_key:
            log.warning(f"RunPod pod {pod_id} will stop in {delay} minutes.")
            _time.sleep(delay * 60)
            query = f'{{"query": "mutation {{ podStop(input: {{podId: \\"{pod_id}\\"}}) {{ id desiredStatus }} }}"}}'
            req   = urllib.request.Request(
                "https://api.runpod.io/graphql",
                data    = query.encode(),
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            )
            try:
                with urllib.request.urlopen(req) as resp:
                    log.info(f"Pod stop response: {_json.loads(resp.read())}")
            except urllib.error.URLError as e:
                log.error(f"Failed to stop pod: {e}")
        else:
            log.warning(f"RUNPOD_POD_ID/RUNPOD_API_KEY not set — falling back to sudo shutdown in {delay} min")
            import subprocess
            subprocess.run(["sudo", "shutdown", "-h", f"+{delay}"], check=False)
