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

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from crop_mapping_pipeline.config import (
    S2_PROCESSED_DIR, CDL_BY_YEAR, MODELS_DIR, FIGURES_DIR, LOGS_DIR,
    S2_BAND_NAMES, N_BANDS_PER_DATE, VEGE_BANDS,
    KEEP_CLASSES, CLASS_REMAP, NUM_CLASSES, CDL_CLASS_NAMES,
    REMAP_LUT, S2_NODATA,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_TRAIN,
    TRAIN_YEARS, TEST_YEAR,
    PATCH_SIZE, STRIDE, MIN_VALID_FRAC, BATCH_SIZE, MAX_EPOCHS, EARLY_STOP,
    VAL_FRAC, SEED, ARCH_CFG,
    STAGE3_EXP_C_BANDS,
)
from geoai.geoai.train import RasterPatchDataset, train_semantic_one_epoch
from geoai.geoai.utils.device import get_device
from crop_mapping_pipeline.models import DeepLabV3PlusCBAM, build_segformer

log = logging.getLogger(__name__)
DEVICE = get_device()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _s2_for_year(s2_processed, yr):
    return sorted([p for p in s2_processed if Path(p).name.split("_")[1] == yr])


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
    """4 phenological dates × 9 vegetation bands = up to 36 channels."""
    available_dates = sorted(local_date_to_idx.keys())
    phenol_targets  = {"Jan": [1], "Mar": [3], "Jul": [7, 8], "Nov": [11, 10]}
    phenol_map      = {}
    for label, months in phenol_targets.items():
        match = next(
            (d for d in available_dates if int(d[4:6]) in months), None
        )
        if match:
            phenol_map[label] = match

    if len(phenol_map) < 2:
        phenol_map = {f"D{i}": d for i, d in enumerate(available_dates)}

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


def build_exp_C_indices(mmdd_to_date, local_band_to_idx):
    """
    Load stage3_exp_c_bands.txt (written by feature_analysis.py / v2 notebook).
    Each line: '<BAND>_<YYYYMMDD>' or '<BAND>_<MMDD>'.
    Maps MMDD to local reference-year date, then to flat local index.
    """
    if not STAGE3_EXP_C_BANDS.exists():
        log.warning(f"Exp C file not found: {STAGE3_EXP_C_BANDS}  — Exp C will be skipped")
        return None, None

    with open(STAGE3_EXP_C_BANDS) as f:
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
        log.warning("Exp C: no bands matched current processed files — skipping")
        return None, None

    if skipped:
        log.warning(f"Exp C: {skipped} band(s) from {STAGE3_EXP_C_BANDS.name} could not be matched")

    log.info(f"Exp C: {len(exp_C_idx)} channels from {STAGE3_EXP_C_BANDS.name}")
    return exp_C_idx, exp_C_names


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
    return {"loss": total_loss / len(loader), "miou": miou, "oa": oa}


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
    band_indices,
    band_names_list,
    description,
    s2_processed,
    class_weights_tensor,
    force=False,
    skip_viz=False,
):
    cfg        = ARCH_CFG[arch]
    exp_dir    = MODELS_DIR / exp_name
    best_ckpt  = exp_dir / "best_model.pth"
    last_ckpt  = exp_dir / "last_model.pth"
    exp_dir.mkdir(parents=True, exist_ok=True)

    if not force and best_ckpt.exists():
        log.info(f"Checkpoint exists — skipping {exp_name}  (use --force to re-run)")
        return None

    in_channels = len(band_indices)
    log.info(f"\n{'='*65}")
    log.info(f" {exp_name}")
    log.info(f"  arch={arch}  in_channels={in_channels}")
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
        ds = RasterPatchDataset(
            s2_paths=yr_s2, cdl_path=str(yr_cdl),
            patch_size=PATCH_SIZE, stride=STRIDE,
            keep_classes=KEEP_CLASSES, remap_lut=REMAP_LUT,
            min_valid_frac=MIN_VALID_FRAC, band_indices=band_indices,
        )
        train_year_datasets.append(ds)
        log.info(f"  [{yr}] {len(ds):,} patches")

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
    test_ds = RasterPatchDataset(
        s2_paths=test_s2, cdl_path=str(test_cdl),
        patch_size=PATCH_SIZE, stride=STRIDE,
        keep_classes=KEEP_CLASSES, remap_lut=REMAP_LUT,
        min_valid_frac=MIN_VALID_FRAC, band_indices=band_indices,
    )

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
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
                model, test_s2, band_indices, patch_size=PATCH_SIZE, stride=PATCH_SIZE
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
    arrays, profile = [], None
    for p in s2_paths:
        with rasterio.open(p) as src:
            arr = src.read().astype(np.float32)
            if profile is None:
                profile = dict(src.profile)
        arr[arr == S2_NODATA] = 0.0
        arrays.append(arr)

    stacked  = np.concatenate(arrays, axis=0)
    selected = stacked[band_indices]
    K, H, W  = selected.shape
    del stacked

    pred_map = np.zeros((H, W), dtype=np.uint8)
    n_rows   = (H + stride - 1) // stride
    n_cols   = (W + stride - 1) // stride
    total    = n_rows * n_cols

    model.eval()
    done = 0
    with torch.no_grad():
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                ph    = min(patch_size, H - y)
                pw    = min(patch_size, W - x)
                patch = selected[:, y:y + ph, x:x + pw]
                if ph < patch_size or pw < patch_size:
                    padded = np.zeros((K, patch_size, patch_size), dtype=np.float32)
                    padded[:, :ph, :pw] = patch
                    patch = padded
                t              = torch.from_numpy(patch).unsqueeze(0).to(DEVICE)
                out            = model(t).argmax(dim=1).squeeze().cpu().numpy()
                pred_map[y:y + ph, x:x + pw] = out[:ph, :pw]
                done          += 1
                if done % 200 == 0 or done == total:
                    log.info(f"  {done}/{total} tiles")

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

    s2_processed = sorted(glob(str(S2_PROCESSED_DIR / "*_processed.tif")))
    if not s2_processed:
        raise FileNotFoundError(f"No processed S2 files in {S2_PROCESSED_DIR}")

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
    exp_C_idx, exp_C_names             = build_exp_C_indices(
        mmdd_to_date, local_band_to_idx
    )

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
            log.warning(f"Exp {exp_key}: no band indices — skipping all architectures")
            continue

        for arch in run_archs:
            plan.append((exp_key, arch, idx, names, desc_fn(arch)))

    log.info(f"Planned {len(plan)} run(s): {[(e, a) for e, a, *_ in plan]}")

    # ── Run experiments ────────────────────────────────────────────────────
    all_results = []
    for exp_key, arch, band_idx, band_names, description in plan:
        exp_name = f"exp_{exp_key}_{arch}"
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
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
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
    log.info(f"Device: {DEVICE}  PyTorch: {torch.__version__}")

    main(
        exps=args.exp,
        archs=args.arch,
        force=args.force,
        data_dir=args.data_dir,
        skip_viz=args.skip_viz,
    )
