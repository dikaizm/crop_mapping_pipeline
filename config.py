"""
Pipeline configuration — edit GDRIVE_FILES IDs and paths before running.
All path settings can be overridden via --data-dir in pipeline.py.
"""

import numpy as np
from pathlib import Path

# ── Project root ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent   # research-crop-mapping-geoai/

# ── Data paths ─────────────────────────────────────────────────────────────────
PROCESSED_DIR    = PROJECT_ROOT / "data" / "processed"
S2_PROCESSED_DIR = PROCESSED_DIR / "s2"
CDL_DIR          = PROCESSED_DIR / "cdl"
MODELS_DIR       = PROJECT_ROOT / "ml_models"
FIGURES_DIR      = PROJECT_ROOT / "documents" / "thesis" / "figures"
LOGS_DIR         = PROJECT_ROOT / "logs"

CDL_BY_YEAR = {
    "2022": CDL_DIR / "cdl_2022_study_area_filtered.tif",
    "2023": CDL_DIR / "cdl_2023_study_area_filtered.tif",
    "2024": CDL_DIR / "cdl_2024_study_area_filtered.tif",
}

# Stage 2 + Stage 3 handoff files
STAGE2_RESULTS_CSV = PROCESSED_DIR / "stage2v2_per_crop_results.csv"
STAGE3_EXP_C_BANDS = PROCESSED_DIR / "stage3_exp_c_bands.txt"

# ── Google Drive file IDs ──────────────────────────────────────────────────────
# Fill in before running fetch_data.py.
#   type="folder" → gdown.download_folder (e.g. GEE export folder per year)
#   type="file"   → gdown.download         (single file)
#
# Example (folder — all S2 files in one GDrive folder):
#   "s2": {"type": "folder", "id": "1ABC...", "output_dir": str(S2_PROCESSED_DIR)}
# Example (file):
#   "cdl_2022": {"type": "file", "id": "1DEF...", "output_path": str(CDL_BY_YEAR["2022"])}

GDRIVE_FILES = {
    "s2": {
        "type":       "folder",
        "id":         "",          # <-- GDrive folder ID for all years' S2 processed TIFs
        "output_dir": str(S2_PROCESSED_DIR),
    },
    "cdl_2022": {
        "type":        "file",
        "id":          "",         # <-- GDrive file ID for cdl_2022_study_area_filtered.tif
        "output_path": str(CDL_BY_YEAR["2022"]),
    },
    "cdl_2023": {
        "type":        "file",
        "id":          "",
        "output_path": str(CDL_BY_YEAR["2023"]),
    },
    "cdl_2024": {
        "type":        "file",
        "id":          "",
        "output_path": str(CDL_BY_YEAR["2024"]),
    },
}

# ── S2 metadata ────────────────────────────────────────────────────────────────
S2_BAND_NAMES    = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
N_BANDS_PER_DATE = len(S2_BAND_NAMES)
S2_NODATA        = -9999.0
# 9 vegetation bands used for Exp A and B (excludes coastal B1 and redundant B8A)
VEGE_BANDS       = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12"]

# ── CDL classes ────────────────────────────────────────────────────────────────
# Fallow/Idle Cropland (61) → background (class 0); not in KEEP_CLASSES
KEEP_CLASSES = [3, 6, 24, 36, 37, 54, 69, 75, 76, 220]
CLASS_REMAP  = {cls_id: i + 1 for i, cls_id in enumerate(KEEP_CLASSES)}
NUM_CLASSES  = len(KEEP_CLASSES) + 1   # 11: 0=bg + 1–10=crops

CDL_CLASS_NAMES = {
    3:   "Rice",         6:   "Sunflower",    24:  "Winter Wheat",
    36:  "Alfalfa",      37:  "Other Hay",    54:  "Tomatoes",
    69:  "Grapes",       75:  "Almonds",      76:  "Walnuts",
    220: "Plums",
}

REMAP_LUT = np.zeros(256, dtype=np.int64)
for _cdl_id, _model_id in CLASS_REMAP.items():
    if _cdl_id < 256:
        REMAP_LUT[_cdl_id] = _model_id

# ── MLflow ─────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI        = "https://mlflow-geoai.stelarea.com"
MLFLOW_EXPERIMENT_PIPELINE = "cropmap_pipeline_runs"
MLFLOW_EXPERIMENT_FEATURE  = "cropmap_feature_analysis_s2"
MLFLOW_EXPERIMENT_TRAIN    = "cropmap_segmentation_s2"

# ── Stage 1 hyperparameters ────────────────────────────────────────────────────
SAMPLE_FRACTION = 0.05   # 5 % of labeled crop pixels for GSI computation
TOP_K_PER_CROP  = 20     # candidates per crop from per-crop SIglobal ranking

# ── Stage 2 hyperparameters ────────────────────────────────────────────────────
S2_ENCODER    = "resnet18"
S2_PATCH_SIZE = 128        # 128 px → more patches → lower IoU variance per step
S2_STRIDE     = 128
S2_MIN_VALID  = 0.3        # min fraction of crop pixels per patch
S2_EPOCHS     = 15
S2_PATIENCE   = 5          # early stopping within each step
S2_DELTA      = 0.005      # min IoU(class 1) gain to accept a band
S2_NO_IMPROVE = 5          # consecutive rejections before stopping per crop
S2_MAX_BANDS  = 20         # max bands selected per crop
S2_BATCH_SIZE = 8

# ── Stage 3 hyperparameters ────────────────────────────────────────────────────
TRAIN_YEARS    = ["2022", "2023"]
TEST_YEAR      = "2024"
PATCH_SIZE     = 256
STRIDE         = 256
MIN_VALID_FRAC = 0.3
BATCH_SIZE     = 8
MAX_EPOCHS     = 100
EARLY_STOP     = 10
VAL_FRAC       = 0.15
SEED           = 42

ARCH_CFG = {
    "deeplabv3plus_cbam": {"lr": 1e-4, "weight_decay": 1e-4, "encoder": "resnet50"},
    "segformer":          {"lr": 6e-5, "weight_decay": 1e-2, "encoder": "mit_b2"},
}
