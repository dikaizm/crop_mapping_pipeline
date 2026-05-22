"""
Pipeline configuration — edit GDrive folder IDs and paths before running.
All path settings can be overridden via --data-dir in pipeline.py.
"""

import numpy as np
from pathlib import Path

# ── Project root ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent   # crop_mapping_pipeline/

# ── Data paths ─────────────────────────────────────────────────────────────────
PROCESSED_DIR    = PROJECT_ROOT / "data" / "processed_v2"
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
STAGE2_RESULTS_CSV          = PROCESSED_DIR / "stage2v2_per_crop_results.csv"
STAGE3_EXP_C_BANDS          = PROCESSED_DIR / "stage3_exp_c_bands.txt"
STAGE3_EXP_C_BANDS_PROJECTED = PROCESSED_DIR / "stage3_exp_c_bands_projected.json"

# ── S2 metadata ────────────────────────────────────────────────────────────────
S2_BAND_NAMES    = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
N_BANDS_PER_DATE = len(S2_BAND_NAMES)
S2_NODATA        = -9999.0
# 9 vegetation bands used for Exp A and B (excludes coastal B1 and redundant B8A)
VEGE_BANDS       = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12"]

# ── CDL classes ────────────────────────────────────────────────────────────────
# Fallow/Idle Cropland (61) → background (class 0); not in KEEP_CLASSES
KEEP_CLASSES = [3, 6, 24, 36, 37, 54, 69, 75, 76, 210]
CLASS_REMAP  = {cls_id: i + 1 for i, cls_id in enumerate(KEEP_CLASSES)}
NUM_CLASSES  = len(KEEP_CLASSES) + 1   # 11: 0=bg + 1–10=crops

CDL_CLASS_NAMES = {
    3:   "Rice",         6:   "Sunflower",    24:  "Winter Wheat",
    36:  "Alfalfa",      37:  "Other Hay",    54:  "Tomatoes",
    69:  "Grapes",       75:  "Almonds",      76:  "Walnuts",
    210: "Prunes",
}

REMAP_LUT = np.zeros(256, dtype=np.int64)
for _cdl_id, _model_id in CLASS_REMAP.items():
    if _cdl_id < 256:
        REMAP_LUT[_cdl_id] = _model_id

# ── Google Drive upload (processed files → GDrive) ────────────────────────────
# Used by process_data.py after local processing.
# GDRIVE_CREDENTIALS: path to a Google service-account JSON key file.
#   Create one at: console.cloud.google.com → IAM → Service Accounts → Keys
#   Share the target GDrive folders with the service-account email.
GDRIVE_CREDENTIALS  = Path(__file__).parent / "ssh" / "gdrive_service_account.json"
GDRIVE_OAUTH_SECRET = Path(__file__).parent / "ssh" / next(
    (f.name for f in (Path(__file__).parent / "ssh").glob("client_secret_*.json")),
    "client_secret.json",
)
GDRIVE_OAUTH_TOKEN  = Path(__file__).parent / "ssh" / "gdrive_token.pickle"
GDRIVE_RAW_S2_V2_FOLDER_ID = "1yZmKDjGnXZH6622d8SU4GDUB1z940HwY"

GDRIVE_PROCESSED_S2_FOLDER_IDS = {
    "2022": "1l1nI9nn4WIfyoYyAUTewrVPooCTrZ6Bx",
    "2023": "1FQaW3NZhuFNl01JGg6yHURFnIL7aYLUL",
    "2024": "1jP7Oz5mPjkTpCxvJOFnN7Ml0kfWkR5m6",
}
GDRIVE_PROCESSED_CDL_FOLDER_ID    = "17a-EkYGDBDluhqTXvQin4UlxMT4X5xt0"
# V2 study area processed data — single parent folder; year subfolders created automatically
GDRIVE_PROCESSED_V2_FOLDER_ID     = "1RepvRly_kh4z54Jum-3F_RBzxsw3wxcS"
GDRIVE_MODELS_FOLDER_ID        = "1Xj8i32ndkc_9wa7LCwBO8LhrpxsZ3jot"

# ── MLflow ─────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI        = "https://mlflow-geoai.stelarea.com"
MLFLOW_EXPERIMENT_PIPELINE = "cropmap_pipeline_runs"
MLFLOW_EXPERIMENT_DATASET  = "cropmap_data_processing"
MLFLOW_EXPERIMENT_FEATURE  = "cropmap_feature_selection_s2"
MLFLOW_EXPERIMENT_TRAIN    = "cropmap_segmentation_s2"
MLFLOW_EXPERIMENT_TRAIN_V2 = "cropmap_segmentation_s2_v2"
MLFLOW_EXPERIMENT_TRAIN_V3 = "cropmap_segmentation_s2_v3"

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
S2_DELTA      = 0.001      # min IoU(class 1) gain to accept a band
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
EARLY_STOP     = 20
EARLY_STOP_DELTA = 0.001   # min mIoU improvement to reset patience
VAL_FRAC       = 0.15
SEED           = 42

ARCH_CFG = {
    "deeplabv3plus_cbam": {"lr": 1e-4, "weight_decay": 1e-4, "encoder": "resnet50"},
    "segformer":          {"lr": 6e-5, "weight_decay": 1e-2, "encoder": "mit_b2"},
}

# ── Stage 1v3 / 2v2 hyperparameters (Feature Analysis v2) ──────────────────
# Top-K selection: always keep the top K dates/bands per crop by mean SI_global.
# No threshold — avoids empty results and removes the need for manual tuning.
TOP_DATES_PER_CROP   = 10    # top dates per crop (Stage 1 → Stage 2 candidates)
TOP_BANDS_PER_CROP   = 9     # top bands per crop (= len(VEGE_BANDS), all bands as candidates)
# Aliases used in code
MAX_DATES_PER_CROP   = TOP_DATES_PER_CROP
MAX_BANDS_PER_CROP   = TOP_BANDS_PER_CROP
S2_DATE_DELTA        = 0.010  # min IoU gain to accept a new date
S2_DATE_NO_IMPROVE   = 4      # consecutive rejections before stopping date selection
S2_MAX_DATES         = 8      # max dates selected per crop
S2_BAND_DELTA        = 0.005  # min IoU gain to accept a new spectral band
S2_BAND_NO_IMPROVE   = 4      # consecutive rejections before stopping band selection
S2_MAX_BANDS_V2      = 9      # max bands selected per crop

STAGE1V3_CANDIDATES_JSON = PROCESSED_DIR / "s2" / "2022" / "stage1v3_candidates.json"
STAGE1V2_CANDIDATES_JSON = PROCESSED_DIR / "s2" / "2022" / "stage1v2_candidates.json"
STAGE2V3_PER_CROP_JSON   = PROCESSED_DIR / "stage2v3_per_crop_results.json"
STAGE3_EXP_C_V2_JSON     = PROCESSED_DIR / "stage3_exp_c_v2.json"
STAGE3_EXP_C_V2_BANDS    = PROCESSED_DIR / "stage3_exp_c_v2_bands.txt"
STAGE3_EXP_D_JSON        = PROCESSED_DIR / "stage3_exp_d.json"
STAGE3_EXP_D_BANDS       = PROCESSED_DIR / "stage3_exp_d_bands.txt"
STAGE2V3_RF_PER_CROP_JSON  = PROCESSED_DIR / "stage2v3_rf_per_crop_results.json"
STAGE3_EXP_C_V2_RF_JSON  = PROCESSED_DIR / "stage3_exp_c_v2_rf.json"
STAGE3_EXP_C_V2_RF_BANDS = PROCESSED_DIR / "stage3_exp_c_v2_rf_bands.txt"
STAGE2V3_SWEEP_PER_CROP_JSON = PROCESSED_DIR / "stage2v3_sweep_per_crop_results.json"
STAGE3_EXP_C_V3_JSON         = PROCESSED_DIR / "stage3_exp_c_v3.json"
STAGE3_EXP_C_V3_BANDS        = PROCESSED_DIR / "stage3_exp_c_v3_bands.txt"

# ── Direct single-stage selectors (no Stage 1 prefilter, no CNN oracle) ────
# GSI-direct: rank all (date × band) channels by per-crop SI_global, take top-K union
# RF-direct:  rank all (date × band) channels by per-crop RF importance, take top-K union
SELECT_TOP_K_PER_CROP    = 20   # channels selected per crop before union
SELECT_GSI_DIRECT_JSON   = PROCESSED_DIR / "select_gsi_direct.json"
SELECT_GSI_DIRECT_BANDS  = PROCESSED_DIR / "select_gsi_direct_bands.txt"
SELECT_RF_DIRECT_JSON    = PROCESSED_DIR / "select_rf_direct.json"
SELECT_RF_DIRECT_BANDS   = PROCESSED_DIR / "select_rf_direct_bands.txt"

# ── RF selector hyperparameters (Stage 2v2-RF and RF-direct) ───────────────
RF_N_ESTIMATORS       = 200    # trees in the binary RF oracle
RF_MAX_PIXELS         = 50_000 # pixel sample cap (crop + rest) to keep RF fast
RF_IMPORTANCE_THRESH  = 0.10   # keep dates/bands with importance >= 10% of max
