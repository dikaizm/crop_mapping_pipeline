"""
Pipeline configuration — edit GDrive folder IDs and paths before running.
All path settings can be overridden via --data-dir in pipeline.py.
"""

import numpy as np
from pathlib import Path

# ── Project root ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent   # crop_mapping_pipeline/

# ── Data paths ─────────────────────────────────────────────────────────────────
PROCESSED_DIR    = PROJECT_ROOT / "data" / "processed"
S2_PROCESSED_DIR = PROCESSED_DIR / "s2"
CDL_DIR          = PROCESSED_DIR / "cdl"
MODELS_DIR       = PROJECT_ROOT / "ml_models"
FIGURES_DIR      = PROJECT_ROOT / "documents" / "thesis" / "figures"
LOGS_DIR         = PROJECT_ROOT / "logs"
PRELOAD_CACHE_DIR = PROCESSED_DIR / "preload_cache"
PRELOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)

CDL_BY_YEAR = {
    "2022": CDL_DIR / "cdl_2022_study_area_filtered.tif",
    "2023": CDL_DIR / "cdl_2023_study_area_filtered.tif",
    "2024": CDL_DIR / "cdl_2024_study_area_filtered.tif",
}

# ── S2 metadata ────────────────────────────────────────────────────────────────
S2_BAND_NAMES    = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
N_BANDS_PER_DATE = len(S2_BAND_NAMES)
S2_NODATA        = -9999.0
# 9 vegetation bands used for Exp A and B (excludes coastal B1 and redundant B8A)
VEGE_BANDS       = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12"]

# ── CDL classes ────────────────────────────────────────────────────────────────
# 6-class experiment: keep only major crops (IoU > 0.4, coverage > 4%).
# Dropped to background: Sunflower(6), Other Hay(37), Grapes(69), Pistachios(204).
# Fallow/Idle Cropland (61) → background (class 0); not in KEEP_CLASSES
KEEP_CLASSES = [3, 24, 36, 54, 75, 76]  # Rice, Winter Wheat, Alfalfa, Tomatoes, Almonds, Walnuts
CLASS_REMAP  = {cls_id: i + 1 for i, cls_id in enumerate(KEEP_CLASSES)}
NUM_CLASSES  = len(KEEP_CLASSES) + 1   # 7: 0=bg + 1–6=crops

CDL_CLASS_NAMES = {
    3:  "Rice",    24: "Winter Wheat",  36: "Alfalfa",
    54: "Tomatoes", 75: "Almonds",      76: "Walnuts",
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
GDRIVE_RAW_S2_V5_FOLDER_ID = "1HZOB1b8eq9sF9dtYhppYQC0jsGPuBZZM"

GDRIVE_RAW_S2_V5_FOLDER_IDS = {
    "2022": "14PE8DRpDJqUlux__bBqd-6oAofJrDioU",
    "2023": "1kP7qv9zvjZ8YRlxhFrrwC0fC3GhHK55S",
    "2024": "1YLfx6b5CXbkeR4lvG2hyoky5KICqHSJr",
}

GDRIVE_PROCESSED_S2_FOLDER_IDS = {
    "2022": "1mgiE8vHXiKZHN-zRc68zYLQOAMtO8hst",
    "2023": "1loxQTczrQ_oje6D3dYxzcU-Eo_tPNfnl",
    "2024": "1Dp--kFrQfqFS7C9osEREy9EZKt7KnN_4",
}
GDRIVE_PROCESSED_CDL_FOLDER_ID_V5 = "1L2vIVTJAuWCpLY9g4wmsWcAF6pXPZsnY"
GDRIVE_RAW_CDL_FOLDER_ID           = ""   # optional GDrive fallback; USDA NASS used by default
CDL_DOWNLOAD_URLS = {
    "2022": "https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets/2022_30m_cdls.zip",
    "2023": "https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets/2023_30m_cdls.zip",
    "2024": "https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets/2024_30m_cdls.zip",
}
GDRIVE_PROCESSED_CDL_FOLDER_ID    = "1limegK5Eu3NpNOKHG9xDPe8RoW1B7qMQ"
# V2 study area processed data — single parent folder; year subfolders created automatically
GDRIVE_PROCESSED_V2_FOLDER_ID     = "1RepvRly_kh4z54Jum-3F_RBzxsw3wxcS"
GDRIVE_PROCESSED_V3_FOLDER_ID     = "1WyMw6j1jRdTeIMrG0rkbRv712RBw5rz_"
GDRIVE_PROCESSED_V5_FOLDER_ID     = "1uIYK2dgfmAKyiw1E-wt0qYvIx0OJhLwy"
GDRIVE_MODELS_FOLDER_ID        = "1R6VbWAJpwEe83iCZX0x2O_m8zLetkH9J"

# ── MLflow ─────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI        = "https://mlflow-geoai.stelarea.com"
MLFLOW_EXPERIMENT_PIPELINE = "cropmap_pipeline_runs"
MLFLOW_EXPERIMENT_DATASET  = "cropmap_data_processing"
MLFLOW_EXPERIMENT_FEATURE  = "cropmap_feature_selection_s2"
MLFLOW_EXPERIMENT_TRAIN        = "cropmap_segmentation_s2"
MLFLOW_EXPERIMENT_TRAIN_V2     = "cropmap_segmentation_s2_v2"
MLFLOW_EXPERIMENT_TRAIN_V3     = "cropmap_segmentation_s2_v3"
MLFLOW_EXPERIMENT_TRAIN_DIRECT = "cropmap_segmentation_s2_direct"
MLFLOW_EXPERIMENT_TRAIN_SINGLE_YEAR = "cropmap_segmentation_s2_single_year"
MLFLOW_EXPERIMENT_TRAIN_6CLASS      = "cropmap_segmentation_s2_6class"

# ── GSI scoring hyperparameters ───────────────────────────────────────────────
SAMPLE_FRACTION = 0.05   # 5% of labeled crop pixels for GSI computation
TOP_K_PER_CROP  = 20     # top-K channels per crop before union

# ── Spatial test areas (held-out geography, same year as TRAIN_YEARS) ─────────
# S2 files: data/processed/s2/test_area_1/*.tif  (same dates as main area)
# CDL files: data/processed/cdl/cdl_2024_test_area_1.tif
SPATIAL_TEST_AREAS = [
    {
        "name":   "test_area_1",
        "s2_dir": S2_PROCESSED_DIR / "test_area_1",
        "cdl":    CDL_DIR / "cdl_2024_test_area_1.tif",
    },
    {
        "name":   "test_area_2",
        "s2_dir": S2_PROCESSED_DIR / "test_area_2",
        "cdl":    CDL_DIR / "cdl_2024_test_area_2.tif",
    },
]

# ── Training hyperparameters ───────────────────────────────────────────────────
TRAIN_YEARS    = ["2024"]
TEST_YEAR      = "2024"
PATCH_SIZE     = 256
STRIDE         = 128
MIN_VALID_FRAC = 0.1
BATCH_SIZE     = 8
MAX_EPOCHS     = 200
EARLY_STOP     = 20
EARLY_STOP_DELTA = 0.001   # min mIoU improvement to reset patience
VAL_FRAC       = 0.15
SEED           = 42

ARCH_CFG = {
    "deeplabv3plus_cbam": {"lr": 1e-4, "weight_decay": 1e-4, "encoder": "resnet50"},
    "segformer":          {"lr": 6e-5, "weight_decay": 1e-2, "encoder": "mit_b2"},
}

# ── Band scoring hyperparameters ───────────────────────────────────────────────
TOP_DATES_PER_CROP = 10   # top dates per crop kept as candidates
TOP_BANDS_PER_CROP = 9    # top bands per crop (= len(VEGE_BANDS))
# Aliases used in scoring code
MAX_DATES_PER_CROP = TOP_DATES_PER_CROP
MAX_BANDS_PER_CROP = TOP_BANDS_PER_CROP

GSI_CANDIDATES_JSON = PROCESSED_DIR / "s2" / "2022" / "gsi_candidates.json"

# ── Band selection outputs ──────────────────────────────────────────────────────
# GSI: rank all (date × band) channels by per-crop SI_global, take top-K union
# RF:  rank all (date × band) channels by per-crop RF importance, take top-K union
SELECT_TOP_K_PER_CROP    = 20   # channels selected per crop before union
SELECT_GSI_DIRECT_JSON   = PROCESSED_DIR / "select_gsi_direct.json"
SELECT_GSI_DIRECT_BANDS  = PROCESSED_DIR / "select_gsi_direct_bands.txt"
SELECT_RF_DIRECT_JSON    = PROCESSED_DIR / "select_rf_direct.json"
SELECT_RF_DIRECT_BANDS   = PROCESSED_DIR / "select_rf_direct_bands.txt"

# ── RF selector hyperparameters ────────────────────────────────────────────────
RF_N_ESTIMATORS       = 200    # trees in the binary RF oracle
RF_MAX_PIXELS         = 50_000 # pixel sample cap (crop + rest) to keep RF fast
RF_IMPORTANCE_THRESH  = 0.10   # keep dates/bands with importance >= 10% of max
