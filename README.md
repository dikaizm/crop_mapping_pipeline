# crop_mapping_pipeline

End-to-end pipeline for crop type mapping using multi-temporal Sentinel-2 imagery and USDA Cropland Data Layer (CDL) labels. Covers data download, band selection, and segmentation model training.

**Study area:** Sacramento Valley, California
**Labels:** 10 crop classes (Rice, Sunflower, Winter Wheat, Alfalfa, Other Hay, Tomatoes, Grapes, Almonds, Walnuts, Plums) + background

---

## Project Structure

```
crop_mapping_pipeline/
├── pipeline.py            # CLI entry point — orchestrates all stages
├── config.py              # All hyperparameters and file paths
├── requirements.txt
├── stages/
│   ├── fetch_data.py      # Stage 0: download S2 + CDL from Google Drive
│   ├── feature_analysis.py  # Stage 1+2: band selection (GSI + CNN forward selection)
│   └── train_segmentation.py  # Stage 3: train Exp A/B/C × 2 architectures
├── models/
│   ├── cbam.py            # CBAM attention module
│   ├── deeplabv3plus.py   # DeepLabV3+ with CBAM
│   └── segformer.py       # SegFormer wrapper
└── utils/
    ├── band_selection.py  # GSI, RF importance, joint score, top-k selection
    ├── constants.py       # CDL class colors and names
    ├── general.py         # Download helpers
    └── label.py           # Label remapping utilities
```

---

## Requirements

- Python >= 3.10
- CUDA-capable GPU recommended (tested on RTX 4090, 24 GB VRAM)
- [`geoai`](https://github.com/opengeos/geoai) library available on `PYTHONPATH`

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone <repo-url>
cd crop_mapping_pipeline

python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

**CPU (testing only):**
```bash
pip install -r requirements.txt
```

**GPU (CUDA 12.4):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 3. Add geoai to PYTHONPATH

`geoai` is an external geospatial AI library required by Stage 3 for patch dataset loading and training utilities. Clone it and add to your environment:

```bash
git clone https://github.com/opengeos/geoai.git /path/to/geoai
export PYTHONPATH="/path/to/geoai:$PYTHONPATH"
```

Or add the export to your `.bashrc` / `.zshrc` for persistence.

### 4. Configure Google Drive file IDs

Edit `config.py` and fill in the Google Drive IDs for each dataset:

```python
GDRIVE_FILES = {
    "s2":      {"type": "folder", "id": "<YOUR_FOLDER_ID>", ...},  # all years in one folder
    "cdl_2022": {"type": "file",  "id": "<YOUR_FILE_ID>",   ...},
    "cdl_2023": {"type": "file",  "id": "<YOUR_FILE_ID>",   ...},
    "cdl_2024": {"type": "file",  "id": "<YOUR_FILE_ID>",   ...},
}
```

### 5. Configure data paths (optional)

By default the pipeline expects data at `../data/processed/` relative to the repo. To use a different location, pass `--data-dir` at runtime (see below) or edit `PROCESSED_DIR` in `config.py`.

### 6. Configure MLflow (optional)

The pipeline logs all experiments to MLflow. Set the tracking URI in `config.py`:

```python
MLFLOW_TRACKING_URI = "http://your-mlflow-server"  # or a local path
```

---

## Running the Pipeline

### Option A — Background launcher (recommended for GPU server)

```bash
chmod +x run.sh

./run.sh                                          # run all stages
./run.sh --stages fetch                           # download data only
./run.sh --stages feature                         # band selection only
./run.sh --stages train                           # model training only
./run.sh --stages feature train                   # skip fetch
./run.sh --stages train --force                   # force re-run even if outputs exist
./run.sh --stages all --data-dir /mnt/data        # override data path
```

Logs are written to `logs/run_YYYYMMDD_HHMMSS.log`. The process PID is saved to `logs/pipeline_YYYYMMDD_HHMMSS.pid`.

Monitor a running job:
```bash
tail -f logs/run_<timestamp>.log
```

Stop a running job:
```bash
kill $(cat logs/pipeline_<timestamp>.pid)
```

### Option B — Direct Python

```bash
source .venv/bin/activate

python -m crop_mapping_pipeline.pipeline                          # all stages
python -m crop_mapping_pipeline.pipeline --stages fetch feature   # fetch + band selection
python -m crop_mapping_pipeline.pipeline --stages train           # training only
python -m crop_mapping_pipeline.pipeline --stages train --force   # force re-train
python -m crop_mapping_pipeline.pipeline --data-dir /mnt/data     # custom data dir
```

### Option C — Run stages individually

```bash
python -m crop_mapping_pipeline.stages.fetch_data
python -m crop_mapping_pipeline.stages.fetch_data --verify-only  # check files only

python -m crop_mapping_pipeline.stages.feature_analysis
python -m crop_mapping_pipeline.stages.feature_analysis --force

python -m crop_mapping_pipeline.stages.train_segmentation
python -m crop_mapping_pipeline.stages.train_segmentation --exp A B C
python -m crop_mapping_pipeline.stages.train_segmentation --arch deeplabv3plus_cbam
python -m crop_mapping_pipeline.stages.train_segmentation --skip-viz
```

---

## Pipeline Stages

### Stage 0 — Fetch (`stages/fetch_data.py`)

Downloads preprocessed Sentinel-2 GeoTIFFs and CDL rasters from Google Drive using `gdown`.

**Inputs:** Google Drive file/folder IDs in `config.py`
**Outputs:** `data/processed/s2/*_processed.tif`, `data/processed/cdl/cdl_*_study_area_filtered.tif`

### Stage 1+2 — Feature Analysis (`stages/feature_analysis.py`)

**Stage 1** ranks all input channels per crop using the SIglobal metric (per-crop GSI), producing a ranked candidate list of up to `TOP_K_PER_CROP=20` bands per crop.

**Stage 2** performs per-crop binary CNN forward selection in Stage 1 rank order. A lightweight U-Net (ResNet-18) is trained for each candidate band set; a band is accepted if IoU gain ≥ `S2_DELTA=0.005`. Selection stops after `S2_NO_IMPROVE=5` consecutive rejections.

MLflow logs one **nested run per crop** under a parent Stage 2 run, with a clean IoU-vs-band-rank curve per crop.

**Inputs:** S2 + CDL from 2022 (training reference year)
**Outputs:**
- `data/processed/stage2v2_per_crop_results.csv` — per-crop K*, key dates, key bands, IoU
- `data/processed/stage3_exp_c_bands.txt` — union of all selected bands (Stage 3 Exp C input)

### Stage 3 — Training (`stages/train_segmentation.py`)

Trains 6 experiments: **Exp A / B / C** × **DeepLabV3+CBAM / SegFormer**.

| Config | Input | Channels | Purpose |
|--------|-------|----------|---------|
| Exp A | Single date (Jul 30, peak season) | 9 | Baseline — no temporal info |
| Exp B | 4 phenological dates (Jan/Mar/Jul/Nov) | 36 | Temporal, no band selection |
| Exp C | Stage 2 output | K* ≤ 36 | Proposed method |

**Train/test split:** 2022 + 2023 → train, 2024 → test (temporal split).
**Loss:** weighted cross-entropy (inverse class frequency).
**Optimizer:** AdamW + PolynomialLR, early stopping patience = 10 epochs.

**Outputs:** `ml_models/<exp>_<arch>/` — checkpoint, segmentation map, per-class IoU CSV, MLflow run

---

## Key Hyperparameters

All hyperparameters are in `config.py`. Key values:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TOP_K_PER_CROP` | 20 | Stage 1 candidate list size per crop |
| `S2_DELTA` | 0.005 | Min IoU gain to accept a band (Stage 2) |
| `S2_NO_IMPROVE` | 5 | Consecutive rejections before stopping (Stage 2) |
| `S2_MAX_BANDS` | 20 | Max bands selected per crop (Stage 2) |
| `MAX_EPOCHS` | 100 | Max training epochs (Stage 3) |
| `EARLY_STOP` | 10 | Early stopping patience (Stage 3) |
| `TRAIN_YEARS` | 2022, 2023 | Training years |
| `TEST_YEAR` | 2024 | Test year |

---

## Expected Outputs

After a full pipeline run:

```
data/processed/
├── s2/                            # downloaded S2 GeoTIFFs
├── cdl/                           # CDL filtered rasters
├── stage2v2_per_crop_results.csv  # per-crop band selection results
└── stage3_exp_c_bands.txt         # union band list for Exp C

ml_models/
├── expA_deeplabv3plus_cbam/
├── expA_segformer/
├── expB_deeplabv3plus_cbam/
├── expB_segformer/
├── expC_deeplabv3plus_cbam/
└── expC_segformer/

logs/
├── run_YYYYMMDD_HHMMSS.log
└── pipeline_YYYYMMDD_HHMMSS.pid
```
