"""
Pipeline orchestrator — runs all stages end-to-end.

Stages:
  fetch    — Stage 0: download S2 + CDL from Google Drive
  feature  — Stage 1+2: GSI ranking + CNN forward selection (feature_analysis.py)
  train    — Stage 3: train Exp A/B/C × 2 architectures (train_segmentation.py)
  all      — run all three stages in order

Usage:
    python crop_mapping_pipeline/pipeline.py                          # run all stages
    python crop_mapping_pipeline/pipeline.py --stages fetch feature  # only fetch + feature
    python crop_mapping_pipeline/pipeline.py --stages train           # only train
    python crop_mapping_pipeline/pipeline.py --force                  # re-run even if outputs exist
    python crop_mapping_pipeline/pipeline.py --data-dir /mnt/data    # override data path

Logs are written to logs/pipeline_YYYYMMDD_HHMMSS.log in addition to stdout.
"""

import sys
import os
import argparse
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from crop_mapping_pipeline.config import LOGS_DIR

log = logging.getLogger(__name__)

VALID_STAGES = ["fetch", "feature", "train", "all"]


# ── Stage runners ─────────────────────────────────────────────────────────────

def run_fetch(force=False, data_dir=None):
    """Stage 0 — download S2 + CDL from Google Drive."""
    log.info("=" * 60)
    log.info("STAGE 0 — Fetch data from Google Drive")
    log.info("=" * 60)
    from crop_mapping_pipeline.stages.fetch_data import main as fetch_main
    fetch_main(overwrite=force)


def run_feature(force=False, data_dir=None):
    """Stage 1+2 — feature analysis (GSI + CNN forward selection)."""
    log.info("=" * 60)
    log.info("STAGE 1+2 — Feature analysis")
    log.info("=" * 60)
    from crop_mapping_pipeline.stages.feature_analysis import main as feature_main
    feature_main(force=force, data_dir=data_dir)


def run_train(force=False, data_dir=None):
    """Stage 3 — train segmentation models (Exp A/B/C)."""
    log.info("=" * 60)
    log.info("STAGE 3 — Train segmentation models")
    log.info("=" * 60)
    from crop_mapping_pipeline.stages.train_segmentation import main as train_main
    train_main(force=force, data_dir=data_dir)


# ── Pipeline runner ───────────────────────────────────────────────────────────

STAGE_FNS = {
    "fetch":   run_fetch,
    "feature": run_feature,
    "train":   run_train,
}


def run_pipeline(stages, force=False, data_dir=None):
    """Execute each stage in order, recording timing and errors."""
    if "all" in stages:
        stages = ["fetch", "feature", "train"]

    results = {}
    pipeline_start = time.time()

    for stage in stages:
        fn = STAGE_FNS.get(stage)
        if fn is None:
            log.error(f"Unknown stage: {stage!r}  — skipping")
            continue

        t0 = time.time()
        log.info(f"\n{'─' * 60}")
        log.info(f"Starting stage: {stage}")
        log.info(f"{'─' * 60}")

        try:
            fn(force=force, data_dir=data_dir)
            elapsed          = time.time() - t0
            results[stage]   = {"status": "ok", "elapsed_s": round(elapsed, 1)}
            log.info(f"Stage '{stage}' completed in {elapsed:.1f}s")
        except Exception:
            elapsed          = time.time() - t0
            results[stage]   = {"status": "error", "elapsed_s": round(elapsed, 1)}
            log.error(f"Stage '{stage}' FAILED after {elapsed:.1f}s")
            log.error(traceback.format_exc())
            # Continue to next stage so partial results are preserved

    total = time.time() - pipeline_start

    log.info("\n" + "=" * 60)
    log.info("PIPELINE SUMMARY")
    log.info("=" * 60)
    any_error = False
    for stage, r in results.items():
        status = "✅ OK" if r["status"] == "ok" else "❌ ERROR"
        log.info(f"  {stage:10s}  {status}  ({r['elapsed_s']}s)")
        if r["status"] != "ok":
            any_error = True
    log.info(f"\nTotal wall time: {total / 60:.1f} min")

    if any_error:
        log.error("One or more stages failed — check logs above")
        sys.exit(1)
    else:
        log.info("All stages completed successfully")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Crop-mapping pipeline orchestrator"
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["all"],
        choices=VALID_STAGES,
        metavar="STAGE",
        help=f"Stages to run: {VALID_STAGES}  (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run stages even if outputs already exist",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        metavar="PATH",
        help="Override data/processed directory (absolute path)",
    )
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )

    log.info(f"Pipeline log: {log_file}")
    log.info(f"Stages: {args.stages}  force={args.force}  data_dir={args.data_dir}")

    run_pipeline(stages=args.stages, force=args.force, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
