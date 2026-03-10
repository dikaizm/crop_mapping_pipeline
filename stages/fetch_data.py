"""
Stage 0 — Download S2 processed TIFs and CDL rasters from Google Drive.

Usage:
    python scripts/fetch_data.py                  # download all configured files
    python scripts/fetch_data.py --overwrite      # re-download even if file exists
    python scripts/fetch_data.py --verify-only    # only check what exists, no download
"""

import os
import sys
import argparse
import logging
from glob import glob
from pathlib import Path

import gdown

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from crop_mapping_pipeline.config import (
    GDRIVE_FILES, S2_PROCESSED_DIR, CDL_BY_YEAR, PROCESSED_DIR,
)

log = logging.getLogger(__name__)


# ── Download helpers ───────────────────────────────────────────────────────────

def download_file(file_id: str, output_path: str, overwrite: bool = False) -> str:
    """Download a single file from Google Drive."""
    if not overwrite and os.path.exists(output_path):
        log.info(f"Already exists — skip: {output_path}")
        return output_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    url    = f"https://drive.google.com/uc?id={file_id}"
    result = gdown.download(url=url, output=output_path, quiet=False, resume=not overwrite)

    if result is None:
        raise RuntimeError(f"Download failed for file_id={file_id}")

    log.info(f"Downloaded: {result}")
    return result


def download_folder(folder_id: str, output_dir: str, overwrite: bool = False) -> None:
    """Download an entire Google Drive folder."""
    os.makedirs(output_dir, exist_ok=True)
    log.info(f"Downloading GDrive folder {folder_id} → {output_dir}")
    gdown.download_folder(
        id=folder_id,
        output=output_dir,
        quiet=False,
        resume=not overwrite,
    )


# ── Verification ───────────────────────────────────────────────────────────────

def verify_data() -> bool:
    """Check all expected files exist. Returns True if everything is present."""
    all_ok = True

    s2_files = sorted(glob(f"{S2_PROCESSED_DIR}/*_processed.tif"))
    by_year: dict[str, list] = {}
    for p in s2_files:
        yr = os.path.basename(p).split("_")[1]
        by_year.setdefault(yr, []).append(p)

    print(f"\nS2 processed files: {len(s2_files)} total")
    for yr in sorted(by_year):
        print(f"  {yr}: {len(by_year[yr])} files")
    if not s2_files:
        print("  ⚠️  No S2 processed files found")
        all_ok = False

    print("\nCDL filtered rasters:")
    for yr, path in sorted(CDL_BY_YEAR.items()):
        exists = os.path.exists(path)
        status = "✅" if exists else "❌ MISSING"
        print(f"  {yr}: {status}  {path}")
        if not exists:
            all_ok = False

    print(f"\nData status: {'✅ All present' if all_ok else '⚠️  Some files missing'}")
    return all_ok


# ── Main ───────────────────────────────────────────────────────────────────────

def main(overwrite: bool = False, verify_only: bool = False) -> None:
    if verify_only:
        ok = verify_data()
        sys.exit(0 if ok else 1)

    # Warn about unconfigured IDs
    missing_ids = [k for k, v in GDRIVE_FILES.items() if not v.get("id")]
    if missing_ids:
        log.warning(
            f"GDrive IDs not set for: {missing_ids}. "
            "Edit scripts/config.py GDRIVE_FILES before running."
        )

    for name, entry in GDRIVE_FILES.items():
        if not entry.get("id"):
            log.warning(f"Skipping '{name}' — GDrive ID not configured")
            continue

        log.info(f"Fetching '{name}' ...")
        try:
            if entry["type"] == "folder":
                download_folder(entry["id"], entry["output_dir"], overwrite=overwrite)
            elif entry["type"] == "file":
                download_file(entry["id"], entry["output_path"], overwrite=overwrite)
            else:
                log.error(f"Unknown type for '{name}': {entry['type']}")
        except Exception as e:
            log.error(f"Failed to download '{name}': {e}")

    verify_data()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download S2 + CDL data from Google Drive")
    parser.add_argument("--overwrite",   action="store_true", help="Re-download even if file exists")
    parser.add_argument("--verify-only", action="store_true", help="Only check if files exist")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )
    main(overwrite=args.overwrite, verify_only=args.verify_only)
