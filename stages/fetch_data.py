"""
Stage 0 — Download processed S2 TIFs and CDL rasters from Google Drive.

Supports year-by-year downloading to minimise server disk usage:

    python fetch_data.py                        # download all years
    python fetch_data.py --years 2022           # download 2022 only
    python fetch_data.py --years 2022 2023      # download 2022 + 2023
    python fetch_data.py --years 2022 --delete  # download then delete after use
    python fetch_data.py --overwrite            # re-download even if files exist
    python fetch_data.py --verify-only          # only check what is present

Recommended server workflow (storage-constrained):

    # Stage 2 needs only 2022
    python fetch_data.py --years 2022
    python feature_analysis.py --stage 2
    python fetch_data.py --years 2022 --delete   # free disk space

    # Stage 3 needs all years
    python fetch_data.py --years 2022 2023 2024
    python train_segmentation.py
    python fetch_data.py --delete                # free disk space after training
"""

import os
import sys
import argparse
import logging
from glob import glob
import pathlib
from pathlib import Path

import gdown

_ROOT = Path(__file__).parent.parent   # crop_mapping_pipeline/
sys.path.insert(0, str(_ROOT.parent))

from crop_mapping_pipeline.config import (
    GDRIVE_FILES, GDRIVE_RAW_S2_FOLDER_IDS,
    S2_PROCESSED_DIR, CDL_BY_YEAR, PROCESSED_DIR,
    GDRIVE_OAUTH_TOKEN,
)

log = logging.getLogger(__name__)

ALL_YEARS = ["2022", "2023", "2024"]


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


# ── GDrive file-by-name download ───────────────────────────────────────────────

def _build_drive_service():
    """Build authenticated GDrive API v3 service (reuses OAuth token from process_data)."""
    import pickle
    from googleapiclient.discovery import build
    from google.auth.transport.requests import Request

    if not GDRIVE_OAUTH_TOKEN.exists():
        raise FileNotFoundError(
            f"OAuth token not found: {GDRIVE_OAUTH_TOKEN}\n"
            "Run: python stages/process_data.py --auth"
        )
    with open(GDRIVE_OAUTH_TOKEN, "rb") as f:
        creds = pickle.load(f)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(GDRIVE_OAUTH_TOKEN, "wb") as f:
            pickle.dump(creds, f)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _get_folder_id(year: str, raw: bool) -> str:
    if raw:
        folder_id = GDRIVE_RAW_S2_FOLDER_IDS.get(year)
        if not folder_id:
            raise ValueError(f"No raw GDrive folder configured for year {year}")
    else:
        folder_entry = GDRIVE_FILES.get(f"s2_{year}")
        if not folder_entry or not folder_entry.get("id"):
            raise ValueError(f"No GDrive folder configured for year {year} (s2_{year})")
        folder_id = folder_entry["id"]
    return folder_id


def _list_folder_gdown(folder_id: str) -> list:
    """Return list of {id, name} dicts by scraping the GDrive folder via gdown internals."""
    import gdown.folder as gdf
    # gdown.folder.get_directory_structure returns a nested dict; use _parse_folder
    files = gdf._get_directory_structure(folder_id, remaining_ok=True, use_cookies=False)
    return files  # list of (title, id) or similar depending on gdown version


def _list_folder_contents(folder_id: str) -> dict:
    """Return {filename: file_id} mapping via OAuth GDrive API."""
    service    = _build_drive_service()
    name_to_id = {}
    page_token = None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken, files(id, name)",
            pageSize=1000,
            pageToken=page_token,
        ).execute()
        for f in resp.get("files", []):
            name_to_id[f["name"]] = f["id"]
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return name_to_id


def list_gdrive_files(year: str, raw: bool = False) -> None:
    """List all filenames in the GDrive folder for a given year."""
    folder_id  = _get_folder_id(year, raw)
    log.info(f"Listing GDrive folder {folder_id} ...")
    name_to_id = _list_folder_contents(folder_id)
    print(f"\nFiles in GDrive folder (year={year}, raw={raw}) — {len(name_to_id)} total:")
    for name in sorted(name_to_id.keys()):
        print(f"  {name}")


def download_files_by_name(filenames: list, year: str, output_dir: str,
                           overwrite: bool = False, raw: bool = False) -> None:
    """
    Download specific files by filename from a GDrive folder via OAuth API.
    Requires token generated with 'drive' scope (python stages/process_data.py --auth).
    """
    from googleapiclient.http import MediaIoBaseDownload
    import io

    folder_id  = _get_folder_id(year, raw)
    service    = _build_drive_service()
    os.makedirs(output_dir, exist_ok=True)

    log.info(f"Fetching file listing for folder {folder_id} ...")
    name_to_id = _list_folder_contents(folder_id)

    for fname in filenames:
        out_path = os.path.join(output_dir, fname)
        if not overwrite and os.path.exists(out_path):
            log.info(f"Already exists — skip: {fname}")
            continue

        file_id = name_to_id.get(fname)
        if not file_id:
            log.error(
                f"File not found in GDrive folder (year={year}): {fname}\n"
                f"  Run --list-files to see available filenames."
            )
            continue

        log.info(f"Downloading {fname}  (id={file_id}) → {out_path}")
        request = service.files().get_media(fileId=file_id)
        with open(out_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request, chunksize=50 * 1024 * 1024)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    log.info(f"  {fname}: {int(status.progress() * 100)}%")
        log.info(f"  Done: {fname}")


# ── Delete helpers ─────────────────────────────────────────────────────────────

def delete_s2_year(year: str) -> None:
    """Delete all processed S2 files for a given year."""
    pattern = str(S2_PROCESSED_DIR / year / f"S2H_{year}_*_processed.tif")
    files   = sorted(glob(pattern))
    if not files:
        log.info(f"  [{year}] No S2 files to delete")
        return
    freed = 0
    for f in files:
        size = os.path.getsize(f)
        os.remove(f)
        freed += size
        log.info(f"  Deleted: {os.path.basename(f)}")
    log.info(f"  [{year}] Freed {freed / 1e9:.2f} GB ({len(files)} S2 files)")


def delete_cdl_year(year: str) -> None:
    """Delete the processed CDL file for a given year."""
    path = CDL_BY_YEAR.get(year)
    if path and os.path.exists(path):
        size = os.path.getsize(path)
        os.remove(path)
        log.info(f"  [{year}] Deleted CDL: {os.path.basename(path)}  ({size / 1e6:.0f} MB)")
    else:
        log.info(f"  [{year}] No CDL file to delete")


# ── Verification ───────────────────────────────────────────────────────────────

def verify_data(years=None) -> bool:
    """Check expected files exist. Returns True if everything is present."""
    years   = years or ALL_YEARS
    all_ok  = True

    s2_files = sorted(glob(str(S2_PROCESSED_DIR / "*" / "*_processed.tif")))
    by_year: dict[str, list] = {}
    for p in s2_files:
        yr = os.path.basename(p).split("_")[1]
        by_year.setdefault(yr, []).append(p)

    print(f"\nS2 processed files: {len(s2_files)} total")
    for yr in sorted(years):
        n = len(by_year.get(yr, []))
        status = "✅" if n > 0 else "❌ MISSING"
        print(f"  {yr}: {status}  ({n} files)")
        if n == 0:
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

def main(
    years       : list = None,
    files       : list = None,
    overwrite   : bool = False,
    verify_only : bool = False,
    list_files  : bool = False,
    delete      : bool = False,
    raw         : bool = False,
    raw_s2_dir  : str  = None,
) -> None:
    years = years or ALL_YEARS

    if verify_only:
        ok = verify_data(years)
        sys.exit(0 if ok else 1)

    if list_files:
        for yr in years:
            list_gdrive_files(yr, raw=raw)
        return

    # ── Download specific files by filename ───────────────────────────────────
    if files:
        for yr in years:
            if raw:
                base_raw_dir = raw_s2_dir or str(PROCESSED_DIR.parent / "raw" / "s2")
                out_dir = str(pathlib.Path(base_raw_dir) / yr)
            else:
                out_dir = str(S2_PROCESSED_DIR / yr)
            log.info(f"Downloading {len(files)} file(s) for year {yr} → {out_dir}")
            download_files_by_name(files, yr, out_dir, overwrite=overwrite, raw=raw)
        return

    if raw:
        # ── Download raw S2 files ─────────────────────────────────────────────
        base_raw_dir = raw_s2_dir or str(PROCESSED_DIR.parent / "raw" / "s2")
        for yr in years:
            folder_id = GDRIVE_RAW_S2_FOLDER_IDS.get(yr)
            if not folder_id:
                log.warning(f"GDRIVE_RAW_S2_FOLDER_IDS['{yr}'] not set — skipping")
                continue
            yr_dir = str(pathlib.Path(base_raw_dir) / yr)
            log.info(f"Downloading raw S2 for {yr} → {yr_dir}")
            download_folder(folder_id, yr_dir, overwrite=overwrite)
    else:
        # ── Download processed files ──────────────────────────────────────────
        missing_ids = [k for k, v in GDRIVE_FILES.items() if not v.get("id")]
        if missing_ids:
            log.warning(
                f"GDrive IDs not set for: {missing_ids}. "
                "Edit config.py GDRIVE_FILES before running."
            )

        for name, entry in GDRIVE_FILES.items():
            entry_year = entry.get("year")
            if entry_year and entry_year not in years:
                continue

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

        verify_data(years)

    # ── Delete ────────────────────────────────────────────────────────────────
    if delete:
        log.info("\nDeleting downloaded files to free disk space...")
        for yr in years:
            delete_s2_year(yr)
            delete_cdl_year(yr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download processed S2 + CDL data from Google Drive."
    )
    parser.add_argument(
        "--years", nargs="+", default=None, choices=ALL_YEARS,
        metavar="YEAR",
        help=f"Years to download (default: all — {ALL_YEARS})",
    )
    parser.add_argument(
        "--files", nargs="+", default=None,
        metavar="FILENAME",
        help="Specific filenames to download from the GDrive S2 folder "
             "(e.g. S2H_2023_20230715_processed.tif). Requires --years to identify the folder.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-download even if files already exist",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Only check if files exist, no download",
    )
    parser.add_argument(
        "--delete", action="store_true",
        help="Delete downloaded files after verification (frees disk space)",
    )
    parser.add_argument(
        "--raw", action="store_true",
        help="Download raw S2 files (from GDRIVE_RAW_S2_FOLDER_IDS) instead of processed files",
    )
    parser.add_argument(
        "--raw-s2-dir", default=None,
        help="Output directory for raw S2 files (default: data/raw/s2/)",
    )
    parser.add_argument(
        "--list-files", action="store_true",
        help="List all filenames in the GDrive folder for the given year(s) without downloading",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )

    main(
        years       = args.years,
        files       = args.files,
        overwrite   = args.overwrite,
        verify_only = args.verify_only,
        list_files  = args.list_files,
        delete      = args.delete,
        raw         = args.raw,
        raw_s2_dir  = args.raw_s2_dir,
    )
