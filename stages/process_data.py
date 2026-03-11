"""
Stage 0.5 — Process raw S2 + CDL files, upload to Google Drive, delete raw.

Designed for a storage-constrained GPU server workflow where keeping all raw
files simultaneously is not feasible.  Processes one year at a time:

  For each year:
    1. Process CDL  — reproject (EPSG:5070 → EPSG:4326) + clip to S2 grid
                      + filter to KEEP_CLASSES  →  cdl_{yr}_study_area_filtered.tif
    2. Process S2   — assign NoData (-9999, cast float32)  →  *_processed.tif
    3. Upload       — push processed files to Google Drive (Drive API v3)
    4. Delete raw   — remove raw S2 TIFs for that year to free disk space

CDL raw files are NOT deleted (they are small, ~1 GB each).

Usage:
    python process_data.py                            # all years
    python process_data.py --years 2022 2023          # specific years
    python process_data.py --skip-upload              # process only, no upload
    python process_data.py --skip-delete              # keep raw S2 after processing
    python process_data.py --data-dir /data/processed # override output dir

Google Drive upload requirements:
    1. Create a service account at console.cloud.google.com
    2. Enable the Google Drive API for the project
    3. Share the target GDrive folders with the service-account email
    4. Download the JSON key → save to  ssh/gdrive_service_account.json
    5. Fill in GDRIVE_PROCESSED_S2_FOLDER_ID and GDRIVE_PROCESSED_CDL_FOLDER_ID
       in crop_mapping_pipeline/config.py
"""

import os
import sys
import logging
import argparse
import pathlib
import subprocess
from glob import glob

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from dotenv import load_dotenv

_ROOT = pathlib.Path(__file__).parent.parent   # crop_mapping_pipeline/
load_dotenv(_ROOT / ".env")
sys.path.insert(0, str(_ROOT.parent))

from crop_mapping_pipeline.config import (
    S2_PROCESSED_DIR, CDL_BY_YEAR, PROCESSED_DIR,
    S2_NODATA, KEEP_CLASSES,
    GDRIVE_OAUTH_TOKEN,
    GDRIVE_PROCESSED_S2_FOLDER_IDS,
    GDRIVE_PROCESSED_CDL_FOLDER_ID,
)
from crop_mapping_pipeline.utils.label import label_filtering

log = logging.getLogger(__name__)

ALL_YEARS = ["2022", "2023", "2024"]


# ── CDL processing ─────────────────────────────────────────────────────────────

def process_cdl(cdl_raw_path: str, s2_ref_path: str,
                out_reprojected: str, out_filtered: str) -> None:
    """
    Step 1+2: reproject CDL to S2 grid, then filter to KEEP_CLASSES.
    Skips individual steps if output already exists.
    """
    # Step 1 — reproject + clip + resample
    if os.path.exists(out_reprojected):
        log.info(f"  CDL reprojected already exists: {os.path.basename(out_reprojected)}")
    else:
        log.info(f"  Reprojecting CDL → {os.path.basename(out_reprojected)}")
        with rasterio.open(s2_ref_path) as s2_ref:
            target_crs       = s2_ref.crs
            target_transform = s2_ref.transform
            target_width     = s2_ref.width
            target_height    = s2_ref.height

        with rasterio.open(cdl_raw_path) as cdl_src:
            dst_data = np.zeros((1, target_height, target_width), dtype=np.uint8)
            reproject(
                source       = rasterio.band(cdl_src, 1),
                destination  = dst_data,
                src_transform= cdl_src.transform,
                src_crs      = cdl_src.crs,
                dst_transform= target_transform,
                dst_crs      = target_crs,
                resampling   = Resampling.nearest,
            )

        profile = {
            "driver": "GTiff", "dtype": "uint8", "nodata": 0,
            "width": target_width, "height": target_height, "count": 1,
            "crs": target_crs, "transform": target_transform, "compress": "lzw",
        }
        os.makedirs(os.path.dirname(out_reprojected), exist_ok=True)
        with rasterio.open(out_reprojected, "w", **profile) as dst:
            dst.write(dst_data)
        log.info(f"  CDL reprojected: {os.path.basename(out_reprojected)}")

    # Step 2 — filter classes
    if os.path.exists(out_filtered):
        log.info(f"  CDL filtered already exists: {os.path.basename(out_filtered)}")
    else:
        log.info(f"  Filtering CDL classes → {os.path.basename(out_filtered)}")
        label_filtering(
            in_path     = out_reprojected,
            out_path    = out_filtered,
            keep_classes= KEEP_CLASSES,
        )
        log.info(f"  CDL filtered: {os.path.basename(out_filtered)}")


# ── S2 processing ──────────────────────────────────────────────────────────────

def process_s2_year(s2_raw_paths: list, out_dir: str) -> list:
    """
    Step 3: assign NoData (negative / NaN / Inf → S2_NODATA) and cast to float32.
    Returns list of output paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_paths = []

    for src_path in s2_raw_paths:
        fname    = os.path.basename(src_path)
        out_path = os.path.join(out_dir, fname.replace(".tif", "_processed.tif"))

        if os.path.exists(out_path):
            log.info(f"  Already processed: {fname}")
            out_paths.append(out_path)
            continue

        with rasterio.open(src_path) as src:
            profile = src.profile.copy()
            profile.update(dtype="float32", nodata=S2_NODATA, compress="lzw")
            data = src.read().astype(np.float32)

        invalid      = (data < 0) | np.isnan(data) | np.isinf(data)
        data[invalid] = S2_NODATA

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data)

        log.info(f"  Processed: {fname}  (invalid_px={invalid.sum():,})")
        out_paths.append(out_path)

    return out_paths


# ── Google Drive upload ────────────────────────────────────────────────────────

def _build_drive_service():
    """Build an authenticated Google Drive API v3 service using OAuth 2.0 token."""
    import pickle
    from googleapiclient.discovery import build
    from google.auth.transport.requests import Request

    if not GDRIVE_OAUTH_TOKEN.exists():
        raise FileNotFoundError(
            f"OAuth token not found: {GDRIVE_OAUTH_TOKEN}\n"
            "Generate it locally with:\n"
            "  python crop_mapping_pipeline/stages/process_data.py --auth\n"
            "Then copy to the server:\n"
            f"  scp {GDRIVE_OAUTH_TOKEN} user@server:{GDRIVE_OAUTH_TOKEN}"
        )

    with open(GDRIVE_OAUTH_TOKEN, "rb") as f:
        creds = pickle.load(f)

    if creds.expired and creds.refresh_token:
        log.info("Refreshing expired OAuth token...")
        creds.refresh(Request())
        with open(GDRIVE_OAUTH_TOKEN, "wb") as f:
            pickle.dump(creds, f)

    return build("drive", "v3", credentials=creds, cache_discovery=False)


def upload_file(local_path: str, folder_id: str, service=None) -> str:
    """
    Upload a single file to a GDrive folder.
    Returns the GDrive file ID.
    Skips upload if a file with the same name already exists in the folder.
    """
    from googleapiclient.http import MediaFileUpload

    if service is None:
        service = _build_drive_service()

    fname = os.path.basename(local_path)

    # Check if file already exists in the folder
    query  = f"name='{fname}' and '{folder_id}' in parents and trashed=false"
    result = service.files().list(q=query, fields="files(id,name)").execute()
    if result.get("files"):
        existing_id = result["files"][0]["id"]
        log.info(f"  Already on GDrive: {fname}  (id={existing_id})")
        return existing_id

    file_size = os.path.getsize(local_path)
    log.info(f"  Uploading: {fname}  ({file_size/1e6:.0f} MB)")

    media    = MediaFileUpload(local_path, mimetype="image/tiff", resumable=True)
    metadata = {"name": fname, "parents": [folder_id]}
    request  = service.files().create(body=metadata, media_body=media, fields="id")

    # Resumable upload with progress
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            log.info(f"    {int(status.progress() * 100)}% uploaded")

    file_id = response.get("id")
    log.info(f"  Uploaded: {fname}  (id={file_id})")
    return file_id


def upload_year(s2_processed_paths: list, cdl_filtered_path: str, year: str) -> None:
    """Upload all processed S2 files + CDL filtered file for one year."""
    s2_folder_id = GDRIVE_PROCESSED_S2_FOLDER_IDS.get(year, "")
    if not s2_folder_id or not GDRIVE_PROCESSED_CDL_FOLDER_ID:
        raise ValueError(
            f"GDRIVE_PROCESSED_S2_FOLDER_IDS['{year}'] and GDRIVE_PROCESSED_CDL_FOLDER_ID "
            "must be set in config.py before uploading."
        )

    service = _build_drive_service()

    log.info(f"  Uploading {len(s2_processed_paths)} S2 files to GDrive (year={year})...")
    for path in s2_processed_paths:
        upload_file(path, s2_folder_id, service)

    log.info(f"  Uploading CDL filtered file to GDrive...")
    upload_file(cdl_filtered_path, GDRIVE_PROCESSED_CDL_FOLDER_ID, service)


# ── Cleanup ────────────────────────────────────────────────────────────────────

def delete_raw_s2(s2_raw_paths: list) -> None:
    """Delete raw S2 TIF files after successful processing + upload."""
    freed = 0
    for path in s2_raw_paths:
        size = os.path.getsize(path)
        os.remove(path)
        freed += size
        log.info(f"  Deleted raw: {os.path.basename(path)}")
    log.info(f"  Freed: {freed/1e9:.2f} GB")


# ── Shutdown ───────────────────────────────────────────────────────────────────

def _schedule_shutdown(delay_min: int = 8) -> None:
    """
    Stop the pod/server after `delay_min` minutes.
    - RunPod: uses RunPod API (requires RUNPOD_API_KEY env var)
    - Other Linux VPS: falls back to `sudo shutdown -h`
    """
    import time, urllib.request, urllib.error, json

    pod_id  = os.environ.get("RUNPOD_POD_ID")
    api_key = os.environ.get("RUNPOD_API_KEY")

    if pod_id and api_key:
        log.warning("=" * 60)
        log.warning(f"RunPod pod {pod_id} will stop in {delay_min} minutes.")
        log.warning("=" * 60)
        time.sleep(delay_min * 60)

        query   = f'{{"query": "mutation {{ podStop(input: {{podId: \\"{pod_id}\\"}}) {{ id desiredStatus }} }}"}}'
        req     = urllib.request.Request(
            "https://api.runpod.io/graphql",
            data    = query.encode(),
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        )
        try:
            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read())
                log.info(f"Pod stop response: {result}")
        except urllib.error.URLError as e:
            log.error(f"Failed to stop pod via RunPod API: {e}")
    else:
        # Fallback for standard Linux VPS with systemd
        log.warning("=" * 60)
        log.warning(f"VPS SHUTDOWN in {delay_min} minutes.")
        log.warning("Cancel with:  sudo shutdown -c")
        log.warning("=" * 60)
        try:
            subprocess.run(["sudo", "shutdown", "-h", f"+{delay_min}"], check=True)
        except Exception as e:
            log.error(f"Failed to schedule shutdown: {e}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(
    years       : list  = None,
    raw_s2_dir  : str   = None,
    raw_cdl_dir : str   = None,
    data_dir    : str   = None,
    skip_upload : bool  = False,
    skip_delete : bool  = False,
    shutdown    : bool  = False,
) -> None:
    global S2_PROCESSED_DIR, CDL_BY_YEAR, PROCESSED_DIR

    if data_dir:
        processed        = pathlib.Path(data_dir)
        PROCESSED_DIR    = processed
        S2_PROCESSED_DIR = processed / "s2"
        CDL_BY_YEAR      = {
            yr: processed / "cdl" / f"cdl_{yr}_study_area_filtered.tif"
            for yr in ALL_YEARS
        }

    years = years or ALL_YEARS

    for yr in years:
        log.info("=" * 60)
        log.info(f"Year: {yr}")
        log.info("=" * 60)

        # ── Locate raw S2 files for this year ─────────────────────────────────
        s2_dir      = pathlib.Path(raw_s2_dir) / yr if raw_s2_dir else _ROOT / "data" / "raw" / "s2" / yr
        s2_raw_year = sorted(glob(str(s2_dir / f"S2H_{yr}_*.tif")))
        if not s2_raw_year:
            log.warning(f"  No raw S2 files found for {yr} in {s2_dir} — skipping")
            continue
        log.info(f"  Raw S2 files: {len(s2_raw_year)}")

        # Use first file as CDL reference grid
        s2_ref = s2_raw_year[0]

        # ── Locate raw CDL ─────────────────────────────────────────────────────
        cdl_dir     = pathlib.Path(raw_cdl_dir) if raw_cdl_dir else _ROOT / "data" / "raw" / "cdl"
        cdl_raw     = next(
            (p for p in glob(str(cdl_dir / f"{yr}_30m_cdls" / "*.tif"))),
            None,
        )
        if not cdl_raw:
            log.warning(f"  Raw CDL for {yr} not found in {cdl_dir} — skipping CDL processing")
        else:
            cdl_out_dir       = S2_PROCESSED_DIR.parent / "cdl"
            cdl_reprojected   = str(cdl_out_dir / f"cdl_{yr}_study_area.tif")
            cdl_filtered      = str(cdl_out_dir / f"cdl_{yr}_study_area_filtered.tif")
            process_cdl(cdl_raw, s2_ref, cdl_reprojected, cdl_filtered)

        # ── Process S2 ────────────────────────────────────────────────────────
        log.info(f"  Processing {len(s2_raw_year)} S2 files...")
        s2_processed = process_s2_year(s2_raw_year, str(S2_PROCESSED_DIR / yr))

        # ── Upload ────────────────────────────────────────────────────────────
        if not skip_upload:
            log.info("  Uploading to Google Drive...")
            upload_year(
                s2_processed_paths = s2_processed,
                cdl_filtered_path  = cdl_filtered if cdl_raw else "",
                year               = yr,
            )
        else:
            log.info("  Upload skipped (--skip-upload)")

        # ── Delete raw S2 ─────────────────────────────────────────────────────
        if not skip_delete:
            log.info("  Deleting raw S2 files...")
            delete_raw_s2(s2_raw_year)
        else:
            log.info("  Raw S2 kept (--skip-delete)")

        log.info(f"Year {yr} done.\n")

    if shutdown:
        _schedule_shutdown(delay_min=8)


def generate_oauth_token():
    """Run OAuth flow in browser to generate gdrive_token.pickle (run locally once)."""
    import pickle
    from google_auth_oauthlib.flow import InstalledAppFlow
    from crop_mapping_pipeline.config import GDRIVE_OAUTH_SECRET, GDRIVE_OAUTH_TOKEN

    if not GDRIVE_OAUTH_SECRET.exists():
        raise FileNotFoundError(
            f"OAuth client secret not found: {GDRIVE_OAUTH_SECRET}\n"
            "Download it from Google Cloud Console → APIs & Services → Credentials."
        )

    flow  = InstalledAppFlow.from_client_secrets_file(
        str(GDRIVE_OAUTH_SECRET),
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    creds = flow.run_local_server(port=0)

    with open(GDRIVE_OAUTH_TOKEN, "wb") as f:
        pickle.dump(creds, f)

    print(f"Token saved to: {GDRIVE_OAUTH_TOKEN}")
    print(f"\nCopy to server:")
    print(f"  scp {GDRIVE_OAUTH_TOKEN} user@server:/workspace/crop_mapping_pipeline/ssh/gdrive_token.pickle")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process raw S2 + CDL files, upload to GDrive, delete raw."
    )
    parser.add_argument(
        "--years", nargs="+", default=None, choices=ALL_YEARS,
        metavar="YEAR",
        help=f"Years to process (default: all — {ALL_YEARS})",
    )
    parser.add_argument(
        "--raw-s2-dir", default=None,
        help="Directory containing raw S2 TIFs (S2H_YYYY_*.tif). "
             "Default: data/raw/s2/ relative to project root.",
    )
    parser.add_argument(
        "--raw-cdl-dir", default=None,
        help="Directory containing raw CDL folders (YYYY_30m_cdls/). "
             "Default: data/raw/cdl/ relative to project root.",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Override processed output directory (default: data/processed/).",
    )
    parser.add_argument(
        "--skip-upload", action="store_true",
        help="Process files but do not upload to Google Drive.",
    )
    parser.add_argument(
        "--skip-delete", action="store_true",
        help="Keep raw S2 files after processing (do not delete).",
    )
    parser.add_argument(
        "--shutdown", action="store_true",
        help="Shut down the VPS 8 minutes after all processing completes (Linux only).",
    )
    parser.add_argument(
        "--auth", action="store_true",
        help="Generate OAuth token via browser (run locally once, then copy token to server).",
    )
    args = parser.parse_args()

    if args.auth:
        generate_oauth_token()
        sys.exit(0)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )

    main(
        years       = args.years,
        raw_s2_dir  = args.raw_s2_dir,
        raw_cdl_dir = args.raw_cdl_dir,
        data_dir    = args.data_dir,
        skip_upload = args.skip_upload,
        skip_delete = args.skip_delete,
        shutdown    = args.shutdown,
    )
