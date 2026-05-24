"""
Upload local processed S2 files to Google Drive (processed_v5 folder).

Mirrors the year/subdir structure: processed_v5/s2/{year}/ on GDrive.

Usage:
    python stages/upload_to_gdrive.py --years 2022
    python stages/upload_to_gdrive.py --years 2022 2023 2024 --workers 2
    python stages/upload_to_gdrive.py --years 2022 --overwrite
    python stages/upload_to_gdrive.py --local-dir /custom/path --years 2022
"""

import os
import sys
import logging
import argparse
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

_ROOT = pathlib.Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")
sys.path.insert(0, str(_ROOT.parent))

from crop_mapping_pipeline.config import (
    GDRIVE_OAUTH_TOKEN,
    GDRIVE_PROCESSED_V5_FOLDER_ID,
    PROCESSED_V5_DIR,
)

log = logging.getLogger(__name__)

ALL_YEARS = ["2022", "2023", "2024"]


# ── GDrive helpers ────────────────────────────────────────────────────────────

def _build_service():
    import pickle
    from googleapiclient.discovery import build
    from google.auth.transport.requests import Request

    if not GDRIVE_OAUTH_TOKEN.exists():
        raise FileNotFoundError(
            f"OAuth token not found: {GDRIVE_OAUTH_TOKEN}\n"
            "Generate it:  python stages/process_data_v5.py --auth"
        )
    with open(GDRIVE_OAUTH_TOKEN, "rb") as f:
        creds = pickle.load(f)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(GDRIVE_OAUTH_TOKEN, "wb") as f:
            pickle.dump(creds, f)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _get_or_create_folder(parent_id: str, name: str, service) -> str:
    q = (f"name='{name}' and '{parent_id}' in parents "
         f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
    res = service.files().list(q=q, fields="files(id)").execute()
    folders = res.get("files", [])
    if folders:
        return folders[0]["id"]
    meta = {"name": name, "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id]}
    f = service.files().create(body=meta, fields="id").execute()
    log.info("  Created GDrive folder: %s", name)
    return f["id"]


def _list_gdrive_names(folder_id: str, service) -> set:
    names, page_token = set(), None
    while True:
        kwargs = dict(q=f"'{folder_id}' in parents and trashed=false",
                      fields="nextPageToken, files(name)", pageSize=1000)
        if page_token:
            kwargs["pageToken"] = page_token
        res = service.files().list(**kwargs).execute()
        names |= {f["name"] for f in res.get("files", [])}
        page_token = res.get("nextPageToken")
        if not page_token:
            break
    return names


def upload_file(local_path: pathlib.Path, folder_id: str, service,
                overwrite: bool = False) -> str:
    from googleapiclient.http import MediaFileUpload

    fname = local_path.name
    q = f"name='{fname}' and '{folder_id}' in parents and trashed=false"
    existing = service.files().list(q=q, fields="files(id,name)").execute().get("files", [])
    size = local_path.stat().st_size
    media = MediaFileUpload(str(local_path), mimetype="image/tiff", resumable=True)

    if existing and overwrite:
        log.info("  Replacing: %s  (%.0f MB)", fname, size / 1e6)
        request = service.files().update(fileId=existing[0]["id"],
                                         media_body=media, fields="id")
    elif existing:
        log.info("  Skip (exists): %s", fname)
        return existing[0]["id"]
    else:
        log.info("  Uploading: %s  (%.0f MB)", fname, size / 1e6)
        request = service.files().create(
            body={"name": fname, "parents": [folder_id]},
            media_body=media, fields="id")

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            log.info("    %s  %d%%", fname, int(status.progress() * 100))
    log.info("  Done: %s", fname)
    return response.get("id")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(years: list, local_dir: pathlib.Path, overwrite: bool, workers: int,
         folder_id: str = None):
    service = _build_service()

    root_id = folder_id or GDRIVE_PROCESSED_V5_FOLDER_ID
    # Resolve GDrive s2/ subfolder under target folder
    s2_folder_id = _get_or_create_folder(root_id, "s2", service)

    for yr in years:
        log.info("=" * 60)
        log.info("Year: %s", yr)
        log.info("=" * 60)

        # Local source dir — try s2/{yr}, then {yr}, then flat
        for candidate in [local_dir / "s2" / yr, local_dir / yr, local_dir]:
            if candidate.exists():
                src_dir = candidate
                break
        files = sorted(src_dir.glob(f"S2H_{yr}_*_processed.tif"))
        if not files:
            log.warning("  No processed files found in %s", src_dir)
            continue
        log.info("  Found %d file(s) in %s", len(files), src_dir)

        # GDrive year subfolder
        yr_folder_id = _get_or_create_folder(s2_folder_id, yr, service)

        # Check what's already uploaded
        if not overwrite:
            existing_names = _list_gdrive_names(yr_folder_id, service)
            files = [f for f in files if f.name not in existing_names]
            log.info("  %d file(s) to upload (skipping already present)", len(files))

        if not files:
            log.info("  All files already on GDrive for year %s", yr)
            continue

        # Upload with thread pool (one service per thread)
        errors = []

        def _upload(path: pathlib.Path):
            svc = _build_service()
            try:
                upload_file(path, yr_folder_id, svc, overwrite=overwrite)
            except Exception as exc:
                log.error("  FAILED %s: %s", path.name, exc)
                errors.append(path.name)

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="upload") as pool:
            futures = {pool.submit(_upload, f): f for f in files}
            for fut in as_completed(futures):
                exc = fut.exception()
                if exc:
                    log.error("  Uncaught: %s", exc)

        if errors:
            log.warning("  %d error(s): %s", len(errors), errors)
        else:
            log.info("  Year %s done.", yr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload local processed S2 to GDrive.")
    parser.add_argument("--years", nargs="+", default=ALL_YEARS, choices=ALL_YEARS)
    parser.add_argument("--local-dir", default=None,
                        help="Root of processed dir (default: PROCESSED_V5_DIR from .env)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace files already on GDrive")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel upload threads (default: 1)")
    parser.add_argument("--folder-id", default=None,
                        help="Target GDrive folder ID (default: GDRIVE_PROCESSED_V5_FOLDER_ID)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )

    local_dir = pathlib.Path(args.local_dir) if args.local_dir else PROCESSED_V5_DIR
    if local_dir is None:
        parser.error("No local dir: pass --local-dir or set PROCESSED_V5_DIR in .env")

    main(years=args.years, local_dir=local_dir, overwrite=args.overwrite,
         workers=args.workers, folder_id=args.folder_id)
