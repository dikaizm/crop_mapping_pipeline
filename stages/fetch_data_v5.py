"""
Stage 0b (v5) — Download raw S2 files from Google Drive.

Unlike v2, GEE exports one file per date (no tile splitting):
    S2H_{year}_{YYYY_MM_DD}.tif

Files are sorted into year subdirectories:
    {output_dir}/2022/S2H_2022_*.tif
    {output_dir}/2023/S2H_2023_*.tif
    {output_dir}/2024/S2H_2024_*.tif

Usage:
    python fetch_data_v5.py --folder-id FOLDER_ID
    python fetch_data_v5.py --folder-id FOLDER_ID --years 2022
    python fetch_data_v5.py --folder-id FOLDER_ID --years 2022 --overwrite
    python fetch_data_v5.py --folder-id FOLDER_ID --list-files
    python fetch_data_v5.py --auth
"""

import os
import re
import sys
import argparse
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT.parent))

from crop_mapping_pipeline.config import GDRIVE_OAUTH_TOKEN

log = logging.getLogger(__name__)

ALL_YEARS = ["2022", "2023", "2024"]

# S2H_{year}_{YYYY_MM_DD}.tif  (no tile offsets)
_FILE_RE = re.compile(r"^S2H_(\d{4})_(\d{4}_\d{2}_\d{2})\.tif$")


# ── Auth ────────────────────────────────────────────────────────────────────────

def _build_drive_service():
    import pickle
    from googleapiclient.discovery import build
    from google.auth.transport.requests import Request

    if not GDRIVE_OAUTH_TOKEN.exists():
        raise FileNotFoundError(
            f"OAuth token not found: {GDRIVE_OAUTH_TOKEN}\n"
            "Run:  python stages/process_data_v5.py --auth"
        )
    with open(GDRIVE_OAUTH_TOKEN, "rb") as f:
        creds = pickle.load(f)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(GDRIVE_OAUTH_TOKEN, "wb") as f:
            pickle.dump(creds, f)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


_thread_local = threading.local()


def _get_thread_service():
    if not hasattr(_thread_local, "service"):
        _thread_local.service = _build_drive_service()
    return _thread_local.service


# ── Filename helpers ────────────────────────────────────────────────────────────

def _year_from_filename(fname: str) -> str:
    m = _FILE_RE.match(fname)
    return m.group(1) if m else ""


def _date_key_from_filename(fname: str) -> str:
    """Return date key, e.g. 'S2H_2022_2022_01_16'."""
    m = _FILE_RE.match(fname)
    if not m:
        return ""
    return f"S2H_{m.group(1)}_{m.group(2)}"


# ── Folder listing ──────────────────────────────────────────────────────────────

def _list_children(service, folder_id: str):
    """Return (tifs, subfolders) as {name: id} dicts for direct children."""
    tifs, subfolders = {}, {}
    page_token = None
    while True:
        resp = service.files().list(
            q         = f"'{folder_id}' in parents and trashed = false",
            fields    = "nextPageToken, files(id, name, mimeType)",
            pageSize  = 1000,
            pageToken = page_token,
        ).execute()
        for f in resp.get("files", []):
            if f["mimeType"] == "application/vnd.google-apps.folder":
                subfolders[f["name"]] = f["id"]
            else:
                tifs[f["name"]] = f["id"]
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return tifs, subfolders


def _find_subfolder(service, folder_id: str, name: str):
    """Return folder ID of a named subfolder, or None."""
    _, subs = _list_children(service, folder_id)
    return subs.get(name)


def list_folder(folder_id: str, years: list = None) -> dict:
    """Return {filename: file_id} for all S2 TIFs reachable from folder_id.

    Handles three layouts automatically:
      - Flat:            folder/S2H_*.tif
      - Year-subdir:     folder/2022/S2H_*.tif
      - s2/year-subdir:  folder/s2/2022/S2H_*.tif  (GDrive processed layout)
    """
    service    = _build_drive_service()
    name_to_id = {}

    def _collect_s2(fid):
        tifs, subs = _list_children(service, fid)
        for name, tid in tifs.items():
            if _FILE_RE.match(name):
                name_to_id[name] = tid
        for sub_name, sub_id in subs.items():
            _collect_s2(sub_id)   # recurse into any subfolders (year dirs, etc.)

    # check if there's an s2/ subfolder — if so, only recurse into that
    s2_sub = _find_subfolder(service, folder_id, "s2")
    _collect_s2(s2_sub if s2_sub else folder_id)

    log.info("  %d S2 file(s) found in folder", len(name_to_id))

    if years:
        years_set  = set(years)
        name_to_id = {n: fid for n, fid in name_to_id.items()
                      if _year_from_filename(n) in years_set}
        log.info("  %d file(s) after year filter=%s", len(name_to_id), years)

    return name_to_id


def list_dates_by_year(folder_id: str, years: list = None) -> dict:
    """Return {year: sorted [date_keys]} from GDrive folder."""
    name_to_id    = list_folder(folder_id, years=years)
    dates_by_year: dict = {}
    for fname in name_to_id:
        yr  = _year_from_filename(fname)
        key = _date_key_from_filename(fname)
        if yr and key:
            dates_by_year.setdefault(yr, set()).add(key)
    return {yr: sorted(keys) for yr, keys in sorted(dates_by_year.items())}


# ── Download ────────────────────────────────────────────────────────────────────

def _download_one(fname: str, file_id: str, output_dir: str,
                  overwrite: bool = False) -> tuple[str, str]:
    """Download one file into {output_dir}/{year}/. Returns (path, status)."""
    from googleapiclient.http import MediaIoBaseDownload

    yr = _year_from_filename(fname)
    if not yr:
        log.warning("  Cannot parse year from '%s' — skipping", fname)
        return "", "error"

    yr_dir   = Path(output_dir) / yr
    yr_dir.mkdir(parents=True, exist_ok=True)
    out_path = yr_dir / fname

    if not overwrite and out_path.exists() and out_path.stat().st_size > 0:
        log.info("  Skip (exists): %s/%s", yr, fname)
        return str(out_path), "skip"

    service = _get_thread_service()
    request = service.files().get_media(fileId=file_id)
    tmp     = out_path.with_suffix(".tmp.tif")
    try:
        with open(tmp, "wb") as fh:
            dl   = MediaIoBaseDownload(fh, request, chunksize=50 * 1024 * 1024)
            done = False
            while not done:
                status, done = dl.next_chunk()
                if status:
                    log.info("  %s: %d%%", fname, int(status.progress() * 100))
        tmp.rename(out_path)
        log.info("  Done: %s/%s  (%.0f MB)", yr, fname, out_path.stat().st_size / 1e6)
        return str(out_path), "new"
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        log.error("  Failed: %s (%s)", fname, exc)
        return "", "error"


def _download_many(name_to_id: dict, output_dir: str,
                   overwrite: bool = False, workers: int = 2) -> list:
    """Parallel download — thread-local Drive service per worker."""
    total     = len(name_to_id)
    results   = []
    new_count = skipped = errors = 0
    lock      = threading.Lock()

    log.info("  Downloading %d file(s) with %d worker(s)...", total, workers)

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dl") as pool:
        futures = {
            pool.submit(_download_one, fname, fid, output_dir, overwrite): fname
            for fname, fid in name_to_id.items()
        }
        done_n = 0
        for fut in as_completed(futures):
            fname   = futures[fut]
            done_n += 1
            try:
                path, status = fut.result()
            except Exception as exc:
                log.error("  [%d/%d] Error %s: %s", done_n, total, fname, exc)
                with lock:
                    errors += 1
                continue
            with lock:
                if status == "new":
                    results.append(path)
                    new_count += 1
                elif status == "skip":
                    results.append(path)
                    skipped += 1
                else:
                    errors += 1

    log.info("  Done: %d new, %d skipped, %d errors", new_count, skipped, errors)
    return results


def download_folder_by_year(folder_id: str, output_dir: str,
                            years: list = None, overwrite: bool = False,
                            workers: int = 2) -> list:
    """Download all S2 files from folder, sorted into year subdirs."""
    name_to_id = list_folder(folder_id, years=years)
    if not name_to_id:
        log.warning("  No files to download.")
        return []
    return _download_many(name_to_id, output_dir, overwrite, workers)


def download_cdl(folder_id: str, output_dir: str,
                 overwrite: bool = False, workers: int = 2) -> list:
    """Download CDL TIFs from folder/cdl/ into {output_dir}/cdl/."""
    service = _build_drive_service()
    cdl_fid = _find_subfolder(service, folder_id, "cdl")
    if not cdl_fid:
        log.warning("  No 'cdl' subfolder found in folder %s", folder_id)
        return []
    tifs, _ = _list_children(service, cdl_fid)
    if not tifs:
        log.warning("  No CDL files found in cdl/ subfolder")
        return []

    from googleapiclient.http import MediaIoBaseDownload
    cdl_dir = Path(output_dir) / "cdl"
    cdl_dir.mkdir(parents=True, exist_ok=True)
    results, new_count, skipped, errors = [], 0, 0, 0

    log.info("  Downloading %d CDL file(s) → %s", len(tifs), cdl_dir)
    for fname, fid in sorted(tifs.items()):
        out_path = cdl_dir / fname
        if not overwrite and out_path.exists() and out_path.stat().st_size > 0:
            log.info("  Skip (exists): cdl/%s", fname)
            skipped += 1
            results.append(str(out_path))
            continue
        tmp = out_path.with_suffix(".tmp.tif")
        try:
            svc     = _build_drive_service()
            request = svc.files().get_media(fileId=fid)
            with open(tmp, "wb") as fh:
                dl   = MediaIoBaseDownload(fh, request, chunksize=50 * 1024 * 1024)
                done = False
                while not done:
                    status, done = dl.next_chunk()
            tmp.rename(out_path)
            log.info("  Done: cdl/%s  (%.0f MB)", fname, out_path.stat().st_size / 1e6)
            new_count += 1
            results.append(str(out_path))
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            log.error("  Failed: cdl/%s (%s)", fname, exc)
            errors += 1

    log.info("  CDL done: %d new, %d skipped, %d errors", new_count, skipped, errors)
    return results


def download_date_keys(folder_id: str, output_dir: str,
                       date_keys: list, overwrite: bool = False,
                       workers: int = 2) -> list:
    """Download only files matching the given date keys."""
    date_keys_set = set(date_keys)
    years         = {_year_from_filename(dk + ".tif") for dk in date_keys}
    years         = {y for y in years if y}

    name_to_id = list_folder(folder_id, years=list(years) if years else None)
    name_to_id = {
        n: fid for n, fid in name_to_id.items()
        if _date_key_from_filename(n) in date_keys_set
    }
    if not name_to_id:
        log.warning("  No files found for date_keys=%s", date_keys)
        return []
    return _download_many(name_to_id, output_dir, overwrite, workers)


# ── Verify ──────────────────────────────────────────────────────────────────────

def verify(output_dir: str, years: list = None) -> bool:
    years  = years or ALL_YEARS
    all_ok = True
    print(f"\nS2 files under {output_dir}/{{year}}/:")
    for yr in sorted(years):
        yr_dir = Path(output_dir) / yr
        files  = sorted(yr_dir.glob(f"S2H_{yr}_*.tif")) if yr_dir.exists() else []
        files  = [f for f in files if _FILE_RE.match(f.name)]
        status = "OK" if files else "MISSING"
        print(f"  {yr}: {status}  {len(files)} date(s)")
        for f in files:
            print(f"    {f.name}  ({f.stat().st_size / 1e6:.0f} MB)")
        if not files:
            all_ok = False
    return all_ok


def generate_oauth_token():
    import pickle
    from google_auth_oauthlib.flow import InstalledAppFlow
    from crop_mapping_pipeline.config import GDRIVE_OAUTH_SECRET

    flow  = InstalledAppFlow.from_client_secrets_file(
        str(GDRIVE_OAUTH_SECRET),
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    creds = flow.run_local_server(port=0)
    with open(GDRIVE_OAUTH_TOKEN, "wb") as f:
        pickle.dump(creds, f)
    print(f"Token saved: {GDRIVE_OAUTH_TOKEN}")


# ── CLI ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download single-file-per-date S2 exports from GDrive."
    )
    parser.add_argument("--folder-id", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--years", nargs="+", default=None, choices=ALL_YEARS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--list-files", action="store_true")
    parser.add_argument("--include-cdl", action="store_true",
                        help="Also download CDL files from cdl/ subfolder")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--auth", action="store_true")
    args = parser.parse_args()

    if args.auth:
        generate_oauth_token()
        sys.exit(0)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )

    folder_id = args.folder_id
    if not folder_id:
        try:
            from crop_mapping_pipeline.config import GDRIVE_RAW_S2_V2_FOLDER_ID
            folder_id = GDRIVE_RAW_S2_V2_FOLDER_ID
        except ImportError:
            parser.error("--folder-id required")

    output_dir = args.output_dir or str(_ROOT / "data" / "raw" / "s2")

    if args.list_files:
        name_to_id = list_folder(folder_id, years=args.years)
        print(f"\n{len(name_to_id)} file(s):")
        for name in sorted(name_to_id):
            print(f"  {name}")
        sys.exit(0)

    if args.verify_only:
        ok = verify(output_dir, years=args.years)
        sys.exit(0 if ok else 1)

    # S2 → {output_dir}/s2/{year}/  to match processed folder structure
    s2_output_dir = str(Path(output_dir) / "s2")
    download_folder_by_year(folder_id, s2_output_dir,
                            years=args.years, overwrite=args.overwrite,
                            workers=args.workers)
    if args.include_cdl:
        # CDL → {output_dir}/cdl/
        download_cdl(folder_id, output_dir, overwrite=args.overwrite,
                     workers=args.workers)
    verify(s2_output_dir, years=args.years)
