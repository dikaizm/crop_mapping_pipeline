"""
Stage 0.5b — Process multi-tile GEE exports for the new (larger) study area.

For large study areas GEE splits exports into tiles named:
    S2H_{year}_{YYYY_MM_DD}-{row_offset:010d}-{col_offset:010d}.tif

This script adds a merge step before the standard NoData assignment:

  For each year:
    1. Group raw tiles by date key
    2. Merge tiles → single mosaic per date  (rasterio.merge)
    3. Assign NoData (-9999, float32)         (same as process_data.py)
    4. Process CDL — reproject + clip to merged S2 grid + filter classes
    5. Upload processed files to Google Drive
    6. Delete raw tiles + temp merged files

Usage:
    python process_data_v2.py --years 2022                      # one year
    python process_data_v2.py --years 2022 2023 2024            # all years
    python process_data_v2.py --years 2022 --skip-upload        # process only
    python process_data_v2.py --years 2022 --skip-delete        # keep raw
    python process_data_v2.py --raw-s2-dir /path/to/tiles       # custom raw dir
    python process_data_v2.py --auth                            # generate OAuth token
    python process_data_v2.py --test-merge --years 2022         # merge + inspect only

Google Drive folder IDs (raw tiles):
    Set GDRIVE_RAW_S2_V2_FOLDER_IDS in config.py before downloading raw tiles.
    Processed outputs go to GDRIVE_PROCESSED_S2_V2_FOLDER_IDS.
"""

import os
import re
import sys
import logging
import argparse
import pathlib
import subprocess
import tempfile
from collections import defaultdict
from glob import glob
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.warp import reproject, Resampling
from dotenv import load_dotenv

_ROOT = pathlib.Path(__file__).parent.parent   # crop_mapping_pipeline/
load_dotenv(_ROOT / ".env")
sys.path.insert(0, str(_ROOT.parent))

from crop_mapping_pipeline.config import (
    S2_PROCESSED_DIR, CDL_BY_YEAR, PROCESSED_DIR,
    S2_NODATA, KEEP_CLASSES,
    GDRIVE_OAUTH_TOKEN,
)
from crop_mapping_pipeline.utils.label import label_filtering

log = logging.getLogger(__name__)

ALL_YEARS = ["2022", "2023", "2024"]

# GEE tile filename pattern: S2H_{year}_{YYYY_MM_DD}-{10digits}-{10digits}.tif
_TILE_RE = re.compile(
    r"^(S2H_\d{4}_\d{4}_\d{2}_\d{2})-(\d{10})-(\d{10})\.tif$"
)


# ── Tile grouping ───────────────────────────────────────────────────────────────

def group_tiles_by_date(raw_dir: str, year: str) -> dict:
    """
    Scan raw_dir for GEE multi-tile files that match:
        S2H_{year}_{YYYY_MM_DD}-{row:010d}-{col:010d}.tif

    Returns OrderedDict  {date_key → [(row, col, path), ...]},
    each list sorted by (row_offset, col_offset) for deterministic merge order.
    date_key example: "S2H_2022_2022_01_01"
    """
    pattern = str(Path(raw_dir) / f"S2H_{year}_*.tif")
    all_files = sorted(glob(pattern))
    if not all_files:
        log.warning("  No tiles found in %s matching year=%s", raw_dir, year)
        return {}

    groups = defaultdict(list)
    unmatched = []
    for fpath in all_files:
        fname = Path(fpath).name
        m = _TILE_RE.match(fname)
        if m:
            date_key  = m.group(1)             # e.g. "S2H_2022_2022_01_01"
            row_off   = int(m.group(2))
            col_off   = int(m.group(3))
            groups[date_key].append((row_off, col_off, fpath))
        else:
            unmatched.append(fname)

    if unmatched:
        log.warning("  %d file(s) did not match tile pattern — skipped: %s",
                    len(unmatched), unmatched[:5])

    # Sort tiles within each date by (row, col)
    sorted_groups = {}
    for dk in sorted(groups):
        sorted_groups[dk] = sorted(groups[dk], key=lambda x: (x[0], x[1]))

    log.info("  Found %d date(s) with tiles (year=%s)", len(sorted_groups), year)
    return sorted_groups


# ── Merge ───────────────────────────────────────────────────────────────────────

def merge_tiles(tile_paths: list, out_path: str) -> None:
    """
    Mosaic a list of TIF tiles into a single file using rasterio.merge.

    Strategy: first valid pixel wins (method="first").
    Output preserves the CRS, pixel size, and band count of the inputs.
    GEE tiles may overlap slightly at boundaries — rasterio.merge handles this.
    """
    if Path(out_path).exists():
        log.info("  Merged already exists: %s", Path(out_path).name)
        return

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    srcs = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic, transform = rio_merge(srcs, method="first")
        profile = srcs[0].profile.copy()
        profile.update(
            width     = mosaic.shape[2],
            height    = mosaic.shape[1],
            transform = transform,
            compress  = "deflate",
            predictor = 2,
            tiled     = True,
            blockxsize= 512,
            blockysize= 512,
        )
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mosaic)
    finally:
        for s in srcs:
            s.close()

    log.info("  Merged → %s  (%d tiles)", Path(out_path).name, len(tile_paths))


# ── NoData assignment ───────────────────────────────────────────────────────────

def assign_nodata(in_path: str, out_path: str) -> str:
    """
    Assign NoData (negative / NaN / Inf → S2_NODATA) and cast to float32.
    Skips if out_path already exists.
    Returns out_path.
    """
    if Path(out_path).exists():
        log.info("  Already processed: %s", Path(out_path).name)
        return out_path

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(in_path) as src:
        profile = src.profile.copy()
        profile.update(dtype="float32", nodata=S2_NODATA,
                       compress="deflate", predictor=2,
                       tiled=True, blockxsize=512, blockysize=512)
        data = src.read().astype(np.float32)

    invalid       = (data < 0) | np.isnan(data) | np.isinf(data)
    data[invalid] = S2_NODATA

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)

    log.info("  Processed: %s  (invalid_px=%s)",
             Path(out_path).name, f"{invalid.sum():,}")
    return out_path


# ── CDL processing ──────────────────────────────────────────────────────────────

def process_cdl(cdl_raw_path: str, s2_ref_path: str,
                out_reprojected: str, out_filtered: str) -> None:
    """Reproject CDL to S2 grid, then filter to KEEP_CLASSES."""
    if Path(out_reprojected).exists():
        log.info("  CDL reprojected already exists: %s",
                 Path(out_reprojected).name)
    else:
        log.info("  Reprojecting CDL → %s", Path(out_reprojected).name)
        with rasterio.open(s2_ref_path) as s2_ref:
            target_crs       = s2_ref.crs
            target_transform = s2_ref.transform
            target_width     = s2_ref.width
            target_height    = s2_ref.height

        with rasterio.open(cdl_raw_path) as cdl_src:
            dst_data = np.zeros((1, target_height, target_width), dtype=np.uint8)
            reproject(
                source        = rasterio.band(cdl_src, 1),
                destination   = dst_data,
                src_transform = cdl_src.transform,
                src_crs       = cdl_src.crs,
                dst_transform = target_transform,
                dst_crs       = target_crs,
                resampling    = Resampling.nearest,
            )

        profile = {
            "driver": "GTiff", "dtype": "uint8", "nodata": 0,
            "width": target_width, "height": target_height, "count": 1,
            "crs": target_crs, "transform": target_transform,
            "compress": "lzw",
        }
        Path(out_reprojected).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_reprojected, "w", **profile) as dst:
            dst.write(dst_data)
        log.info("  CDL reprojected: %s", Path(out_reprojected).name)

    if Path(out_filtered).exists():
        log.info("  CDL filtered already exists: %s", Path(out_filtered).name)
    else:
        log.info("  Filtering CDL classes → %s", Path(out_filtered).name)
        label_filtering(
            in_path      = out_reprojected,
            out_path     = out_filtered,
            keep_classes = KEEP_CLASSES,
        )
        log.info("  CDL filtered: %s", Path(out_filtered).name)


# ── Google Drive upload ─────────────────────────────────────────────────────────

def _build_drive_service():
    """Build an authenticated Google Drive API v3 service (OAuth token)."""
    import pickle
    from googleapiclient.discovery import build
    from google.auth.transport.requests import Request

    if not GDRIVE_OAUTH_TOKEN.exists():
        raise FileNotFoundError(
            f"OAuth token not found: {GDRIVE_OAUTH_TOKEN}\n"
            "Generate it locally:  python process_data_v2.py --auth\n"
            f"Then copy to server:  scp {GDRIVE_OAUTH_TOKEN} user@server:{GDRIVE_OAUTH_TOKEN}"
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
    """Upload a single file. Skips if already present. Returns GDrive file ID."""
    from googleapiclient.http import MediaFileUpload

    if service is None:
        service = _build_drive_service()

    fname  = Path(local_path).name
    query  = f"name='{fname}' and '{folder_id}' in parents and trashed=false"
    result = service.files().list(q=query, fields="files(id,name)").execute()
    if result.get("files"):
        log.info("  Already on GDrive: %s", fname)
        return result["files"][0]["id"]

    size    = os.path.getsize(local_path)
    log.info("  Uploading: %s  (%.0f MB)", fname, size / 1e6)
    media   = MediaFileUpload(local_path, mimetype="image/tiff", resumable=True)
    meta    = {"name": fname, "parents": [folder_id]}
    request = service.files().create(body=meta, media_body=media, fields="id")

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            log.info("    %d%% uploaded", int(status.progress() * 100))

    file_id = response.get("id")
    log.info("  Uploaded: %s  (id=%s)", fname, file_id)
    return file_id


def upload_year(s2_processed_paths: list, cdl_filtered_path: str,
                year: str, s2_folder_ids: dict, cdl_folder_id: str) -> None:
    """Upload all processed S2 + CDL files for one year."""
    s2_folder = s2_folder_ids.get(year, "")
    if not s2_folder or not cdl_folder_id:
        raise ValueError(
            f"s2_folder_ids['{year}'] and cdl_folder_id must be set before uploading. "
            "Pass --skip-upload to skip."
        )

    service = _build_drive_service()
    log.info("  Uploading %d S2 files (year=%s)...", len(s2_processed_paths), year)
    for path in s2_processed_paths:
        upload_file(path, s2_folder, service)

    if cdl_filtered_path and Path(cdl_filtered_path).exists():
        log.info("  Uploading CDL filtered file...")
        upload_file(cdl_filtered_path, cdl_folder_id, service)


# ── Cleanup ─────────────────────────────────────────────────────────────────────

def delete_files(paths: list, label: str = "raw") -> None:
    freed = 0
    for p in paths:
        if Path(p).exists():
            size = os.path.getsize(p)
            os.remove(p)
            freed += size
            log.info("  Deleted %s: %s", label, Path(p).name)
    log.info("  Freed: %.2f GB", freed / 1e9)


# ── Shutdown ────────────────────────────────────────────────────────────────────

def _schedule_shutdown(delay_min: int = 8) -> None:
    import time, urllib.request, urllib.error, json

    pod_id  = os.environ.get("RUNPOD_POD_ID")
    api_key = os.environ.get("RUNPOD_API_KEY")

    if pod_id and api_key:
        log.warning("=" * 60)
        log.warning("RunPod pod %s will stop in %d minutes.", pod_id, delay_min)
        log.warning("=" * 60)
        time.sleep(delay_min * 60)
        query = (f'{{"query": "mutation {{ podStop(input: {{podId: \\"{pod_id}\\"}}) '
                 f'{{ id desiredStatus }} }}"}}')
        req   = urllib.request.Request(
            "https://api.runpod.io/graphql",
            data    = query.encode(),
            headers = {"Content-Type": "application/json",
                       "Authorization": f"Bearer {api_key}"},
        )
        try:
            with urllib.request.urlopen(req) as resp:
                log.info("Pod stop response: %s", json.loads(resp.read()))
        except urllib.error.URLError as e:
            log.error("Failed to stop pod: %s", e)
    else:
        log.warning("=" * 60)
        log.warning("VPS SHUTDOWN in %d minutes.  Cancel: sudo shutdown -c", delay_min)
        log.warning("=" * 60)
        try:
            subprocess.run(["sudo", "shutdown", "-h", f"+{delay_min}"], check=True)
        except Exception as e:
            log.error("Shutdown failed: %s", e)


# ── Test helper ─────────────────────────────────────────────────────────────────

def process_date_batch(
    date_groups  : dict,
    yr           : str,
    s2_out_dir,
    merge_tmp_dir,
    keep_merged  : bool = False,
) -> tuple:
    """
    Merge + assign NoData for a subset of dates already downloaded locally.

    Parameters
    ----------
    date_groups   : {date_key: [(row, col, path), ...]} — output of group_tiles_by_date(),
                    filtered to the desired batch.
    yr            : year string (used for logging).
    s2_out_dir    : Path — directory to write *_processed.tif files.
    merge_tmp_dir : Path — temporary directory for intermediate merged TIFs.
    keep_merged   : bool — keep intermediate merged files.

    Returns
    -------
    (raw_tile_paths, processed_paths, s2_ref_path)
        raw_tile_paths : list[str] — all raw tile paths consumed (for deletion).
        processed_paths: list[str] — all *_processed.tif paths produced.
        s2_ref_path    : str | None — first processed file (CDL grid reference).
    """
    Path(s2_out_dir).mkdir(parents=True, exist_ok=True)
    Path(merge_tmp_dir).mkdir(parents=True, exist_ok=True)

    raw_tile_paths  = []
    processed_paths = []
    s2_ref_path     = None

    for date_key, tiles in date_groups.items():
        tile_paths = [t[2] for t in tiles]
        raw_tile_paths.extend(tile_paths)

        merged_path    = str(Path(merge_tmp_dir) / f"{date_key}_merged.tif")
        processed_path = str(Path(s2_out_dir)   / f"{date_key}_processed.tif")

        merge_tiles(tile_paths, merged_path)
        assign_nodata(merged_path, processed_path)
        processed_paths.append(processed_path)

        if s2_ref_path is None:
            s2_ref_path = processed_path

        if not keep_merged and Path(merged_path).exists():
            os.remove(merged_path)

    log.info("  Batch: processed %d date(s) for year %s", len(processed_paths), yr)
    return raw_tile_paths, processed_paths, s2_ref_path


def test_merge(raw_dir: str, year: str, out_dir: str) -> None:
    """
    Merge + inspect only — no CDL, no upload, no delete.
    Useful for verifying bounds and band count before full processing.
    """
    groups = group_tiles_by_date(raw_dir, year)
    if not groups:
        log.error("No tile groups found.")
        return

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for date_key, tiles in groups.items():
        tile_paths  = [t[2] for t in tiles]
        merged_path = str(Path(out_dir) / f"{date_key}_merged.tif")

        log.info("─" * 60)
        log.info("Date key : %s", date_key)
        log.info("Tiles    : %d", len(tiles))
        for row, col, p in tiles:
            log.info("  offset=(%d,%d)  %s", row, col, Path(p).name)

        merge_tiles(tile_paths, merged_path)

        with rasterio.open(merged_path) as src:
            b = src.bounds
            log.info("Merged   : %dx%d  bands=%d  dtype=%s",
                     src.width, src.height, src.count, src.dtypes[0])
            log.info("SW       : (%.5f, %.5f)", b.left, b.bottom)
            log.info("NE       : (%.5f, %.5f)", b.right, b.top)
            log.info("CRS      : %s", src.crs)


# ── Main ────────────────────────────────────────────────────────────────────────

def main(
    years              : list = None,
    raw_s2_dir         : str  = None,
    raw_cdl_dir        : str  = None,
    data_dir           : str  = None,
    s2_folder_ids      : dict = None,   # GDrive folder IDs for processed S2 per year
    cdl_folder_id      : str  = None,   # GDrive folder ID for processed CDL
    skip_upload        : bool = False,
    skip_delete        : bool = False,
    keep_merged        : bool = False,
    flat_dir           : bool = False,  # raw_s2_dir has no year subdir
    shutdown           : bool = False,
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
        log.info("Year: %s", yr)
        log.info("=" * 60)

        # ── Locate raw S2 tile directory ───────────────────────────────────────
        if flat_dir and raw_s2_dir:
            s2_raw_dir = pathlib.Path(raw_s2_dir)
        elif raw_s2_dir:
            s2_raw_dir = pathlib.Path(raw_s2_dir) / yr
        else:
            s2_raw_dir = _ROOT / "data" / "raw" / "s2" / yr

        # ── Group tiles by date ────────────────────────────────────────────────
        groups = group_tiles_by_date(str(s2_raw_dir), yr)
        if not groups:
            log.warning("  No tile groups for year %s — skipping", yr)
            continue

        # Temp dir for merged (pre-nodata) files; cleaned up unless --keep-merged
        merge_tmp_dir = _ROOT / "data" / "raw" / "s2" / yr / "_merged"
        merge_tmp_dir.mkdir(parents=True, exist_ok=True)

        s2_out_dir   = S2_PROCESSED_DIR / yr
        s2_out_dir.mkdir(parents=True, exist_ok=True)

        all_raw_tiles, all_processed, s2_ref_path = process_date_batch(
            date_groups   = groups,
            yr            = yr,
            s2_out_dir    = s2_out_dir,
            merge_tmp_dir = merge_tmp_dir,
            keep_merged   = keep_merged,
        )
        log.info("  Processed %d date(s) for year %s", len(all_processed), yr)

        # ── CDL processing ─────────────────────────────────────────────────────
        cdl_dir  = (pathlib.Path(raw_cdl_dir) if raw_cdl_dir
                    else _ROOT / "data" / "raw" / "cdl")
        cdl_raw  = next(
            (p for p in glob(str(cdl_dir / f"{yr}_30m_cdls" / "*.tif"))),
            None,
        )
        cdl_filtered = None
        if not cdl_raw:
            log.warning("  Raw CDL for %s not found — skipping CDL processing", yr)
        elif s2_ref_path is None:
            log.warning("  No processed S2 reference — skipping CDL processing")
        else:
            cdl_out_dir     = S2_PROCESSED_DIR.parent / "cdl"
            cdl_reprojected = str(cdl_out_dir / f"cdl_{yr}_study_area.tif")
            cdl_filtered    = str(cdl_out_dir / f"cdl_{yr}_study_area_filtered.tif")
            process_cdl(cdl_raw, s2_ref_path, cdl_reprojected, cdl_filtered)

        # ── Upload ─────────────────────────────────────────────────────────────
        if not skip_upload:
            if s2_folder_ids and cdl_folder_id:
                log.info("  Uploading to Google Drive...")
                upload_year(
                    s2_processed_paths = all_processed,
                    cdl_filtered_path  = cdl_filtered or "",
                    year               = yr,
                    s2_folder_ids      = s2_folder_ids,
                    cdl_folder_id      = cdl_folder_id,
                )
            else:
                log.warning(
                    "  --skip-upload not set but no GDrive folder IDs provided — "
                    "pass --s2-folder-ids / --cdl-folder-id or use --skip-upload"
                )
        else:
            log.info("  Upload skipped (--skip-upload)")

        # ── Delete raw tiles ───────────────────────────────────────────────────
        if not skip_delete:
            log.info("  Deleting raw tiles...")
            delete_files(all_raw_tiles, label="tile")
        else:
            log.info("  Raw tiles kept (--skip-delete)")

        log.info("Year %s done.\n", yr)

    if shutdown:
        _schedule_shutdown(delay_min=8)


def generate_oauth_token():
    """Run OAuth flow in browser (run locally once, then copy token to server)."""
    import pickle
    from google_auth_oauthlib.flow import InstalledAppFlow
    from crop_mapping_pipeline.config import GDRIVE_OAUTH_SECRET, GDRIVE_OAUTH_TOKEN

    if not GDRIVE_OAUTH_SECRET.exists():
        raise FileNotFoundError(
            f"OAuth client secret not found: {GDRIVE_OAUTH_SECRET}\n"
            "Download from Google Cloud Console → APIs & Services → Credentials."
        )

    flow  = InstalledAppFlow.from_client_secrets_file(
        str(GDRIVE_OAUTH_SECRET),
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    creds = flow.run_local_server(port=0)

    with open(GDRIVE_OAUTH_TOKEN, "wb") as f:
        pickle.dump(creds, f)

    print(f"Token saved: {GDRIVE_OAUTH_TOKEN}")
    print(f"Copy to server:\n  scp {GDRIVE_OAUTH_TOKEN} user@server:{GDRIVE_OAUTH_TOKEN}")


# ── CLI ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process multi-tile GEE S2 exports: merge → NoData → CDL → upload."
    )
    parser.add_argument(
        "--years", nargs="+", default=None, choices=ALL_YEARS, metavar="YEAR",
        help=f"Years to process (default: all — {ALL_YEARS})",
    )
    parser.add_argument(
        "--raw-s2-dir", default=None,
        help=(
            "Directory containing raw S2 tile dirs. "
            "By default the script appends /{year}/ to this path. "
            "Use --flat-dir if tiles are directly in this directory (no year subdir)."
        ),
    )
    parser.add_argument(
        "--flat-dir", action="store_true",
        help=(
            "Treat --raw-s2-dir as the direct tile directory (no year subdir). "
            "Useful when pointing at a GDrive-mounted export folder."
        ),
    )
    parser.add_argument(
        "--raw-cdl-dir", default=None,
        help="Directory containing raw CDL folders (YYYY_30m_cdls/).",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Override processed output directory (default: data/processed/).",
    )
    parser.add_argument(
        "--s2-folder-ids", nargs="+", metavar="YEAR:FOLDER_ID",
        default=None,
        help=(
            "GDrive folder IDs for processed S2 output, one per year. "
            "Format: 2022:FOLDER_ID 2023:FOLDER_ID 2024:FOLDER_ID"
        ),
    )
    parser.add_argument(
        "--cdl-folder-id", default=None,
        help="GDrive folder ID for processed CDL output.",
    )
    parser.add_argument(
        "--skip-upload", action="store_true",
        help="Process files but do not upload to Google Drive.",
    )
    parser.add_argument(
        "--skip-delete", action="store_true",
        help="Keep raw tile files after processing.",
    )
    parser.add_argument(
        "--keep-merged", action="store_true",
        help="Keep intermediate merged (pre-NoData) TIFs for inspection.",
    )
    parser.add_argument(
        "--shutdown", action="store_true",
        help="Stop the VPS 8 minutes after processing (Linux / RunPod).",
    )
    parser.add_argument(
        "--auth", action="store_true",
        help="Generate OAuth token via browser (run locally once).",
    )
    parser.add_argument(
        "--test-merge", action="store_true",
        help=(
            "Merge tiles and print bounds/shape only — no CDL, upload, or delete. "
            "Output goes to data/raw/s2/{year}/_merged/. "
            "Requires --years."
        ),
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

    # Parse --s2-folder-ids YEAR:ID pairs
    s2_folder_ids = None
    if args.s2_folder_ids:
        s2_folder_ids = {}
        for item in args.s2_folder_ids:
            yr, fid = item.split(":", 1)
            s2_folder_ids[yr] = fid

    if args.test_merge:
        if not args.years:
            parser.error("--test-merge requires --years")
        for yr in (args.years or ALL_YEARS):
            if args.flat_dir and args.raw_s2_dir:
                s2_raw_dir = pathlib.Path(args.raw_s2_dir)
            elif args.raw_s2_dir:
                s2_raw_dir = pathlib.Path(args.raw_s2_dir) / yr
            else:
                s2_raw_dir = _ROOT / "data" / "raw" / "s2" / yr
            out_dir = _ROOT / "data" / "raw" / "s2" / yr / "_merged"
            test_merge(str(s2_raw_dir), yr, str(out_dir))
        sys.exit(0)

    main(
        years         = args.years,
        raw_s2_dir    = args.raw_s2_dir,
        raw_cdl_dir   = args.raw_cdl_dir,
        data_dir      = args.data_dir,
        s2_folder_ids = s2_folder_ids,
        cdl_folder_id = args.cdl_folder_id,
        skip_upload   = args.skip_upload,
        skip_delete   = args.skip_delete,
        keep_merged   = args.keep_merged,
        flat_dir      = args.flat_dir,
        shutdown      = args.shutdown,
    )
