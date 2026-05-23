"""Full tile scan — reads every 256×256 tile across all bands for each processed S2 file.

Usage:
    python stages/verify_tiles.py
    python stages/verify_tiles.py --data-dir /workspace/crop_mapping_pipeline/data/processed
    python stages/verify_tiles.py --years 2022
    python stages/verify_tiles.py --years 2022 2023 --tile-size 512
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import rasterio
import rasterio.windows

TILE = 256
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


def scan_file(path: Path, tile_size: int = TILE) -> tuple[Path, bool, str]:
    """Read every tile of every band. Return (path, ok, error_msg)."""
    try:
        with rasterio.open(path) as src:
            h, w, nb = src.height, src.width, src.count
            for band in range(1, nb + 1):
                for y in range(0, h, tile_size):
                    for x in range(0, w, tile_size):
                        ph = min(tile_size, h - y)
                        pw = min(tile_size, w - x)
                        src.read(band, window=rasterio.windows.Window(x, y, pw, ph))
        return path, True, ""
    except Exception as e:
        return path, False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Full tile scan of processed S2 files")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path to processed/ directory (default: pipeline data/processed/)",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        default=["2022", "2023"],
        help="Years to scan (default: 2022 2023)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=TILE,
        help=f"Tile size in pixels (default: {TILE})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel file workers (default: 4)",
    )
    args = parser.parse_args()

    s2_dir = args.data_dir / "s2"
    if not s2_dir.exists():
        print(f"ERROR: S2 directory not found: {s2_dir}", file=sys.stderr)
        sys.exit(1)

    files = []
    for yr in args.years:
        yr_dir = s2_dir / yr
        if not yr_dir.exists():
            print(f"WARN: year dir missing: {yr_dir}", file=sys.stderr)
            continue
        yr_files = sorted(yr_dir.glob("S2H_*_processed.tif"))
        if not yr_files:
            print(f"WARN: no processed TIFs in {yr_dir}", file=sys.stderr)
        files.extend((yr, f) for f in yr_files)

    if not files:
        print("No files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {len(files)} files with tile={args.tile_size}px, workers={args.workers}...")
    print()

    failed = []
    ok_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {
            pool.submit(scan_file, f, args.tile_size): (yr, f)
            for yr, f in files
        }
        # preserve order for display
        results = {}
        for future in as_completed(future_map):
            yr, f = future_map[future]
            path, ok, err = future.result()
            results[(yr, f)] = (ok, err)

    for yr, f in files:
        ok, err = results[(yr, f)]
        date = f.stem.replace("S2H_", "").replace("_processed", "").replace("_", "-")
        if ok:
            print(f"OK    {yr}/{date}")
            ok_count += 1
        else:
            print(f"FAIL  {yr}/{date}  —  {err}")
            failed.append((yr, f, err))

    print()
    print(f"{'='*50}")
    print(f"Result: {ok_count}/{len(files)} files OK")

    if failed:
        print(f"\nFAILED ({len(failed)}):")
        for yr, f, err in failed:
            print(f"  {yr}/{f.name}")
            print(f"    {err}")
        sys.exit(1)
    else:
        print("All files clean.")


if __name__ == "__main__":
    main()
