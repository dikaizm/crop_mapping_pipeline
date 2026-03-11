import os
import glob
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import argparse

def check_files(directory):
    print(f"Scanning for corrupt TIF files in: {directory}")
    tif_files = glob.glob(os.path.join(directory, "**/*.tif"), recursive=True)
    
    corrupt_files = []
    
    for tif in tqdm(tif_files):
        try:
            with rasterio.open(tif) as src:
                # Just opening isn't enough, we need to try reading a bit of data
                # to trigger the decompression error. We read a small 100x100 window.
                src.read(1, window=Window(0, 0, 100, 100))
        except Exception as e:
            print(f"\n[CORRUPT] {tif}")
            print(f"Error: {e}")
            corrupt_files.append(tif)
            
    if corrupt_files:
        print(f"\nFound {len(corrupt_files)} corrupt files:")
        for f in corrupt_files:
            print(f)
        print("\nRecommendation: Delete these files and re-run your data processing stage.")
    else:
        print("\nNo corrupt files found!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check for corrupt GeoTIFF files")
    parser.add_argument("dir", help="Directory to scan (e.g. data/processed)")
    args = parser.parse_args()
    
    check_files(args.dir)
