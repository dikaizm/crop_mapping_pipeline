import rasterio
import numpy as np

def label_filtering(in_path, out_path, keep_classes=[]):
    # Open raster
    with rasterio.open(in_path) as src:
        profile = src.profile.copy()
        data = src.read(1)  # read first band
        nodata_val = src.nodata if src.nodata is not None else 0

    # Ensure nodata is declared in the output profile
    profile.update(nodata=nodata_val)

    # Keep selected classes; everything else becomes nodata
    filtered = np.where(np.isin(data, keep_classes), data, nodata_val)

    # Save filtered raster
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(filtered, 1)

    print("✅ Saved filtered raster:", out_path)