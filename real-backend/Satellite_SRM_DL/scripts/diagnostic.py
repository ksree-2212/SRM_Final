import os
import numpy as np
import rasterio

def diagnostic():
    sr_paths = [
        r"C:\Users\mouni\OneDrive\Documents\sih 142\backend\Satellite_SRM_DL\outputs\sr\agriculture_sr.tif",
        r"C:\Users\mouni\OneDrive\Documents\sih 142\backend\Satellite_SRM_DL\outputs\sr\urban_sr.tif",
        r"C:\Users\mouni\OneDrive\Documents\sih 142\backend\Satellite_SRM_DL\outputs\sr\disaster_sr.tif"
    ]
    for path in sr_paths:
        if not os.path.exists(path):
            continue
        print(f"=== {os.path.basename(path)} ===")
        with rasterio.open(path) as src:
            data = src.read()
            for b in range(4):
                band = data[b]
                print(f"B{b}: min={band.min():.4f}, max={band.max():.4f}, mean={band.mean():.4f}, "
                      f"median={np.median(band):.4f}, p5={np.percentile(band, 5):.4f}, "
                      f"p25={np.percentile(band, 25):.4f}, p75={np.percentile(band, 75):.4f}, "
                      f"p95={np.percentile(band, 95):.4f}")
            print(f"Count of exactly 1.0 in B04: {(data[2] >= 0.999).sum()} out of {data[2].size}")
            print(f"Count of exactly 1.0 in B08: {(data[3] >= 0.999).sum()} out of {data[3].size}")

if __name__ == '__main__':
    diagnostic()
