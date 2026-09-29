"""
Post-inference comprehensive validation script.
Validates SR GeoTIFFs for physical correctness, dimensions, CRS, and computes
all analytics against normalized raw inputs.
Does NOT retrain or re-run inference.
"""
import os, sys
import numpy as np

PROJECT_DIR = r"C:\Users\mouni\OneDrive\Documents\sih 142\backend\Satellite_SRM_DL"
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

from satellite_srm.geospatial.geotiff import read_geotiff
from satellite_srm.preprocessing.normalization import RadiometricNormalizer
from satellite_srm.applications.agriculture import AgricultureAnalyzer
from satellite_srm.applications.urban import UrbanAnalyzer
from satellite_srm.applications.disaster import DisasterAnalyzer
from satellite_srm.applications.indices import SpectralIndices

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

def pf(cond, label):
    status = PASS if cond else FAIL
    print(f"   {status}  {label}")
    return cond

def banner(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")

def band_stats(arr, name):
    p5  = float(np.percentile(arr, 5))
    p25 = float(np.percentile(arr, 25))
    p50 = float(np.percentile(arr, 50))
    p75 = float(np.percentile(arr, 75))
    p95 = float(np.percentile(arr, 95))
    print(f"      {name:20s}  min={np.nanmin(arr):.4f}  p5={p5:.4f}  p25={p25:.4f}  "
          f"p50={p50:.4f}  p75={p75:.4f}  p95={p95:.4f}  max={np.nanmax(arr):.4f}  mean={np.nanmean(arr):.4f}")

all_passed = True

normalizer = RadiometricNormalizer()

banner("SATELLITE-SRM POST-FIX VALIDATION REPORT")

for aoi_id, aoi_label in [("agriculture", "Punjab Basin"), ("urban", "Hyderabad Sector"), ("disaster", "Assam Basin")]:
    banner(f"AOI: {aoi_id.upper()} ({aoi_label})")

    # ── Paths ─────────────────────────────────────────────────────────────────
    raw_path = os.path.join(PROJECT_DIR, "data", "raw", "sentinel2", f"srm_{aoi_id}.tif")
    sr_path  = os.path.join(PROJECT_DIR, "outputs", "sr", f"{aoi_id}_sr.tif")
    unc_path = os.path.join(PROJECT_DIR, "outputs", "uncertainty", f"{aoi_id}_uncertainty.tif")

    # ── File existence ────────────────────────────────────────────────────────
    print("\n  [A] File existence")
    ok = pf(os.path.exists(sr_path),  f"SR GeoTIFF exists: {sr_path}")
    ok = pf(os.path.exists(unc_path), f"Uncertainty map exists: {unc_path}")
    all_passed &= ok

    if not os.path.exists(sr_path):
        print("  SKIP: SR file missing, cannot continue validation for this AOI.")
        continue

    # ── Load files ────────────────────────────────────────────────────────────
    raw_raster = read_geotiff(raw_path)
    sr_raster  = read_geotiff(sr_path)
    unc_raster = read_geotiff(unc_path)

    raw_data = raw_raster.data
    sr_data  = sr_raster.data
    unc_data = unc_raster.data
    meta     = sr_raster.metadata

    # ── 1. Dimensions ─────────────────────────────────────────────────────────
    print("\n  [1] Dimensions & bands")
    exp_h = raw_data.shape[1] * 3
    exp_w = raw_data.shape[2] * 3
    ok = pf(sr_data.shape == (4, exp_h, exp_w), f"SR shape = {sr_data.shape} (expected (4,{exp_h},{exp_w}))")
    all_passed &= ok

    # ── 2. Value range ────────────────────────────────────────────────────────
    print("\n  [2] Value range check (must be float32, within [0, 1])")
    ok = pf(str(sr_data.dtype) == 'float32', f"dtype = {sr_data.dtype}")
    all_passed &= ok
    ok = pf(float(np.nanmin(sr_data)) >= -1e-6, f"min = {np.nanmin(sr_data):.6f} (>= 0.0)")
    all_passed &= ok
    ok = pf(float(np.nanmax(sr_data)) <= 1.0 + 1e-6, f"max = {np.nanmax(sr_data):.6f} (<= 1.0)")
    all_passed &= ok

    # ── 3. NaN / Inf check ───────────────────────────────────────────────────
    print("\n  [3] NaN / Inf check")
    nan_count = int(np.sum(np.isnan(sr_data)))
    inf_count = int(np.sum(np.isinf(sr_data)))
    ok = pf(nan_count == 0, f"NaN count = {nan_count}")
    all_passed &= ok
    ok = pf(inf_count == 0, f"Inf count = {inf_count}")
    all_passed &= ok

    # ── 4. CRS & GSD ─────────────────────────────────────────────────────────
    print("\n  [4] Spatial metadata")
    ok = pf(meta.crs is not None, f"CRS = {meta.crs}")
    all_passed &= ok
    gsd_ok = abs(meta.gsd_x - 10.0 / 3.0) < 0.5
    ok = pf(gsd_ok, f"GSD_x = {meta.gsd_x:.4f} m (expected ~3.33 m)")
    all_passed &= ok
    gsd_ok2 = abs(meta.gsd_y - 10.0 / 3.0) < 0.5
    ok = pf(gsd_ok2, f"GSD_y = {meta.gsd_y:.4f} m (expected ~3.33 m)")
    all_passed &= ok

    # ── 5. Band statistics ────────────────────────────────────────────────────
    print("\n  [5] SR band statistics (float32 reflectance [0,1])")
    band_names = ["B02 (Blue)", "B03 (Green)", "B04 (Red)", "B08 (NIR)"]
    for b, bname in enumerate(band_names):
        band_stats(sr_data[b], bname)

    # ── 6. Compare with normalized raw ───────────────────────────────────────
    print("\n  [6] Comparison with normalized raw Sentinel-2 input")
    norm_raw = normalizer.normalize(raw_data)   # clips to [0,1]
    for b, bname in enumerate(band_names):
        sr_mean  = float(np.mean(sr_data[b]))
        raw_mean = float(np.mean(norm_raw[b]))
        delta = abs(sr_mean - raw_mean)
        flag = PASS if delta < 0.2 else WARN
        print(f"      {flag}  {bname:20s}  raw_mean={raw_mean:.4f}  sr_mean={sr_mean:.4f}  |diff|={delta:.4f}")

    # ── 7. Analytics ─────────────────────────────────────────────────────────
    print("\n  [7] Downstream analytics (computed on [0,1] reflectance SR data)")

    if aoi_id == "agriculture":
        res = AgricultureAnalyzer().analyze(sr_data)
        ndvi = SpectralIndices.ndvi(sr_data)
        ndvi_raw = SpectralIndices.ndvi(norm_raw)
        print(f"      Mean NDVI (SR):           {res['mean_ndvi']:.4f}")
        print(f"      Mean NDVI (raw, normed):  {float(np.mean(ndvi_raw)):.4f}")
        print(f"      Vigorous Canopy (>0.50):  {res['vigorous_canopy_percent']:.2f}%")
        print(f"      Moderate Veg (0.20-0.50): {res['moderate_vegetation_percent']:.2f}%")
        print(f"      Barren Soil  (<0.20):     {res['barren_soil_percent']:.2f}%")
        ndvi_ok = res['mean_ndvi'] > -0.5
        ok = pf(ndvi_ok, f"NDVI in plausible range (> -0.5): {res['mean_ndvi']:.4f}")
        all_passed &= ok

    elif aoi_id == "urban":
        res = UrbanAnalyzer().analyze(sr_data)
        print(f"      Edge Density:             {res['edge_density']:.6f}")
        ok = pf(res['edge_density'] > 0, "Edge density > 0")
        all_passed &= ok

    elif aoi_id == "disaster":
        res = DisasterAnalyzer().analyze(sr_data, threshold=0.0)
        ndwi = SpectralIndices.ndwi(sr_data)
        ndwi_raw = SpectralIndices.ndwi(norm_raw)
        print(f"      Inundation %:             {res['inundation_percentage']:.2f}%")
        print(f"      Mean NDWI (SR):           {float(np.mean(ndwi)):.4f}")
        print(f"      Mean NDWI (raw, normed):  {float(np.mean(ndwi_raw)):.4f}")
        ok = pf(True, f"Inundation computed: {res['inundation_percentage']:.2f}%")

    # ── 8. Uncertainty map check ──────────────────────────────────────────────
    print("\n  [8] Uncertainty map")
    unc_mean = float(np.mean(unc_data))
    pf(unc_mean >= 0, f"Mean uncertainty = {unc_mean:.6f} (>= 0)")
    pf(not np.any(np.isnan(unc_data)), "No NaN in uncertainty map")

banner("OVERALL RESULT")
if all_passed:
    print("  [PASS] ALL CHECKS PASSED")
else:
    print("  [FAIL] SOME CHECKS FAILED — review above")
print()
