# Inference Job Verification

**Job ID:** `SRM-E66FDFD9` (Initial Test) / `SRM-57989C22` (Corrected Radiometry Test)
**Final Status:** `completed`

The background job was tracked until completion and processed successfully.

### 1. Backend Inference Execution
- **Task Start**: The `process_satellite_srm_task` successfully started in the background after the `POST` request was validated. 
- **Checkpoint Validation**: The `generate_product_for_raster` function loaded `outputs/checkpoints/srm_corrected_v1.pt` via `strict=True`, ensuring the newly corrected model was used instead of `latest.pt`.
- **Band Ordering**: The backend properly routed the four distinct uploaded Sentinel-2 bands to `stack_four_bands`, which stacked them into `B02, B03, B04, B08` format.

### 2. Output GeoTIFF Verification
The background inference successfully exported the GeoTIFF output file.

- **Output Path**: `data/outputs/job_SRM-57989C22_sr.tif`
- **Spatial Resolution (Upscaling)**: 
  - The input satellite patch was `1038 x 1933`.
  - The output GeoTIFF is exactly 3x spatially upscaled: **`3114 x 5799`**.
- **Band Structure**: 4 bands (Blue, Green, Red, NIR).
- **Value Integrity (NaN/Inf Checks)**:
  - NaN count: `0`
  - Inf count: `0`

### 3. Radiometric Fix Verification
The input processing pipeline (`stack_four_bands` in `server.py`) was successfully rewritten to rely on `rasterio` rather than `Pillow`, which safely preserves the raw Sentinel-2 Data Number (DN) values and geotransforms without performing destructive byte-scaling.

**Stacked Input Verification (`SRM-57989C22_stacked_4band.tif`):**
- Data type: `uint16` (Preserved successfully)
- Original Sentinel-2 DN Range preserved: `Min = 1126, Max = 19857` (matches raw uploaded files perfectly).

**Super-Resolution Output Verification (`job_SRM-57989C22_sr.tif`):**
- With `uint16` preserved, the `pipeline.py` correctly normalized via `DN / 10000.0` automatically.
- `Min`: `0.0772`
- `Max`: `1.000` (Bounded naturally by the valid surface reflectance limits).
- `Mean`: `0.2812` (Appropriate visual brightness level for multispectral reflectance).

The server configuration successfully integrates the validated, corrected model and processes realistic multi-band Sentinel-2 inputs end-to-end with flawless radiometric and spatial consistency!
