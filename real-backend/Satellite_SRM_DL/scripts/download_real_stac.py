"""
Download Sentinel-2 L2A data from the Copernicus Data Space Ecosystem (CDSE).

Uses the CDSE STAC API for discovery and the OData Nodes endpoint
(alternate HTTPS links) with OIDC Bearer token for downloading
individual band files.

Strategy: Download each band JP2 via requests.get() with Bearer token
to a temporary local file, then open with rioxarray for clipping.
(GDAL's /vsicurl/ HEAD-first behaviour is rejected by the OData endpoint.)
"""
import os
import tempfile
import yaml
import rioxarray
import xarray as xr
import requests
import pystac_client
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env")))


def get_cdse_token():
    """Generates an OAuth token for Copernicus Data Space Ecosystem."""
    username = os.environ.get("CDSE_USERNAME")
    password = os.environ.get("CDSE_PASSWORD")

    if not username or not password:
        raise ValueError("CDSE_USERNAME and CDSE_PASSWORD must be set in .env")

    url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }

    response = requests.post(url, data=data)
    response.raise_for_status()
    token_info = response.json()

    if "access_token" in token_info:
        return token_info["access_token"]
    else:
        raise RuntimeError(f"Failed to get access token: {token_info}")


def get_alternate_https_href(asset):
    """Extract the OData HTTPS download URL from a STAC asset's alternate links."""
    alt = asset.extra_fields.get("alternate", {})
    https_info = alt.get("https", {})
    return https_info.get("href")


def download_band_to_tempfile(href, token, band_name):
    """Download a single band JP2 file via HTTPS with Bearer token.

    Returns the path to the downloaded temporary file.
    """
    headers = {"Authorization": f"Bearer {token}"}
    print(f"    GET {href[:120]}...")
    resp = requests.get(href, headers=headers, stream=True, timeout=300)
    resp.raise_for_status()

    # Write to a temp file preserving the .jp2 extension so GDAL picks the right driver
    tmp = tempfile.NamedTemporaryFile(
        suffix=f"_{band_name}.jp2", delete=False, dir=tempfile.gettempdir()
    )
    total = 0
    for chunk in resp.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
        tmp.write(chunk)
        total += len(chunk)
    tmp.close()
    print(f"    Downloaded {total / (1024*1024):.1f} MB -> {tmp.name}")
    return tmp.name


def download_stac_data():
    CONFIG_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "configs", "data.yaml")
    )
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    aois = config.get("aois", {})
    OUTPUT_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "raw", "sentinel2")
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Generating CDSE OAuth token...")
    token = get_cdse_token()
    print("Token obtained successfully.")

    print("Connecting to CDSE STAC API...")
    catalog = pystac_client.Client.open("https://stac.dataspace.copernicus.eu/v1/")

    band_mapping = {
        "B02": "B02_10m",
        "B03": "B03_10m",
        "B04": "B04_10m",
        "B08": "B08_10m",
    }

    for aoi_id, aoi_info in aois.items():
        bbox = aoi_info["bbox"]
        print(f"\n{'='*60}")
        print(f"Processing {aoi_id} ({aoi_info['name']}) with bbox {bbox}")
        print(f"{'='*60}")

        out_path = os.path.join(OUTPUT_DIR, f"srm_{aoi_id}.tif")

        # Search CDSE catalog for Sentinel-2 Level-2A
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime="2024-01-01/2026-09-01",
            query={"eo:cloud_cover": {"lt": 10}},
            max_items=10,
        )

        items = list(search.items())

        if not items:
            print(f"  No Level-2A items found for {aoi_id}!")
            continue

        # Pick the item whose tile has the best spatial overlap with our AOI
        def overlap_area(item_bbox, aoi_bbox):
            """Compute intersection area between two [minx,miny,maxx,maxy] bboxes."""
            ix0 = max(item_bbox[0], aoi_bbox[0])
            iy0 = max(item_bbox[1], aoi_bbox[1])
            ix1 = min(item_bbox[2], aoi_bbox[2])
            iy1 = min(item_bbox[3], aoi_bbox[3])
            if ix1 > ix0 and iy1 > iy0:
                return (ix1 - ix0) * (iy1 - iy0)
            return 0.0

        items.sort(key=lambda it: overlap_area(it.bbox, bbox), reverse=True)
        item = items[0]
        best_overlap = overlap_area(item.bbox, bbox)
        aoi_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        print(f"  Selected item: {item.id}")
        print(f"  AOI coverage:     {best_overlap / aoi_area * 100:.1f}%")

        # Print source verification
        props = item.properties
        print(f"  Platform:         {props.get('platform', 'N/A')}")
        print(f"  Processing level: {props.get('processing:level', 'N/A')}")
        print(f"  Cloud cover:      {props.get('eo:cloud_cover', 'N/A')}%")
        print(f"  Datetime:         {props.get('datetime', 'N/A')}")

        bands_data = []
        temp_files = []

        for band_key, band_asset_name in band_mapping.items():
            print(f"\n  Band {band_key} ({band_asset_name}):")

            if band_asset_name not in item.assets:
                print(f"    WARNING: {band_asset_name} not found in assets.")
                print(f"    Available assets: {list(item.assets.keys())[:10]}")
                continue

            asset = item.assets[band_asset_name]
            href = get_alternate_https_href(asset)

            if not href:
                # Fall back to the s3:// href and convert to HTTPS
                s3_href = asset.href
                if s3_href and s3_href.startswith("s3://eodata/"):
                    href = s3_href.replace(
                        "s3://eodata/",
                        "https://eodata.dataspace.copernicus.eu/"
                    )
                    print(f"    Using converted S3→HTTPS URL")
                else:
                    print(f"    WARNING: No download URL for {band_asset_name}.")
                    continue

            try:
                # Step 1: Download the JP2 file locally
                tmp_path = download_band_to_tempfile(href, token, band_key)
                temp_files.append(tmp_path)

                # Step 2: Open with rioxarray from local file
                da = rioxarray.open_rasterio(tmp_path)
                print(f"    Full tile shape: {da.shape}, CRS: {da.rio.crs}")

                # Step 3: Clip to AOI bounding box
                da_clipped = da.rio.clip_box(
                    minx=bbox[0],
                    miny=bbox[1],
                    maxx=bbox[2],
                    maxy=bbox[3],
                    crs="EPSG:4326",
                )
                # Load into memory and close the file handle
                da_clipped = da_clipped.load()
                bands_data.append(da_clipped)
                print(f"    Clipped shape:   {da_clipped.shape}")
                print(f"    Value range:     {float(da_clipped.min()):.0f} – {float(da_clipped.max()):.0f}")

            except Exception as e:
                print(f"    ERROR processing band {band_asset_name}: {e}")

        # Clean up temp files
        for tf in temp_files:
            try:
                os.remove(tf)
            except OSError:
                pass

        if len(bands_data) == 4:
            combined = xr.concat(bands_data, dim="band")
            combined.rio.to_raster(out_path)
            size_mb = os.path.getsize(out_path) / (1024 * 1024)
            print(f"\n  [OK] Saved {out_path} ({size_mb:.1f} MB)")
            print(f"     Final shape: {combined.shape}")
        else:
            print(
                f"\n  [FAIL] Only got {len(bands_data)}/4 bands for {aoi_id}."
            )

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)


if __name__ == "__main__":
    download_stac_data()
