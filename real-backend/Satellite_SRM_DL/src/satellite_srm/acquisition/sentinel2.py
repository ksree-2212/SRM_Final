"""Sentinel-2 L2A acquisition engine – strict real-data mode (no synthetic fallback)."""
import os
import glob
import tempfile
import zipfile
import numpy as np
from typing import List
from satellite_srm.acquisition.copernicus import CopernicusClient
from satellite_srm.acquisition.manifests import AcquisitionManifest
from satellite_srm.acquisition.checksums import compute_sha256
from satellite_srm.geospatial.metadata import GeoMetadata
from satellite_srm.geospatial.geotiff import write_geotiff
from satellite_srm.compat import Affine, CRS
from satellite_srm.logging_config import get_logger

logger = get_logger("sentinel2")


class Sentinel2Downloader:
    """Manages Sentinel-2 L2A asset discovery, download, and verification.

    Strict mode: synthetic data generation is entirely removed.
    Every scene must come from the Copernicus Data Space Ecosystem.
    """

    def __init__(self, output_dir: str = "data/raw/sentinel2"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.client = CopernicusClient()

    def acquire_scene(
        self,
        scene_id: str,
        bbox: List[float],
        bands: List[str] = ["B02", "B03", "B04", "B08"],
    ) -> str:
        """
        Downloads and returns the path to a Sentinel-2 L2A GeoTIFF for the given bbox.
        Raises RuntimeError immediately if authentication or download fails.
        No synthetic data is ever generated.

        Returns
        -------
        str : absolute path to the written .tif file.
        """
        # ── Check for cached scene (keyed by scene_id stub) ──────────────────
        stub_path = os.path.join(self.output_dir, f"{scene_id}.tif")
        if os.path.exists(stub_path):
            logger.info(f"Cache hit – scene already exists at {stub_path}")
            return stub_path

        # ── Authenticate ──────────────────────────────────────────────────────
        if not self.client.authenticate():
            raise RuntimeError(
                "Authentication with Copernicus Data Space Ecosystem failed. "
                "Check CDSE_USERNAME / CDSE_PASSWORD in your .env file."
            )

        # ── Search catalogue ──────────────────────────────────────────────────
        logger.info(f"Searching Copernicus catalogue for bbox {bbox} ...")
        scenes = self.client.search_scenes(bbox, "2020-01-01", "2027-12-31")
        if not scenes:
            raise RuntimeError(
                f"No Sentinel-2 L2A scenes found for bbox {bbox}. "
                "Try widening the bounding box or adjusting the date range."
            )

        # ── Pick the most-recent scene ────────────────────────────────────────
        latest_scene = scenes[0]
        product_id = latest_scene.get("Id")
        real_scene_id = latest_scene.get("Name", scene_id).replace(".SAFE", "")
        scene_path = os.path.join(self.output_dir, f"{real_scene_id}.tif")

        if os.path.exists(scene_path):
            logger.info(f"Cache hit – scene already exists at {scene_path}")
            return scene_path

        zip_path = os.path.join(self.output_dir, f"{real_scene_id}.zip")
        logger.info(f"Downloading real scene {real_scene_id} (may take several minutes)...")

        # ── Download ──────────────────────────────────────────────────────────
        success = self.client.download_product(product_id, zip_path)
        if not success:
            raise RuntimeError(
                f"download_product() failed for product {product_id}. "
                "Check network connectivity, disk space, and CDSE credentials."
            )

        # ── Extract bands from .SAFE archive ─────────────────────────────────
        logger.info("Extracting and processing bands from SAFE archive...")
        try:
            import rasterio
        except ImportError:
            raise RuntimeError("rasterio is required to extract Sentinel-2 JP2 bands. pip install rasterio")

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(tmpdir)

                band_data = []
                band_meta = None

                for b in bands:
                    pattern = os.path.join(tmpdir, "**", "R10m", f"*_{b}_10m.jp2")
                    matches = glob.glob(pattern, recursive=True)
                    if not matches:
                        raise FileNotFoundError(
                            f"Band {b} JP2 file not found in SAFE archive. "
                            f"Pattern tried: {pattern}"
                        )
                    with rasterio.open(matches[0]) as src:
                        band_data.append(src.read(1))
                        if band_meta is None:
                            band_meta = src.profile

                stacked = np.stack(band_data, axis=0).astype(np.float32)

                meta = GeoMetadata(
                    width=band_meta["width"],
                    height=band_meta["height"],
                    count=len(bands),
                    crs=CRS.from_epsg(band_meta["crs"].to_epsg()),
                    transform=Affine(*band_meta["transform"][:6]),
                    dtype="float32",
                    band_names=bands,
                    gsd_x=10.0,
                    gsd_y=10.0,
                )
                write_geotiff(scene_path, stacked, meta)

                sha = compute_sha256(scene_path)
                manifest = AcquisitionManifest(
                    scene_id=real_scene_id,
                    sensor="Sentinel-2",
                    product="L2A",
                    acquisition_date=(
                        latest_scene.get("ContentDate", {}).get("Start", "2026-01-01")[:10]
                    ),
                    cloud_cover=float(
                        next(
                            (
                                a["Value"]
                                for a in latest_scene.get("Attributes", [])
                                if a.get("Name") == "cloudCover"
                            ),
                            0.0,
                        )
                    ),
                    bands=bands,
                    crs=f"EPSG:{band_meta['crs'].to_epsg()}",
                    source="Copernicus Data Space Ecosystem",
                    checksum=sha,
                    file_path=scene_path,
                )
                manifest.save()
                logger.info(f"Scene saved: {scene_path}  (SHA-256: {sha[:12]}...)")

        finally:
            # Always clean up the large zip even on failure
            if os.path.exists(zip_path):
                os.remove(zip_path)
                logger.info(f"Cleaned up archive: {zip_path}")

        return scene_path
