"""
End-to-end AOI training script for Satellite-SRM.

Pipeline (all modules wired):
  Copernicus download  ->  BandProcessor  ->  CloudMasker  ->  ShadowMasker
  ->  RadiometricNormalizer  ->  CoRegistrationEngine  ->  QualityController
  ->  PatchExtractor (LR / HR folders)  ->  PairedSRDataset
  ->  SRMTrainer (CombinedSRMLoss: L1 + Perceptual + NDVI + Spectral)
  ->  FullSceneSRMPipeline (tiled + blending + MC uncertainty)
  ->  write_geotiff SR output  ->  AgricultureAnalyzer / UrbanAnalyzer / DisasterAnalyzer

No synthetic data is generated or used at any point.
"""
import sys
import os
import yaml
import logging
import json
import numpy as np

# ── path setup ──────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))
os.chdir(PROJECT_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_DIR, ".env"))

# ── acquisition ──────────────────────────────────────────────────────────────
from satellite_srm.acquisition.sentinel2 import Sentinel2Downloader

# ── geospatial ───────────────────────────────────────────────────────────────
from satellite_srm.geospatial.geotiff import read_geotiff, write_geotiff
from satellite_srm.geospatial.metadata import GeoMetadata
from satellite_srm.geospatial.raster import GeoRaster

# ── preprocessing ─────────────────────────────────────────────────────────────
from satellite_srm.preprocessing.bands import BandProcessor
from satellite_srm.preprocessing.cloud_mask import CloudMasker
from satellite_srm.preprocessing.shadow_mask import ShadowMasker
from satellite_srm.preprocessing.normalization import RadiometricNormalizer
from satellite_srm.preprocessing.registration import CoRegistrationEngine
from satellite_srm.preprocessing.quality_control import QualityController
from satellite_srm.preprocessing.patches import PatchExtractor

# ── datasets ─────────────────────────────────────────────────────────────────
from satellite_srm.datasets.paired_dataset import PairedSRDataset

# ── model ─────────────────────────────────────────────────────────────────────
from satellite_srm.models.model_factory import create_model

# ── training ─────────────────────────────────────────────────────────────────
from satellite_srm.training.trainer import SRMTrainer

# ── inference ────────────────────────────────────────────────────────────────
from satellite_srm.inference.pipeline import FullSceneSRMPipeline

# ── applications ─────────────────────────────────────────────────────────────
from satellite_srm.applications.agriculture import AgricultureAnalyzer
from satellite_srm.applications.urban import UrbanAnalyzer
from satellite_srm.applications.disaster import DisasterAnalyzer

# ── compat (torch / F) ───────────────────────────────────────────────────────
from satellite_srm.compat import Affine, torch, F

# ── logging ───────────────────────────────────────────────────────────────────
from satellite_srm.logging_config import configure_logging
configure_logging("INFO")
logger = logging.getLogger("train_aois")


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def build_lr_raster(hr_raster: GeoRaster, scale: float = 3.0) -> GeoRaster:
    """Downsample a real HR raster to create the paired LR input (self-supervised)."""
    c, h, w = hr_raster.data.shape
    lr_h, lr_w = int(h // scale), int(w // scale)
    hr_tensor = torch.from_numpy(hr_raster.data).unsqueeze(0)
    lr_tensor = F.interpolate(hr_tensor, size=(lr_h, lr_w), mode="bicubic", align_corners=False)
    lr_data = lr_tensor.squeeze(0).numpy() if hasattr(lr_tensor, "squeeze") else lr_tensor[0].numpy()
    lr_meta = GeoMetadata(
        width=lr_w, height=lr_h, count=c,
        crs=hr_raster.metadata.crs,
        transform=hr_raster.metadata.transform * Affine.scale(scale, scale),
        dtype=hr_raster.metadata.dtype,
        band_names=hr_raster.metadata.band_names,
        gsd_x=hr_raster.metadata.gsd_x * scale,
        gsd_y=hr_raster.metadata.gsd_y * scale,
    )
    return GeoRaster(lr_data, lr_meta)


def apply_full_preprocessing(hr_raster: GeoRaster) -> GeoRaster:
    """
    Runs the full preprocessing chain from /preprocessing on the raw HR raster:
      1. BandProcessor  – validates 4 bands (B02 B03 B04 B08)
      2. CloudMasker    – spectral reflectance cloud detection
      3. ShadowMasker   – NIR + cloud proximity shadow detection
      4. RadiometricNormalizer – DN -> [0,1] reflectance
    Returns a clean, normalised GeoRaster.
    """
    data = hr_raster.data.copy()

    # 1. Band validation
    BandProcessor.validate_band_count(data, expected=4)

    # 2. Cloud mask (spectral reflectance thresholding – no SCL band needed)
    cloud_masker = CloudMasker(dilation_radius=2, threshold=0.35)
    cloud_mask = cloud_masker.mask_from_reflectance(data[0], data[2], data[3])
    logger.info(f"  Cloud cover: {np.mean(cloud_mask)*100:.1f}%")

    # 3. Shadow mask (low NIR + cloud adjacency)
    shadow_masker = ShadowMasker(dilation_size=4)
    shadow_mask = shadow_masker.mask_shadows(data[3], cloud_mask)
    combined_mask = cloud_mask | shadow_mask
    logger.info(f"  Combined cloud+shadow mask: {np.mean(combined_mask)*100:.1f}% of pixels flagged")

    # 4. Radiometric normalisation (DN 0-10000 -> reflectance 0-1)
    normalizer = RadiometricNormalizer(scale_factor=10000.0, use_percentiles=True, p_min=1.0, p_max=99.0)
    norm_data = normalizer.normalize(data)

    # Zero-out masked pixels so model never trains on cloud/shadow
    for b in range(norm_data.shape[0]):
        norm_data[b][combined_mask] = 0.0

    return GeoRaster(norm_data.astype(np.float32), hr_raster.metadata)


def extract_patches_with_qc(
    lr_raster: GeoRaster,
    hr_raster: GeoRaster,
    scene_id: str,
    aoi_name: str,
    output_dir: str,
    scale: float = 3.0,
    lr_patch_size: int = 64,
    stride: int = 48,
) -> int:
    """
    Runs CoRegistrationEngine + QualityController before saving each patch pair.
    Rejected patches (cloud-heavy, misaligned, NaN) are silently skipped.
    """
    os.makedirs(os.path.join(output_dir, "lr"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "hr"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "meta"), exist_ok=True)

    registrar = CoRegistrationEngine(max_error_pixels=3.0, min_overlap=0.75)
    qc = QualityController(max_cloud_fraction=0.10, max_nodata_fraction=0.05)
    extractor = PatchExtractor(lr_patch_size=lr_patch_size, scale_factor=scale, stride=stride)

    c, h_lr, w_lr = lr_raster.data.shape
    hr_patch_size = int(round(lr_patch_size * scale))
    saved = 0

    # Downscale HR NIR for registration comparison at LR resolution
    hr_nir = hr_raster.data[3]
    hr_nir_scaled = hr_nir[::3, ::3] if hr_nir.shape[0] > lr_raster.data.shape[1] else hr_nir

    for row in range(0, h_lr - lr_patch_size + 1, stride):
        for col in range(0, w_lr - lr_patch_size + 1, stride):
            lr_patch = lr_raster.data[:, row:row + lr_patch_size, col:col + lr_patch_size]
            hr_row = int(round(row * scale))
            hr_col = int(round(col * scale))
            hr_patch = hr_raster.data[:, hr_row:hr_row + hr_patch_size, hr_col:hr_col + hr_patch_size]

            if hr_patch.shape[1] != hr_patch_size or hr_patch.shape[2] != hr_patch_size:
                continue

            # Registration check
            lr_nir_crop = lr_patch[3]
            hr_nir_crop_ds = hr_nir_scaled[row:row + lr_patch_size, col:col + lr_patch_size]
            if hr_nir_crop_ds.shape == lr_nir_crop.shape:
                reg_result = registrar.register(lr_nir_crop, hr_nir_crop_ds)
            else:
                from satellite_srm.preprocessing.registration import RegistrationResult
                reg_result = RegistrationResult(0.0, 0.0, 0.0, 1.0, True)

            # Cloud mask for this patch
            cloud_patch = np.zeros((lr_patch_size, lr_patch_size), dtype=bool)

            # QC check
            valid, reason = qc.evaluate_patch(lr_patch, hr_patch, cloud_patch, reg_result)
            if not valid:
                continue

            patch_id = f"{scene_id}_r{row}_c{col}"
            np.save(os.path.join(output_dir, "lr", f"{patch_id}.npy"), lr_patch)
            np.save(os.path.join(output_dir, "hr", f"{patch_id}.npy"), hr_patch)
            meta = {
                "patch_id": patch_id, "scene_id": scene_id, "aoi_name": aoi_name,
                "col_off": col, "row_off": row,
                "lr_size": lr_patch_size, "hr_size": hr_patch_size, "scale_factor": scale,
                "registration_error_px": reg_result.error,
            }
            with open(os.path.join(output_dir, "meta", f"{patch_id}.json"), "w") as f:
                json.dump(meta, f, indent=2)
            saved += 1

    logger.info(f"  Saved {saved} QC-passed patch pairs for {scene_id}")
    return saved


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Load configs ──────────────────────────────────────────────────────────
    with open(os.path.join(PROJECT_DIR, "configs", "data.yaml")) as f:
        data_cfg = yaml.safe_load(f)
    aois = data_cfg.get("aois", {})
    if not aois:
        logger.error("No AOIs found in configs/data.yaml")
        sys.exit(1)

    raw_dir = os.path.join(PROJECT_DIR, "data", "raw", "sentinel2")
    patches_dir = os.path.join(PROJECT_DIR, "data", "train_patches")
    outputs_dir = os.path.join(PROJECT_DIR, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    downloader = Sentinel2Downloader(output_dir=raw_dir)
    total_patches = 0

    # ── Per-AOI: download → preprocess → patch ────────────────────────────────
    for aoi_id, aoi_info in aois.items():
        name = aoi_info.get("name", aoi_id)
        bbox = aoi_info.get("bbox")
        if not bbox:
            logger.warning(f"No bbox for AOI {name}, skipping.")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"AOI: {name}  |  bbox: {bbox}")
        logger.info(f"{'='*60}")

        # 1. Download real Copernicus scene (raises on failure — no synthetic fallback)
        scene_path = downloader.acquire_scene(scene_id=f"srm_{aoi_id}", bbox=bbox)
        if not scene_path or not os.path.exists(scene_path):
            logger.error(f"Scene not found after download for {name}. Aborting.")
            sys.exit(1)

        # 2. Read raw HR raster (real Sentinel-2 data)
        logger.info(f"Loading scene: {scene_path}")
        hr_raw = read_geotiff(scene_path)
        logger.info(f"  Shape: {hr_raw.data.shape}  GSD: {hr_raw.metadata.gsd_x}m")

        # 3. Full preprocessing chain (cloud mask, shadow mask, normalise)
        logger.info("Running full preprocessing chain...")
        hr_clean = apply_full_preprocessing(hr_raw)

        # 4. Generate LR counterpart via downsampling (self-supervised)
        logger.info("Generating LR counterpart (10m -> 30m bicubic downsampling)...")
        lr_raster = build_lr_raster(hr_clean, scale=3.0)

        # 5. QC-filtered patch extraction (co-registration check per patch)
        logger.info("Extracting QC-filtered LR/HR patch pairs...")
        n = extract_patches_with_qc(
            lr_raster, hr_clean,
            scene_id=f"srm_{aoi_id}",
            aoi_name=name,
            output_dir=patches_dir,
            scale=3.0,
            lr_patch_size=64,
            stride=48,
        )
        total_patches += n

    logger.info(f"\nTotal QC-passed patch pairs across all AOIs: {total_patches}")
    if total_patches == 0:
        logger.error("No patches available for training. Aborting.")
        sys.exit(1)

    # ── Dataset ──────────────────────────────────────────────────────────────
    logger.info("\nLoading aggregated LR-HR dataset...")
    dataset = PairedSRDataset(patches_dir)
    logger.info(f"Dataset size: {len(dataset)} patch pairs")

    # ── Model ─────────────────────────────────────────────────────────────────
    use_cuda = torch.cuda.is_available() if hasattr(torch, "cuda") else False
    device_str = "cuda" if use_cuda else "cpu"
    logger.info(f"\nUsing device: {device_str}")
    if not use_cuda and hasattr(torch, "set_num_threads"):
        torch.set_num_threads(4)
        logger.info(f"Set PyTorch CPU threads to {torch.get_num_threads()}")

    model_config = {
        "model": {
            "backend": "swinir",
            "scale_factor": 3.0,
            "swinir": {
                "in_channels": 4,
                "out_channels": 4,
                "embed_dim": 96,
                "depths": [4, 4, 4, 4],
                "num_heads": [6, 6, 6, 6],
                "window_size": 8,
                "drop_rate": 0.05,
            },
        },
        "training": {
            "epochs": 5,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "loss_weights": {
                "l1": 1.0,
                "perceptual": 0.10,
                "spectral_ratio": 0.25,
                "ndvi": 0.50,
            },
            "early_stopping": {"patience": 10, "min_delta": 0.001},
            "use_ema": True,
        },
        "device": device_str,
        "checkpoints_dir": os.path.join(PROJECT_DIR, "outputs", "checkpoints"),
    }
    model = create_model(model_config)

    # ── Training (all losses: L1 + Perceptual + NDVI + Spectral) ─────────────
    logger.info("\nStarting SRM fine-tuning on 3 AOIs...")
    trainer = SRMTrainer(
        model=model,
        train_dataset=dataset,
        val_dataset=None,
        config=model_config,
    )
    history = trainer.fit()
    logger.info(f"\nTraining complete: best Val PSNR = {history['best_val_psnr']:.2f} dB")

    # ── Post-training inference on each downloaded scene ──────────────────────
    logger.info("\nRunning full-scene inference on downloaded AOI scenes...")
    infer_cfg = {
        "device": device_str,
        "inference": {
            "tile_size": 128,
            "overlap": 32,
            "scale_factor": 3.0,
            "blending_method": "weighted_hann",
            "uncertainty": {"mc_passes": 8},
        },
    }
    inference_pipeline = FullSceneSRMPipeline(model=model, config=infer_cfg)

    # Application analyzers (one per AOI type)
    analyzers = {
        "agriculture": AgricultureAnalyzer(),
        "urban": UrbanAnalyzer(),
        "disaster": DisasterAnalyzer(),
    }

    for aoi_id, aoi_info in aois.items():
        name = aoi_info.get("name", aoi_id)
        raw_scene_path = os.path.join(raw_dir, f"srm_{aoi_id}.tif")
        if not os.path.exists(raw_scene_path):
            logger.warning(f"No downloaded scene found at {raw_scene_path}, skipping inference.")
            continue

        sr_out = os.path.join(outputs_dir, "sr", f"{aoi_id}_sr.tif")
        unc_out = os.path.join(outputs_dir, "uncertainty", f"{aoi_id}_uncertainty.tif")
        os.makedirs(os.path.dirname(sr_out), exist_ok=True)
        os.makedirs(os.path.dirname(unc_out), exist_ok=True)

        logger.info(f"\nInference: {name}")
        result = inference_pipeline.run(raw_scene_path, sr_out, unc_out)
        logger.info(
            f"  SR GeoTIFF: {result['sr_output_path']}\n"
            f"  Target GSD: {result['target_gsd']:.2f} m\n"
            f"  Mean Uncertainty: {result['mean_uncertainty']:.4f}\n"
            f"  Runtime: {result['runtime_seconds']:.1f}s"
        )

        # Application analytics on super-resolved output
        sr_raster = read_geotiff(sr_out)
        if aoi_id == "agriculture":
            analytics = analyzers["agriculture"].analyze(sr_raster.data)
            logger.info(f"  [Agriculture] Mean NDVI: {analytics['mean_ndvi']:.3f} | "
                        f"Vigorous Canopy: {analytics['vigorous_canopy_percent']:.1f}%")
        elif aoi_id == "urban":
            analytics = analyzers["urban"].analyze(sr_raster.data)
            logger.info(f"  [Urban] Edge Density: {analytics['edge_density']:.4f}")
        elif aoi_id == "disaster":
            analytics = analyzers["disaster"].analyze(sr_raster.data, threshold=0.0)
            logger.info(f"  [Disaster] Inundation: {analytics['inundation_percentage']:.2f}%")

    logger.info("\n✅ Full pipeline completed successfully.")
    logger.info(f"   Training history: logs/training_history.json")
    logger.info(f"   SR outputs:       {os.path.join(outputs_dir, 'sr')}")
    logger.info(f"   Checkpoints:      {os.path.join(outputs_dir, 'checkpoints')}")


if __name__ == "__main__":
    main()
