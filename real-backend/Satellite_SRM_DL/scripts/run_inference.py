"""
Standalone full-scene SRM inference runner.

Loads existing trained model weights from outputs/checkpoints/latest.pt (or best.pt),
runs tiled Hannibal-blended super-resolution inference on raw Sentinel-2 AOI scenes,
writes georeferenced sub-4m GeoTIFF products, and outputs domain analytics.
"""
import sys
import os
import yaml
import logging
import numpy as np

# ── path setup ──────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))
os.chdir(PROJECT_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_DIR, ".env"))

# ── geospatial & model components ───────────────────────────────────────────
from satellite_srm.geospatial.geotiff import read_geotiff
from satellite_srm.models.model_factory import create_model
from satellite_srm.models.checkpoint import load_checkpoint_weights
from satellite_srm.inference.pipeline import FullSceneSRMPipeline
from satellite_srm.applications.agriculture import AgricultureAnalyzer
from satellite_srm.applications.urban import UrbanAnalyzer
from satellite_srm.applications.disaster import DisasterAnalyzer
from satellite_srm.compat import torch
from satellite_srm.logging_config import configure_logging

configure_logging("INFO")
logger = logging.getLogger("run_inference")


def main():
    # ── Checkpoint path resolution ──────────────────────────────────────────
    checkpoints_dir = os.path.join(PROJECT_DIR, "outputs", "checkpoints")
    latest_ckpt = os.path.join(checkpoints_dir, "latest.pt")
    best_ckpt = os.path.join(checkpoints_dir, "best.pt")

    if os.path.exists(latest_ckpt):
        ckpt_path = latest_ckpt
    elif os.path.exists(best_ckpt):
        ckpt_path = best_ckpt
    else:
        logger.error(f"No checkpoint found in {checkpoints_dir}. Aborting.")
        sys.exit(1)

    logger.info(f"Verified checkpoint file: {ckpt_path} ({os.path.getsize(ckpt_path)} bytes)")

    # ── Device selection ─────────────────────────────────────────────────────
    use_cuda = torch.cuda.is_available() if hasattr(torch, "cuda") else False
    device_str = "cuda" if use_cuda else "cpu"
    logger.info(f"Using execution device: {device_str}")
    if not use_cuda and hasattr(torch, "set_num_threads"):
        torch.set_num_threads(4)
        logger.info(f"Set PyTorch CPU threads to {torch.get_num_threads()}")

    # ── Load model architecture & weights ────────────────────────────────────
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
        "device": device_str,
    }

    logger.info("Instantiating Multispectral SwinIR architecture...")
    model = create_model(model_config)

    from satellite_srm.models.pytorch_loader import load_pytorch_checkpoint
    logger.info(f"Loading trained model weights from {ckpt_path}...")
    checkpoint_dict = load_pytorch_checkpoint(ckpt_path)
    state_dict = checkpoint_dict.get("model_state_dict", checkpoint_dict)
    model.load_state_dict(state_dict, strict=False)
    epoch = checkpoint_dict.get("epoch", 0)
    metrics = {"loaded": True}
    logger.info(f"Loaded checkpoint trained up to Epoch {epoch} with metrics: {metrics}")

    # ── Pipeline setup ───────────────────────────────────────────────────────
    infer_cfg = {
        "device": device_str,
        "inference": {
            "tile_size": 128,
            "overlap": 32,
            "scale_factor": 3.0,
            "blending_method": "weighted_hann",
            "uncertainty": {"mc_passes": 4},
        },
    }
    inference_pipeline = FullSceneSRMPipeline(model=model, config=infer_cfg)

    # ── Load AOI metadata ────────────────────────────────────────────────────
    data_cfg_path = os.path.join(PROJECT_DIR, "configs", "data.yaml")
    with open(data_cfg_path) as f:
        data_cfg = yaml.safe_load(f)
    aois = data_cfg.get("aois", {})

    raw_dir = os.path.join(PROJECT_DIR, "data", "raw", "sentinel2")
    outputs_dir = os.path.join(PROJECT_DIR, "outputs")
    sr_dir = os.path.join(outputs_dir, "sr")
    unc_dir = os.path.join(outputs_dir, "uncertainty")
    os.makedirs(sr_dir, exist_ok=True)
    os.makedirs(unc_dir, exist_ok=True)

    analyzers = {
        "agriculture": AgricultureAnalyzer(),
        "urban": UrbanAnalyzer(),
        "disaster": DisasterAnalyzer(),
    }

    results = {}

    # ── Process each AOI scene ──────────────────────────────────────────────
    for aoi_id, aoi_info in aois.items():
        name = aoi_info.get("name", aoi_id)
        raw_scene_path = os.path.join(raw_dir, f"srm_{aoi_id}.tif")

        if not os.path.exists(raw_scene_path):
            logger.warning(f"Raw scene missing at {raw_scene_path}, skipping.")
            continue

        sr_out = os.path.join(sr_dir, f"{aoi_id}_sr.tif")
        unc_out = os.path.join(unc_dir, f"{aoi_id}_uncertainty.tif")

        logger.info(f"\n{'='*60}")
        logger.info(f"Running Full-Scene Inference: {name} ({aoi_id})")
        logger.info(f"{'='*60}")

        if os.path.exists(sr_out) and os.path.getsize(sr_out) > 1000000 and os.path.exists(unc_out) and os.path.getsize(unc_out) > 1000000:
            logger.info(f"Existing completed SR GeoTIFF found for {name} ({sr_out}). Reusing cached outputs.")
            unc_raster = read_geotiff(unc_out)
            result = {
                "target_gsd": read_geotiff(sr_out).metadata.gsd_x,
                "mean_uncertainty": float(np.mean(unc_raster.data)),
                "runtime_seconds": 0.0
            }
        else:
            result = inference_pipeline.run(raw_scene_path, sr_out, unc_out)

        # ── Domain analytics on super-resolved output ────────────────────────
        sr_raster = read_geotiff(sr_out)
        analytics_summary = {}

        if aoi_id == "agriculture":
            analytics = analyzers["agriculture"].analyze(sr_raster.data)
            analytics_summary = {
                "mean_ndvi": round(analytics["mean_ndvi"], 4),
                "vigorous_canopy_percent": round(analytics["vigorous_canopy_percent"], 2),
                "moderate_vegetation_percent": round(analytics["moderate_vegetation_percent"], 2),
                "barren_soil_percent": round(analytics["barren_soil_percent"], 2),
            }
            logger.info(
                f"  [Agriculture Analytics] Mean NDVI: {analytics_summary['mean_ndvi']} | "
                f"Vigorous Canopy: {analytics_summary['vigorous_canopy_percent']}% | "
                f"Moderate Veg: {analytics_summary['moderate_vegetation_percent']}% | "
                f"Barren Soil: {analytics_summary['barren_soil_percent']}%"
            )

        elif aoi_id == "urban":
            analytics = analyzers["urban"].analyze(sr_raster.data)
            analytics_summary = {
                "edge_density": round(analytics["edge_density"], 6),
            }
            logger.info(
                f"  [Urban Analytics] Edge Density: {analytics_summary['edge_density']}"
            )

        elif aoi_id == "disaster":
            analytics = analyzers["disaster"].analyze(sr_raster.data, threshold=0.0)
            analytics_summary = {
                "inundation_percentage": round(analytics["inundation_percentage"], 2),
                "mean_ndwi": round(float(np.mean(analytics["ndwi_map"])), 4),
            }
            logger.info(
                f"  [Disaster Analytics] Flood Inundation: {analytics_summary['inundation_percentage']}% | "
                f"Mean NDWI: {analytics_summary['mean_ndwi']}"
            )

        results[aoi_id] = {
            "sr_output": sr_out,
            "uncertainty_output": unc_out,
            "target_gsd": result["target_gsd"],
            "mean_uncertainty": result["mean_uncertainty"],
            "runtime_seconds": result["runtime_seconds"],
            "analytics": analytics_summary,
        }

    # ── Final Summary ────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("🎉 FULL-SCENE INFERENCE & ANALYTICS COMPLETE")
    logger.info(f"{'='*60}")
    for aoi_id, res in results.items():
        logger.info(
            f"🔹 [{aoi_id.upper()}] SR GeoTIFF: {res['sr_output']} | "
            f"Target GSD: {res['target_gsd']:.2f}m | "
            f"Mean Uncertainty: {res['mean_uncertainty']:.4f} | "
            f"Time: {res['runtime_seconds']:.1f}s"
        )

    return results


if __name__ == "__main__":
    main()
