"""Combined multi-objective loss: L_total = λ1*L_L1 + λ2*L_perceptual + λ3*L_spectral + λ4*L_ndvi.

Weight rationale
----------------
L1 (1.0)        — primary pixel-level fidelity anchor
perceptual (0.05) — lightweight texture regulariser (no VGG download)
spectral (0.10)  — log-ratio inter-band consistency (stable, bounded)
ndvi (0.10)      — vegetation index alignment

Spectral and NDVI are intentionally small secondaries. The previous weight
of 0.25 on an unstable raw-ratio spectral loss caused it to contribute ~96%
of total loss, collapsing training. These values keep spectral physics
meaningful without letting it overwhelm the spatial reconstruction objective.
"""
from typing import Dict, Any
from satellite_srm.compat import nn
from satellite_srm.losses.l1 import MaskedL1Loss
from satellite_srm.losses.perceptual import PerceptualVGGLoss
from satellite_srm.losses.ndvi import NDVIConsistencyLoss
from satellite_srm.losses.spectral import SpectralConsistencyLoss


class CombinedSRMLoss(nn.Module):
    """Orchestrates multi-objective training balancing spatial fidelity with spectral integrity."""

    def __init__(self, l1_weight: float = 1.0, perceptual_weight: float = 0.05,
                 spectral_weight: float = 0.10, ndvi_weight: float = 0.10):
        """
        Args:
            l1_weight:         Weight for pixel-level L1 reconstruction loss.
            perceptual_weight: Weight for feature-level perceptual loss.
            spectral_weight:   Weight for log-ratio spectral consistency loss.
            ndvi_weight:       Weight for NDVI index consistency loss.
        """
        super().__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.spectral_weight = spectral_weight
        self.ndvi_weight = ndvi_weight

        self.l1_criterion = MaskedL1Loss()
        self.perceptual_criterion = PerceptualVGGLoss()
        self.ndvi_criterion = NDVIConsistencyLoss()
        self.spectral_criterion = SpectralConsistencyLoss()

    def forward(self, pred, target, mask=None) -> Dict[str, Any]:
        """
        Args:
            pred:   (B, 4, H, W) SR output
            target: (B, 4, H, W) HR target
            mask:   Optional spatial mask for L1 computation
        Returns:
            Dict with 'total_loss' and per-component losses for logging.
        """
        l1_val   = self.l1_criterion(pred, target, mask=mask)
        perc_val = self.perceptual_criterion(pred, target)
        ndvi_val = self.ndvi_criterion(pred, target)
        spec_val = self.spectral_criterion(pred, target)

        total_loss = (
            self.l1_weight      * l1_val   +
            self.perceptual_weight * perc_val +
            self.ndvi_weight    * ndvi_val +
            self.spectral_weight * spec_val
        )

        return {
            "total_loss":      total_loss,
            "l1_loss":         l1_val,
            "perceptual_loss": perc_val,
            "ndvi_loss":       ndvi_val,
            "spectral_loss":   spec_val,
        }
