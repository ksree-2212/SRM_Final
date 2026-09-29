"""Cross-band spectral consistency loss using numerically stable log-ratio formulation.

Replaces raw ratio NIR/Red (which explodes when Red→0) with log-difference:
  L = |log(NIR+ε) - log(Red+ε) - [log(NIR_hr+ε) - log(Red_hr+ε)]|

Log differences are inherently bounded and produce smooth, finite gradients
even when individual bands approach zero. Epsilon=1e-3 in [0,1] reflectance
domain ensures the denominator is never dangerously small.
"""
from satellite_srm.compat import torch, nn


class SpectralConsistencyLoss(nn.Module):
    """
    Numerically stable spectral band-relationship loss.

    Uses log-ratio instead of raw ratio to prevent gradient explosion:
      log_ratio(A, B) = log(A + eps) - log(B + eps)
    This is bounded even as B → 0, unlike A / (B + eps) which can be huge.

    Enforces three inter-band relationships:
      - NIR vs Red   (B08 / B04): vegetation sensitivity
      - Green vs Red (B03 / B04): greenness index
      - NIR vs Green (B08 / B03): secondary spectral check
    """

    def __init__(self, epsilon: float = 1e-3):
        """
        Args:
            epsilon: Small additive constant before log. Use 1e-3 for [0,1]
                     reflectance — keeps log values in a compact range and
                     prevents log(0) without biasing near-zero bands heavily.
        """
        super().__init__()
        self.epsilon = epsilon

    def _log_ratio(self, a, b):
        """Compute log(clamp(a, 0) + eps) - log(clamp(b, 0) + eps) — bounded, finite, and differentiable."""
        a_pos = torch.clamp(a, min=0.0)
        b_pos = torch.clamp(b, min=0.0)
        return torch.log(a_pos + self.epsilon) - torch.log(b_pos + self.epsilon)

    def forward(self, pred, target):
        """
        Args:
            pred:   (B, 4, H, W) SR output in [0, 1] reflectance
            target: (B, 4, H, W) HR target in [0, 1] reflectance
        Returns:
            Scalar loss value (mean over batch, pixels, and ratio pairs).
        """
        # Band indices: 0=B02(Blue), 1=B03(Green), 2=B04(Red), 3=B08(NIR)
        pred_g   = pred[:, 1:2]
        pred_r   = pred[:, 2:3]
        pred_nir = pred[:, 3:4]

        targ_g   = target[:, 1:2]
        targ_r   = target[:, 2:3]
        targ_nir = target[:, 3:4]

        # Log-ratio differences: pred vs target
        # Each is |log(A/B)_pred - log(A/B)_target| — scale-invariant, bounded
        loss_nir_r = (self._log_ratio(pred_nir, pred_r) -
                      self._log_ratio(targ_nir, targ_r)).abs().mean()

        loss_g_r   = (self._log_ratio(pred_g, pred_r) -
                      self._log_ratio(targ_g, targ_r)).abs().mean()

        loss_nir_g = (self._log_ratio(pred_nir, pred_g) -
                      self._log_ratio(targ_nir, targ_g)).abs().mean()

        return (loss_nir_r + loss_g_r + loss_nir_g) / 3.0
