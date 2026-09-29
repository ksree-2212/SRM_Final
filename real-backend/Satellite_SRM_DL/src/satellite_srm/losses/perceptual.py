"""Perceptual feature loss for visible RGB channels (B04-B03-B02).

Uses a lightweight learned feature extractor (no VGG download required).
Operates entirely in-graph — no detach/numpy round-trips — so gradients
flow correctly through this loss during backward().
"""
from satellite_srm.compat import torch, nn


class PerceptualVGGLoss(nn.Module):
    """
    Lightweight multi-scale perceptual loss for 4-band multispectral SR.

    Operates on the visible RGB subset [B04, B03, B02] to capture spatial
    high-frequency detail without pushing NIR through an RGB network.

    Implementation notes
    --------------------
    * Pure PyTorch tensor ops — NO detach/numpy round-trips.
      Gradients are preserved end-to-end for all bands.
    * No external model download. Weights are randomly initialized and
      trained jointly with the super-resolution model.
    * Two conv layers give a ~3×3 and ~5×5 effective receptive field,
      which captures edge-level and texture-level structure.
    """

    def __init__(self):
        super().__init__()
        # Lightweight 3-channel feature extractor (no VGG, no download)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, pred, target):
        """
        Args:
            pred:   (B, 4, H, W) — SR output  in [0,1] reflectance
            target: (B, 4, H, W) — HR target  in [0,1] reflectance
        Returns:
            Scalar perceptual feature-matching loss (stays fully differentiable).
        """
        # Select and reorder to [R=B04, G=B03, B=B02] — pure tensor indexing,
        # gradient graph is NOT broken here (no detach/numpy).
        pred_rgb = pred[:, [2, 1, 0], :, :]    # shape (B, 3, H, W)
        targ_rgb = target[:, [2, 1, 0], :, :]  # shape (B, 3, H, W)

        f_pred = self.relu(self.conv2(self.relu(self.conv1(pred_rgb))))
        f_targ = self.relu(self.conv2(self.relu(self.conv1(targ_rgb))))

        # Detach target features — we only backprop through pred
        loss = (f_pred - f_targ.detach()).abs().mean()
        return loss
