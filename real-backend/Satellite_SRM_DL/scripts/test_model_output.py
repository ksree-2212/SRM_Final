"""Quick diagnostic: test that the model produces non-black (non-zero) output."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

# Force fallback to compat (no real torch)
from satellite_srm.compat import torch, nn, HAS_TORCH
print(f"Using real PyTorch: {HAS_TORCH}")

from satellite_srm.models.multispectral_swinir import MultispectralSwinIR

# Create model with small config
model = MultispectralSwinIR(
    scale_factor=3.0,
    in_channels=4,
    out_channels=4,
    embed_dim=48,
    depths=[2, 2],
    num_heads=[6, 6],
    window_size=8,
    drop_rate=0.05
)
model.eval()

# Create a synthetic input (4 bands, 16x16 pixels, values in [0,1] like normalized Sentinel-2)
np.random.seed(42)
test_input = np.random.rand(1, 4, 16, 16).astype(np.float32) * 0.5 + 0.1  # range ~0.1-0.6
input_tensor = torch.from_numpy(test_input)

print(f"Input shape: {input_tensor.shape}")
print(f"Input range: [{float(np.min(test_input)):.4f}, {float(np.max(test_input)):.4f}]")
print(f"Input mean: {float(np.mean(test_input)):.4f}")

# Run forward pass
output = model(input_tensor)
out_arr = output.numpy() if hasattr(output, 'numpy') else output

print(f"\nOutput shape: {out_arr.shape}")
print(f"Output range: [{float(np.min(out_arr)):.4f}, {float(np.max(out_arr)):.4f}]")
print(f"Output mean: {float(np.mean(out_arr)):.4f}")
print(f"Output std: {float(np.std(out_arr)):.4f}")

# Check if output is essentially black (all near-zero)
if np.max(np.abs(out_arr)) < 0.01:
    print("\n[FAIL] Output is essentially BLACK (all values near zero)")
    sys.exit(1)
elif np.std(out_arr) < 0.001:
    print("\n[FAIL] Output has NO spatial variation (flat/uniform)")
    sys.exit(1)
else:
    print("\n[PASS] Output has meaningful non-zero values with spatial variation")
    
    # Check per-band stats
    for b in range(min(4, out_arr.shape[1])):
        band = out_arr[0, b]
        print(f"  Band {b}: mean={float(np.mean(band)):.4f}, std={float(np.std(band)):.4f}, range=[{float(np.min(band)):.4f}, {float(np.max(band)):.4f}]")
