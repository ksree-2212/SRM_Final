import os
import numpy as np
import glob

def diagnostic_targets():
    hr_dir = r"C:\Users\mouni\OneDrive\Documents\sih 142\backend\Satellite_SRM_DL\data\train_patches\hr"
    hr_files = sorted(glob.glob(os.path.join(hr_dir, "*.npy")))
    if not hr_files:
        print("No HR patches found.")
        return
        
    print(f"Found {len(hr_files)} HR patches. Checking first 100...")
    all_hr = []
    for f in hr_files[:100]:
        hr = np.load(f)
        all_hr.append(hr)
    
    all_hr = np.stack(all_hr)  # (100, 4, H, W)
    
    print("HR Raw (from disk) Stats:")
    for b in range(4):
        band = all_hr[:, b, :, :]
        print(f"B{b}: min={band.min():.4f}, max={band.max():.4f}, mean={band.mean():.4f}, "
              f"median={np.median(band):.4f}, p5={np.percentile(band, 5):.4f}, "
              f"p95={np.percentile(band, 95):.4f}")

    # Paired dataset also normalizes:
    # if np.max(hr) > 10.0: hr = hr / 10000.0
    # hr = np.clip(hr, 0.0, 1.0)
    print("\nHR After PairedSRDataset Normalization:")
    for i in range(100):
        if np.max(all_hr[i]) > 10.0:
            all_hr[i] = all_hr[i] / 10000.0
        all_hr[i] = np.clip(all_hr[i], 0.0, 1.0)
        
    for b in range(4):
        band = all_hr[:, b, :, :]
        print(f"B{b}: min={band.min():.4f}, max={band.max():.4f}, mean={band.mean():.4f}, "
              f"median={np.median(band):.4f}, p5={np.percentile(band, 5):.4f}, "
              f"p95={np.percentile(band, 95):.4f}")

if __name__ == '__main__':
    diagnostic_targets()
