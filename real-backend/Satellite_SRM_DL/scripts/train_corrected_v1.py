"""
train_corrected_v1.py
=====================
Fresh training run from random initialization.
Saves checkpoint as:  outputs/checkpoints/srm_corrected_v1.pt

Fixes applied vs latest.pt
---------------------------
1. SpectralConsistencyLoss  — log-ratio (bounded) instead of raw ratio
2. PerceptualVGGLoss        — pure tensor ops, no detach/numpy gradient break
3. CombinedSRMLoss weights  — L1=1.0, perc=0.05, spectral=0.10, ndvi=0.10
4. Normalization             — /10000 (DN→reflectance) everywhere, same in
                               training and inference (no percentile stretch)
5. Model init               — completely fresh random weights; latest.pt is
                               NOT loaded or fine-tuned.

Usage
-----
    cd "c:/Users/mouni/OneDrive/Documents/sih 142/backend/Satellite_SRM_DL"
    .venv\Scripts\python.exe scripts/train_corrected_v1.py

Python environment
------------------
    .venv\Scripts\python.exe  —  PyTorch 2.14.0+cpu  (CPU-only build)
    Confirmed working: 2026-09-26
"""

import os
import sys
import json
import time
import math

# ── resolve project root and package path ──────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH     = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import numpy as np

# ── torch ──────────────────────────────────────────────────────────────────
from satellite_srm.compat import torch, nn, HAS_TORCH
if not HAS_TORCH:
    raise RuntimeError("PyTorch is required for training. Please install torch.")

print(f"[INFO] PyTorch {torch.__version__}")
print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ── imports ────────────────────────────────────────────────────────────────
from satellite_srm.models.multispectral_swinir import MultispectralSwinIR
from satellite_srm.losses.combined import CombinedSRMLoss
from satellite_srm.datasets.paired_dataset import PairedSRDataset
from satellite_srm.datasets.transforms import get_transforms

# ── paths ──────────────────────────────────────────────────────────────────
DATA_DIR        = os.path.join(PROJECT_ROOT, "data", "train_patches")
VAL_DIR         = os.path.join(PROJECT_ROOT, "data", "val_patches")
CHECKPOINT_DIR  = os.path.join(PROJECT_ROOT, "outputs", "checkpoints")
LOG_DIR         = os.path.join(PROJECT_ROOT, "logs")
CHECKPOINT_OUT  = os.path.join(CHECKPOINT_DIR, "srm_corrected_v1.pt")
HISTORY_OUT     = os.path.join(LOG_DIR, "training_history_v1.json")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ── hyper-parameters ───────────────────────────────────────────────────────
EPOCHS        = 60
BATCH_SIZE    = 4
LR            = 2e-4
WEIGHT_DECAY  = 1e-4
PATIENCE      = 12          # early-stopping patience (epochs w/o improvement)
MIN_DELTA     = 5e-4        # minimum PSNR improvement to count as progress
GRAD_CLIP     = 1.0         # max gradient norm

# ── dataset ────────────────────────────────────────────────────────────────
print(f"\n[INFO] Loading training data from: {DATA_DIR}")
train_ds = PairedSRDataset(DATA_DIR, transform=get_transforms(is_train=True))
print(f"[INFO] Training patches : {len(train_ds)}")

val_ds = None
if os.path.isdir(VAL_DIR):
    lr_val = os.path.join(VAL_DIR, "lr")
    hr_val = os.path.join(VAL_DIR, "hr")
    if os.path.isdir(lr_val) and os.path.isdir(hr_val):
        val_ds = PairedSRDataset(VAL_DIR, transform=get_transforms(is_train=False))
        print(f"[INFO] Validation patches : {len(val_ds)}")

if len(train_ds) == 0:
    raise RuntimeError(f"No training patches found in {DATA_DIR}. "
                       "Run patch extraction first.")

# ── model — fresh random init ──────────────────────────────────────────────
print("\n[INFO] Initialising fresh MultispectralSwinIR (random weights, NOT from latest.pt)")
model = MultispectralSwinIR(
    scale_factor=3,
    in_channels=4,
    out_channels=4,
    embed_dim=60,
    depths=(4, 4, 4, 4),
    num_heads=(6, 6, 6, 6),
    window_size=4,
)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[INFO] Total parameters  : {total_params:,}")
print(f"[INFO] Trainable params  : {trainable_params:,}")

# ── loss ───────────────────────────────────────────────────────────────────
loss_fn = CombinedSRMLoss(
    l1_weight=1.0,
    perceptual_weight=0.05,
    spectral_weight=0.10,
    ndvi_weight=0.10,
).to(device)

# ── optimiser + scheduler ──────────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=LR * 0.01
)

# ── loss sanity check on first batch ──────────────────────────────────────
print("\n[SANITY] Running single-batch sanity check before full training ...")
model.train()
first_items = [train_ds[i] for i in range(min(BATCH_SIZE, len(train_ds)))]
lr_batch = torch.stack([it["lr"] for it in first_items]).to(device)
hr_batch = torch.stack([it["hr"] for it in first_items]).to(device)

print(f"  LR batch shape : {lr_batch.shape}  dtype={lr_batch.dtype}")
print(f"  HR batch shape : {hr_batch.shape}  dtype={hr_batch.dtype}")
print(f"  LR value range : [{lr_batch.min():.4f}, {lr_batch.max():.4f}]")
print(f"  HR value range : [{hr_batch.min():.4f}, {hr_batch.max():.4f}]")

with torch.no_grad():
    pred_sanity = model(lr_batch)
print(f"  SR output shape: {pred_sanity.shape}")
print(f"  SR value range : [{pred_sanity.min():.4f}, {pred_sanity.max():.4f}]")

optimizer.zero_grad()
pred_sanity = model(lr_batch)
loss_dict_sanity = loss_fn(pred_sanity, hr_batch)
total_s = loss_dict_sanity["total_loss"]
print(f"\n  [SANITY] Loss components (before any training):")
print(f"    L1         : {float(loss_dict_sanity['l1_loss']):.6f}  (weight=1.00)")
print(f"    Perceptual : {float(loss_dict_sanity['perceptual_loss']):.6f}  (weight=0.05)")
print(f"    NDVI       : {float(loss_dict_sanity['ndvi_loss']):.6f}  (weight=0.10)")
print(f"    Spectral   : {float(loss_dict_sanity['spectral_loss']):.6f}  (weight=0.10)")
print(f"    TOTAL      : {float(total_s):.6f}")

# gradient sanity
total_s.backward()
grad_norm = 0.0
for p in model.parameters():
    if p.grad is not None:
        grad_norm += p.grad.data.norm(2).item() ** 2
grad_norm = math.sqrt(grad_norm)
print(f"    Gradient norm (pre-clip): {grad_norm:.6f}")

if not math.isfinite(grad_norm):
    raise RuntimeError("[SANITY FAILED] Non-finite gradient norm detected. "
                       "Abort before training.")
if grad_norm > 1e4:
    print(f"  [WARN] Gradient norm {grad_norm:.2f} is large but finite - clipping will handle it.")
print("  [SANITY] PASSED - proceeding with full training.\n")

# ── training loop ──────────────────────────────────────────────────────────

def compute_psnr(pred_np, hr_np):
    mse = np.mean((pred_np - hr_np) ** 2)
    if mse == 0:
        return 100.0
    return float(10 * np.log10(1.0 / mse))


def run_validation(model, val_ds, device, n=50):
    model.eval()
    psnr_list = []
    n_eval = min(n, len(val_ds))
    with torch.no_grad():
        for i in range(n_eval):
            item  = val_ds[i]
            lr_t  = item["lr"].unsqueeze(0).to(device)
            hr_np = item["hr"].numpy()
            pred  = model(lr_t)
            pred_np = pred.squeeze(0).cpu().numpy()
            pred_np = np.clip(pred_np, 0.0, 1.0)
            psnr_list.append(compute_psnr(pred_np, hr_np))
    model.train()
    return float(np.mean(psnr_list))


history = []
best_psnr = -float("inf")
no_improve = 0
num_batches = max(1, len(train_ds) // BATCH_SIZE)

print(f"[TRAIN] Starting {EPOCHS}-epoch training run")
print(f"        Batches/epoch: {num_batches}  |  Batch size: {BATCH_SIZE}")
print(f"        LR: {LR}  |  Grad clip: {GRAD_CLIP}  |  Patience: {PATIENCE}\n")

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    model.train()

    epoch_total = 0.0
    epoch_l1    = 0.0
    epoch_perc  = 0.0
    epoch_ndvi  = 0.0
    epoch_spec  = 0.0
    epoch_gnorm = 0.0

    # Shuffle indices each epoch
    idx_order = np.random.permutation(len(train_ds))

    for b in range(num_batches):
        start = b * BATCH_SIZE
        end   = min(start + BATCH_SIZE, len(train_ds))
        idxs  = idx_order[start:end]

        batch_items = [train_ds[int(i)] for i in idxs]
        lr_b = torch.stack([it["lr"] for it in batch_items]).to(device)
        hr_b = torch.stack([it["hr"] for it in batch_items]).to(device)

        optimizer.zero_grad()
        pred = model(lr_b)
        ld   = loss_fn(pred, hr_b)
        loss = ld["total_loss"]
        loss.backward()

        # Gradient clipping
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        epoch_total += float(loss.detach().cpu())
        epoch_l1    += float(ld["l1_loss"].detach().cpu())
        epoch_perc  += float(ld["perceptual_loss"].detach().cpu())
        epoch_ndvi  += float(ld["ndvi_loss"].detach().cpu())
        epoch_spec  += float(ld["spectral_loss"].detach().cpu())
        epoch_gnorm += float(gnorm)

        if (b + 1) % max(1, num_batches // 4) == 0:
            print(f"  Epoch {epoch:3d} [{b+1:4d}/{num_batches}] "
                  f"loss={epoch_total/(b+1):.4f}  "
                  f"gnorm={epoch_gnorm/(b+1):.3f}")

    scheduler.step()
    elapsed = time.time() - t0

    # Per-epoch averages
    avg = lambda s: s / num_batches
    val_psnr = run_validation(model, val_ds, device) if val_ds else 0.0
    current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else LR

    rec = {
        "epoch":       epoch,
        "time_s":      round(elapsed, 2),
        "lr":          current_lr,
        "total_loss":  round(avg(epoch_total), 6),
        "l1_loss":     round(avg(epoch_l1),    6),
        "perc_loss":   round(avg(epoch_perc),  6),
        "ndvi_loss":   round(avg(epoch_ndvi),  6),
        "spec_loss":   round(avg(epoch_spec),  6),
        "grad_norm":   round(avg(epoch_gnorm), 4),
        "val_psnr":    round(val_psnr,         4),
    }
    history.append(rec)

    print(
        f"[Epoch {epoch:3d}/{EPOCHS}] ({elapsed:.1f}s) "
        f"loss={rec['total_loss']:.4f} "
        f"(L1={rec['l1_loss']:.4f} perc={rec['perc_loss']:.4f} "
        f"ndvi={rec['ndvi_loss']:.4f} spec={rec['spec_loss']:.4f}) "
        f"gnorm={rec['grad_norm']:.3f} "
        f"val_psnr={rec['val_psnr']:.2f} "
        f"lr={current_lr:.2e}"
    )

    # Early stopping
    improved = val_psnr > best_psnr + MIN_DELTA
    if improved:
        best_psnr  = val_psnr
        no_improve = 0
        # Save best checkpoint
        torch.save({
            "epoch":        epoch,
            "model_state":  model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "scheduler":    scheduler.state_dict(),
            "val_psnr":     val_psnr,
            "history":      history,
            "config": {
                "in_channels":  4,
                "out_channels": 4,
                "scale_factor": 3,
                "loss_weights": {
                    "l1": 1.0, "perceptual": 0.05,
                    "spectral": 0.10, "ndvi": 0.10,
                },
                "normalization": "DN/10000_reflectance",
                "note": "srm_corrected_v1 - fresh init, log-ratio spectral loss",
            }
        }, CHECKPOINT_OUT)
        print(f"  [BEST] New best PSNR {val_psnr:.2f} dB - checkpoint saved -> {CHECKPOINT_OUT}")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"\n[TRAIN] Early stopping at epoch {epoch} "
                  f"(no improvement for {PATIENCE} epochs).")
            break

# ── save final history ─────────────────────────────────────────────────────
with open(HISTORY_OUT, "w") as f:
    json.dump(history, f, indent=2)
print(f"\n[DONE] Training history -> {HISTORY_OUT}")
print(f"[DONE] Best val PSNR    : {best_psnr:.4f} dB")
print(f"[DONE] Checkpoint       : {CHECKPOINT_OUT}")
