"""
train_unet.py
-------------
Train a simplified U-Net (Jones et al. 2026 architecture) for lightning
density prediction on ERA5 data.

Architecture (Table 2, Jones et al. 2026):
  - Input:      7 channels × spatial grid (H × W)
  - Encoder:    Block1: Conv3×3+ReLU → MaxPool2×2  (7 → 32 ch, H→H/2)
                Block2: Conv3×3+ReLU → MaxPool2×2  (32 → 16 ch, H/2→H/4)
  - Bottleneck: Conv3×3+ReLU  (16 → 8 ch)
  - Decoder:    Block1: TranspConv2×2 → Conv3×3+ReLU  (8 → 16 ch, H/4→H/2)
                Block2: TranspConv2×2 → Conv3×3+ReLU  (16 → 32 ch, H/2→H)
  - Output:     Conv1×1  (32 → 1 ch) — lightning density map
  - No skip connections (intentional, as per Jones et al.)
  - Loss:       MSE
  - Norm:       z-score on both inputs and output

Input channels (top-7 by feature importance from LightGBM):
  1. specific_cloud_ice_water_content_600hPa
  2. specific_cloud_ice_water_content_550hPa
  3. specific_cloud_ice_water_content_650hPa
  4. total_totals_index
  5. specific_cloud_ice_water_content_500hPa
  6. specific_cloud_liquid_water_content_700hPa
  7. convective_available_potential_energy

Usage:
    python train_unet.py
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────────────────
FEATURE_COLS = [
    'specific_cloud_ice_water_content_600hPa',
    'specific_cloud_ice_water_content_550hPa',
    'specific_cloud_ice_water_content_650hPa',
    'total_totals_index',
    'specific_cloud_ice_water_content_500hPa',
    'specific_cloud_liquid_water_content_700hPa',
    'convective_available_potential_energy',
]

# ERA5 spatial grid over Israel/E. Med domain (lat/lon from ds_single)
# Will be inferred from data at runtime — set to None to auto-detect
GRID_H = None
GRID_W = None

TRAIN_PARQUETS = [
    'data/tabular_dataset_2004.parquet',
    'data/tabular_dataset_2005.parquet',
    'data/tabular_dataset_2006.parquet',
    'data/tabular_dataset_2008.parquet',
    'data/tabular_dataset_2009.parquet',
    'data/tabular_dataset_2023.parquet',
    'data/tabular_dataset_2024.parquet',
]
TEST_PARQUET = 'data/tabular_dataset_2025.parquet'

BATCH_SIZE  = 32
EPOCHS      = 50
LR          = 1e-3
OUT_DIR     = 'results/unet'
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── U-Net Architecture ────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """3×3 Conv + ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LightningUNet(nn.Module):
    """
    Simplified U-Net from Jones et al. (2026).
    No skip connections. MSE loss target.
    """
    def __init__(self, in_channels=7):
        super().__init__()

        # Encoder
        self.enc1   = ConvBlock(in_channels, 32)
        self.pool1  = nn.MaxPool2d(2)

        self.enc2   = ConvBlock(32, 16)
        self.pool2  = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(16, 8)

        # Decoder
        self.up1    = nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2)
        self.dec1   = ConvBlock(16, 16)

        self.up2    = nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2)
        self.dec2   = ConvBlock(32, 32)

        # Output — sigmoid for binary classification
        self.out = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        x = self.enc1(x)
        x = self.pool1(x)
        x = self.enc2(x)
        x = self.pool2(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.up1(x)
        x = self.dec1(x)
        x = self.up2(x)
        x = self.dec2(x)

        # Output
        x = self.out(x)
        return x  # shape: (B, 1, H, W)


# ── Dataset ───────────────────────────────────────────────────────────────────

class LightningGridDataset(Dataset):
    """
    Reads ERA5 tabular parquet(s), reshapes each timestep into a spatial
    grid (H × W), and returns (input_tensor [7, H, W], target_tensor [1, H, W]).

    Applies z-score normalization using precomputed stats.
    """
    def __init__(self, parquet_paths, feature_cols, grid_h, grid_w,
                 feat_mean=None, feat_std=None, tgt_mean=None, tgt_std=None):
        self.feature_cols = feature_cols
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.feat_mean = feat_mean
        self.feat_std  = feat_std
        self.tgt_mean  = tgt_mean
        self.tgt_std   = tgt_std

        # Load all parquets, keep only needed columns
        cols = feature_cols + ['lightning_count', 'time']
        dfs = []
        for path in parquet_paths:
            if not os.path.exists(path):
                print(f"  WARNING: missing {path}, skipping")
                continue
            dfs.append(pd.read_parquet(path, columns=cols))
        self.df = pd.concat(dfs, ignore_index=True)

        # Sort once, build (start, end) index per timestep — O(1) lookup, no extra RAM
        self.df = self.df.sort_values('time').reset_index(drop=True)
        time_arr   = self.df['time'].values
        boundaries = np.where(time_arr[:-1] != time_arr[1:])[0] + 1
        starts     = np.concatenate([[0], boundaries])
        ends       = np.concatenate([boundaries, [len(self.df)]])
        self.times = np.sort(time_arr[starts])
        self.time_slices = {t: (int(s), int(e)) for t, s, e in zip(self.times, starts, ends)}
        print(f"  Dataset: {len(self.times):,} timesteps × {grid_h}×{grid_w} grid")

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        t = self.times[idx]
        s, e = self.time_slices[t]
        snap = self.df.iloc[s:e]

        # Features: (H*W, 7) → (7, H, W)
        X = snap[self.feature_cols].values.reshape(
            self.grid_h, self.grid_w, len(self.feature_cols)
        )
        X = X.transpose(2, 0, 1).astype(np.float32)   # (7, H, W)

        # Target: binary presence/absence (H*W,) → (1, H, W)
        y = (snap['lightning_count'].values > 0).reshape(
            self.grid_h, self.grid_w
        ).astype(np.float32)[np.newaxis]               # (1, H, W)

        # z-score normalization on features only (target is binary)
        if self.feat_mean is not None:
            X = (X - self.feat_mean[:, None, None]) / (self.feat_std[:, None, None] + 1e-8)

        return torch.from_numpy(X), torch.from_numpy(y)


# ── Normalization stats ───────────────────────────────────────────────────────

def compute_norm_stats(parquet_paths, feature_cols, sample_rows=500_000):
    """Compute mean/std over a sample of the training data."""
    print("Computing normalization statistics...")
    dfs = []
    rows_left = sample_rows
    for path in parquet_paths:
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path, columns=feature_cols + ['lightning_count'])
        df = df.sample(min(len(df), rows_left // len(parquet_paths)), random_state=42)
        dfs.append(df)
        rows_left -= len(df)
        if rows_left <= 0:
            break
    df_sample = pd.concat(dfs, ignore_index=True)

    feat_mean = df_sample[feature_cols].mean().values.astype(np.float32)
    feat_std  = df_sample[feature_cols].std().values.astype(np.float32)
    tgt_mean  = float(df_sample['lightning_count'].mean())
    tgt_std   = float(df_sample['lightning_count'].std())

    print(f"  Feature means: {feat_mean}")
    print(f"  Target mean/std: {tgt_mean:.4f} / {tgt_std:.4f}")
    return feat_mean, feat_std, tgt_mean, tgt_std


# ── Detect grid size from parquet ─────────────────────────────────────────────

def detect_grid_size(parquet_path):
    """Infer H, W from the number of rows per unique timestep."""
    df = pd.read_parquet(parquet_path, columns=['time', 'lat', 'lon'])
    t0 = df['time'].iloc[0]
    snap = df[df['time'] == t0]
    n_lats = snap['lat'].nunique()
    n_lons = snap['lon'].nunique()
    print(f"Detected grid: {n_lats} lat × {n_lons} lon")
    return n_lats, n_lons


# ── Training loop ─────────────────────────────────────────────────────────────

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        # crop y to pred's size (pred may be smaller due to odd spatial dims)
        if pred.shape != y.shape:
            y = y[:, :, :pred.shape[2], :pred.shape[3]]
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)


def compute_fss(pred, target, threshold=0.0, window=3):
    """
    Fractions Skill Score (Roberts & Lean, 2008) with a square neighbourhood.
    FSS = 1 - MSE(fractions) / ref_MSE
    where ref_MSE = (mean(O_frac²) + mean(M_frac²)).

    pred, target : torch tensors shape (B, 1, H, W), in normalised space.
    threshold    : scalar in normalised space (default 0 = above-mean density).
    window       : neighbourhood size (Jones et al. use 3×3).
    """
    import torch.nn.functional as F
    with torch.no_grad():
        obs_bin  = (target > threshold).float()
        pred_bin = (pred   > threshold).float()

        kernel = torch.ones(1, 1, window, window, device=pred.device) / (window * window)
        pad    = window // 2

        obs_frac  = F.conv2d(obs_bin,  kernel, padding=pad, groups=1)
        pred_frac = F.conv2d(pred_bin, kernel, padding=pad, groups=1)

        mse_frac = ((pred_frac - obs_frac) ** 2).mean().item()
        ref      = (pred_frac ** 2 + obs_frac ** 2).mean().item()

        if ref < 1e-12:
            return 1.0   # no lightning in either field → perfect score
        return 1.0 - mse_frac / ref


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_fss  = 0.0
    n_batches  = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            if pred.shape != y.shape:
                y = y[:, :, :pred.shape[2], :pred.shape[3]]
            loss = criterion(pred, y)
            total_loss += loss.item() * X.size(0)
            total_fss  += compute_fss(pred, y, threshold=0.5)
            n_batches  += 1
    mean_fss = total_fss / n_batches if n_batches > 0 else 0.0
    return total_loss / len(loader.dataset), mean_fss


# ── Padding helper ────────────────────────────────────────────────────────────

def pad_to_divisible(tensor, divisor=4):
    """Pad spatial dims to be divisible by `divisor` (needed for 2 pool ops)."""
    _, _, h, w = tensor.shape
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    if pad_h > 0 or pad_w > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h))
    return tensor


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")

    # Auto-detect grid size
    ref_parquet = next((p for p in TRAIN_PARQUETS if os.path.exists(p)), None)
    if ref_parquet is None:
        raise FileNotFoundError("No training parquet files found.")
    grid_h, grid_w = detect_grid_size(ref_parquet)

    # Pad to divisible by 4 for U-Net pooling
    pad_h = (4 - grid_h % 4) % 4
    pad_w = (4 - grid_w % 4) % 4
    grid_h_pad = grid_h + pad_h
    grid_w_pad = grid_w + pad_w
    print(f"Padded grid: {grid_h_pad} × {grid_w_pad}")

    # Normalization stats from training data
    feat_mean, feat_std, tgt_mean, tgt_std = compute_norm_stats(
        TRAIN_PARQUETS, FEATURE_COLS
    )
    stats = {
        'feat_mean': feat_mean.tolist(),
        'feat_std':  feat_std.tolist(),
        'tgt_mean':  tgt_mean,
        'tgt_std':   tgt_std,
        'grid_h':    grid_h,
        'grid_w':    grid_w,
        'features':  FEATURE_COLS,
    }
    with open(os.path.join(OUT_DIR, 'norm_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    # Datasets
    train_ds = LightningGridDataset(
        TRAIN_PARQUETS, FEATURE_COLS, grid_h, grid_w,
        feat_mean, feat_std, tgt_mean, tgt_std,
    )
    test_ds = LightningGridDataset(
        [TEST_PARQUET], FEATURE_COLS, grid_h, grid_w,
        feat_mean, feat_std, tgt_mean, tgt_std,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Model
    model     = LightningUNet(in_channels=len(FEATURE_COLS)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training on {len(train_ds):,} timesteps, testing on {len(test_ds):,}")

    # Training loop
    train_losses, test_losses, fss_scores = [], [], []
    best_test_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        train_loss            = train(model, train_loader, optimizer, criterion, DEVICE)
        test_loss, mean_fss   = evaluate(model, test_loader, criterion, DEVICE)
        scheduler.step(test_loss)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        fss_scores.append(mean_fss)

        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"train_loss={train_loss:.6f}  test_loss={test_loss:.6f}  FSS={mean_fss:.4f}")

        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(),
                       os.path.join(OUT_DIR, 'unet_best.pt'))

    # Save final model
    torch.save(model.state_dict(), os.path.join(OUT_DIR, 'unet_final.pt'))

    # Loss curve
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(train_losses, label='Train MSE')
    ax1.plot(test_losses,  label='Test MSE')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('MSE Loss')
    ax1.set_title('U-Net Training — Jones et al. (2026) Architecture')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(fss_scores, color='green', label='FSS (3×3 window, threshold=0)')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('FSS')
    ax2.set_title('Fractions Skill Score on Test Set')
    ax2.set_ylim([0, 1]); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'loss_curve.png'), dpi=150)
    plt.close()

    print(f"\nBest test loss: {best_test_loss:.6f}")
    print(f"Best FSS:       {max(fss_scores):.4f}")
    print(f"Outputs saved to {OUT_DIR}/")
