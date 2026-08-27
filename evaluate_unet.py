"""
evaluate_unet.py
----------------
Post-training evaluation of the best U-Net model.

Computes:
  1. Per-timestep r² — predicted probability vs binary observed, averaged over time
  2. Climatological r² — mean predicted map vs mean observed map over all test timesteps
     (this is the metric Jones et al. report, r²=0.92)
  3. Scatter plot: predicted climatology vs observed climatology (per grid cell)
  4. Spatial maps: mean observed vs mean predicted

Usage:
    python evaluate_unet.py
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Reuse classes from train_unet.py ─────────────────────────────────────────

FEATURE_COLS = [
    'specific_cloud_ice_water_content_600hPa',
    'specific_cloud_ice_water_content_550hPa',
    'specific_cloud_ice_water_content_650hPa',
    'total_totals_index',
    'specific_cloud_ice_water_content_500hPa',
    'specific_cloud_liquid_water_content_700hPa',
    'convective_available_potential_energy',
]

TEST_PARQUET = 'data/tabular_dataset_2025.parquet'
MODEL_PATH   = 'results/unet/unet_best.pt'
STATS_PATH   = 'results/unet/norm_stats.json'
OUT_DIR      = 'results/unet'
BATCH_SIZE   = 32
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class LightningUNet(nn.Module):
    def __init__(self, in_channels=7):
        super().__init__()
        self.enc1       = ConvBlock(in_channels, 32)
        self.pool1      = nn.MaxPool2d(2)
        self.enc2       = ConvBlock(32, 16)
        self.pool2      = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(16, 8)
        self.up1        = nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2)
        self.dec1       = ConvBlock(16, 16)
        self.up2        = nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2)
        self.dec2       = ConvBlock(32, 32)
        self.out        = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        x = self.pool1(self.enc1(x))
        x = self.pool2(self.enc2(x))
        x = self.bottleneck(x)
        x = self.dec1(self.up1(x))
        x = self.dec2(self.up2(x))
        return self.out(x)


class LightningGridDataset(Dataset):
    def __init__(self, parquet_paths, feature_cols, grid_h, grid_w,
                 feat_mean=None, feat_std=None):
        self.feature_cols = feature_cols
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.feat_mean = feat_mean
        self.feat_std  = feat_std

        cols = feature_cols + ['lightning_count', 'time']
        dfs = []
        for path in parquet_paths:
            if not os.path.exists(path):
                print(f"  WARNING: missing {path}")
                continue
            dfs.append(pd.read_parquet(path, columns=cols))
        self.df = pd.concat(dfs, ignore_index=True)

        self.df = self.df.sort_values('time').reset_index(drop=True)
        time_arr   = self.df['time'].values
        boundaries = np.where(time_arr[:-1] != time_arr[1:])[0] + 1
        starts     = np.concatenate([[0], boundaries])
        ends       = np.concatenate([boundaries, [len(self.df)]])
        self.times = np.sort(time_arr[starts])
        self.time_slices = {t: (int(s), int(e)) for t, s, e in zip(self.times, starts, ends)}

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        t = self.times[idx]
        s, e = self.time_slices[t]
        snap = self.df.iloc[s:e]

        X = snap[self.feature_cols].values.reshape(
            self.grid_h, self.grid_w, len(self.feature_cols)
        ).transpose(2, 0, 1).astype(np.float32)

        y = (snap['lightning_count'].values > 0).reshape(
            self.grid_h, self.grid_w
        ).astype(np.float32)[np.newaxis]

        if self.feat_mean is not None:
            X = (X - self.feat_mean[:, None, None]) / (self.feat_std[:, None, None] + 1e-8)

        return torch.from_numpy(X), torch.from_numpy(y)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"Device: {DEVICE}")

    # Load norm stats
    with open(STATS_PATH) as f:
        stats = json.load(f)
    feat_mean = np.array(stats['feat_mean'], dtype=np.float32)
    feat_std  = np.array(stats['feat_std'],  dtype=np.float32)
    grid_h    = stats['grid_h']
    grid_w    = stats['grid_w']
    print(f"Grid: {grid_h}×{grid_w}")

    # Dataset
    ds = LightningGridDataset(
        [TEST_PARQUET], FEATURE_COLS, grid_h, grid_w,
        feat_mean, feat_std,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Load model
    model = LightningUNet(in_channels=len(FEATURE_COLS)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Loaded model from {MODEL_PATH}")

    # ── Collect all predictions and targets ───────────────────────────────────
    all_preds   = []   # list of (1, H', W') arrays
    all_targets = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            pred = model(X)
            # crop target to pred size (odd dims)
            if pred.shape != y.shape:
                y = y[:, :, :pred.shape[2], :pred.shape[3]]
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.numpy())

    # shape: (N_timesteps, 1, H', W')
    preds   = np.concatenate(all_preds,   axis=0)
    targets = np.concatenate(all_targets, axis=0)
    H, W    = preds.shape[2], preds.shape[3]
    print(f"Collected {len(preds):,} timesteps, spatial size {H}×{W}")

    # ── 1. Per-timestep r² ────────────────────────────────────────────────────
    # For each timestep, flatten and compute r² between predicted prob and binary obs
    r2_per_timestep = []
    for i in range(len(preds)):
        p = preds[i, 0].ravel()
        t = targets[i, 0].ravel()
        if t.sum() == 0:
            continue   # skip timesteps with no lightning (r² undefined)
        r2_per_timestep.append(r2_score(t, p))

    mean_r2 = np.mean(r2_per_timestep)
    print(f"\nPer-timestep r² (lightning timesteps only): {mean_r2:.4f}")

    # ── 2. Climatological r² (Jones et al. style) ─────────────────────────────
    # Mean over all timesteps → one spatial map each
    mean_pred   = preds[:, 0, :, :].mean(axis=0)    # (H', W')
    mean_target = targets[:, 0, :, :].mean(axis=0)

    clim_r2 = r2_score(mean_target.ravel(), mean_pred.ravel())
    print(f"Climatological r² (Jones et al. style):     {clim_r2:.4f}")

    # ── 3. Scatter plot: predicted vs observed climatology ────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(mean_target.ravel(), mean_pred.ravel(), alpha=0.3, s=10, color='steelblue')
    lim = max(mean_target.max(), mean_pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', lw=1, label='1:1 line')
    ax.set_xlabel('Observed mean lightning frequency')
    ax.set_ylabel('Predicted mean lightning probability')
    ax.set_title(f'Climatological r² = {clim_r2:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    scatter_path = os.path.join(OUT_DIR, 'climatological_scatter.png')
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"Scatter plot saved to {scatter_path}")

    # ── 4. Spatial maps: observed vs predicted climatology ────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax1.imshow(mean_target, origin='lower', cmap='Blues')
    ax1.set_title('Observed — mean lightning frequency (2025)')
    plt.colorbar(im1, ax=ax1, label='fraction of hours with lightning')

    im2 = ax2.imshow(mean_pred, origin='lower', cmap='Reds')
    ax2.set_title('Predicted — mean lightning probability (2025)')
    plt.colorbar(im2, ax=ax2, label='predicted probability')

    plt.suptitle(f'Climatological r² = {clim_r2:.4f}  |  Mean per-timestep r² = {mean_r2:.4f}')
    plt.tight_layout()
    maps_path = os.path.join(OUT_DIR, 'climatological_maps.png')
    plt.savefig(maps_path, dpi=150)
    plt.close()
    print(f"Spatial maps saved to {maps_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Per-timestep r²  (lightning steps): {mean_r2:.4f}")
    print(f"Climatological r² (Jones style):    {clim_r2:.4f}")
    print(f"Jones et al. best (CPLRSTW):        0.92 (ocean), 0.77 (land)")
    print(f"{'='*50}")
