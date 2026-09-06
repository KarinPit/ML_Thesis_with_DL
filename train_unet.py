"""
train_unet.py
-------------
Train a simplified U-Net (Jones et al. 2026 architecture) for lightning
prediction on ERA5 + IMERG data using Jones et al. CPLRSTW features.

Architecture (Table 2, Jones et al. 2026):
  - Input:      7 channels × spatial grid (H × W)
  - Encoder:    Block1: Conv3×3+ReLU → MaxPool2×2  (7 → 32 ch, H→H/2)
                Block2: Conv3×3+ReLU → MaxPool2×2  (32 → 16 ch, H/2→H/4)
  - Bottleneck: Conv3×3+ReLU  (16 → 8 ch)
  - Decoder:    Block1: TranspConv2×2 → ReLU → Conv3×3+ReLU  (8 → 16 ch, H/4→H/2)
                Block2: TranspConv2×2 → ReLU → Conv3×3+ReLU  (16 → 32 ch, H/2→H)
  - Output:     Conv1×1  (32 → 1 ch) — raw z-scored lightning density
  - No skip connections (intentional, as per Jones et al.)
  - Loss:       MSE on z-scored lightning density (matching Jones et al.)
  - Norm:       z-score on both inputs and target

Input channels (Jones et al. CPLRSTW — 7 variables):
  C 1. cape                ← ERA5 single-level (original download)
  P 2. precipitation       ← NASA GPM IMERG V07 hourly
  L 3. land_sea_mask       ← ERA5 single-level
  R 4. rh_avg              ← derived from q+T at 500 & 1000 hPa (avg)
  S 5. wind_shear          ← sqrt((u500−u1000)²+(v500−v1000)²)
  T 6. 2m_temperature      ← ERA5 single-level
  W 7. wcd                 ← Zero Degree Level − Cloud Base Height

Data: jones_tabular_dataset_{year}.parquet (from build_jones_tabular_dataset.py)

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
# FEATURE_COLS = [
#     # Jones et al. CPLRSTW (7) — from combined_tabular_dataset_{year}.parquet
#     'cape',            # C — convective available potential energy
#     'precipitation',   # P — IMERG hourly precipitation
#     'land_sea_mask',   # L
#     'rh_avg',          # R — mean relative humidity (500 & 1000 hPa)
#     'wind_shear',      # S — deep-layer wind shear (500–1000 hPa)
#     '2m_temperature',  # T
#     'wcd',             # W — warm cloud depth (ZDL − CBH)
#     # Exp 7b top-6 microphysical features (excl. cape = already above)
#     'specific_cloud_ice_water_content_600hPa',   # rank 1 (gain 0.531)
#     'specific_cloud_ice_water_content_550hPa',   # rank 2 (gain 0.150)
#     'specific_cloud_ice_water_content_650hPa',   # rank 3 (gain 0.039)
#     'total_totals_index',                         # rank 4 (gain 0.026)
#     'specific_cloud_ice_water_content_500hPa',   # rank 5 (gain 0.023)
#     'specific_cloud_liquid_water_content_700hPa', # rank 6 (gain 0.022)
# ]

FEATURE_COLS = [
    # Top-20 XGBoost feature importance (Exp 12)
    'specific_cloud_ice_water_content_600hPa',      # rank  1 (0.428)
    'specific_cloud_ice_water_content_550hPa',      # rank  2 (0.130)
    'specific_cloud_ice_water_content_650hPa',      # rank  3 (0.055)
    'total_totals_index',                            # rank  4 (0.034)
    'specific_cloud_ice_water_content_500hPa',      # rank  5 (0.013)
    'specific_cloud_liquid_water_content_700hPa',   # rank  6 (0.013)
    'convective_available_potential_energy',         # rank  7 (0.013)
    'total_column_cloud_ice_water',                  # rank  8 (0.012)
    'total_column_cloud_liquid_water',               # rank  9 (0.009)
    'specific_cloud_liquid_water_content_775hPa',   # rank 10 (0.007)
    'specific_cloud_liquid_water_content_750hPa',   # rank 11 (0.007)
    'specific_cloud_liquid_water_content_850hPa',   # rank 12 (0.005)
    'specific_cloud_liquid_water_content_825hPa',   # rank 13 (0.005)
    'proxy_lpi',                                     # rank 14 (0.004)
    'vertical_velocity_850hPa',                      # rank 15 (0.004)
    'specific_cloud_ice_water_content_400hPa',      # rank 16 (0.004)
    'k_index',                                       # rank 17 (0.004)
    'temperature_250hPa',                            # rank 18 (0.003)
    'temperature_225hPa',                            # rank 19 (0.003)
    'specific_cloud_ice_water_content_450hPa',      # rank 20 (0.003)
]

# No aux parquets needed — CIWC already baked into jones_ciwc_tabular_dataset files
AUX_TRAIN_PARQUETS = None
AUX_TEST_PARQUET   = None
AUX_COLS           = None

# ERA5 spatial grid over Israel/E. Med domain (lat/lon from ds_single)
# Will be inferred from data at runtime — set to None to auto-detect
GRID_H = None
GRID_W = None

# TRAIN_PARQUETS = [
#     'data/combined_tabular_dataset_2004.parquet',
#     'data/combined_tabular_dataset_2005.parquet',
#     'data/combined_tabular_dataset_2006.parquet',
#     'data/combined_tabular_dataset_2008.parquet',
#     'data/combined_tabular_dataset_2009.parquet',
#     'data/combined_tabular_dataset_2023.parquet',
#     'data/combined_tabular_dataset_2024.parquet',
# ]
# TEST_PARQUET = 'data/combined_tabular_dataset_2025.parquet'

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
LR          = 1e-3   # Exp 7b LR (training from scratch)
OUT_DIR     = 'results/unet_exp13_top20_seasonal'
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
AGG_HOURS     = 1           # 1 = every hour is one sample; 3/6/12 = aggregate N hours into one window
SEED          = 42          # set to None to disable fixed seed
ACTIVE_MONTHS = [10, 11, 12, 1, 2, 3]  # Oct–Mar lightning season; set to None to use all months
BINARY_TARGET = True        # True = BCE binary classification; False = MSE z-scored density
# Unweighted BCE — same as Exp 7b (FSS 0.613)
# Set to a value (e.g. 40) to weight false negatives more strongly
BCE_POS_WEIGHT = None

# ── Transfer learning ─────────────────────────────────────────────────────────
PRETRAINED_WEIGHTS = None  # train from scratch — 13-ch input differs from Jones 7-ch
FREEZE_ENCODER     = False

# Key mapping: Jones .pth → our model attribute names
JONES_KEY_MAP = {
    'encoder0.0.0.weight': 'enc1.block.0.weight',
    'encoder0.0.0.bias':   'enc1.block.0.bias',
    'encoder1.0.0.weight': 'enc2.block.0.weight',
    'encoder1.0.0.bias':   'enc2.block.0.bias',
    'center.0.weight':     'bottleneck.block.0.weight',
    'center.0.bias':       'bottleneck.block.0.bias',
    'decoder1.0.weight':   'up1.0.weight',
    'decoder1.0.bias':     'up1.0.bias',
    'decoder1.2.weight':   'dec1.block.0.weight',
    'decoder1.2.bias':     'dec1.block.0.bias',
    'decoder0.0.weight':   'up2.0.weight',
    'decoder0.0.bias':     'up2.0.bias',
    'decoder0.2.weight':   'dec2.block.0.weight',
    'decoder0.2.bias':     'dec2.block.0.bias',
    'outputs.0.weight':    'out.weight',
    'outputs.0.bias':      'out.bias',
}

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

        # Decoder — Jones uses TranspConv+ReLU → Conv+ReLU per block
        self.up1    = nn.Sequential(nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2), nn.ReLU(inplace=True))
        self.dec1   = ConvBlock(16, 16)

        self.up2    = nn.Sequential(nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2), nn.ReLU(inplace=True))
        self.dec2   = ConvBlock(32, 32)

        # Output — raw logits (no activation), MSE on z-scored density (Jones et al.)
        self.out = nn.Conv2d(32, 1, kernel_size=1)

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
    Reads ERA5 tabular parquet(s), aggregates to 12-hourly windows (Jones et al.),
    reshapes each window into a spatial grid (H × W), and returns
    (input_tensor [7, H, W], target_tensor [1, H, W]).

    Aggregation strategy (matching Jones et al.):
      - ERA5 features:   snapshot at the first hour of each 12-hour window (00Z / 12Z)
      - Precipitation:   mean rate over the 12-hour window
      - Lightning:       sum of counts over the 12-hour window
    """
    def __init__(self, parquet_paths, feature_cols, grid_h, grid_w,
                 feat_mean=None, feat_std=None, tgt_mean=None, tgt_std=None,
                 agg_hours=12, binary_target=False,
                 aux_parquet_paths=None, aux_cols=None,
                 active_months=None):
        self.feature_cols = feature_cols
        self.grid_h   = grid_h
        self.grid_w   = grid_w
        self.feat_mean = feat_mean
        self.feat_std  = feat_std
        self.tgt_mean  = tgt_mean
        self.tgt_std   = tgt_std
        self.agg_hours = agg_hours
        self.binary_target = binary_target
        self.precip_idx = feature_cols.index('precipitation') if 'precipitation' in feature_cols else None

        # Jones feature cols (excluding any aux cols)
        jones_cols = [c for c in feature_cols if c not in (aux_cols or [])]

        # Load Jones parquets — include lat/lon only if we need to align aux features
        need_latlon = bool(aux_parquet_paths and aux_cols)
        sort_keys   = ['time', 'lat', 'lon'] if need_latlon else ['time']
        load_cols   = jones_cols + ['lightning_count', 'time'] + (['lat', 'lon'] if need_latlon else [])

        dfs = []
        for path in parquet_paths:
            if not os.path.exists(path):
                print(f"  WARNING: missing {path}, skipping")
                continue
            dfs.append(pd.read_parquet(path, columns=load_cols))
        self.df = pd.concat(dfs, ignore_index=True).sort_values(sort_keys).reset_index(drop=True)

        # Align and attach auxiliary features via column-concat (no merge — avoids OOM)
        # Both parquets share the same (time, lat, lon) grid; sorting aligns rows exactly.
        if aux_parquet_paths and aux_cols:
            aux_dfs = []
            for path in aux_parquet_paths:
                if not os.path.exists(path):
                    print(f"  WARNING: missing aux {path}, skipping")
                    continue
                aux_dfs.append(pd.read_parquet(path, columns=aux_cols + ['time', 'lat', 'lon']))
            if aux_dfs:
                aux_df = pd.concat(aux_dfs, ignore_index=True).sort_values(sort_keys).reset_index(drop=True)
                self.df = pd.concat([self.df.drop(columns=['lat', 'lon']), aux_df[aux_cols]], axis=1)
                print(f"  Attached {aux_cols} from {len(aux_dfs)} aux parquet(s) via column-concat")

        # Sort and build per-hour index
        self.df = self.df.sort_values('time').reset_index(drop=True)
        time_arr   = self.df['time'].values
        boundaries = np.where(time_arr[:-1] != time_arr[1:])[0] + 1
        starts     = np.concatenate([[0], boundaries])
        ends       = np.concatenate([boundaries, [len(self.df)]])
        all_times  = np.sort(time_arr[starts])
        self.time_slices = {t: (int(s), int(e)) for t, s, e in zip(all_times, starts, ends)}

        # Group individual hours into agg_hours-hour windows aligned to 00Z
        times_pd    = pd.DatetimeIndex([pd.Timestamp(t) for t in all_times])
        win_labels  = times_pd.floor(f'{agg_hours}h')
        from collections import defaultdict
        win_dict = defaultdict(list)
        for t, wl in zip(all_times, win_labels):
            win_dict[wl.to_datetime64()].append(t)

        all_windows = np.array(sorted(win_dict.keys()))

        # Filter to active months (e.g. Oct–Mar lightning season)
        if active_months:
            keep = np.array([pd.Timestamp(w).month in active_months for w in all_windows])
            all_windows = all_windows[keep]
            print(f"  Seasonal filter: months {active_months} → {keep.sum():,}/{len(keep):,} windows kept")

        self.windows         = all_windows
        self.window_to_times = {w: sorted(ts) for w, ts in win_dict.items()}
        print(f"  Dataset: {len(self.windows):,} {agg_hours}-hour windows × {grid_h}×{grid_w} grid")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window_start = self.windows[idx]
        hour_times   = self.window_to_times[window_start]
        H, W, C      = self.grid_h, self.grid_w, len(self.feature_cols)

        # Load all hourly grids in this window
        feat_stack = []
        light_sum  = np.zeros((H, W), dtype=np.float32)
        for i, t in enumerate(hour_times):
            s, e = self.time_slices[t]
            snap = self.df.iloc[s:e]

            f = snap[self.feature_cols].values.reshape(H, W, C).transpose(2, 0, 1).astype(np.float32)
            f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
            feat_stack.append(f)

            light_sum += snap['lightning_count'].values.reshape(H, W).astype(np.float32)

        feat_stack = np.stack(feat_stack, axis=0)   # (n_hours, C, H, W)

        # ERA5 features: snapshot at first hour of window
        X = feat_stack[0].copy()                    # (C, H, W)

        # Precipitation: mean rate over all hours in window
        if self.precip_idx is not None:
            X[self.precip_idx] = feat_stack[:, self.precip_idx, :, :].mean(axis=0)

        # Target: binary (any lightning in window) or z-scored density
        if self.binary_target:
            y = (light_sum > 0).astype(np.float32)[np.newaxis]   # (1, H, W) binary
        else:
            y = light_sum[np.newaxis]                             # (1, H, W) counts

        # z-score normalization on features (always); target only for MSE mode
        if self.feat_mean is not None:
            X = (X - self.feat_mean[:, None, None]) / (self.feat_std[:, None, None] + 1e-8)
        if self.tgt_mean is not None and not self.binary_target:
            y = (y - self.tgt_mean) / (self.tgt_std + 1e-8)

        return torch.from_numpy(X), torch.from_numpy(y)


# ── Normalization stats ───────────────────────────────────────────────────────

def compute_norm_stats(parquet_paths, feature_cols, agg_hours=12, sample_rows=500_000,
                       aux_parquet_paths=None, aux_cols=None):
    """
    Compute feature mean/std from hourly data (ERA5 snapshot values).
    Compute target mean/std from 12-hourly SUMMED lightning (matching Jones et al.).
    Optionally merges aux_cols from aux_parquet_paths before computing stats.
    """
    print("Computing normalization statistics...")
    jones_cols = [c for c in feature_cols if c not in (aux_cols or [])]
    need_latlon = bool(aux_parquet_paths and aux_cols)
    load_cols   = jones_cols + ['lightning_count', 'time'] + (['lat', 'lon'] if need_latlon else [])

    dfs = []
    rows_left = sample_rows
    for path in parquet_paths:
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path, columns=load_cols)
        df = df.sample(min(len(df), rows_left // len(parquet_paths)), random_state=42)
        dfs.append(df)
        rows_left -= len(df)
        if rows_left <= 0:
            break
    df_sample = pd.concat(dfs, ignore_index=True)

    if aux_parquet_paths and aux_cols:
        # Sample the same row indices from aux parquets, align by sort, column-concat
        sort_keys = ['time', 'lat', 'lon']
        df_sample = df_sample.sort_values(sort_keys).reset_index(drop=True)
        aux_dfs = []
        for path in aux_parquet_paths:
            if not os.path.exists(path):
                continue
            # Sample same fraction as main (approximation — stats only need rough values)
            adf = pd.read_parquet(path, columns=aux_cols + sort_keys)
            adf = adf.sample(min(len(adf), sample_rows // len(parquet_paths)), random_state=42)
            aux_dfs.append(adf)
        if aux_dfs:
            aux_df = pd.concat(aux_dfs, ignore_index=True).sort_values(sort_keys).reset_index(drop=True)
            # Trim to same length in case of minor count mismatch from sampling
            n = min(len(df_sample), len(aux_df))
            df_sample = pd.concat([df_sample.iloc[:n].drop(columns=['lat', 'lon']),
                                    aux_df.iloc[:n][aux_cols]], axis=1)

    feat_mean = df_sample[feature_cols].mean().values.astype(np.float32)
    feat_std  = df_sample[feature_cols].std().values.astype(np.float32)

    # Target stats from aggregated lightning (sum over agg_hours-hour windows)
    df_sample['time'] = pd.to_datetime(df_sample['time'])
    df_sample['window'] = df_sample['time'].dt.floor(f'{agg_hours}h')
    agg_lightning = df_sample.groupby('window')['lightning_count'].sum()
    tgt_mean = float(agg_lightning.mean())
    tgt_std  = float(agg_lightning.std())

    print(f"  Feature means: {feat_mean}")
    print(f"  Target mean/std (12-hr aggregated): {tgt_mean:.4f} / {tgt_std:.4f}")
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


def load_jones_weights(model, path, key_map):
    """
    Load Jones et al. pre-trained weights into our model using a key remapping.
    Encoder weights transfer directly (same 7-channel CPLRSTW input).
    Decoder weights provide a warm start but will be fine-tuned.
    """
    jones_state = torch.load(path, map_location='cpu')
    our_state   = model.state_dict()

    loaded, skipped = [], []
    for jones_key, our_key in key_map.items():
        if jones_key not in jones_state or our_key not in our_state:
            skipped.append((our_key, 'missing in jones', 'n/a'))
            continue

        j_w = jones_state[jones_key]
        o_w = our_state[our_key]

        if j_w.shape == o_w.shape:
            # Exact match — copy directly
            our_state[our_key] = j_w
            loaded.append(our_key)
        elif j_w.ndim == 4 and j_w.shape[1] < o_w.shape[1]:
            # Input-channel mismatch (e.g. enc1 conv: Jones 7-ch, ours 10-ch).
            # Copy Jones weights for the first N channels; leave extra channels
            # at their current random init so the model can learn from new features.
            n = j_w.shape[1]
            our_state[our_key][:, :n, :, :] = j_w
            loaded.append(f"{our_key} (partial {n}/{o_w.shape[1]} in-channels)")
        else:
            skipped.append((our_key, j_w.shape, o_w.shape))

    model.load_state_dict(our_state)
    print(f"Loaded {len(loaded)}/{len(key_map)} weight tensors from {path}")
    if skipped:
        print(f"  Skipped (shape mismatch or missing): {skipped}")
    return model


def freeze_encoder(model):
    """Freeze encoder layers so only decoder is trained."""
    for module in [model.enc1, model.enc2, model.bottleneck]:
        for param in module.parameters():
            param.requires_grad = False
    frozen   = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Encoder frozen: {frozen:,} frozen params, {trainable:,} trainable params")


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
            total_fss  += compute_fss(pred, y, threshold=0.0)
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

    if SEED is not None:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        print(f"Random seed: {SEED}")

    print(f"Device: {DEVICE}")
    print(f"AGG_HOURS: {AGG_HOURS}")

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

    # Normalization stats from training data (target stats use 12-hr aggregated lightning)
    feat_mean, feat_std, tgt_mean, tgt_std = compute_norm_stats(
        TRAIN_PARQUETS, FEATURE_COLS, agg_hours=AGG_HOURS,
        aux_parquet_paths=AUX_TRAIN_PARQUETS, aux_cols=AUX_COLS,
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
        feat_mean, feat_std, tgt_mean, tgt_std, agg_hours=AGG_HOURS,
        binary_target=BINARY_TARGET,
        aux_parquet_paths=AUX_TRAIN_PARQUETS, aux_cols=AUX_COLS,
        active_months=ACTIVE_MONTHS,
    )
    test_ds = LightningGridDataset(
        [TEST_PARQUET], FEATURE_COLS, grid_h, grid_w,
        feat_mean, feat_std, tgt_mean, tgt_std, agg_hours=AGG_HOURS,
        binary_target=BINARY_TARGET,
        aux_parquet_paths=[AUX_TEST_PARQUET], aux_cols=AUX_COLS,
        active_months=ACTIVE_MONTHS,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Model
    model = LightningUNet(in_channels=len(FEATURE_COLS)).to(DEVICE)

    # Transfer learning: load Jones pre-trained weights, freeze encoder
    if PRETRAINED_WEIGHTS and os.path.exists(PRETRAINED_WEIGHTS):
        print(f"\nLoading Jones pre-trained weights from {PRETRAINED_WEIGHTS}...")
        load_jones_weights(model, PRETRAINED_WEIGHTS, JONES_KEY_MAP)
        if FREEZE_ENCODER:
            freeze_encoder(model)
    else:
        print("\nNo pretrained weights found — training from scratch.")

    # Only pass trainable parameters to the optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    if BINARY_TARGET:
        if BCE_POS_WEIGHT is not None:
            pw = torch.tensor([BCE_POS_WEIGHT], device=DEVICE)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
            print(f"BCE pos_weight={BCE_POS_WEIGHT} (penalises false positives)")
        else:
            criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    print(f"\nTotal model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training on {len(train_ds):,} timesteps, testing on {len(test_ds):,}")

    # Training loop
    train_losses, test_losses, fss_scores = [], [], []
    best_test_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        train_loss       = train(model, train_loader, optimizer, criterion, DEVICE)
        test_loss, mean_fss = evaluate(model, test_loader, criterion, DEVICE)
        scheduler.step(test_loss)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        fss_scores.append(mean_fss)

        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"train_loss={train_loss:.6f}  test_loss={test_loss:.6f}  "
              f"FSS={mean_fss:.4f}")

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
    ax1.set_title('U-Net Training — Jones CPLRSTW Features + Pretrained Weights (MSE)')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(fss_scores, color='green', label='FSS (3×3 window, threshold=0)')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('FSS')
    ax2.set_title('Fractions Skill Score on Test Set')
    ax2.set_ylim([0, 1]); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'loss_curve.png'), dpi=150)
    plt.close()

    print(f"\nBest test loss:   {best_test_loss:.6f}")
    print(f"Best FSS (thr=0): {max(fss_scores):.4f}")
    print(f"Outputs saved to {OUT_DIR}/")
