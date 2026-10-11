import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.stats as stats
import netCDF4
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import sys
import random
import os
import xarray as xr
import numpy as np
import numpy.ma as ma
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from eofs.xarray import Eof
import time
import h5py
from scipy.signal import correlation_lags
from scipy.stats import pearsonr
from scipy.stats import linregress
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import gc

os.chdir('/home/ec2-user/ML_Thesis_with_DL/data/')

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}", flush=True)

# ── Geographic crop (set to None to use full domain) ──────────────────────────
# Full domain: lat 28–43.25°N, lon 0–39.25°E
# Tip: raise LON_MIN to ~10 to cut out the Sahara; raise LAT_MIN to ~30 to drop deep Negev/desert
LAT_MIN = 30.0   # °N  — southern boundary
LAT_MAX = 43.0   # °N  — northern boundary
LON_MIN = 10.0   # °E  — western boundary (0 = full domain incl. Sahara)
LON_MAX = 39.0   # °E  — eastern boundary

# ── Load dataset ───────────────────────────────────────────────────────────────
ds_daily = xr.open_dataset('jones_israel_all_years.nc')

# Apply geographic crop before divisibility trim
ds_daily = ds_daily.sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))
print(f"Geographic crop: lat {float(ds_daily.lat.min()):.2f}–{float(ds_daily.lat.max()):.2f}°N, "
      f"lon {float(ds_daily.lon.min()):.2f}–{float(ds_daily.lon.max()):.2f}°E", flush=True)

# Trim to dimensions divisible by 4 (required for encoder/decoder pooling)
n_lat = (ds_daily.sizes['lat'] // 4) * 4
n_lon = (ds_daily.sizes['lon'] // 4) * 4
ds_daily = ds_daily.isel(lat=slice(0, n_lat), lon=slice(0, n_lon))
print(f"Grid trimmed to: {n_lat} × {n_lon}", flush=True)

dataset_cnn = ds_daily

feature_name = ['cape', 'precipitation', 'rh', 'shear', 't2m', 'wcd', 'vertical_velocity_500hPa']
output_name  = ['ltg']

# ── Temporal split: 2024-2025 = test, rest = train+val ────────────────────────
all_times = ds_daily.time.values
test_mask = all_times >= np.datetime64('2024-01-01')
idx_test  = np.where(test_mask)[0]
idx_tv    = np.where(~test_mask)[0]

# ── Season filter: Oct-Nov-Dec only ───────────────────────────────────────────
ACTIVE_MONTHS = [10, 11, 12]
all_months = pd.DatetimeIndex(all_times).month
idx_test = idx_test[np.isin(all_months[idx_test], ACTIVE_MONTHS)]
idx_tv   = idx_tv[np.isin(all_months[idx_tv],   ACTIVE_MONTHS)]
print(f"Season filter {ACTIVE_MONTHS}: {len(idx_tv)} train+val, {len(idx_test)} test timesteps", flush=True)

idx_train, idx_val = train_test_split(idx_tv, test_size=0.25, random_state=13)

# ── Derive dims dynamically ────────────────────────────────────────────────────
zdim = len(feature_name)
ydim = n_lat
xdim = n_lon

seed = 13
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

print(f"Splits — train: {len(idx_train)}, val: {len(idx_val)}, test: {len(idx_test)}", flush=True)
print(f"Model dims: zdim={zdim}, ydim={ydim}, xdim={xdim}", flush=True)

# ── Dataset slices ─────────────────────────────────────────────────────────────
train_dataset_cnn_X = dataset_cnn[feature_name].isel(time=idx_train)
val_dataset_cnn_X   = dataset_cnn[feature_name].isel(time=idx_val)
test_dataset_cnn_X  = dataset_cnn[feature_name].isel(time=idx_test)
train_dataset_cnn_y = dataset_cnn[output_name].isel(time=idx_train)
val_dataset_cnn_y   = dataset_cnn[output_name].isel(time=idx_val)
test_dataset_cnn_y  = dataset_cnn[output_name].isel(time=idx_test)

# ── Model ──────────────────────────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self, zdim, ydim, xdim):
        super(SimpleCNN, self).__init__()
        self.encoder0 = self.encoder_block(zdim, 32)
        self.encoder1 = self.encoder_block(32, 16)
        self.center = self.conv_block(16, 8)
        self.decoder1 = self.decoder_block(8, 16)
        self.decoder0 = self.decoder_block(16, 32)
        self.outputs = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
#            nn.ReLU()  # Ensures non-negative output
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            self.conv_block(in_channels, out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        encoder0 = self.encoder0(x)
        encoder1 = self.encoder1(encoder0)
        center = self.center(encoder1)
        decoder1 = self.decoder1(center)
        decoder0 = self.decoder0(decoder1)
        outputs = self.outputs(decoder0)
        return outputs

model = SimpleCNN(zdim, ydim, xdim).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.005)

# ── Normalisation (Jones: per-month z-score on features only) ─────────────────
print("Computing normalisation stats...", flush=True)
train_X_mean = train_dataset_cnn_X.groupby('time.month').mean(dim=['time', 'lat', 'lon'])
train_X_std  = train_dataset_cnn_X.groupby('time.month').std(dim=['time', 'lat', 'lon'])

def zscore(ds, mean, std):
    return ((ds.groupby('time.month') - mean).groupby('time.month') / std).fillna(0)

print("Normalising splits...", flush=True)
train_dataset_cnn_X_norm = zscore(train_dataset_cnn_X, train_X_mean, train_X_std)
val_dataset_cnn_X_norm   = zscore(val_dataset_cnn_X,   train_X_mean, train_X_std)
test_dataset_cnn_X_norm  = zscore(test_dataset_cnn_X,  train_X_mean, train_X_std)

# ── BCE loss with pos_weight (sqrt of neg/pos ratio balances precision/recall) ─
y_train_flat  = (train_dataset_cnn_y['ltg'].values > 0).ravel()
n_pos = float(y_train_flat.sum())
n_neg = float((~y_train_flat.astype(bool)).sum())
pos_weight_val = float(np.sqrt(n_neg / n_pos))
print(f"pos_weight (sqrt): {pos_weight_val:.2f}  (neg/pos ratio: {n_neg/n_pos:.1f})", flush=True)
# criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=DEVICE))
criterion = nn.BCEWithLogitsLoss()

# ── Convert to tensors ─────────────────────────────────────────────────────────
print("Converting to tensors...", flush=True)
def to_tensor(ds_norm):
    """Convert normalised xarray dataset to CPU tensor — batches moved to GPU inside training loop."""
    arr = np.moveaxis(np.array(ds_norm.to_array()), 0, -1)
    arr = np.nan_to_num(arr, nan=0.0)   # replace any residual NaNs with 0
    return torch.tensor(arr, dtype=torch.float32).permute(0, 3, 1, 2)  # stays on CPU

def to_binary_tensor(ds_y):
    """Binarise lightning target: >0 → 1, else 0. Shape (T, 1, H, W)."""
    arr = (ds_y['ltg'].values > 0).astype('float32')
    return torch.tensor(arr[:, np.newaxis, :, :])  # stays on CPU

# Load train + val into RAM; test loaded after training to save GPU memory
tr_X_tensor  = to_tensor(train_dataset_cnn_X_norm)
val_X_tensor = to_tensor(val_dataset_cnn_X_norm)
tr_y_tensor  = to_binary_tensor(train_dataset_cnn_y)
val_y_tensor = to_binary_tensor(val_dataset_cnn_y)

# ── Training ───────────────────────────────────────────────────────────────────
def train(model, criterion, optimizer, train_inputs, train_targets, val_inputs, val_targets,
          batch_size=128, epochs=50, patience=5):
    dataset_size = len(train_inputs)
    val_size     = len(val_inputs)
    best_val_loss    = float('inf')
    patience_counter = 0
    ckpt_path = '/home/ec2-user/ML_Thesis_with_DL/data/unet_best.pt'

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(dataset_size)
        for i in range(0, dataset_size, batch_size):
            indices = permutation[i:i + batch_size]
            batch_inputs  = train_inputs[indices].to(DEVICE)
            batch_targets = train_targets[indices].to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(batch_inputs), batch_targets)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss      = 0.0
        n_val_batches = math.ceil(val_size / batch_size)
        with torch.no_grad():
            for i in range(0, val_size, batch_size):
                val_out  = model(val_inputs[i:i + batch_size].to(DEVICE))
                val_loss += criterion(val_out, val_targets[i:i + batch_size].to(DEVICE)).item()
        val_loss /= max(1, n_val_batches)

        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {loss.item():.6f}, Validation Loss: {val_loss:.6f}', flush=True)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            # print(f"  ✓ best model saved (val={val_loss:.6f})", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}", flush=True)
                break

train(model, criterion, optimizer, tr_X_tensor, tr_y_tensor, val_X_tensor, val_y_tensor)

# ── Inference ──────────────────────────────────────────────────────────────────
print("Loading best checkpoint for inference...", flush=True)
model.load_state_dict(torch.load('/home/ec2-user/ML_Thesis_with_DL/data/unet_best.pt', map_location=DEVICE))
model.eval()

# Load test data now (after training, so GPU memory is free)
print("Loading test tensors...", flush=True)
test_X_tensor = to_tensor(test_dataset_cnn_X_norm)

with torch.no_grad():
    pred_parts = []
    for i in range(0, len(test_X_tensor), 128):
        logits = model(test_X_tensor[i:i+128].to(DEVICE))
        pred_parts.append(torch.sigmoid(logits).cpu())
    prob_tensor = torch.cat(pred_parts, dim=0)

prob_np        = prob_tensor.numpy().squeeze(1)          # (T, H, W)  probabilities 0–1
pred_binary_np = (prob_np >= 0.5).astype('float32')

# ── Save predictions ───────────────────────────────────────────────────────────
out_path = '/home/ec2-user/ML_Thesis_with_DL/data/unet_binary_OND_predictions.nc'
dims   = ['time', 'lat', 'lon']
coords = {
    'time': test_dataset_cnn_y['time'].values,
    'lat':  test_dataset_cnn_y['lat'].values,
    'lon':  test_dataset_cnn_y['lon'].values,
}

ds_out = xr.Dataset({
    'prob': xr.DataArray(prob_np,        dims=dims, coords=coords),
    'pred': xr.DataArray(pred_binary_np, dims=dims, coords=coords),
})
ds_out.to_netcdf(out_path)
print(f"Saved → {out_path}", flush=True)

# ── Metrics ────────────────────────────────────────────────────────────────────
from scipy.ndimage import uniform_filter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

obs_count_np  = test_dataset_cnn_y['ltg'].values.astype('float32')
obs_binary_np = (obs_count_np > 0).astype('float32')
T_test        = len(obs_count_np)

obs_flat  = obs_binary_np.ravel().astype(int)
pred_flat = pred_binary_np.ravel().astype(int)
acc  = accuracy_score(obs_flat, pred_flat)
prec = precision_score(obs_flat, pred_flat, zero_division=0)
rec  = recall_score(obs_flat, pred_flat, zero_division=0)
f1   = f1_score(obs_flat, pred_flat, zero_division=0)

def compute_fss(obs, pred, threshold, window_size):
    obs_bin   = (obs  >= threshold).astype(int)
    pred_bin  = (pred >= threshold).astype(int)
    obs_frac  = uniform_filter(obs_bin.astype(float),  size=window_size, mode='constant', cval=0)
    pred_frac = uniform_filter(pred_bin.astype(float), size=window_size, mode='constant', cval=0)
    num = np.sum((pred_frac - obs_frac) ** 2)
    den = np.sum(pred_frac**2 + obs_frac**2)
    return np.nan if den == 0 else 1 - num / den

THRESHOLD, WINDOW = 0.5, 3
fss_scores = [compute_fss(obs_binary_np[i], pred_binary_np[i], THRESHOLD, WINDOW) for i in range(T_test)]
fss_arr    = np.array(fss_scores)
fss_valid  = fss_arr[np.isfinite(fss_arr)]

print(f"\n{'='*52}")
print(f"  TEST SET METRICS  (2024-2025, {T_test} timesteps)")
print(f"{'='*52}")
print(f"  Accuracy            : {acc:.4f}")
print(f"  Precision           : {prec:.4f}")
print(f"  Recall              : {rec:.4f}")
print(f"  F1 score            : {f1:.4f}")
print(f"  Mean FSS (3×3)      : {np.nanmean(fss_valid):.4f}")
print(f"  Median FSS          : {np.nanmedian(fss_valid):.4f}")
print(f"{'='*52}", flush=True)
