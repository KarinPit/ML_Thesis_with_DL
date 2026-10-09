import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.stats as stats
import netCDF4
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from collections import defaultdict

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


# os.chdir('/home/ec2-user/ML_Thesis_with_DL/Jones_Replicated/Lightning-CNN-main/data/input_data')
os.chdir('/home/ec2-user/ML_Thesis_with_DL/data')

ds_hourly = xr.open_dataset('jones_israel_all_years.nc')
# Trim to dimensions divisible by 4
n_lat = (ds_hourly.sizes['lat'] // 4) * 4
n_lon = (ds_hourly.sizes['lon'] // 4) * 4

ds_hourly = ds_hourly.isel(lat=slice(0, n_lat), lon=slice(0, n_lon))
print(f"Grid trimmed to: {n_lat} × {n_lon}", flush=True)

# ── Resample to 12-hourly ───────────────────────────────────────────────────────
# ERA5 features: average over each 12h window
# ltg (stroke counts): sum over each 12h window (counts accumulate)
# lsm is constant in time — mean is fine
print("Resampling to 12-hourly...", flush=True)
feature_name = ['cape', 'precipitation', 'lsm', 'rh', 'shear', 't2m', 'wcd']
output_name  = ['ltg']

ds_feat = ds_hourly[feature_name].resample(time='12h').mean()
ds_ltg  = ds_hourly[output_name].resample(time='12h').sum()
ds_daily = xr.merge([ds_feat, ds_ltg])
print(f"After resampling: {ds_daily.sizes['time']} 12-hourly timesteps", flush=True)

dataset_cnn = ds_daily

# ── Temporal split: 2024-2025 = test, 2013-2023 = train+val ────────────────────
all_times  = ds_daily.time.values
test_mask  = all_times >= np.datetime64('2024-01-01')
idx_test   = np.where(test_mask)[0]
idx_tv     = np.where(~test_mask)[0]

# ── Season filter: Oct-Nov-Dec only (Israel's active lightning season) ──────────
ACTIVE_MONTHS = [10, 11, 12]
all_months    = pd.DatetimeIndex(all_times).month
idx_test = idx_test[np.isin(all_months[idx_test], ACTIVE_MONTHS)]
idx_tv   = idx_tv[np.isin(all_months[idx_tv],   ACTIVE_MONTHS)]
print(f"Season filter ({ACTIVE_MONTHS}): {len(idx_tv)} train+val, {len(idx_test)} test timesteps", flush=True)

idx_train, idx_val = train_test_split(idx_tv, test_size=0.25, random_state=13)

zdim = len(feature_name)

seed = 13
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

print(f"Splits — train: {len(idx_train)}, val: {len(idx_val)}, test: {len(idx_test)}", flush=True)
print(f"Test period: {pd.Timestamp(all_times[idx_test[0]]).date()} → {pd.Timestamp(all_times[idx_test[-1]]).date()}", flush=True)

# Define the PyTorch model
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

# Initialize model, loss function, and optimizer
# zdim = number of input features; ydim/xdim = spatial grid (unused inside the model,
# but kept for clarity — SimpleCNN's conv layers work on any spatial size)
zdim_m = len(feature_name)   # 7
ydim_m = n_lat               # trimmed lat dimension
xdim_m = n_lon               # trimmed lon dimension
print(f"Model dims: zdim={zdim_m}, ydim={ydim_m}, xdim={xdim_m}", flush=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SimpleCNN(zdim_m, ydim_m, xdim_m).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.005)
# criterion is defined after the stats loop, once pos_weight is known

# ── Helper: group global time indices by year (for sequential disk reads) ──────
def group_by_year(indices, all_times):
    """Returns sorted dict: year_str -> sorted array of global indices."""
    year_map = defaultdict(list)
    for i in indices:
        yr = str(all_times[i].astype('datetime64[Y]'))
        year_map[yr].append(i)
    return {yr: np.sort(np.array(idxs)) for yr, idxs in sorted(year_map.items())}

all_times     = ds_daily.time.values
train_by_year = group_by_year(idx_train, all_times)
val_by_year   = group_by_year(idx_val,   all_times)
test_by_year  = group_by_year(idx_test,  all_times)

# ── Compute train normalisation stats incrementally (year-by-year) ─────────────
# Jones normalises per (month, variable) with mean/std over (time, lat, lon).
# We reproduce this exactly using running sums so we never load all train at once.
n_months = 12
n_X_vars = len(feature_name)
n_y_vars = len(output_name)

X_sum    = np.zeros((n_months, n_X_vars), dtype=np.float64)
X_sum_sq = np.zeros((n_months, n_X_vars), dtype=np.float64)
X_count  = np.zeros((n_months, n_X_vars), dtype=np.float64)
# For binary target: count positive (ltg > 0) and total pixels to compute pos_weight
y_pos_count = 0.0
y_tot_count = 0.0

print("Computing train stats (year-by-year to save RAM)...", flush=True)
for yr, yr_indices in train_by_year.items():
    print(f"  Stats {yr}: {len(yr_indices)} timesteps", flush=True)
    sub_X = dataset_cnn[feature_name].isel(time=yr_indices).load()
    sub_y = dataset_cnn[output_name].isel(time=yr_indices).load()
    months_arr = sub_X.time.dt.month.values  # shape (T,)

    for m_idx in range(1, 13):
        mask = (months_arr == m_idx)
        if not mask.any():
            continue
        for vi, vname in enumerate(feature_name):
            v = sub_X[vname].values[mask].astype(np.float64)
            v_fin = v[np.isfinite(v)]
            X_sum[m_idx-1, vi]    += v_fin.sum()
            X_sum_sq[m_idx-1, vi] += (v_fin**2).sum()
            X_count[m_idx-1, vi]  += len(v_fin)

    # Count positives for pos_weight (over all pixels in this year)
    ltg_vals = sub_y['ltg'].values
    y_pos_count += float((ltg_vals > 0).sum())
    y_tot_count += float(np.isfinite(ltg_vals).sum())

    del sub_X, sub_y; gc.collect()

# Final mean/std for features (shape 12×n_vars)
X_mean = X_sum    / X_count
X_std  = np.sqrt(np.maximum(X_sum_sq / X_count - X_mean**2, 1e-12))

# pos_weight = (# negative pixels) / (# positive pixels)
# Tells BCEWithLogitsLoss how much more to penalise missing a positive
pos_rate = y_pos_count / y_tot_count
print(f"Train stats done.  pos_rate={pos_rate:.4f}  ({y_pos_count:.0f} / {y_tot_count:.0f} pixels)", flush=True)

# Plain BCE loss — no pos_weight
criterion = nn.BCEWithLogitsLoss()

print("Train stats done.", flush=True)

# ── Load, normalise, and convert a split to tensors year-by-year ───────────────
def norm_to_tensor(by_year_dict, label):
    """Normalise and convert to float32 tensors, one year at a time."""
    print(f"\nNormalising + converting {label}...", flush=True)
    X_parts, y_parts = [], []
    for yr, yr_indices in by_year_dict.items():
        print(f"  {yr}: {len(yr_indices)} timesteps", flush=True)
        sub_X = dataset_cnn[feature_name].isel(time=yr_indices).load()
        sub_y = dataset_cnn[output_name].isel(time=yr_indices).load()
        months_arr = sub_X.time.dt.month.values  # (T,)
        T = len(months_arr)

        # Stack features: (T, lat, lon, n_X_vars)
        X_arr = np.stack([sub_X[v].values for v in feature_name], axis=-1).astype(np.float32)
        # Binary target: 1 if any lightning, 0 otherwise — shape (T, lat, lon, 1)
        y_arr = (sub_y['ltg'].values > 0).astype(np.float32)[:, :, :, np.newaxis]
        del sub_X, sub_y

        # Normalise features per month; y is already binary — no normalisation needed
        for t in range(T):
            m = months_arr[t] - 1  # 0-indexed
            X_arr[t] = ((X_arr[t].astype(np.float64) - X_mean[m]) / X_std[m]).astype(np.float32)

        np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

        # (T, lat, lon, C) → (T, C, lat, lon)
        X_parts.append(torch.tensor(X_arr.transpose(0, 3, 1, 2)))
        y_parts.append(torch.tensor(y_arr.transpose(0, 3, 1, 2)))
        del X_arr, y_arr; gc.collect()

    return torch.cat(X_parts, dim=0), torch.cat(y_parts, dim=0)

tr_X_tensor,   tr_y_tensor   = norm_to_tensor(train_by_year, 'train')
val_X_tensor,  val_y_tensor  = norm_to_tensor(val_by_year,   'val')
test_X_tensor, test_y_tensor = norm_to_tensor(test_by_year,  'test')

# Collect the sorted test times (used for saving predictions)
test_times_sorted = np.concatenate(
    [all_times[idxs] for yr, idxs in test_by_year.items()]
)

print("\nTensors ready. Starting training...", flush=True)

def train(model, criterion, optimizer, train_inputs, train_targets, val_inputs, val_targets, batch_size=128, epochs=50, patience=5):
    model.train()
    dataset_size = len(train_inputs)
    val_size = len(val_inputs)

    best_val_loss = float('inf')
    patience_counter = 5

    for epoch in range(epochs):
        permutation = torch.randperm(dataset_size)
        for i in range(0, dataset_size, batch_size):
            indices = permutation[i:i + batch_size]
            batch_inputs, batch_targets = train_inputs[indices], train_targets[indices]

            optimizer.zero_grad()
            outputs = model(batch_inputs.to(DEVICE))
            loss = criterion(outputs, batch_targets.to(DEVICE))
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i in range(0, val_size, batch_size):
                val_batch_inputs = val_inputs[i:i + batch_size]
                val_batch_targets = val_targets[i:i + batch_size]
                val_outputs = model(val_batch_inputs.to(DEVICE))
                val_loss += criterion(val_outputs, val_batch_targets.to(DEVICE)).item()

        val_loss /= max(1, val_size // batch_size)
        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {loss.item():.6f}, Validation Loss: {val_loss:.6f}', flush=True)

train(model, criterion, optimizer, tr_X_tensor, tr_y_tensor, val_X_tensor, val_y_tensor)

print("Training done. Running inference on test set...", flush=True)
model.eval()

with torch.no_grad():
    logit_parts = []
    for i in range(0, len(test_X_tensor), 128):
        logit_parts.append(model(test_X_tensor[i:i+128].to(DEVICE)).cpu())
    logits = torch.cat(logit_parts, dim=0)   # raw logits, shape (T, 1, lat, lon)

# Sigmoid → probabilities; threshold at 0.5 → binary predictions
prob_np   = torch.sigmoid(logits).numpy().squeeze(1)   # (T, lat, lon), values 0–1
pred_binary_np = (prob_np >= 0.5).astype(np.float32)   # (T, lat, lon), 0 or 1

T_test = len(test_times_sorted)

# ── Save predictions ────────────────────────────────────────────────────────────
dims   = ['time', 'lat', 'lon']
coords = {
    'time': test_times_sorted,
    'lat':  ds_daily.lat.values[:n_lat],
    'lon':  ds_daily.lon.values[:n_lon],
}
ds_out = xr.Dataset({
    'prob': xr.DataArray(prob_np,        dims=dims, coords=coords),  # predicted probability
    'pred': xr.DataArray(pred_binary_np, dims=dims, coords=coords),  # binary prediction (0/1)
})
out_path = '/home/ec2-user/ML_Thesis_with_DL/data/unet_binary_OND_predictions.nc'
print("Saving predictions...", flush=True)
ds_out.to_netcdf(out_path)
print(f"Done! Saved → {out_path}", flush=True)
print(f"(Oct-Nov-Dec season model)", flush=True)

# ── Evaluation metrics ──────────────────────────────────────────────────────────
print("\nComputing test metrics...", flush=True)
from scipy.ndimage import uniform_filter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# True binary labels for the test set
test_obs_parts = []
for yr, yr_indices in test_by_year.items():
    sub_y = dataset_cnn[output_name].isel(time=yr_indices).load()
    test_obs_parts.append((sub_y['ltg'].values > 0).astype(np.float32))
    del sub_y; gc.collect()
obs_binary_np = np.concatenate(test_obs_parts, axis=0)   # (T, lat, lon), 0 or 1

# Pixel-level classification metrics (flattened)
obs_flat  = obs_binary_np.ravel().astype(int)
pred_flat = pred_binary_np.ravel().astype(int)
acc  = accuracy_score(obs_flat, pred_flat)
prec = precision_score(obs_flat, pred_flat, zero_division=0)
rec  = recall_score(obs_flat, pred_flat, zero_division=0)
f1   = f1_score(obs_flat, pred_flat, zero_division=0)

# FSS (Jones metric): obs threshold=0.5 on binary obs, pred threshold=0.5 on probs
def compute_fss(obs, pred, threshold, window_size):
    obs_bin  = (obs  >= threshold).astype(int)
    pred_bin = (pred >= threshold).astype(int)
    obs_frac  = uniform_filter(obs_bin.astype(float),  size=window_size, mode='constant', cval=0)
    pred_frac = uniform_filter(pred_bin.astype(float), size=window_size, mode='constant', cval=0)
    num = np.sum((pred_frac - obs_frac) ** 2)
    den = np.sum(pred_frac**2 + obs_frac**2)
    return np.nan if den == 0 else 1 - num / den

THRESHOLD, WINDOW = 0.5, 3
fss_scores = [
    compute_fss(obs_binary_np[i], prob_np[i], THRESHOLD, WINDOW)
    for i in range(T_test)
]
fss_arr   = np.array(fss_scores)
fss_valid = fss_arr[np.isfinite(fss_arr)]

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
