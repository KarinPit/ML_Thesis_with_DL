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
zdim_m, ydim_m, xdim_m = 7, 104, 148  # Adjust these based on your data
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SimpleCNN(zdim_m, ydim_m, xdim_m).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# ── Helper: group global time indices by year (for sequential disk reads) ──────
def group_by_year(indices, all_times):
    """Returns sorted dict: year_str -> sorted array of global indices."""
    from collections import defaultdict
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
y_sum    = np.zeros((n_months, n_y_vars), dtype=np.float64)
y_sum_sq = np.zeros((n_months, n_y_vars), dtype=np.float64)
y_count  = np.zeros((n_months, n_y_vars), dtype=np.float64)

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
            v = sub_X[vname].values[mask].astype(np.float64)   # (T_m, lat, lon)
            v_fin = v[np.isfinite(v)]
            X_sum[m_idx-1, vi]    += v_fin.sum()
            X_sum_sq[m_idx-1, vi] += (v_fin**2).sum()
            X_count[m_idx-1, vi]  += len(v_fin)
        for vi, vname in enumerate(output_name):
            v = sub_y[vname].values[mask].astype(np.float64)
            v_fin = v[np.isfinite(v)]
            y_sum[m_idx-1, vi]    += v_fin.sum()
            y_sum_sq[m_idx-1, vi] += (v_fin**2).sum()
            y_count[m_idx-1, vi]  += len(v_fin)

    del sub_X, sub_y; gc.collect()

# Final mean/std per (month, variable): shape (12, n_vars)
X_mean = X_sum    / X_count
X_std  = np.sqrt(np.maximum(X_sum_sq / X_count - X_mean**2, 1e-12))
y_mean = y_sum    / y_count
y_std  = np.sqrt(np.maximum(y_sum_sq / y_count - y_mean**2, 1e-12))

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
        y_arr = np.stack([sub_y[v].values for v in output_name], axis=-1).astype(np.float32)
        del sub_X, sub_y

        # Apply per-month normalisation
        for t in range(T):
            m = months_arr[t] - 1  # 0-indexed
            X_arr[t] = ((X_arr[t].astype(np.float64) - X_mean[m]) / X_std[m]).astype(np.float32)
            y_arr[t] = ((y_arr[t].astype(np.float64) - y_mean[m]) / y_std[m]).astype(np.float32)

        np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        np.nan_to_num(y_arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

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
    pred_parts = []
    for i in range(0, len(test_X_tensor), 128):
        pred_parts.append(model(test_X_tensor[i:i+128].to(DEVICE)).cpu())
    predictions_norm = torch.cat(pred_parts, dim=0)

# Convert predictions to numpy array
predictions_norm_np = predictions_norm.numpy()
predictions_norm_np = predictions_norm_np.squeeze(1)

# Unnormalise predictions using train stats
# Match Jones: re-standardise predictions then apply train month std/mean
# Jones: re-normalize pred to z-score, then multiply by train_y std, add train_y mean
pred_mean = predictions_norm_np.mean()
pred_std  = predictions_norm_np.std()
pred_z    = (predictions_norm_np - pred_mean) / (pred_std + 1e-12)  # (T, lat, lon)

T_test = len(test_times_sorted)
predictions_unnorm_np = np.zeros_like(pred_z, dtype=np.float32)
for t in range(T_test):
    m = int(pd.Timestamp(test_times_sorted[t]).month) - 1  # 0-indexed
    # y_std and y_mean are shape (12, 1); pick month m, variable 0
    predictions_unnorm_np[t] = pred_z[t] * y_std[m, 0] + y_mean[m, 0]

predictions_unnorm_np = np.maximum(predictions_unnorm_np, 0)

# ── Save predictions ────────────────────────────────────────────────────────────
dims   = ['time', 'lat', 'lon']
coords = {
    'time': test_times_sorted,
    'lat':  ds_daily.lat.values[:n_lat],
    'lon':  ds_daily.lon.values[:n_lon],
}
predictions_unnorm_da = xr.DataArray(predictions_unnorm_np, dims=dims, coords=coords)
predictions_unnorm_ds = predictions_unnorm_da.to_dataset(name='ltg')

print("Saving predictions...", flush=True)
predictions_unnorm_ds.to_netcdf('/home/ec2-user/ML_Thesis_with_DL/data/jones_original_predictions.nc')
print("Done! Saved to data/jones_original_predictions.nc", flush=True)
