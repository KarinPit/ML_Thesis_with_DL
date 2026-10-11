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


os.chdir('/home/ec2-user/ML_Thesis_with_DL/data')

# ── Domain bounds ──────────────────────────────────────────────────────────────
LAT_MIN, LAT_MAX = 30.970, 43.435
LON_MIN, LON_MAX = -9.142, 39.292

ds_hourly = xr.open_dataset('jones_israel_all_years.nc')

# ── Spatial subset to domain bounds ───────────────────────────────────────────
lats = ds_hourly.lat.values
if lats[0] > lats[-1]:  # descending lat (ERA5 native)
    ds_hourly = ds_hourly.sel(lat=slice(LAT_MAX, LAT_MIN), lon=slice(LON_MIN, LON_MAX))
else:                    # ascending lat
    ds_hourly = ds_hourly.sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))
print(f"Domain: lat [{LAT_MIN}, {LAT_MAX}], lon [{LON_MIN}, {LON_MAX}]", flush=True)

# Trim to dimensions divisible by 4
n_lat = (ds_hourly.sizes['lat'] // 4) * 4
n_lon = (ds_hourly.sizes['lon'] // 4) * 4

ds_hourly = ds_hourly.isel(lat=slice(0, n_lat), lon=slice(0, n_lon))
print(f"Grid trimmed to: {n_lat} × {n_lon}", flush=True)

# ── Resample to 12-hourly ───────────────────────────────────────────────────────
print("Resampling to 12-hourly...", flush=True)
feature_name = ['cape', 'precipitation', 'lsm', 'rh', 'shear', 't2m', 'wcd', 'vertical_velocity_500hPa']
output_name  = ['ltg']

ds_feat = ds_hourly[feature_name].resample(time='12h').mean()
ds_ltg  = ds_hourly[output_name].resample(time='12h').sum()
ds_daily = xr.merge([ds_feat, ds_ltg])
print(f"After resampling: {ds_daily.sizes['time']} 12-hourly timesteps", flush=True)

dataset_cnn = ds_daily

# ── Print available variables so you can verify feature list ──────────────────
print(f"\nVariables in dataset: {list(ds_daily.data_vars)}", flush=True)

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

# ── Jones architecture: plain encoder-decoder (no skip connections) ────────────
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
zdim_m = len(feature_name)
ydim_m = n_lat
xdim_m = n_lon
print(f"Model dims: zdim={zdim_m}, ydim={ydim_m}, xdim={xdim_m}", flush=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SimpleCNN(zdim_m, ydim_m, xdim_m).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# ── Helper: group global time indices by year ──────────────────────────────────
def group_by_year(indices, all_times):
    year_map = defaultdict(list)
    for i in indices:
        yr = str(all_times[i].astype('datetime64[Y]'))
        year_map[yr].append(i)
    return {yr: np.sort(np.array(idxs)) for yr, idxs in sorted(year_map.items())}

all_times     = ds_daily.time.values
train_by_year = group_by_year(idx_train, all_times)
val_by_year   = group_by_year(idx_val,   all_times)
test_by_year  = group_by_year(idx_test,  all_times)

# ── Compute train normalisation stats (Jones: per-month mean/std over time×lat×lon) ──
n_months = 12
n_X_vars = len(feature_name)

X_sum    = np.zeros((n_months, n_X_vars), dtype=np.float64)
X_sum_sq = np.zeros((n_months, n_X_vars), dtype=np.float64)
X_count  = np.zeros((n_months, n_X_vars), dtype=np.float64)
y_sum    = np.zeros(n_months, dtype=np.float64)
y_sum_sq = np.zeros(n_months, dtype=np.float64)
y_count  = np.zeros(n_months, dtype=np.float64)

print("Computing train stats (year-by-year)...", flush=True)
for yr, yr_indices in train_by_year.items():
    print(f"  Stats {yr}: {len(yr_indices)} timesteps", flush=True)
    sub_X = dataset_cnn[feature_name].isel(time=yr_indices).load()
    sub_y = dataset_cnn[output_name].isel(time=yr_indices).load()
    months_arr = sub_X.time.dt.month.values

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

        yv = sub_y['ltg'].values[mask].astype(np.float64)
        yv_fin = yv[np.isfinite(yv)]
        y_sum[m_idx-1]    += yv_fin.sum()
        y_sum_sq[m_idx-1] += (yv_fin**2).sum()
        y_count[m_idx-1]  += len(yv_fin)

    del sub_X, sub_y; gc.collect()

X_mean = X_sum    / X_count
X_std  = np.sqrt(np.maximum(X_sum_sq / X_count - X_mean**2, 1e-12))
y_mean = y_sum    / y_count
y_std  = np.sqrt(np.maximum(y_sum_sq / y_count - y_mean**2, 1e-12))
print("Train stats done.", flush=True)
for m in [10, 11, 12]:
    print(f"  ltg month {m}: mean={y_mean[m-1]:.4f}  std={y_std[m-1]:.4f}", flush=True)

# ── Load, normalise, and convert a split to tensors (Jones z-score on both X and y) ──
def norm_to_tensor(by_year_dict, label):
    print(f"\nNormalising + converting {label}...", flush=True)
    X_parts, y_parts = [], []
    for yr, yr_indices in by_year_dict.items():
        print(f"  {yr}: {len(yr_indices)} timesteps", flush=True)
        sub_X = dataset_cnn[feature_name].isel(time=yr_indices).load()
        sub_y = dataset_cnn[output_name].isel(time=yr_indices).load()
        months_arr = sub_X.time.dt.month.values
        T = len(months_arr)

        X_arr = np.stack([sub_X[v].values for v in feature_name], axis=-1).astype(np.float32)
        # Binary target: 1 if any lightning, 0 otherwise
        y_arr = (sub_y['ltg'].values > 0).astype(np.float32)[:, :, :, np.newaxis]
        del sub_X, sub_y

        # Z-score normalise features per month; target is already binary
        for t in range(T):
            m = months_arr[t] - 1
            X_arr[t] = ((X_arr[t].astype(np.float64) - X_mean[m]) / X_std[m]).astype(np.float32)

        np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

        X_parts.append(torch.tensor(X_arr.transpose(0, 3, 1, 2)))
        y_parts.append(torch.tensor(y_arr.transpose(0, 3, 1, 2)))
        del X_arr, y_arr; gc.collect()

    return torch.cat(X_parts, dim=0), torch.cat(y_parts, dim=0)

tr_X_tensor,   tr_y_tensor   = norm_to_tensor(train_by_year, 'train')
val_X_tensor,  val_y_tensor  = norm_to_tensor(val_by_year,   'val')
test_X_tensor, test_y_tensor = norm_to_tensor(test_by_year,  'test')

test_times_sorted = np.concatenate(
    [all_times[idxs] for yr, idxs in test_by_year.items()]
)

print("\nTensors ready. Starting training...", flush=True)

def train(model, criterion, optimizer, train_inputs, train_targets, val_inputs, val_targets, batch_size=128, epochs=50, patience=5):
    dataset_size = len(train_inputs)
    val_size = len(val_inputs)

    best_val_loss = float('inf')
    patience_counter = 0
    ckpt_path = '/home/ec2-user/ML_Thesis_with_DL/data/unet_best.pt'

    for epoch in range(epochs):
        # ── Training phase ──────────────────────────────────────────────────────
        model.train()
        permutation = torch.randperm(dataset_size)
        for i in range(0, dataset_size, batch_size):
            indices = permutation[i:i + batch_size]
            batch_inputs, batch_targets = train_inputs[indices], train_targets[indices]

            optimizer.zero_grad()
            outputs = model(batch_inputs.to(DEVICE))
            loss = criterion(outputs, batch_targets.to(DEVICE))
            loss.backward()
            optimizer.step()

        # ── Validation phase ────────────────────────────────────────────────────
        model.eval()
        val_loss = 0
        n_val_batches = math.ceil(val_size / batch_size)
        with torch.no_grad():
            for i in range(0, val_size, batch_size):
                val_batch_inputs  = val_inputs[i:i + batch_size]
                val_batch_targets = val_targets[i:i + batch_size]
                val_outputs = model(val_batch_inputs.to(DEVICE))
                val_loss += criterion(val_outputs, val_batch_targets.to(DEVICE)).item()

        val_loss /= max(1, n_val_batches)
        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {loss.item():.6f}, Validation Loss: {val_loss:.6f}', flush=True)

        # ── Early stopping + checkpoint ─────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ best model saved (val={val_loss:.6f})", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)", flush=True)
                break

train(model, criterion, optimizer, tr_X_tensor, tr_y_tensor, val_X_tensor, val_y_tensor)

print("Training done. Loading best checkpoint for inference...", flush=True)
model.load_state_dict(torch.load('/home/ec2-user/ML_Thesis_with_DL/data/unet_best.pt', map_location=DEVICE))
model.eval()

with torch.no_grad():
    logit_parts = []
    for i in range(0, len(test_X_tensor), 128):
        logit_parts.append(model(test_X_tensor[i:i+128].to(DEVICE)).cpu())
    logits = torch.cat(logit_parts, dim=0)   # (T, 1, lat, lon), raw logits

# ── Binary inference: sigmoid → probability, threshold at 0.5 ─────────────────
prob_np        = torch.sigmoid(logits).numpy().squeeze(1)   # (T, lat, lon), 0–1
pred_count_np  = prob_np                                     # saved as "prob" in output
pred_binary_np = (prob_np >= 0.5).astype(np.float32)        # binary prediction

T_test = len(test_times_sorted)

# ── Save predictions ────────────────────────────────────────────────────────────
dims   = ['time', 'lat', 'lon']
coords = {
    'time': test_times_sorted,
    'lat':  ds_daily.lat.values[:n_lat],
    'lon':  ds_daily.lon.values[:n_lon],
}
ds_out = xr.Dataset({
    'prob': xr.DataArray(pred_count_np,  dims=dims, coords=coords),  # predicted stroke count
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

test_obs_parts = []
for yr, yr_indices in test_by_year.items():
    sub_y = dataset_cnn[output_name].isel(time=yr_indices).load()
    test_obs_parts.append(sub_y['ltg'].values.astype(np.float32))
    del sub_y; gc.collect()
obs_count_np  = np.concatenate(test_obs_parts, axis=0)
obs_binary_np = (obs_count_np > 0).astype(np.float32)

obs_flat  = obs_binary_np.ravel().astype(int)
pred_flat = pred_binary_np.ravel().astype(int)
acc  = accuracy_score(obs_flat, pred_flat)
prec = precision_score(obs_flat, pred_flat, zero_division=0)
rec  = recall_score(obs_flat, pred_flat, zero_division=0)
f1   = f1_score(obs_flat, pred_flat, zero_division=0)

def compute_fss(obs, pred, threshold, window_size):
    obs_bin  = (obs  >= threshold).astype(int)
    pred_bin = (pred >= threshold).astype(int)
    obs_frac  = uniform_filter(obs_bin.astype(float),  size=window_size, mode='constant', cval=0)
    pred_frac = uniform_filter(pred_bin.astype(float), size=window_size, mode='constant', cval=0)
    num = np.sum((pred_frac - obs_frac) ** 2)
    den = np.sum(pred_frac**2 + obs_frac**2)
    return np.nan if den == 0 else 1 - num / den

THRESHOLD, WINDOW = 1.0, 3
fss_scores = [
    compute_fss(obs_count_np[i], pred_count_np[i], THRESHOLD, WINDOW)
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
