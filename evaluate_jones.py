"""
evaluate_jones.py
-----------------
Computes FSS (Fractions Skill Score) exactly as Jones et al. (2026) did,
using his compute_fss function from the sample figures notebook.

Usage:
    cd /home/ec2-user/ML_Thesis_with_DL
    python evaluate_jones.py
"""

import xarray as xr
import numpy as np
import pandas as pd
import os
from scipy.ndimage import uniform_filter

os.chdir('/home/ec2-user/ML_Thesis_with_DL/data')

# ── Jones' exact FSS function ───────────────────────────────────────────────────
def compute_fss(obs, pred, threshold, window_size):
    obs_bin  = (obs  >= threshold).astype(int)
    pred_bin = (pred >= threshold).astype(int)
    obs_frac  = uniform_filter(obs_bin.astype(float),  size=window_size, mode='constant', cval=0)
    pred_frac = uniform_filter(pred_bin.astype(float), size=window_size, mode='constant', cval=0)
    numerator   = np.sum((pred_frac - obs_frac) ** 2)
    denominator = np.sum(pred_frac**2 + obs_frac**2)
    if denominator == 0:
        return np.nan
    return 1 - (numerator / denominator)

# ── Load predictions ────────────────────────────────────────────────────────────
print("Loading predictions...", flush=True)
ds_pred = xr.open_dataset('jones_original_predictions.nc')
pred_da = ds_pred['ltg']           # (T, lat, lon) — unnormalised stroke counts
times   = ds_pred['time'].values

# ── Load actual observations for the same times ─────────────────────────────────
print("Loading observations...", flush=True)
ds_obs = xr.open_dataset('jones_israel_all_years.nc')
n_lat  = (ds_obs.sizes['lat'] // 4) * 4
n_lon  = (ds_obs.sizes['lon'] // 4) * 4
ds_obs = ds_obs.isel(lat=slice(0, n_lat), lon=slice(0, n_lon))
obs_da = ds_obs['ltg'].sel(time=times)

# ── Compute FSS per timestep (Jones: threshold=1e-5, window=3) ──────────────────
# Jones' data is in strokes/km²/yr; ours is raw counts per grid cell per hour.
# We use threshold=0.5 (≡ "at least 1 stroke") which is equivalent for integer counts.
# Also test threshold=1e-5 if your data has been density-normalised.
THRESHOLD   = 0.5    # at least 1 stroke in the grid cell
WINDOW_SIZE = 3      # 3×3 neighbourhood (Jones' default)

print(f"Computing FSS (threshold={THRESHOLD}, window={WINDOW_SIZE})...", flush=True)
fss_scores = []
for i in range(len(times)):
    obs_t  = obs_da.isel(time=i).values
    pred_t = pred_da.isel(time=i).values
    fss    = compute_fss(obs_t, pred_t, THRESHOLD, WINDOW_SIZE)
    fss_scores.append(fss)
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(times)} done...", flush=True)

fss_arr = np.array(fss_scores)
fss_arr = fss_arr[np.isfinite(fss_arr)]   # drop timesteps where denominator=0 (no lightning)

# ── Also compute R² on climatology (Jones' Figure 4 style) ──────────────────────
obs_clim  = obs_da.mean(dim='time').values.ravel()
pred_clim = pred_da.mean(dim='time').values.ravel()
mask = np.isfinite(obs_clim) & np.isfinite(pred_clim)
from sklearn.metrics import r2_score
r2_clim = r2_score(obs_clim[mask], pred_clim[mask])

# ── Print results ───────────────────────────────────────────────────────────────
print(f"\n{'='*45}")
print(f"  Test period: {pd.Timestamp(times[0]).date()} → {pd.Timestamp(times[-1]).date()}")
print(f"  Timesteps evaluated: {len(fss_arr)} (non-trivial)")
print(f"{'='*45}")
print(f"  Mean FSS (3×3, thr={THRESHOLD}) : {np.nanmean(fss_arr):.4f}")
print(f"  Median FSS                      : {np.nanmedian(fss_arr):.4f}")
print(f"  R² on mean climatology          : {r2_clim:.4f}")
print(f"{'='*45}", flush=True)

# ── Save FSS time series for plotting ───────────────────────────────────────────
fss_da = xr.DataArray(
    np.array(fss_scores),
    dims=['time'],
    coords={'time': times},
    name='fss'
)
fss_da.to_dataset().to_netcdf('jones_fss_timeseries.nc')
print("FSS time series saved → data/jones_fss_timeseries.nc", flush=True)
