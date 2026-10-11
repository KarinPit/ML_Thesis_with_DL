"""
check_data.py
-------------
Quick sanity check on jones_israel_all_years.nc

Usage (on EC2):
    cd /home/ec2-user/ML_Thesis_with_DL
    python check_data.py
"""

import xarray as xr
import numpy as np
import pandas as pd

ds = xr.open_dataset('data/jones_israel_all_years.nc')

print("=== DATASET INFO ===")
print(ds)
print(f"\nTime range: {pd.Timestamp(ds.time.values[0])} → {pd.Timestamp(ds.time.values[-1])}")
print(f"Time steps: {ds.sizes['time']}")
print(f"Grid: {ds.sizes['lat']} lat × {ds.sizes['lon']} lon")
print(f"Lat: {float(ds.lat.min()):.3f} → {float(ds.lat.max()):.3f}")
print(f"Lon: {float(ds.lon.min()):.3f} → {float(ds.lon.max()):.3f}")

print("\n=== NaN COUNTS PER VARIABLE ===")
for v in ds.data_vars:
    n = int(ds[v].isnull().sum())
    print(f"  {v:30s}: {n} NaNs")

print("\n=== LTG STATS ===")
ltg = ds['ltg'].values
print(f"  min={ltg.min():.2f}  max={ltg.max():.2f}  mean={ltg.mean():.4f}")
print(f"  % zero cells : {100*(ltg==0).mean():.1f}%")
print(f"  % >0 cells   : {100*(ltg>0).mean():.1f}%")
print(f"  % negative   : {100*(ltg<0).mean():.2f}%")

print("\n=== LTG BY MONTH (OND only) ===")
times = pd.DatetimeIndex(ds.time.values)
for m in [10, 11, 12]:
    mask = times.month == m
    v = ltg[mask]
    print(f"  Month {m}: {mask.sum()} steps, mean={v.mean():.4f}, max={v.max():.1f}, %pos={100*(v>0).mean():.1f}%")

print("\n=== FEATURE STATS (OND timesteps) ===")
ond_mask = times.month.isin([10, 11, 12])
for var in ['cape', 'precipitation', 'lsm', 'rh', 'shear', 't2m', 'wcd', 'vertical_velocity_500hPa']:
    if var in ds.data_vars:
        v = ds[var].values[ond_mask]
        nans = int(np.isnan(v).sum())
        pct  = 100 * nans / v.size
        print(f"  {var:30s}: min={np.nanmin(v):.3f}  max={np.nanmax(v):.3f}  mean={np.nanmean(v):.3f}  NaN={nans} ({pct:.1f}%)")

print("\n=== WCD NaN DETAIL ===")
v = ds['wcd'].values[ond_mask]
total = v.size
nans  = int(np.isnan(v).sum())
print(f"  wcd OND pixels  : {total:,}")
print(f"  NaN count       : {nans:,}  ({100*nans/total:.1f}%)")
print(f"  Non-NaN count   : {total-nans:,}  ({100*(total-nans)/total:.1f}%)")
if total - nans > 0:
    print(f"  Non-NaN min/max : {np.nanmin(v):.1f} / {np.nanmax(v):.1f}")
    print(f"  Non-NaN mean    : {np.nanmean(v):.1f}")
