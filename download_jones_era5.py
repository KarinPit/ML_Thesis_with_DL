"""
download_jones_era5.py
----------------------
Downloads ERA5 variables used by Jones et al. (2025) for their CPLRSTW lightning U-Net.

Variables downloaded from ARCO-ERA5:
  Single-level : 2m_temperature, cloud_base_height, zero_degree_level, land_sea_mask
  Pressure-level: u/v wind + relative_humidity at 500 hPa & 1000 hPa

Derived variables (computed after download and saved into the output file):
  WCD   = zero_degree_level - cloud_base_height          (warm cloud depth)
  SHEAR = sqrt((u500 - u1000)^2 + (v500 - v1000)^2)    (deep-layer wind shear)
  RH    = mean(RH_500, RH_1000)                          (average relative humidity)

NOTE: Precipitation in Jones et al. comes from NASA IMERG, NOT ERA5.
      Download IMERG separately (see download_imerg.py — TODO).

Output files (per year/period):
  data/jones_single_level_{ts}.nc    — T2M, CBH, ZDL, LSM, WCD
  data/jones_pressure_level_{ts}.nc  — u, v, RH at 500 & 1000 hPa, SHEAR, RH_avg

Usage:
    python download_jones_era5.py
"""

import os
import numpy as np
import xarray as xr
import gcsfs
from datetime import datetime
import pandas as pd

LAT_MIN, LAT_MAX = 27.296, 36.598
LON_MIN, LON_MAX = 27.954, 39.292

PRESSURE_LEVELS = [500, 1000]   # hPa

SINGLE_LEVEL_VARS = [
    '2m_temperature',        # T2M
    'cloud_base_height',     # CBH  — needed to compute WCD
    'zero_degree_level',     # ZDL  — needed to compute WCD
    'land_sea_mask',         # LSM  (static, but stored per-timestep in ARCO)
]

PRESSURE_LEVEL_VARS = [
    'u_component_of_wind',   # u at 500 & 1000 hPa — needed for SHEAR
    'v_component_of_wind',   # v at 500 & 1000 hPa — needed for SHEAR
    'relative_humidity',     # RH at 500 & 1000 hPa
]


def download_jones_era5(time_range, out_dir='data'):
    """Download Jones et al. ERA5 variables, derive WCD/SHEAR/RH, save to NetCDF."""
    fs = gcsfs.GCSFileSystem(token='anon')
    store = fs.get_mapper('gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3')
    ds = xr.open_zarr(store, consolidated=True)

    spatial = dict(latitude=slice(LAT_MAX, LAT_MIN), longitude=slice(LON_MIN, LON_MAX))

    start_year = datetime.fromisoformat(time_range.start).year
    end_year   = datetime.fromisoformat(time_range.stop).year
    ts = f"{start_year}_{end_year}" if end_year != start_year else str(start_year)

    single_out   = f'{out_dir}/jones_single_level_{ts}.nc'
    pressure_out = f'{out_dir}/jones_pressure_level_{ts}.nc'

    months = pd.date_range(start=time_range.start, end=time_range.stop, freq='MS')

    single_monthly   = []
    pressure_monthly = []
    tmp_files        = []

    for month_start in months:
        month_end   = (month_start + pd.offsets.MonthEnd(1)).replace(hour=23)
        month_slice = slice(month_start.isoformat(), month_end.isoformat())
        month_str   = month_start.strftime('%Y_%m')

        tmp_single   = f'{out_dir}/tmp_jones_single_{month_str}.nc'
        tmp_pressure = f'{out_dir}/tmp_jones_pressure_{month_str}.nc'

        if os.path.exists(tmp_single) and os.path.exists(tmp_pressure):
            print(f"  {month_str} already downloaded, skipping.")
        else:
            print(f"  Downloading {month_str}...")

            # ── Single-level ──────────────────────────────────────────────────
            sl = ds[SINGLE_LEVEL_VARS].sel(time=month_slice, **spatial)
            sl.to_netcdf(tmp_single, format='NETCDF4')

            # ── Pressure-level (500 & 1000 hPa only) ─────────────────────────
            pl = (ds[PRESSURE_LEVEL_VARS]
                  .sel(time=month_slice, **spatial)
                  .sel(level=PRESSURE_LEVELS))
            pl.to_netcdf(tmp_pressure, format='NETCDF4')

        single_monthly.append(tmp_single)
        pressure_monthly.append(tmp_pressure)
        tmp_files.extend([tmp_single, tmp_pressure])

    # ── Merge monthly files ───────────────────────────────────────────────────
    print(f"\nMerging {len(months)} months...")
    sl_merged = xr.open_mfdataset(single_monthly,   combine='by_coords', chunks={'time': 100})
    pl_merged = xr.open_mfdataset(pressure_monthly, combine='by_coords', chunks={'time': 100})

    # ── Derive WCD, SHEAR, RH_avg ─────────────────────────────────────────────
    print("Computing derived variables (WCD, SHEAR, RH_avg)...")

    # WCD = Zero Degree Level - Cloud Base Height  [metres]
    sl_merged['wcd'] = sl_merged['zero_degree_level'] - sl_merged['cloud_base_height']
    sl_merged['wcd'].attrs = {'long_name': 'Warm Cloud Depth', 'units': 'm'}

    # Wind shear = sqrt((u500 - u1000)^2 + (v500 - v1000)^2)  [m/s]
    u500  = pl_merged['u_component_of_wind'].sel(level=500)
    u1000 = pl_merged['u_component_of_wind'].sel(level=1000)
    v500  = pl_merged['v_component_of_wind'].sel(level=500)
    v1000 = pl_merged['v_component_of_wind'].sel(level=1000)
    pl_merged['wind_shear'] = np.sqrt((u500 - u1000)**2 + (v500 - v1000)**2)
    pl_merged['wind_shear'].attrs = {'long_name': 'Deep-layer wind shear (500-1000 hPa)', 'units': 'm/s'}

    # RH average of 500 and 1000 hPa
    rh500  = pl_merged['relative_humidity'].sel(level=500)
    rh1000 = pl_merged['relative_humidity'].sel(level=1000)
    pl_merged['rh_avg'] = (rh500 + rh1000) / 2.0
    pl_merged['rh_avg'].attrs = {'long_name': 'Mean relative humidity (500 & 1000 hPa)', 'units': '%'}

    # ── Save ──────────────────────────────────────────────────────────────────
    print("Saving...")
    sl_merged.to_netcdf(single_out,   format='NETCDF4')
    pl_merged.to_netcdf(pressure_out, format='NETCDF4')

    # ── Clean up temp files ───────────────────────────────────────────────────
    print("Cleaning up temporary files...")
    for f in tmp_files:
        if os.path.exists(f):
            os.remove(f)

    print(f"\nDone! Saved:\n  {single_out}\n  {pressure_out}")
    print("\nReminder: download IMERG precipitation separately for the full CPLRSTW feature set.")
    return single_out, pressure_out


if __name__ == "__main__":
    # Uncomment the year you want to download:
    time_range = slice('2004-09-01T00:00', '2004-12-31T23:00')
    # time_range = slice('2005-01-01T00:00', '2005-11-30T23:00')
    # time_range = slice('2006-01-01T00:00', '2006-08-31T23:00')
    # time_range = slice('2008-09-01T00:00', '2008-12-31T23:00')
    # time_range = slice('2009-01-01T00:00', '2009-09-30T23:00')
    # time_range = slice('2023-01-01T00:00', '2023-12-31T23:00')
    # time_range = slice('2024-01-01T00:00', '2024-12-31T23:00')
    # time_range = slice('2025-01-01T00:00', '2025-12-31T23:00')

    download_jones_era5(time_range=time_range)
