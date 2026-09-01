"""
download_imerg.py
-----------------
Downloads NASA GPM IMERG Final Precipitation L3 Half-Hourly 0.1° V07 data
for the Israel/Eastern Mediterranean domain, aggregates to hourly resolution,
regrids to the ERA5 0.25° grid, and saves as NetCDF.

This is the precipitation source used by Jones et al. (2025) — NOT ERA5 precipitation.

Prerequisites:
  1. Create a free NASA Earthdata account: https://urs.earthdata.nasa.gov
  2. Install earthaccess: pip install earthaccess h5py scipy --break-system-packages

First-time setup:
  Run `earthaccess.login(strategy="interactive")` once to save credentials locally.
  After that, subsequent runs authenticate automatically.

Output:
  data/imerg_hourly_{ts}.nc  — hourly precipitation (mm/hr) on 0.25° ERA5 grid

Usage:
    python download_imerg.py
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
import h5py
from scipy.interpolate import RegularGridInterpolator
import earthaccess

# ── Domain (match ERA5 grid exactly) ──────────────────────────────────────────
LAT_MIN, LAT_MAX = 27.296, 36.598
LON_MIN, LON_MAX = 27.954, 39.292

# ERA5 0.25° target grid for this domain
ERA5_LATS = np.arange(np.ceil(LAT_MIN * 4) / 4, LAT_MAX + 0.001, 0.25)
ERA5_LONS = np.arange(np.ceil(LON_MIN * 4) / 4, LON_MAX + 0.001, 0.25)

# ── IMERG product ──────────────────────────────────────────────────────────────
IMERG_SHORT_NAME = 'GPM_3IMERGHH'
IMERG_VERSION    = '07'
PRECIP_VAR       = 'precipitationCal'   # calibrated precipitation, mm/hr


def read_imerg_granule(filepath):
    """Read one IMERG HDF5 half-hourly file → (time, lat, lon, precip_mm_hr)."""
    with h5py.File(filepath, 'r') as f:
        # IMERG HDF5 structure: /Grid/precipitationCal shape (1, lon, lat)
        precip = f[f'Grid/{PRECIP_VAR}'][0]   # shape: (lon, lat)
        lons   = f['Grid/lon'][:]             # 0.1° grid, -180 to 180
        lats   = f['Grid/lat'][:]             # 0.1° grid, -90 to 90

        # Read time (minutes since 1970-01-01)
        t_ms   = f['Grid/time'][0]
        time   = pd.Timestamp('1970-01-01') + pd.Timedelta(minutes=int(t_ms))

    # precip shape is (lon, lat) — transpose to (lat, lon)
    precip = precip.T   # now (lat, lon)
    precip = np.where(precip < 0, 0.0, precip)   # mask fill values (-9999)

    return time, lats, lons, precip


def crop_and_regrid(lats_src, lons_src, precip, target_lats, target_lons):
    """Crop to domain and regrid from 0.1° IMERG grid to 0.25° ERA5 grid."""
    # Crop source to slightly wider than domain (for interpolation boundary)
    lat_mask = (lats_src >= LAT_MIN - 0.2) & (lats_src <= LAT_MAX + 0.2)
    lon_mask = (lons_src >= LON_MIN - 0.2) & (lons_src <= LON_MAX + 0.2)

    lats_c  = lats_src[lat_mask]
    lons_c  = lons_src[lon_mask]
    precip_c = precip[np.ix_(lat_mask, lon_mask)]

    # Bilinear interpolation onto ERA5 0.25° grid
    interp = RegularGridInterpolator(
        (lats_c, lons_c), precip_c, method='linear', bounds_error=False, fill_value=0.0
    )
    grid_lats, grid_lons = np.meshgrid(target_lats, target_lons, indexing='ij')
    regridded = interp(np.stack([grid_lats.ravel(), grid_lons.ravel()], axis=-1))
    return regridded.reshape(len(target_lats), len(target_lons))


def download_imerg(time_range, out_dir='data', tmp_dir='data/tmp_imerg'):
    """
    Download IMERG half-hourly granules, aggregate to hourly, regrid, save NetCDF.
    """
    os.makedirs(tmp_dir, exist_ok=True)

    start_year = pd.Timestamp(time_range.start).year
    end_year   = pd.Timestamp(time_range.stop).year
    ts = f"{start_year}_{end_year}" if end_year != start_year else str(start_year)
    out_path = f'{out_dir}/imerg_hourly_{ts}.nc'

    if os.path.exists(out_path):
        print(f"Output already exists: {out_path}")
        return out_path

    # ── Authenticate ──────────────────────────────────────────────────────────
    print("Authenticating with NASA Earthdata...")
    earthaccess.login(strategy="netrc")   # uses ~/.netrc after first login

    # ── Search for granules ───────────────────────────────────────────────────
    print(f"Searching IMERG granules: {time_range.start} → {time_range.stop}")
    results = earthaccess.search_data(
        short_name=IMERG_SHORT_NAME,
        version=IMERG_VERSION,
        temporal=(time_range.start, time_range.stop),
        bounding_box=(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX),
    )
    print(f"Found {len(results)} granules (2 per hour × ~{len(results)//2} hours)")

    # ── Download granules ─────────────────────────────────────────────────────
    print(f"Downloading to {tmp_dir}/ ...")
    files = earthaccess.download(results, tmp_dir)
    files = sorted(files)
    print(f"Downloaded {len(files)} files")

    # ── Read, crop, regrid all granules ──────────────────────────────────────
    print("Reading and regridding granules...")
    records = []   # list of (time, precip_array)
    for fpath in files:
        try:
            t, lats_src, lons_src, precip = read_imerg_granule(fpath)
            p_regrid = crop_and_regrid(lats_src, lons_src, precip, ERA5_LATS, ERA5_LONS)
            records.append((t, p_regrid))
        except Exception as e:
            print(f"  WARNING: could not read {fpath}: {e}")

    # ── Aggregate two 30-min granules → one hourly value ─────────────────────
    print("Aggregating to hourly...")
    df_times = pd.Series([r[0] for r in records])
    hour_keys = df_times.dt.floor('h')
    unique_hours = hour_keys.unique()

    hourly_precip = []
    hourly_times  = []
    for hr in sorted(unique_hours):
        idx = hour_keys[hour_keys == hr].index.tolist()
        stacked = np.stack([records[i][1] for i in idx], axis=0)
        hourly_precip.append(stacked.mean(axis=0))   # mean of 30-min rates → hourly rate
        hourly_times.append(hr)

    precip_arr = np.stack(hourly_precip, axis=0)   # (time, lat, lon)

    # ── Build xarray Dataset and save ────────────────────────────────────────
    ds_out = xr.Dataset(
        {'precipitation': (['time', 'latitude', 'longitude'], precip_arr,
                           {'long_name': 'IMERG calibrated precipitation rate',
                            'units': 'mm/hr'})},
        coords={
            'time':      hourly_times,
            'latitude':  ERA5_LATS,
            'longitude': ERA5_LONS,
        }
    )

    print(f"Saving to {out_path}...")
    ds_out.to_netcdf(out_path, format='NETCDF4')

    # ── Clean up raw HDF5 files ───────────────────────────────────────────────
    print("Cleaning up raw granule files...")
    for f in files:
        if os.path.exists(f):
            os.remove(f)

    print(f"\nDone! {len(hourly_times)} hourly timesteps saved to {out_path}")
    return out_path


if __name__ == "__main__":
    # First time only — run this interactively to save credentials:
    #   import earthaccess; earthaccess.login(strategy="interactive")

    # Uncomment the year you want:
    # time_range = slice('2004-09-01', '2004-12-31')
    # time_range = slice('2005-01-01', '2005-11-30')
    # time_range = slice('2006-01-01', '2006-08-31')
    # time_range = slice('2008-09-01', '2008-12-31')
    # time_range = slice('2009-01-01', '2009-09-30')
    # time_range = slice('2023-01-01', '2023-12-31')
    # time_range = slice('2024-01-01', '2024-12-31')
    time_range = slice('2025-01-01', '2025-12-31')

    download_imerg(time_range=time_range)
