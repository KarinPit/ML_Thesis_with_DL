import os
import numpy as np
import xarray as xr
import gcsfs
from datetime import datetime
import pandas as pd

LAT_MIN, LAT_MAX = 27.296, 36.598
LON_MIN, LON_MAX = 27.954, 39.292

PRESSURE_LEVELS = [500, 1000]  # hPa — Jones et al. use 500 & 1000 hPa for wind/RH

# single-level variables (2D: lat × lon, no pressure dimension)
SINGLE_LEVEL_VARS = [
    # ── Original thesis variables ──────────────────────────────────────────────
    # 'convective_available_potential_energy',
    # 'k_index',
    # 'total_totals_index',
    # 'total_column_cloud_ice_water',
    # 'total_column_cloud_liquid_water',
    # 'surface_pressure',

    # ── Jones et al. (2025) CPLRSTW single-level variables ────────────────────
    '2m_temperature',        # T2M
    'cloud_base_height',     # CBH  — used to derive WCD = ZDL - CBH
    'zero_degree_level',     # ZDL  — used to derive WCD = ZDL - CBH
    'land_sea_mask',         # LSM
]

# pressure-level variables (3D: pressure × lat × lon)
PRESSURE_LEVEL_VARS = [
    # ── Original thesis variables ──────────────────────────────────────────────
    # 'temperature',
    # 'specific_humidity',
    # 'vertical_velocity',
    # 'geopotential',
    # 'specific_cloud_ice_water_content',
    # 'specific_cloud_liquid_water_content',

    # ── Jones et al. (2025) CPLRSTW pressure-level variables ──────────────────
    'u_component_of_wind',   # u at 500 & 1000 hPa — used to derive SHEAR
    'v_component_of_wind',   # v at 500 & 1000 hPa — used to derive SHEAR
    'specific_humidity',     # q at 500 & 1000 hPa — used to derive RH
    'temperature',           # T at 500 & 1000 hPa — used to derive RH
    # Note: relative_humidity not in ARCO-ERA5; derived from q and T instead
]


def download_era5(time_range, out_dir='data'):
    """Download Jones et al. ERA5 variables month by month, derive WCD/SHEAR/RH, save."""
    fs = gcsfs.GCSFileSystem(token='anon')
    store = fs.get_mapper('gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3')
    ds = xr.open_zarr(store, consolidated=True)

    spatial = dict(latitude=slice(LAT_MAX, LAT_MIN), longitude=slice(LON_MIN, LON_MAX))

    start_year = datetime.fromisoformat(time_range.start).year
    end_year   = datetime.fromisoformat(time_range.stop).year
    ts = f"{start_year}_{end_year}" if end_year != start_year else str(start_year)

    single_path   = f'{out_dir}/jones_single_level_{ts}.nc'
    pressure_path = f'{out_dir}/jones_pressure_level_{ts}.nc'

    months = pd.date_range(start=time_range.start, end=time_range.stop, freq='MS')

    single_monthly   = []
    pressure_monthly = []
    tmp_files        = []

    # Use seconds since 1970 to avoid xarray time-decoding overflow on merge
    time_enc = {'time': {'units': 'seconds since 1970-01-01', 'dtype': 'int64'}}

    for month_start in months:
        month_end   = month_start + pd.offsets.MonthEnd(1)
        month_end   = month_end.replace(hour=23)
        month_slice = slice(month_start.isoformat(), month_end.isoformat())
        month_str   = month_start.strftime('%Y_%m')

        tmp_single   = f'{out_dir}/tmp_jones_single_{month_str}.nc'
        tmp_pressure = f'{out_dir}/tmp_jones_pressure_{month_str}.nc'

        if os.path.exists(tmp_single) and os.path.exists(tmp_pressure):
            print(f"  {month_str} already downloaded, skipping.")
        else:
            print(f"  Downloading {month_str}...")
            ds[SINGLE_LEVEL_VARS].sel(time=month_slice, **spatial).to_netcdf(
                tmp_single, format='NETCDF4', encoding=time_enc)
            (ds[PRESSURE_LEVEL_VARS]
                .sel(time=month_slice, **spatial)
                .sel(level=PRESSURE_LEVELS)
             ).to_netcdf(tmp_pressure, format='NETCDF4', encoding=time_enc)

        single_monthly.append(tmp_single)
        pressure_monthly.append(tmp_pressure)
        tmp_files.extend([tmp_single, tmp_pressure])

    # ── Merge months ──────────────────────────────────────────────────────────
    print(f"\nMerging {len(months)} months into yearly files...")
    sl = xr.open_mfdataset(single_monthly,   combine='by_coords', chunks={'time': 100})
    pl = xr.open_mfdataset(pressure_monthly, combine='by_coords', chunks={'time': 100})

    # ── Derive WCD, SHEAR, RH_avg ─────────────────────────────────────────────
    print("Computing derived variables (WCD, SHEAR, RH_avg)...")

    # WCD = Zero Degree Level - Cloud Base Height  [m]
    sl['wcd'] = sl['zero_degree_level'] - sl['cloud_base_height']
    sl['wcd'].attrs = {'long_name': 'Warm Cloud Depth (ZDL - CBH)', 'units': 'm'}

    # SHEAR = sqrt((u500 - u1000)^2 + (v500 - v1000)^2)  [m/s]
    pl['wind_shear'] = np.sqrt(
        (pl['u_component_of_wind'].sel(level=500) - pl['u_component_of_wind'].sel(level=1000))**2 +
        (pl['v_component_of_wind'].sel(level=500) - pl['v_component_of_wind'].sel(level=1000))**2
    )
    pl['wind_shear'].attrs = {'long_name': 'Deep-layer wind shear (500-1000 hPa)', 'units': 'm/s'}

    # RH from q and T (ARCO doesn't store RH directly)
    # e  = q * p / (0.622 + 0.378 * q)   actual vapour pressure [hPa]
    # es = 6.112 * exp(17.67 * (T-273.15) / (T-29.65))  saturation VP [hPa]
    def compute_rh(q, T, p_hpa):
        e  = q * p_hpa / (0.622 + 0.378 * q)
        es = 6.112 * np.exp(17.67 * (T - 273.15) / (T - 29.65))
        return (e / es * 100).clip(0, 100)

    rh500  = compute_rh(pl['specific_humidity'].sel(level=500),  pl['temperature'].sel(level=500),  500)
    rh1000 = compute_rh(pl['specific_humidity'].sel(level=1000), pl['temperature'].sel(level=1000), 1000)
    pl['rh_avg'] = (rh500 + rh1000) / 2.0
    pl['rh_avg'].attrs = {'long_name': 'Mean relative humidity (500 & 1000 hPa)', 'units': '%'}

    # ── Save ──────────────────────────────────────────────────────────────────
    sl.to_netcdf(single_path,   format='NETCDF4')
    pl.to_netcdf(pressure_path, format='NETCDF4')

    # ── Cleanup ───────────────────────────────────────────────────────────────
    print("Cleaning up temporary monthly files...")
    for f in tmp_files:
        os.remove(f)

    print(f"Done! Saved:\n  {single_path}\n  {pressure_path}")
    print("Reminder: download IMERG precipitation separately (download_imerg.py).")
    return single_path, pressure_path


if __name__ == "__main__":
    TIME_RANGES = [
        # slice('2004-09-01', '2004-12-31'),
        # slice('2005-01-01', '2005-11-30'),
        slice('2006-01-01', '2006-08-31'),
        slice('2008-09-01', '2008-12-31'),
        slice('2009-01-01', '2009-09-30'),
        slice('2023-01-01', '2023-12-31'),
        slice('2024-01-01', '2024-12-31'),
        slice('2025-01-01', '2025-12-31'),
    ]

    for time_range in TIME_RANGES:
        download_era5(time_range=time_range)
