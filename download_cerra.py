"""
download_cerra.py
-----------------
Download CERRA reanalysis pressure-level data via CDS API.

Downloads one month at a time to stay within CDS size limits, then merges
into yearly NetCDF files (same pattern as download_era5_arco.py).

CERRA differences from ERA5:
  - 5.5 km resolution, Lambert conformal conic projection
  - 3-hourly temporal resolution (analysis product)
  - No geographic subsetting — full European domain downloaded, clipped later
  - No CAPE, K-index, Total Totals Index

Usage:
    python download_cerra.py
"""

import os
import cdsapi
import xarray as xr

# ── Variables ─────────────────────────────────────────────────────────────────
PRESSURE_VARS = [
    'specific_cloud_ice_water_content',
    'specific_cloud_liquid_water_content',
    'specific_rain_water_content',
    'specific_snow_water_content',
    'temperature',
    'u_component_of_wind',
    'v_component_of_wind',
    'relative_humidity',
    'turbulent_kinetic_energy',
    'geopotential',
    'cloud_cover',
]

# Only the 3 levels in the mixed-phase / charge-separation zone (500–700 hPa).
# Cloud ice at 500–600 hPa was the dominant feature in ERA5 experiments
# (>53% XGBoost gain). Adding more levels can be done later if needed.
PRESSURE_LEVELS = ['500', '600', '700']

# 3-hourly analysis times
TIMES = ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00']

ALL_DAYS = [f'{d:02d}' for d in range(1, 32)]


def _months_for_year(year):
    """Months to download per year — mirrors ERA5 LPATS partial-year ranges."""
    ranges = {
        2004: ['09', '10', '11', '12'],
        2005: ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11'],
        2006: ['01', '02', '03', '04', '05', '06', '07', '08'],
        2008: ['09', '10', '11', '12'],
        2009: ['01', '02', '03', '04', '05', '06', '07', '08', '09'],
        2023: [f'{m:02d}' for m in range(1, 13)],
        2024: [f'{m:02d}' for m in range(1, 13)],
        2025: [f'{m:02d}' for m in range(1, 13)],
    }
    return ranges.get(year, [f'{m:02d}' for m in range(1, 13)])


def download_cerra_year(year, out_dir='data'):
    """Download CERRA pressure-level data for one year, month by month."""
    c = cdsapi.Client()
    os.makedirs(out_dir, exist_ok=True)

    final_path = f'{out_dir}/cerra_pressure_level_{year}.nc'
    if os.path.exists(final_path):
        print(f"  {final_path} already exists, skipping.")
        return final_path

    months = _months_for_year(year)
    tmp_files = []

    for month in months:
        tmp_path = f'{out_dir}/tmp_cerra_{year}_{month}.nc'
        tmp_files.append(tmp_path)

        if os.path.exists(tmp_path):
            print(f"  {year}-{month} already downloaded, skipping.")
            continue

        print(f"  Downloading {year}-{month}...")
        c.retrieve(
            'reanalysis-cerra-pressure-levels',
            {
                'variable':       PRESSURE_VARS,
                'pressure_level': PRESSURE_LEVELS,
                'data_type':      ['reanalysis'],
                'product_type':   ['analysis'],
                'year':           str(year),
                'month':          month,
                'day':            ALL_DAYS,
                'time':           TIMES,
                'data_format':    'netcdf',
            },
            tmp_path,
        )
        print(f"    Saved: {tmp_path}")

    # merge monthly files into one yearly file
    print(f"\nMerging {len(tmp_files)} months into {final_path}...")
    xr.open_mfdataset(tmp_files, combine='by_coords',
                      chunks={'time': 50}).to_netcdf(final_path)

    # clean up monthly temp files
    print("Cleaning up monthly temp files...")
    for f in tmp_files:
        if os.path.exists(f):
            os.remove(f)

    print(f"Done: {final_path}")
    return final_path


if __name__ == '__main__':
    LPATS_YEARS = [2004, 2005, 2006, 2008, 2009]
    ILDN_YEARS  = [2023, 2024, 2025]

    for year in LPATS_YEARS + ILDN_YEARS:
        print(f"\n{'='*60}\nYear {year}\n{'='*60}")
        download_cerra_year(year)

    print('\nAll downloads complete.')
