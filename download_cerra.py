"""
download_cerra.py
-----------------
Download CERRA reanalysis pressure-level data via CDS API.

CERRA differences from ERA5:
  - 5.5 km resolution, Lambert conformal conic projection
  - 3-hourly temporal resolution (analysis product)
  - 29 pressure levels (vs ERA5's 37)
  - NetCDF4 format
  - No CAPE, K-index, Total Totals Index (pressure levels only)
  - Has: rain/snow water content, turbulent kinetic energy, cloud cover

Note: CERRA has no useful single-level stability indices (no CAPE etc.).
All features come from pressure levels. Cloud ice at 500-700 hPa was the
dominant ERA5 feature anyway (>53% gain in XGBoost), so this is fine.

Usage:
    python download_cerra.py
"""

import os
import cdsapi

# ── Domain (same as ERA5) ─────────────────────────────────────────────────────
# CERRA area format: [North, West, South, East]
AREA = [37.0, 27.0, 27.0, 40.0]

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

# All 29 CERRA pressure levels (hPa)
PRESSURE_LEVELS = [
    '1000', '975', '950', '925', '900', '875', '850', '825', '800',
    '750', '700', '600', '500', '400', '300', '250', '200', '150',
    '100', '70', '50', '30', '20', '10', '7', '5', '3', '2', '1',
]

# 3-hourly analysis times
TIMES = ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00']

ALL_DAYS = [f'{d:02d}' for d in range(1, 32)]
ALL_MONTHS = [f'{m:02d}' for m in range(1, 13)]


def _months_for_year(year):
    """Return (months, days) to download for each LPATS/ILDN year.
    Mirrors the partial-year ranges used for ERA5 downloads."""
    ranges = {
        2004: (['09', '10', '11', '12'], ALL_DAYS),
        2005: (['01', '02', '03', '04', '05', '06',
                '07', '08', '09', '10', '11'], ALL_DAYS),
        2006: (['01', '02', '03', '04', '05', '06', '07', '08'], ALL_DAYS),
        2008: (['09', '10', '11', '12'], ALL_DAYS),
        2009: (['01', '02', '03', '04', '05', '06', '07', '08', '09'], ALL_DAYS),
        # ILDN years — full year
        2023: (ALL_MONTHS, ALL_DAYS),
        2024: (ALL_MONTHS, ALL_DAYS),
        2025: (ALL_MONTHS, ALL_DAYS),
    }
    return ranges.get(year, (ALL_MONTHS, ALL_DAYS))


def download_cerra_year(year, out_dir='data'):
    """Download CERRA pressure-level data for one year."""
    c = cdsapi.Client()

    months, days = _months_for_year(year)
    pressure_path = f'{out_dir}/cerra_pressure_level_{year}.nc'

    if os.path.exists(pressure_path):
        print(f"  {pressure_path} already exists, skipping.")
    else:
        print(f"\nDownloading CERRA pressure levels for {year}...")
        c.retrieve(
            'reanalysis-cerra-pressure-levels',
            {
                'variable':       PRESSURE_VARS,
                'pressure_level': PRESSURE_LEVELS,
                'product_type':   'reanalysis',
                'year':           str(year),
                'month':          months,
                'day':            days,
                'time':           TIMES,
                'data_format':    'netcdf',
                'area':           AREA,
            },
            pressure_path,
        )
        print(f"  Saved: {pressure_path}")

    return pressure_path


if __name__ == '__main__':
    LPATS_YEARS = [2004, 2005, 2006, 2008, 2009]
    ILDN_YEARS  = [2023, 2024, 2025]

    for year in LPATS_YEARS + ILDN_YEARS:
        print(f"\n{'='*60}\nYear {year}\n{'='*60}")
        download_cerra_year(year)

    print('\nAll downloads complete.')
