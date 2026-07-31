import os
import xarray as xr
import gcsfs
from datetime import datetime
import pandas as pd

LAT_MIN, LAT_MAX = 27.296, 36.598
LON_MIN, LON_MAX = 27.954, 39.292

# single-level variables (2D: lat × lon, no pressure dimension)
SINGLE_LEVEL_VARS = [
    'convective_available_potential_energy',
    'k_index',
    'total_totals_index',
    'total_column_cloud_ice_water',
    'total_column_cloud_liquid_water',
    'surface_pressure',
]

# pressure-level variables (3D: pressure × lat × lon)
PRESSURE_LEVEL_VARS = [
    'temperature',
    'specific_humidity',
    'vertical_velocity',
    'geopotential',
    'specific_cloud_ice_water_content',
    'specific_cloud_liquid_water_content',
]


def download_era5(time_range, out_dir='data'):
    """Download ERA5 data month by month and merge into yearly NetCDF files.
    Downloads each month separately for resilience — if one month fails,
    only that month needs to be re-downloaded.
    """
    fs = gcsfs.GCSFileSystem(token='anon')
    store = fs.get_mapper('gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3')
    ds = xr.open_zarr(store, consolidated=True)

    spatial = dict(latitude=slice(LAT_MAX, LAT_MIN), longitude=slice(LON_MIN, LON_MAX))

    start_year = datetime.fromisoformat(time_range.start).year
    end_year   = datetime.fromisoformat(time_range.stop).year
    ts = f"{start_year}_{end_year}" if end_year != start_year else str(start_year)

    single_path   = f'{out_dir}/era5_single_level_{ts}.nc'
    pressure_path = f'{out_dir}/era5_pressure_level_{ts}.nc'

    # generate list of months in the time range
    months = pd.date_range(
        start=time_range.start,
        end=time_range.stop,
        freq='MS'  # month start
    )

    single_monthly   = []
    pressure_monthly = []
    tmp_files        = []

    for month_start in months:
        month_end = month_start + pd.offsets.MonthEnd(1)
        month_end = month_end.replace(hour=23)
        month_slice = slice(month_start.isoformat(), month_end.isoformat())
        month_str   = month_start.strftime('%Y_%m')

        tmp_single   = f'{out_dir}/tmp_single_{month_str}.nc'
        tmp_pressure = f'{out_dir}/tmp_pressure_{month_str}.nc'

        # skip if already downloaded (resume support)
        if os.path.exists(tmp_single) and os.path.exists(tmp_pressure):
            print(f"  {month_str} already downloaded, skipping.")
        else:
            print(f"  Downloading {month_str}...")
            ds[SINGLE_LEVEL_VARS].sel(time=month_slice, **spatial).to_netcdf(tmp_single,   format='NETCDF4')
            ds[PRESSURE_LEVEL_VARS].sel(time=month_slice, **spatial).to_netcdf(tmp_pressure, format='NETCDF4')

        single_monthly.append(tmp_single)
        pressure_monthly.append(tmp_pressure)
        tmp_files.extend([tmp_single, tmp_pressure])

    # merge all months into final yearly files
    print(f"\nMerging {len(months)} months into yearly files...")
    xr.open_mfdataset(single_monthly,   combine='by_coords').to_netcdf(single_path,   format='NETCDF4')
    xr.open_mfdataset(pressure_monthly, combine='by_coords').to_netcdf(pressure_path, format='NETCDF4')

    # clean up monthly temp files
    print("Cleaning up temporary monthly files...")
    for f in tmp_files:
        os.remove(f)

    print(f"Done! Saved:\n  {single_path}\n  {pressure_path}")
    return single_path, pressure_path


if __name__ == "__main__":
    # ── Modern ILDN years (full year) ─────────────────────────────────────────
    # time_range = slice('2023-01-01T00:00', '2023-12-31T23:00')
    # time_range = slice('2024-01-01T00:00', '2024-12-31T23:00')
    # time_range = slice('2025-01-01T00:00', '2025-12-31T23:00')

    # ── LPATS years (partial — only months with lightning data) ───────────────
    # Uncomment the year you want to download:
    # time_range = slice('2004-09-01T00:00', '2004-12-31T23:00')  # 2004: Sep-Dec
    # time_range = slice('2005-01-01T00:00', '2005-11-30T23:00')  # 2005: Jan-Nov
    # time_range = slice('2006-01-01T00:00', '2006-08-31T23:00')  # 2006: Jan-Aug
    # time_range = slice('2008-09-01T00:00', '2008-12-31T23:00')  # 2008: Sep-Dec
    # time_range = slice('2009-01-01T00:00', '2009-09-30T23:00')  # 2009: Jan-Sep

    time_range = slice('2025-01-01T00:00', '2025-12-31T23:00')
    download_era5(time_range=time_range)
