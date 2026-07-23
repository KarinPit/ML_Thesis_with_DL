import xarray as xr
import gcsfs
import numpy as np
from datetime import datetime

LAT_MIN, LAT_MAX = 27.296, 36.598
LON_MIN, LON_MAX = 27.954, 39.292

TIME_RANGE = slice('2025-01-01T00:00', '2025-12-31T23:00')

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
]

def download_era5(time_range=TIME_RANGE, out_dir='data'):
    # connect to the ERA5 ARCO database
    fs = gcsfs.GCSFileSystem(token='anon')
    store = fs.get_mapper('gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3')
    ds = xr.open_zarr(store, consolidated=True)

    spatial = dict(latitude=slice(LAT_MAX, LAT_MIN), longitude=slice(LON_MIN, LON_MAX))

    print("Downloading single-level variables...")
    ds_single = ds[SINGLE_LEVEL_VARS].sel(time=time_range, **spatial).compute()

    print("Downloading pressure-level variables (all 37 levels)...")
    ds_pressure = ds[PRESSURE_LEVEL_VARS].sel(time=time_range, **spatial).compute()

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    single_path   = f'{out_dir}/era5_single_level_{ts}.nc'
    pressure_path = f'{out_dir}/era5_pressure_level_{ts}.nc'
    ds_single.to_netcdf(single_path, format='NETCDF4')
    ds_pressure.to_netcdf(pressure_path, format='NETCDF4')
    print(f"Done! Saved:\n  {single_path}\n  {pressure_path}")

    return single_path, pressure_path

if __name__ == "__main__":
    download_era5()
