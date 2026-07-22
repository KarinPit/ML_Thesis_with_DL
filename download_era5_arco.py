import xarray as xr
import gcsfs
import numpy as np

LAT_MIN, LAT_MAX = 27.296, 36.598
LON_MIN, LON_MAX = 27.954, 39.292

TIME_RANGE = slice('2025-11-24T00:00', '2025-11-26T00:00')

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

def download_era5():
    # connect to the ERA5 ARCO database
    fs = gcsfs.GCSFileSystem(token='anon')
    store = fs.get_mapper('gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3')
    ds = xr.open_zarr(store, consolidated=True)

    spatial = dict(latitude=slice(LAT_MAX, LAT_MIN), longitude=slice(LON_MIN, LON_MAX))

    print("Downloading single-level variables...")
    ds_single = ds[SINGLE_LEVEL_VARS].sel(time=TIME_RANGE, **spatial).compute()

    print("Downloading pressure-level variables (all 37 levels)...")
    ds_pressure = ds[PRESSURE_LEVEL_VARS].sel(time=TIME_RANGE, **spatial).compute()

    ds_single.to_netcdf('data/era5_single_level_sample.nc')
    ds_pressure.to_netcdf('data/era5_pressure_level_sample.nc')
    print("Done! Saved to data/")

if __name__ == "__main__":
    download_era5()
