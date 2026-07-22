import xarray as xr
import gcsfs
import numpy as np

LAT_MIN, LAT_MAX = 27.296, 36.598
LON_MIN, LON_MAX = 27.954, 39.292

# connect and access the ERA5 ARCO database 
fs = gcsfs.GCSFileSystem(token='anon')
store = fs.get_mapper('gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3')
ds = xr.open_zarr(store, consolidated=True)

# define time range of data
times = slice('2025-11-25T00:00', '2025-11-25T03:00')  # 4 timestamps: 00,01,02,03

# single-level variables (2D: lat × lon, no pressure dimension)
single_level_vars = [
    'convective_available_potential_energy',   # Convective Available Potential Energy
    'k_index',     # K-index
    'total_totals_index',  # Total Totals Index
    'total_column_cloud_ice_water',   # Total column cloud ice water
    'total_column_cloud_liquid_water',   # Total column cloud liquid water
    'surface_pressure',     # Surface pressure
]

# pressure-level variables (3D: pressure × lat × lon)
pressure_level_vars = [
    'temperature',   # Temperature
    # 'r',   # Relative humidity
    'specific_humidity',
    'vertical_velocity',   # Vertical velocity / omega
]

# load data to memory
print("Downloading single-level variables...")
ds_single = ds[single_level_vars].sel(
    time=times,
    latitude=slice(LAT_MAX, LAT_MIN), 
    longitude=slice(LON_MIN, LON_MAX),
).compute()

print("Downloading pressure-level variables (all 37 levels)...")
ds_pressure = ds[pressure_level_vars].sel(
    time=times,
    latitude=slice(LAT_MAX, LAT_MIN),
    longitude=slice(LON_MIN, LON_MAX),
).compute()

print("\nDone!")

# # ── 6. Access individual arrays ────────────────────────────────────────────
# Shape: (time=4, lat, lon)
cape_array = ds_single['k_index'].values
print(f"\ncape shape: {cape_array.shape}")   # (4, lat, lon)

# Shape: (time=4, level=37, lat, lon)
temp_array = ds_pressure['temperature'].values
print(f"temperature shape: {temp_array.shape}")   # (4, 37, lat, lon)


# save to NetCDF files
ds_single.to_netcdf('era5_single_level_sample.nc')
ds_pressure.to_netcdf('era5_pressure_level_sample.nc')
