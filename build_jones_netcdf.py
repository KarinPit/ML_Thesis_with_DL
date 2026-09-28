"""
build_jones_netcdf_direct.py
-----------------------------
Builds one jones_israel_{year}.nc per year DIRECTLY from downloaded NetCDF files,
skipping the parquet intermediate step entirely.

Input files (per year, all in data/):
  jones_single_level_{year}.nc    → CAPE, LSM, T2M, WCD (+ k_index, etc.)
  jones_pressure_level_{year}.nc  → RH avg, wind shear (pre-derived),
                                    vertical_velocity at levels 500+1000 hPa
  imerg_hourly_{year}.nc          → precipitation
  wwln_on_era5_grid_{year}.nc     → lightning_count (optional — fills 0 if absent)

Output (one file per year):
  data/jones_israel_{year}.nc
  Dims:  time × lat × lon
  Vars:  cape, precipitation, lsm, rh, shear, t2m, wcd,
         vertical_velocity_500hPa, ltg

To load all years lazily during training:
  ds = xr.open_mfdataset('data/jones_israel_*.nc', chunks={'time': 24})
"""

import os
import numpy as np
import xarray as xr

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR = 'data'

ALL_YEARS = list(range(2013, 2026))   # 2013–2025 inclusive


def build_year(year: int) -> xr.Dataset:
    """Read all source NetCDFs for one year and return a combined xr.Dataset."""

    # ── Paths ──────────────────────────────────────────────────────────────────
    jones_single_path   = os.path.join(DATA_DIR, f'jones_single_level_{year}.nc')
    jones_pressure_path = os.path.join(DATA_DIR, f'jones_pressure_level_{year}.nc')
    imerg_path          = os.path.join(DATA_DIR, f'imerg_hourly_{year}.nc')
    lightning_path      = os.path.join(DATA_DIR, f'wwln_on_era5_grid_{year}.nc')

    # ── Check required files (lightning is optional) ───────────────────────────
    required = [jones_single_path, jones_pressure_path, imerg_path]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        print(f"  SKIP {year} — missing required files:")
        for p in missing:
            print(f"    {p}")
        return None

    lightning_available = os.path.exists(lightning_path)
    if not lightning_available:
        print(f"  NOTE {year}: {os.path.basename(lightning_path)} not found — ltg will be 0")

    print(f"\n{'='*50}")
    print(f"  Processing year {year}")
    print(f"{'='*50}")

    # ── Load datasets ──────────────────────────────────────────────────────────
    ds_jones_s = xr.open_dataset(jones_single_path)
    ds_jones_p = xr.open_dataset(jones_pressure_path)
    ds_imerg   = xr.open_dataset(imerg_path)
    ds_light   = xr.open_dataset(lightning_path) if lightning_available else None

    # ── Use Jones single-level grid as reference (ERA5 0.25°) ─────────────────
    times = ds_jones_s.time.values
    lats  = ds_jones_s.latitude.values  if 'latitude'  in ds_jones_s.coords else ds_jones_s.lat.values
    lons  = ds_jones_s.longitude.values if 'longitude' in ds_jones_s.coords else ds_jones_s.lon.values

    print(f"  Grid: {len(times)} timesteps × {len(lats)} lat × {len(lons)} lon")

    def align(ds, fill=np.nan):
        """Reindex dataset to reference time/lat/lon grid."""
        lat_name = 'latitude'  if 'latitude'  in ds.coords else 'lat'
        lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'
        rename = {}
        if lat_name != 'lat':
            rename[lat_name] = 'lat'
        if lon_name != 'lon':
            rename[lon_name] = 'lon'
        if rename:
            ds = ds.rename(rename)
        ds = ds.reindex(time=times, lat=lats, lon=lons,
                        method='nearest', tolerance=1e9)  # 1e9 ns ≈ 1 s
        return ds.fillna(fill)

    ds_jones_s = align(ds_jones_s)
    ds_jones_p = align(ds_jones_p)
    ds_imerg   = align(ds_imerg,  fill=0.0)
    if ds_light is not None:
        ds_light = align(ds_light, fill=0)

    # ── Extract variables ──────────────────────────────────────────────────────

    # C — CAPE (from jones_single_level)
    cape = ds_jones_s['convective_available_potential_energy'].values.astype('float32')

    # P — Precipitation (IMERG)
    precip_var = 'precipitation' if 'precipitation' in ds_imerg else list(ds_imerg.data_vars)[0]
    precip = ds_imerg[precip_var].values.astype('float32')

    # L — Land-Sea Mask
    lsm_var = 'land_sea_mask' if 'land_sea_mask' in ds_jones_s else 'lsm'
    lsm = ds_jones_s[lsm_var].values.astype('float32')

    # T — 2m Temperature
    t2m_var = '2m_temperature' if '2m_temperature' in ds_jones_s else 't2m'
    t2m = ds_jones_s[t2m_var].values.astype('float32')

    # W — Warm Cloud Depth
    wcd = ds_jones_s['wcd'].values.astype('float32')

    # R — RH average (pre-derived in jones_pressure_level, no pressure dim)
    rh_var = 'rh_avg' if 'rh_avg' in ds_jones_p else 'rh'
    rh = ds_jones_p[rh_var].values.astype('float32')

    # S — Wind Shear (pre-derived in jones_pressure_level, no pressure dim)
    shear_var = 'wind_shear' if 'wind_shear' in ds_jones_p else 'shear'
    shear = ds_jones_p[shear_var].values.astype('float32')

    # V — Vertical velocity at 500 hPa (from jones_pressure_level, levels: 500 & 1000)
    level_dim = 'level' if 'level' in ds_jones_p.dims else 'pressure_level'
    vv = ds_jones_p['vertical_velocity'].sel({level_dim: 500}).values.astype('float32')

    # Target — lightning count
    if ds_light is not None:
        ltg_var = 'lightning_count' if 'lightning_count' in ds_light else list(ds_light.data_vars)[0]
        ltg = ds_light[ltg_var].values.astype('float32')
    else:
        ltg = np.zeros((len(times), len(lats), len(lons)), dtype='float32')

    # ── Build output dataset ───────────────────────────────────────────────────
    coords = {'time': times, 'lat': lats, 'lon': lons}

    data_vars = {
        'cape':                     xr.DataArray(cape,   dims=['time','lat','lon'], attrs={'units': 'J kg-1',   'long_name': 'CAPE'}),
        'precipitation':            xr.DataArray(precip, dims=['time','lat','lon'], attrs={'units': 'mm hr-1',  'long_name': 'IMERG precipitation'}),
        'lsm':                      xr.DataArray(lsm,    dims=['time','lat','lon'], attrs={'units': '0-1',      'long_name': 'Land-sea mask'}),
        't2m':                      xr.DataArray(t2m,    dims=['time','lat','lon'], attrs={'units': 'K',        'long_name': '2m temperature'}),
        'wcd':                      xr.DataArray(wcd,    dims=['time','lat','lon'], attrs={'units': 'm',        'long_name': 'Warm cloud depth'}),
        'rh':                       xr.DataArray(rh,     dims=['time','lat','lon'], attrs={'units': '%',        'long_name': 'RH avg 500-1000 hPa'}),
        'shear':                    xr.DataArray(shear,  dims=['time','lat','lon'], attrs={'units': 'm s-1',    'long_name': 'Wind shear 500-1000 hPa'}),
        'vertical_velocity_500hPa': xr.DataArray(vv,     dims=['time','lat','lon'], attrs={'units': 'Pa s-1',   'long_name': 'Vertical velocity at 500 hPa'}),
        'ltg':                      xr.DataArray(ltg,    dims=['time','lat','lon'], attrs={'units': 'count',    'long_name': 'Lightning count', 'source': 'WWLN'}),
    }

    ds_year = xr.Dataset(data_vars, coords=coords)
    print(f"  Variables: {list(ds_year.data_vars)}")
    return ds_year


# ── Main ───────────────────────────────────────────────────────────────────────
# One file per year — avoids OOM from concatenating all years in memory.
# Re-running is safe: existing files are skipped automatically.

saved = []

for year in ALL_YEARS:
    out_path = os.path.join(DATA_DIR, f'jones_israel_{year}.nc')

    if os.path.exists(out_path):
        print(f"  SKIP {year} — already exists ({out_path})")
        saved.append(out_path)
        continue

    ds = build_year(year)
    if ds is None:
        continue

    ds.attrs['description']  = 'Jones et al. CPLRSTW features + lightning target, Israel/E. Med domain'
    ds.attrs['created_by']   = 'Karin Pitlik'
    ds.attrs['source_files'] = 'ERA5 (Jones download), IMERG, WWLN'
    ds.attrs['year']         = str(year)

    print(f"  Saving → {out_path} ...")
    ds.to_netcdf(out_path)
    ds.close()   # free memory before next year
    saved.append(out_path)
    print(f"  Done {year}.")

if not saved:
    raise RuntimeError("No data found! Check that your NetCDF files are in data/")

print(f"\nAll done! {len(saved)} year file(s) written.")
print(f"\nTo load all years lazily during training:")
print(f"  import xarray as xr")
print(f"  ds = xr.open_mfdataset('data/jones_israel_*.nc', chunks={{'time': 24}})")