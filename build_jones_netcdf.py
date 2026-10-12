"""
build_jones_netcdf.py
---------------------
Builds one jones_israel_{year}.nc per year DIRECTLY from downloaded NetCDF files,
skipping the parquet intermediate step entirely.

Usage:
    python build_jones_netcdf.py                           # build missing years, then combine
    python build_jones_netcdf.py --years 2025 --overwrite  # rebuild a year (e.g. after re-downloading IMERG)
    python build_jones_netcdf.py --allow-gaps              # keep years with missing IMERG hours (as NaN)

Missing data stays NaN; it is never filled with 0 (a missing hour of rain or lightning is not
"no rain" / "no lightning"). A year whose IMERG file does not cover every hour is skipped with
an error, unless --allow-gaps is given.

Input files (per year, all in data/):
  jones_single_level_{year}.nc    → CAPE, LSM, T2M, WCD (+ k_index, etc.)
  jones_pressure_level_{year}.nc  → RH avg, wind shear (pre-derived),
                                    vertical_velocity at levels 500+1000 hPa
  imerg_hourly_{year}.nc          → precipitation
  wwln_on_era5_grid_{year}.nc     → lightning_count (optional — NaN if absent)

Output (one file per year):
  data/jones_israel_{year}.nc
  Dims:  time × lat × lon
  Vars:  cape, precipitation, lsm, rh, shear, t2m, wcd,
         vertical_velocity_500hPa, ltg

To load all years lazily during training:
  ds = xr.open_mfdataset('data/jones_israel_*.nc', chunks={'time': 24})
"""

import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR = 'data'

ALL_YEARS = list(range(2013, 2026))   # 2013–2025 inclusive


def build_year(year: int, allow_gaps: bool = False) -> xr.Dataset:
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
        print(f"  WARNING {year}: {os.path.basename(lightning_path)} not found — ltg will be NaN (missing)")

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

    def align(ds):
        """Reindex dataset to the reference time/lat/lon grid. Anything without a match stays
        NaN (= missing); it is never filled with 0."""
        lat_name = 'latitude'  if 'latitude'  in ds.coords else 'lat'
        lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'
        rename = {}
        if lat_name != 'lat':
            rename[lat_name] = 'lat'
        if lon_name != 'lon':
            rename[lon_name] = 'lon'
        if rename:
            ds = ds.rename(rename)
        # separate tolerances: 1 s for time, 0.01° for lat/lon (a shifted grid → NaN, not a silent match)
        ds = ds.reindex(time=times, method='nearest', tolerance=np.timedelta64(1, 's'))
        ds = ds.reindex(lat=lats, lon=lons, method='nearest', tolerance=0.01)
        return ds

    ds_jones_s = align(ds_jones_s)
    ds_jones_p = align(ds_jones_p)
    ds_imerg   = align(ds_imerg)
    if ds_light is not None:
        ds_light = align(ds_light)

    # ── Helper: safely extract a variable (returns NaN array if missing) ─────────
    def get_sl(name, alt=None):
        """Get single-level variable by name or alt name."""
        for n in ([name] + ([alt] if alt else [])):
            if n in ds_jones_s:
                return ds_jones_s[n].values.astype('float32')
        print(f"    WARNING: {name} not found in single-level file — filling NaN")
        return np.full((len(times), len(lats), len(lons)), np.nan, dtype='float32')

    def get_pl(name, level):
        """Get pressure-level variable at a specific hPa level."""
        level_dim = 'level' if 'level' in ds_jones_p.dims else 'pressure_level'
        if name in ds_jones_p:
            return ds_jones_p[name].sel({level_dim: level}).values.astype('float32')
        print(f"    WARNING: {name}@{level}hPa not found — filling NaN")
        return np.full((len(times), len(lats), len(lons)), np.nan, dtype='float32')

    level_dim = 'level' if 'level' in ds_jones_p.dims else 'pressure_level'

    # ── Extract variables ──────────────────────────────────────────────────────

    # ── Single-level ──────────────────────────────────────────────────────────
    cape   = get_sl('convective_available_potential_energy')
    lsm    = get_sl('land_sea_mask', 'lsm')
    t2m    = get_sl('2m_temperature', 't2m')
    wcd    = get_sl('wcd')                        # derived in download script
    ki     = get_sl('k_index')
    tti    = get_sl('total_totals_index')
    tciw   = get_sl('total_column_cloud_ice_water')
    tclw   = get_sl('total_column_cloud_liquid_water')
    sp     = get_sl('surface_pressure')
    cbh    = get_sl('cloud_base_height')
    zdl    = get_sl('zero_degree_level')

    # ── Derived (pre-computed in download script) ──────────────────────────────
    rh_var    = 'rh_avg'    if 'rh_avg'    in ds_jones_p else 'rh'
    shear_var = 'wind_shear' if 'wind_shear' in ds_jones_p else 'shear'
    rh    = ds_jones_p[rh_var].values.astype('float32')    if rh_var    in ds_jones_p else np.full((len(times), len(lats), len(lons)), np.nan, dtype='float32')
    shear = ds_jones_p[shear_var].values.astype('float32') if shear_var in ds_jones_p else np.full((len(times), len(lats), len(lons)), np.nan, dtype='float32')

    # ── Pressure-level @ 500 & 1000 hPa ──────────────────────────────────────
    vv500   = get_pl('vertical_velocity',              500)
    vv1000  = get_pl('vertical_velocity',              1000)
    t500    = get_pl('temperature',                    500)
    t1000   = get_pl('temperature',                    1000)
    q500    = get_pl('specific_humidity',              500)
    q1000   = get_pl('specific_humidity',              1000)
    z500    = get_pl('geopotential',                   500)
    z1000   = get_pl('geopotential',                   1000)
    ciwc500 = get_pl('specific_cloud_ice_water_content',    500)
    ciwc1000= get_pl('specific_cloud_ice_water_content',    1000)
    clwc500 = get_pl('specific_cloud_liquid_water_content', 500)
    clwc1000= get_pl('specific_cloud_liquid_water_content', 1000)
    u500    = get_pl('u_component_of_wind',            500)
    u1000   = get_pl('u_component_of_wind',            1000)
    v500    = get_pl('v_component_of_wind',            500)
    v1000   = get_pl('v_component_of_wind',            1000)

    # ── Precipitation (IMERG) ─────────────────────────────────────────────────
    precip_var = 'precipitation' if 'precipitation' in ds_imerg else list(ds_imerg.data_vars)[0]
    precip = ds_imerg[precip_var].values.astype('float32')

    # IMERG coverage check: every hour must be present. (Missing hours used to be filled with
    # 0 mm/h — that is how Oct–Dec 2025 ended up as "no rain".)
    hour_missing = np.isnan(precip).all(axis=(1, 2))
    if hour_missing.any():
        by_month = pd.DatetimeIndex(times[hour_missing]).to_period('M').value_counts().sort_index()
        print(f"  WARNING {year}: IMERG is missing for {int(hour_missing.sum())} of {len(times)} hours:")
        for month, n in by_month.items():
            print(f"    {month}: {n} hours missing")
        if not allow_gaps:
            print(f"  SKIP {year} — re-download IMERG first: "
                  f"python download_imerg.py --years {year} --overwrite "
                  f"(or rerun with --allow-gaps to keep the gaps as NaN)")
            return None

    # ── Target — lightning count ──────────────────────────────────────────────
    if ds_light is not None:
        ltg_var = 'lightning_count' if 'lightning_count' in ds_light else list(ds_light.data_vars)[0]
        ltg = ds_light[ltg_var].values.astype('float32')
        if 'data_available' in ds_light:
            no_data = ~(ds_light['data_available'].values == 1)   # also True for NaN (hour not in file)
            ltg[no_data] = np.nan
            if no_data.any():
                print(f"  NOTE {year}: no WWLN data for {int(no_data.sum())} hours — ltg set to NaN there")
        else:
            print(f"  NOTE {year}: WWLN file has no 'data_available' flag, so days without a .loc file "
                  f"look like 'no lightning'. Re-run: python process_wwln.py --year {year}")
    else:
        ltg = np.full((len(times), len(lats), len(lons)), np.nan, dtype='float32')

    # ── Build output dataset ───────────────────────────────────────────────────
    coords = {'time': times, 'lat': lats, 'lon': lons}
    D = ['time', 'lat', 'lon']

    def da(arr, units, long_name):
        return xr.DataArray(arr, dims=D, attrs={'units': units, 'long_name': long_name})

    data_vars = {
        # ── Jones CPLRSTW features ──────────────────────────────────────────
        'cape':                      da(cape,    'J kg-1',  'CAPE'),
        'precipitation':             da(precip,  'mm hr-1', 'IMERG precipitation'),
        'lsm':                       da(lsm,     '0-1',     'Land-sea mask'),
        't2m':                       da(t2m,     'K',       '2m temperature'),
        'wcd':                       da(wcd,     'm',       'Warm cloud depth (ZDL-CBH)'),
        'rh':                        da(rh,      '%',       'RH avg 500-1000 hPa'),
        'shear':                     da(shear,   'm s-1',   'Wind shear 500-1000 hPa'),
        'vertical_velocity_500hPa':  da(vv500,   'Pa s-1',  'Vertical velocity 500 hPa'),
        # ── Additional single-level ─────────────────────────────────────────
        'k_index':                   da(ki,      'K',       'K-index'),
        'total_totals_index':        da(tti,     'K',       'Total Totals Index'),
        'total_column_cloud_ice_water':    da(tciw, 'kg m-2', 'Total column cloud ice water'),
        'total_column_cloud_liquid_water': da(tclw, 'kg m-2', 'Total column cloud liquid water'),
        'surface_pressure':          da(sp,      'Pa',      'Surface pressure'),
        'cloud_base_height':         da(cbh,     'm',       'Cloud base height'),
        'zero_degree_level':         da(zdl,     'm',       'Zero degree level'),
        # ── Additional pressure-level ───────────────────────────────────────
        'vertical_velocity_1000hPa': da(vv1000,   'Pa s-1',  'Vertical velocity 1000 hPa'),
        't500':                      da(t500,    'K',       'Temperature 500 hPa'),
        't1000':                     da(t1000,   'K',       'Temperature 1000 hPa'),
        'q500':                      da(q500,    'kg kg-1', 'Specific humidity 500 hPa'),
        'q1000':                     da(q1000,   'kg kg-1', 'Specific humidity 1000 hPa'),
        'z500':                      da(z500,    'm2 s-2',  'Geopotential 500 hPa'),
        'z1000':                     da(z1000,   'm2 s-2',  'Geopotential 1000 hPa'),
        'ciwc500':                   da(ciwc500, 'kg kg-1', 'Cloud ice water content 500 hPa'),
        'ciwc1000':                  da(ciwc1000,'kg kg-1', 'Cloud ice water content 1000 hPa'),
        'clwc500':                   da(clwc500, 'kg kg-1', 'Cloud liquid water content 500 hPa'),
        'clwc1000':                  da(clwc1000,'kg kg-1', 'Cloud liquid water content 1000 hPa'),
        'u500':                      da(u500,    'm s-1',   'U wind 500 hPa'),
        'u1000':                     da(u1000,   'm s-1',   'U wind 1000 hPa'),
        'v500':                      da(v500,    'm s-1',   'V wind 500 hPa'),
        'v1000':                     da(v1000,   'm s-1',   'V wind 1000 hPa'),
        # ── Target ──────────────────────────────────────────────────────────
        'ltg':                       da(ltg,     'count',   'Lightning count (WWLN)'),
    }

    ds_year = xr.Dataset(data_vars, coords=coords)
    print(f"  Variables: {list(ds_year.data_vars)}")
    return ds_year


# ── Main ───────────────────────────────────────────────────────────────────────
# One file per year — avoids OOM from concatenating all years in memory.
# Re-running is safe: existing files are skipped automatically.

parser = argparse.ArgumentParser(description='Build jones_israel_{year}.nc files and combine them.')
parser.add_argument('--years', type=int, nargs='+', default=None,
                    help='years to build/rebuild (default: all years whose file does not exist yet)')
parser.add_argument('--overwrite', action='store_true',
                    help='rebuild year files that already exist (only the years given with --years)')
parser.add_argument('--allow-gaps', action='store_true',
                    help='keep years with missing IMERG hours (gaps stay NaN) instead of skipping them')
args = parser.parse_args()
build_years = args.years or ALL_YEARS
if args.overwrite and not args.years:
    parser.error('--overwrite needs --years (so existing years are not rebuilt by accident)')

saved = []

for year in ALL_YEARS:
    out_path = os.path.join(DATA_DIR, f'jones_israel_{year}.nc')

    if os.path.exists(out_path) and not (args.overwrite and year in build_years):
        with xr.open_dataset(out_path) as old:
            built_by_old_version = 'missing_data' not in old.attrs
        print(f"  SKIP {year} — already exists ({out_path})"
              + ("\n         built by an older version that filled missing data with 0; "
                 f"rebuild with: --years {year} --overwrite" if built_by_old_version else ""))
        saved.append(out_path)
        continue
    if year not in build_years:
        continue

    ds = build_year(year, allow_gaps=args.allow_gaps)
    if ds is None:
        if os.path.exists(out_path):
            print(f"  NOTE: the old {out_path} was NOT replaced and is left out of the combined file.")
        continue

    ds.attrs['description']  = 'Jones et al. CPLRSTW features + lightning target, Israel/E. Med domain'
    ds.attrs['created_by']   = 'Karin Pitlik'
    ds.attrs['source_files'] = 'ERA5 (Jones download), IMERG, WWLN'
    ds.attrs['year']         = str(year)
    ds.attrs['missing_data'] = 'kept as NaN (never filled with 0)'

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

# ── Combine all per-year files into one jones_israel_all_years.nc ─────────────
ALL_YEARS_PATH = os.path.join(DATA_DIR, 'jones_israel_all_years.nc')
print(f"\nCombining {len(saved)} yearly files → {ALL_YEARS_PATH} ...")

import xarray as xr
ds_all = xr.open_mfdataset(sorted(saved), combine='by_coords', chunks={'time': 24})
ds_all.attrs['description'] = 'Jones et al. CPLRSTW features + lightning target, all years combined'
ds_all.attrs['years']       = f"{ALL_YEARS[0]}–{ALL_YEARS[-1]}"

# Write as a single NetCDF (this may take a few minutes for 13 years)
ds_all.to_netcdf(ALL_YEARS_PATH)
ds_all.close()
print(f"Saved: {ALL_YEARS_PATH}")