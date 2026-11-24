"""
download_jones_era5.py — ERA5 fields for the Israel/E. Med domain from the public ARCO-ERA5 copy on Google Cloud.

Downloads month by month and adds the months to one file per year (data/jones_single_level_{year}.nc and
data/jones_pressure_level_{year}.nc). Months already in a yearly file are skipped, so the script can be
stopped and started again, and new months (e.g. January–August) are merged into the existing Sep–Dec files.

Usage (run it in the terminal; it prints what it is doing as it goes):
    python download_jones_era5.py --years $(seq 2013 2024) --months $(seq 1 8) --model-vars-only
    python download_jones_era5.py --years 2025 --months 9 10 11 12          # all variables, as before

Ctrl+C stops it safely: months that finished downloading are kept (data/tmp_jones_*), the yearly files are
only replaced after a merge has fully finished, and running the same command again continues where it
stopped. The same happens if the connection to the machine drops. Only one copy can run at a time.

--model-vars-only  downloads only the variables train_unet.py uses (CAPE, K index, Total Totals, total column
                   cloud ice and liquid, land-sea mask, cloud ice at 500/1000 hPa). The pressure-level fields
                   are the slow part (every hour reads the whole 3-D field), so this is several times faster.
                   The other variables of the yearly file are NaN for those months.
"""
import os
import numpy as np
import xarray as xr
import time
import gcsfs
import dask
from dask.diagnostics import ProgressBar
from datetime import datetime
import pandas as pd

LAT_MIN, LAT_MAX = 28.0, 43.435   # extended south to include Negev / Eilat (~29.5°N)
LON_MIN, LON_MAX = -9.142, 39.292

PRESSURE_LEVELS = [500, 1000]  # hPa — Jones et al. use 500 & 1000 hPa for wind/RH

# single-level variables (2D: lat × lon, no pressure dimension)
SINGLE_LEVEL_VARS = [
    # ── Original thesis variables ──────────────────────────────────────────────
    'convective_available_potential_energy',
    'k_index',
    'total_totals_index',
    'total_column_cloud_ice_water',
    'total_column_cloud_liquid_water',
    'surface_pressure',

    # ── Jones et al. (2025) CPLRSTW single-level variables ────────────────────
    '2m_temperature',        # T2M
    'cloud_base_height',     # CBH  — used to derive WCD = ZDL - CBH
    'zero_degree_level',     # ZDL  — used to derive WCD = ZDL - CBH
    'land_sea_mask',         # LSM
]

# only what train_unet.py uses (--model-vars-only)
MODEL_SINGLE_VARS = ['convective_available_potential_energy', 'k_index', 'total_totals_index',
                     'total_column_cloud_ice_water', 'total_column_cloud_liquid_water', 'land_sea_mask']
MODEL_PRESSURE_VARS = ['specific_cloud_ice_water_content']

# pressure-level variables (3D: pressure × lat × lon)
PRESSURE_LEVEL_VARS = [
    # ── Original thesis variables ──────────────────────────────────────────────
    'temperature',
    'specific_humidity',
    'vertical_velocity',
    'geopotential',
    'specific_cloud_ice_water_content',
    'specific_cloud_liquid_water_content',

    # ── Jones et al. (2025) CPLRSTW pressure-level variables ──────────────────
    'u_component_of_wind',   # u at 500 & 1000 hPa — used to derive SHEAR
    'v_component_of_wind',   # v at 500 & 1000 hPa — used to derive SHEAR
    # temperature and specific_humidity (listed above) are also used to derive RH
    # Note: relative_humidity not in ARCO-ERA5; derived from q and T instead
]


TIME_ENCODING = {'units': 'seconds since 1970-01-01', 'dtype': 'int64'}


def _add_missing_vars(ds, ref):
    """Variables of `ref` that `ds` lacks are added to `ds` as NaN (lazily), with `ds`'s times."""
    import dask.array as dsa
    for v in ref.data_vars:
        if v in ds.data_vars:
            continue
        dims = ref[v].dims
        shape = tuple(ds.sizes['time'] if d == 'time' else ref.sizes[d] for d in dims)
        chunks = tuple(min(168, n) if d == 'time' else n for d, n in zip(dims, shape))
        coords = {d: (ds[d] if d == 'time' else ref[d]) for d in dims if d in ref.coords or d == 'time'}
        ds[v] = xr.DataArray(dsa.full(shape, np.nan, dtype='float32', chunks=chunks), dims=dims, coords=coords,
                             attrs=ref[v].attrs)
    return ds


def merge_into_yearly(new, path, replace=False):
    """Add the times in `new` to the yearly file `path` (created if missing). Variables present in only one
    of the two are kept and are NaN where missing. Times already in the file are kept unless replace=True.
    Written to a temporary file first, so an interrupted run never damages the existing file."""
    old = xr.open_dataset(path, chunks={'time': 168}) if os.path.exists(path) else None
    if old is not None:
        if replace:
            old = old.sel(time=~old['time'].isin(new['time'].values))
        new = _add_missing_vars(new, old)
        old = _add_missing_vars(old, new)
        both = xr.concat([old, new[list(old.data_vars)]], dim='time', data_vars='all', coords='minimal',
                         compat='override', join='outer')
    else:
        both = new
    both = both.sortby('time')
    _, first = np.unique(both['time'].values, return_index=True)
    both = both.isel(time=np.sort(first))
    enc = {v: {'zlib': True, 'complevel': 1,                 # stored in day-long pieces (24 h)
               'chunksizes': tuple(min(24, n) if d == 'time' else n for d, n in zip(both[v].dims, both[v].shape))}
           for v in both.data_vars}
    enc['time'] = TIME_ENCODING
    tmp = path + '.tmp'
    # one piece at a time, no threads: reading and writing netCDF from several threads can hang on some
    # netCDF/HDF5 builds, and it keeps memory low. The bar shows that it is moving (updated every 30 s).
    t0 = time.time()
    print(f"    writing {os.path.basename(path)}", flush=True)
    with dask.config.set(scheduler='synchronous'), ProgressBar(dt=2):
        both.to_netcdf(tmp, format='NETCDF4', encoding=enc)
    if old is not None:
        old.close()
    os.replace(tmp, path)
    months = sorted(set(pd.DatetimeIndex(both['time'].values).month))
    print(f"    {os.path.basename(path)}: months {months[0]}–{months[-1]} ({len(months)}), {both.sizes['time']} hours, "
          f"{os.path.getsize(path) / 1e9:.1f} GB, {_dur(time.time() - t0)}", flush=True)


def months_in_file(path):
    """Months that are complete (every hour present) in an existing yearly file."""
    if not os.path.exists(path):
        return set()
    with xr.open_dataset(path) as f:
        t = pd.DatetimeIndex(f['time'].values)
    per = t.to_period('M').value_counts()
    return {p.month for p, n in per.items() if n == p.days_in_month * 24}


def _dur(sec):
    sec = int(sec)
    return f"{sec // 3600} h {sec % 3600 // 60:02d} min" if sec >= 3600 else f"{sec // 60} min {sec % 60:02d} s"


def open_arco():
    print("Opening ARCO-ERA5 on Google Cloud ...", flush=True)
    fs = gcsfs.GCSFileSystem(token='anon')
    store = fs.get_mapper('gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3')
    return xr.open_zarr(store, consolidated=True)


class Clock:
    """Months done so far and their average time, for the 'time left' estimate."""
    def __init__(self, total):
        self.total, self.done, self.spent, self.t0 = total, 0, 0.0, time.time()

    def month_done(self, sec):
        self.done += 1
        self.spent += sec
        left = self.total - self.done
        return (f"{self.done}/{self.total} months downloaded"
                + (f", about {_dur(left * self.spent / self.done)} left" if left else ''))


PARTS = []          # the .part file being written right now (removed if you press Ctrl+C)


def download_era5(year, months, out_dir='data', model_vars_only=False, ds=None, clock=None):
    """Download the given months of one year and merge them into the yearly files."""
    sl_vars = MODEL_SINGLE_VARS if model_vars_only else SINGLE_LEVEL_VARS
    pl_vars = MODEL_PRESSURE_VARS if model_vars_only else PRESSURE_LEVEL_VARS
    tag = '_model' if model_vars_only else ''
    single_path   = f'{out_dir}/jones_single_level_{year}.nc'
    pressure_path = f'{out_dir}/jones_pressure_level_{year}.nc'

    have = months_in_file(single_path) & months_in_file(pressure_path)
    todo = [m for m in months if m not in have]
    print(f"\n── {year} " + '─' * 60, flush=True)
    if not todo:
        print(f"  months {months} are already in the yearly files — nothing to do.", flush=True)
        return
    print(f"  to download: months {todo}" + (f"   (already there: {sorted(set(months) & have)})" if set(months) & have else ''),
          flush=True)
    ds = ds if ds is not None else open_arco()
    spatial = dict(latitude=slice(LAT_MAX, LAT_MIN), longitude=slice(LON_MIN, LON_MAX))
    time_enc = {'time': TIME_ENCODING}
    t_year = time.time()

    single_monthly, pressure_monthly = [], []
    for m in todo:
        month_start = pd.Timestamp(year, m, 1)
        month_end   = (month_start + pd.offsets.MonthEnd(1)).replace(hour=23)
        month_slice = slice(month_start.isoformat(), month_end.isoformat())
        month_str   = month_start.strftime('%Y_%m')
        tmp_single   = f'{out_dir}/tmp_jones_single_{month_str}{tag}.nc'
        tmp_pressure = f'{out_dir}/tmp_jones_pressure_{month_str}{tag}.nc'
        if os.path.exists(tmp_single) and os.path.exists(tmp_pressure):
            print(f"  {month_start:%b %Y}: downloaded in an earlier run, kept", flush=True)
            if clock:
                clock.total -= 1
        else:
            t0 = time.time()
            for label, data, tmp in (
                    (f'{len(sl_vars)} single-level fields', ds[sl_vars].sel(time=month_slice, **spatial), tmp_single),
                    (f'{len(pl_vars)} pressure-level field(s) at {PRESSURE_LEVELS} hPa',
                     ds[pl_vars].sel(time=month_slice, **spatial).sel(level=PRESSURE_LEVELS), tmp_pressure)):
                if os.path.exists(tmp):
                    continue
                print(f"  {month_start:%b %Y}: {label}", flush=True)
                PARTS[:] = [tmp + '.part']
                with ProgressBar(dt=2):
                    data.to_netcdf(tmp + '.part', format='NETCDF4', encoding=time_enc)
                os.replace(tmp + '.part', tmp)          # a month counts only when complete
                PARTS[:] = []
            sec = time.time() - t0
            size = (os.path.getsize(tmp_single) + os.path.getsize(tmp_pressure)) / 1e6
            print(f"  {month_start:%b %Y}: done in {_dur(sec)} ({size:.0f} MB)"
                  + (f"  —  {clock.month_done(sec)}" if clock else ''), flush=True)
        single_monthly.append(tmp_single)
        pressure_monthly.append(tmp_pressure)

    print(f"  merging {len(todo)} month(s) into the yearly files ...", flush=True)
    sl = xr.open_mfdataset(single_monthly,   combine='by_coords', chunks={'time': 168})
    pl = xr.open_mfdataset(pressure_monthly, combine='by_coords', chunks={'time': 168})
    # derived variables, where their ingredients were downloaded
    if {'zero_degree_level', 'cloud_base_height'} <= set(sl.data_vars):
        sl['wcd'] = sl['zero_degree_level'] - sl['cloud_base_height']          # warm cloud depth [m]
        sl['wcd'].attrs = {'long_name': 'Warm Cloud Depth (ZDL - CBH)', 'units': 'm'}
    if {'u_component_of_wind', 'v_component_of_wind'} <= set(pl.data_vars):
        pl['wind_shear'] = np.sqrt(
            (pl['u_component_of_wind'].sel(level=500) - pl['u_component_of_wind'].sel(level=1000))**2 +
            (pl['v_component_of_wind'].sel(level=500) - pl['v_component_of_wind'].sel(level=1000))**2)
        pl['wind_shear'].attrs = {'long_name': 'Deep-layer wind shear (500-1000 hPa)', 'units': 'm/s'}
    if {'specific_humidity', 'temperature'} <= set(pl.data_vars):
        def compute_rh(q, T, p_hpa):     # RH from q and T (ARCO doesn't store RH)
            e  = q * p_hpa / (0.622 + 0.378 * q)
            es = 6.112 * np.exp(17.67 * (T - 273.15) / (T - 29.65))
            return (e / es * 100).clip(0, 100)
        rh500  = compute_rh(pl['specific_humidity'].sel(level=500),  pl['temperature'].sel(level=500),  500)
        rh1000 = compute_rh(pl['specific_humidity'].sel(level=1000), pl['temperature'].sel(level=1000), 1000)
        pl['rh_avg'] = (rh500 + rh1000) / 2.0
        pl['rh_avg'].attrs = {'long_name': 'Mean relative humidity (500 & 1000 hPa)', 'units': '%'}

    merge_into_yearly(sl, single_path)
    merge_into_yearly(pl, pressure_path)
    sl.close(); pl.close()

    for f in single_monthly + pressure_monthly:          # the months are now in the yearly files
        os.remove(f)
    print(f"  {year} done in {_dur(time.time() - t_year)}", flush=True)


if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser(description='Download ERA5 (ARCO) months and merge them into the yearly files.')
    ap.add_argument('--years', type=int, nargs='+', default=list(range(2013, 2025)),
                    help='years (default 2013–2024)')
    ap.add_argument('--months', type=int, nargs='+', default=list(range(1, 13)),
                    help='months, 1–12 (default: all; months already in the yearly file are skipped)')
    ap.add_argument('--model-vars-only', action='store_true',
                    help='only the variables train_unet.py uses (much faster)')
    ap.add_argument('--out-dir', default='data')
    args = ap.parse_args()
    months = sorted(set(args.months))

    # only one copy at a time (two copies would write the same files)
    lock = os.path.join(args.out_dir, '.download_era5.lock')
    if os.path.exists(lock):
        pid = int(open(lock).read().strip() or 0)
        if pid and os.path.exists(f'/proc/{pid}') and 'download_jones_era5' in open(f'/proc/{pid}/cmdline').read():
            sys.exit(f"Another ERA5 download is already running (process {pid}). Stop it first with:  kill {pid}")
    open(lock, 'w').write(str(os.getpid()))

    # the plan: what is missing in each yearly file
    plan = {}
    for y in args.years:
        have = (months_in_file(f'{args.out_dir}/jones_single_level_{y}.nc')
                & months_in_file(f'{args.out_dir}/jones_pressure_level_{y}.nc'))
        plan[y] = [m for m in months if m not in have]
    n = sum(len(v) for v in plan.values())
    print('ERA5 download' + (' — model variables only' if args.model_vars_only else ' — all variables')
          + f"\n  years {args.years[0]}–{args.years[-1]}, months {months}: {n} month(s) to download"
          + f"\n  into {args.out_dir}/jones_single_level_<year>.nc and jones_pressure_level_<year>.nc"
          + "\n  Ctrl+C stops it safely; run the same command again to continue.", flush=True)
    t_all = time.time()
    try:
        arco = open_arco() if n else None
        clock = Clock(n)
        for year in args.years:
            download_era5(year, months, out_dir=args.out_dir, model_vars_only=args.model_vars_only,
                          ds=arco, clock=clock)
        print(f"\nAll done in {_dur(time.time() - t_all)}.", flush=True)
    except KeyboardInterrupt:
        for f in PARTS:
            if os.path.exists(f):
                os.remove(f)
        print("\n\nStopped (Ctrl+C). Months that finished are kept; the yearly files are unchanged unless their "
              "merge had finished.\nRun the same command again to continue.", flush=True)
        sys.exit(130)
    finally:
        if os.path.exists(lock):
            os.remove(lock)
