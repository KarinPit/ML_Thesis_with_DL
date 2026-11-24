"""
download_imerg.py
-----------------
Downloads NASA GPM IMERG Final Precipitation L3 Half-Hourly 0.1° V07 data
for the Israel/Eastern Mediterranean domain, aggregates to hourly resolution,
regrids to the ERA5 0.25° grid, and saves as NetCDF.

This is the precipitation source used by Jones et al. (2025) — NOT ERA5 precipitation.

Prerequisites:
  1. Create a free NASA Earthdata account: https://urs.earthdata.nasa.gov
  2. Install earthaccess: pip install earthaccess h5py scipy --break-system-packages

First-time setup:
  Run `earthaccess.login(strategy="interactive")` once to save credentials locally.
  After that, subsequent runs authenticate automatically.

Output:
  data/imerg_hourly_{year}.nc — hourly precipitation (mm/hr) on 0.25° ERA5 grid, one file per year
                                (missing hours/pixels are NaN, never 0). Months are downloaded one at a
                                time and merged into the yearly file; months already in it are skipped.

Regridding (--regrid):
  conservative (default) — area-weighted average of the 0.1° IMERG cells inside each 0.25° cell
                           (first-order conservative, as in Jones et al.)
  bilinear               — the original method (point interpolation). Use this if you re-download
                           only some years and want them consistent with files made earlier.

Usage (run it in the terminal; it prints what it is doing as it goes):
    python download_imerg.py                                          # all years, Sep–Dec, skip what exists
    python download_imerg.py --years $(seq 2013 2024) --months $(seq 1 8)   # add Jan–Aug to the yearly files
    python download_imerg.py --years 2025 --overwrite                 # re-download Sep–Dec 2025
    python download_imerg.py --years 2025 --overwrite --regrid bilinear

Ctrl+C stops it safely: finished months are kept (data/imerg_hourly_<year>_<month>.nc until they are merged),
the yearly file is only replaced after a merge has fully finished, and running the same command again
continues where it stopped. The same happens if the connection to the machine drops. Only one copy can run
at a time.
"""

import os
import argparse
import warnings
import numpy as np
import xarray as xr
import pandas as pd
import h5py
from scipy.interpolate import RegularGridInterpolator
import earthaccess
import dask
dask.config.set(scheduler='synchronous')   # merging: no threads (netCDF from several threads can hang)

# ── Domain (match ERA5 grid exactly) ──────────────────────────────────────────
LAT_MIN, LAT_MAX = 28.0, 43.435   # extended south to include Negev / Eilat (~29.5°N)
LON_MIN, LON_MAX = -9.142, 39.292

# ERA5 0.25° target grid for this domain
ERA5_LATS = np.arange(np.ceil(LAT_MIN * 4) / 4, LAT_MAX + 0.001, 0.25)
ERA5_LONS = np.arange(np.ceil(LON_MIN * 4) / 4, LON_MAX + 0.001, 0.25)

# ── IMERG product ──────────────────────────────────────────────────────────────
IMERG_SHORT_NAME = 'GPM_3IMERGHH'
IMERG_VERSION    = '07'
PRECIP_VAR       = 'precipitation'   # V07: 'precipitation'; V06: 'precipitationCal'
REGRID           = 'conservative'    # 'conservative' (area average) or 'bilinear' (original)


def inspect_imerg_file(filepath):
    """Print HDF5 structure of one IMERG file — use for debugging."""
    def print_tree(name, obj):
        print(f"  {name}: {obj}")
    with h5py.File(filepath, 'r') as f:
        print(f"\n=== IMERG file structure: {os.path.basename(filepath)} ===")
        f.visititems(print_tree)
        if 'Grid' in f:
            print(f"\nGrid keys: {list(f['Grid'].keys())}")
            if PRECIP_VAR in f['Grid']:
                print(f"  {PRECIP_VAR} shape: {f['Grid'][PRECIP_VAR].shape}")
            if 'time' in f['Grid']:
                print(f"  time: {f['Grid']['time'][:]}")
                print(f"  time attrs: {dict(f['Grid']['time'].attrs)}")


def read_imerg_granule(filepath):
    """Read one IMERG HDF5 half-hourly file → (time, lat, lon, precip_mm_hr)."""
    with h5py.File(filepath, 'r') as f:
        # V07: precipitation shape is (1, lon, lat)
        grid   = f['Grid']
        var    = PRECIP_VAR if PRECIP_VAR in grid else 'precipitationCal'
        precip = grid[var][0]          # (lon, lat) after dropping time dim
        lons   = f['Grid/lon'][:]      # shape (3600,)
        lats   = f['Grid/lat'][:]      # shape (1800,)

        # Time: IMERG V07 uses GPS epoch = 1980-01-06 00:00:00 UTC
        t_val  = int(f['Grid/time'][0])
        t_unit = f['Grid/time'].attrs.get('units', b'seconds since 1980-01-06 00:00:00 UTC')
        t_unit = t_unit.decode() if isinstance(t_unit, bytes) else t_unit
        # Parse reference date from units string e.g. "seconds since 1980-01-06 00:00:00 UTC"
        ref_str = t_unit.split('since')[-1].strip().replace(' UTC', '')
        ref     = pd.Timestamp(ref_str)
        time    = ref + pd.Timedelta(seconds=t_val)

    # shape after [0]: (lon=3600, lat=1800) — transpose to (lat, lon)
    precip = precip.T   # now (lat=1800, lon=3600)

    precip = np.where(precip < 0, np.nan, precip)   # fill values (-9999.9) = missing → NaN, not 0
    return time, lats, lons, precip


def _cell_edges(centers):
    """Cell edges for a regular 1-D grid of cell centres (ascending)."""
    d = np.diff(centers).mean()
    return np.concatenate([centers - d / 2, [centers[-1] + d / 2]])


def _overlap_weights(src_edges, dst_edges):
    """W[i, j] = fraction of destination cell i covered by source cell j (1-D, ascending edges)."""
    lo = np.maximum(dst_edges[:-1, None], src_edges[None, :-1])
    hi = np.minimum(dst_edges[1:, None],  src_edges[None, 1:])
    return np.clip(hi - lo, 0, None) / (dst_edges[1:] - dst_edges[:-1])[:, None]


def crop_and_regrid(lats_src, lons_src, precip, target_lats, target_lons, method=REGRID):
    """Crop to domain and regrid from the 0.1° IMERG grid to the 0.25° ERA5 grid.
    NaN (missing) source pixels are ignored; a target cell with less than half of its area
    covered by valid IMERG pixels becomes NaN."""
    # Crop source to slightly wider than domain (for interpolation boundary)
    lat_mask = (lats_src >= LAT_MIN - 0.2) & (lats_src <= LAT_MAX + 0.2)
    lon_mask = (lons_src >= LON_MIN - 0.2) & (lons_src <= LON_MAX + 0.2)

    lats_c  = lats_src[lat_mask]
    lons_c  = lons_src[lon_mask]
    precip_c = precip[np.ix_(lat_mask, lon_mask)]

    if method == 'bilinear':
        # Original method: bilinear point interpolation onto the ERA5 0.25° grid
        interp = RegularGridInterpolator(
            (lats_c, lons_c), precip_c, method='linear', bounds_error=False, fill_value=np.nan
        )
        grid_lats, grid_lons = np.meshgrid(target_lats, target_lons, indexing='ij')
        regridded = interp(np.stack([grid_lats.ravel(), grid_lons.ravel()], axis=-1))
        return regridded.reshape(len(target_lats), len(target_lons))

    # First-order conservative: area-weighted mean of the IMERG cells overlapping each ERA5 cell.
    # Latitude overlaps are measured in sin(lat), which makes them proportional to true area.
    w_lat = _overlap_weights(np.sin(np.deg2rad(_cell_edges(lats_c))),
                             np.sin(np.deg2rad(_cell_edges(np.asarray(target_lats)))))
    w_lon = _overlap_weights(_cell_edges(lons_c), _cell_edges(np.asarray(target_lons)))
    valid = np.isfinite(precip_c)
    num = w_lat @ np.where(valid, precip_c, 0.0) @ w_lon.T
    cov = w_lat @ valid.astype(float) @ w_lon.T            # fraction of each cell with valid data
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(cov >= 0.5, num / cov, np.nan)


def _dur(sec):
    sec = int(sec)
    return f"{sec // 3600} h {sec % 3600 // 60:02d} min" if sec >= 3600 else f"{sec // 60} min {sec % 60:02d} s"


def download_imerg(time_range, out_dir='data', tmp_dir='data/tmp_imerg',
                   overwrite=False, regrid=REGRID, out_path=None):
    """
    Download IMERG half-hourly granules, aggregate to hourly, regrid, save NetCDF.
    Every hour of the period is written; hours without IMERG data are NaN.
    Returns the file written, or None if there is no IMERG data for the period.
    """
    os.makedirs(tmp_dir, exist_ok=True)

    start_year = pd.Timestamp(time_range.start).year
    end_year   = pd.Timestamp(time_range.stop).year
    ts = f"{start_year}_{end_year}" if end_year != start_year else str(start_year)
    out_path = out_path or f'{out_dir}/imerg_hourly_{ts}.nc'
    label = (f"{pd.Timestamp(time_range.start):%b %Y}" if pd.Timestamp(time_range.start).to_period('M')
             == pd.Timestamp(time_range.stop).to_period('M') else f"{time_range.start} → {time_range.stop}")

    if os.path.exists(out_path) and not overwrite:
        print(f"  {label}: downloaded in an earlier run, kept", flush=True)
        return out_path

    # A date-only end ('2025-12-31') can be read as midnight, which drops the last day.
    t_end = time_range.stop if 'T' in str(time_range.stop) else f"{time_range.stop}T23:59:59"

    # ── Search and download the granules (earthaccess shows its own progress bars) ──
    results = earthaccess.search_data(
        short_name=IMERG_SHORT_NAME,
        version=IMERG_VERSION,
        temporal=(time_range.start, t_end),
        bounding_box=(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX),
    )
    print(f"  {label}: downloading {len(results)} half-hourly granules", flush=True)
    files = sorted(earthaccess.download(results, tmp_dir))

    # ── Read, crop, regrid ──
    records = []   # list of (time, precip_array)
    n = len(files)
    for k, fpath in enumerate(files, 1):
        try:
            t, lats_src, lons_src, precip = read_imerg_granule(fpath)
            p_regrid = crop_and_regrid(lats_src, lons_src, precip, ERA5_LATS, ERA5_LONS, method=regrid)
            records.append((t, p_regrid))
        except Exception as e:
            print(f"\n    WARNING: could not read {os.path.basename(fpath)}: {e}")
        if k == n or k % max(1, n // 50) == 0:
            print(f"\r  {label}: regridding  {k}/{n} granules ({100 * k // max(n, 1)} %)", end='', flush=True)
    print(flush=True)
    if not records:
        print(f"  {label}: ERROR — no IMERG data. IMERG Final is published ~3.5 months after the fact; "
              f"this period may not be available yet.", flush=True)
        return None

    # ── Aggregate two 30-min granules → one hourly value (mean of the 30-min rates) ──
    by_hour = {}
    for t, arr in records:
        by_hour.setdefault(pd.Timestamp(t).floor('h'), []).append(arr)
    hourly_times = sorted(by_hour)
    with warnings.catch_warnings():               # all-NaN pixels → NaN, silently
        warnings.simplefilter('ignore', RuntimeWarning)
        precip_arr = np.stack([np.nanmean(np.stack(by_hour[h]), axis=0) for h in hourly_times])

    ds_out = xr.Dataset(
        {'precipitation': (['time', 'latitude', 'longitude'], precip_arr,
                           {'long_name': 'IMERG calibrated precipitation rate',
                            'units': 'mm/hr'})},
        coords={
            'time':      hourly_times,
            'latitude':  ERA5_LATS,
            'longitude': ERA5_LONS,
        }
    )

    # ── Coverage check: write every hour of the period, NaN where IMERG had nothing ──
    expected = pd.date_range(pd.Timestamp(time_range.start).floor('D'),
                             pd.Timestamp(t_end).floor('h'), freq='h')
    ds_out = ds_out.reindex(time=expected)
    missing = np.isnan(ds_out['precipitation'].values).all(axis=(1, 2))
    if missing.any():
        print(f"  {label}: WARNING — {int(missing.sum())} of {len(expected)} hours have no IMERG data (kept as NaN). "
              f"IMERG Final appears ~3.5 months after the fact; re-run later with --overwrite if this is recent.",
              flush=True)
    ds_out.attrs['regrid'] = regrid
    ds_out.attrs['missing_hours'] = int(missing.sum())

    ds_out.to_netcdf(out_path + '.part', format='NETCDF4')
    os.replace(out_path + '.part', out_path)              # a month counts only when complete
    for f in files:                                       # raw granules are no longer needed
        if os.path.exists(f):
            os.remove(f)
    return out_path


def _months_in_file(path):
    """Months that are complete (every hour present, not all NaN) in an existing yearly file."""
    if not os.path.exists(path):
        return set()
    with xr.open_dataset(path) as f:
        t = pd.DatetimeIndex(f['time'].values)
        has = ~np.isnan(f['precipitation'].values).all(axis=(1, 2))
    per = pd.Series(has, index=t).groupby(t.to_period('M')).sum()
    return {p.month for p, n in per.items() if n == p.days_in_month * 24}


def merge_into_yearly(monthly_paths, path, replace=True):
    """Merge monthly files into the yearly file (created if missing). The new months replace the same
    hours in the file (a month is only downloaded when it is missing or incomplete there, or with
    --overwrite). Written to a temporary file first, so an interrupted run never damages the file."""
    new = xr.open_mfdataset(sorted(monthly_paths), combine='by_coords')
    old = xr.open_dataset(path, chunks={'time': 168}) if os.path.exists(path) else None
    if old is not None:
        if replace:
            old = old.sel(time=~old['time'].isin(new['time'].values))
        both = xr.concat([old, new], dim='time', data_vars='all', coords='minimal', compat='override', join='outer')
    else:
        both = new
    both = both.sortby('time')
    _, first = np.unique(both['time'].values, return_index=True)
    both = both.isel(time=np.sort(first))
    both.attrs['missing_hours'] = int(np.isnan(both['precipitation'].values).all(axis=(1, 2)).sum())
    tmp = path + '.tmp'
    with dask.config.set(scheduler='synchronous'):       # no threads: can hang on some netCDF/HDF5 builds
        both.to_netcdf(tmp, format='NETCDF4', encoding={'precipitation': {'zlib': True, 'complevel': 1,
                                                                      'chunksizes': (24,) + both['precipitation'].shape[1:]},
                                                    'time': {'units': 'seconds since 1970-01-01', 'dtype': 'int64'}})
    new.close()
    if old is not None:
        old.close()
    os.replace(tmp, path)
    months = sorted(set(pd.DatetimeIndex(both['time'].values).month))
    print(f"    {os.path.basename(path)}: months {months[0]}–{months[-1]} ({len(months)}), {both.sizes['time']} hours, "
          f"{both.attrs['missing_hours']} without data", flush=True)


if __name__ == "__main__":
    import sys, time
    parser = argparse.ArgumentParser(description='Download IMERG Final and regrid to the ERA5 grid.')
    parser.add_argument('--years', type=int, nargs='+', default=list(range(2013, 2026)),
                        help='years to download (default: 2013–2025)')
    parser.add_argument('--months', type=int, nargs='+', default=[9, 10, 11, 12],
                        help='months to download, 1–12 (default: 9 10 11 12, as before)')
    parser.add_argument('--overwrite', action='store_true',
                        help='download the months again even if they are already in the yearly file')
    parser.add_argument('--regrid', choices=['conservative', 'bilinear'], default=REGRID,
                        help='regridding method (default: conservative)')
    args = parser.parse_args()
    months = sorted(set(args.months))

    # only one copy at a time (two copies would write the same files)
    os.makedirs('data', exist_ok=True)
    lock = 'data/.download_imerg.lock'
    if os.path.exists(lock):
        pid = int(open(lock).read().strip() or 0)
        if pid and os.path.exists(f'/proc/{pid}') and 'download_imerg' in open(f'/proc/{pid}/cmdline').read():
            sys.exit(f"Another IMERG download is already running (process {pid}). Stop it first with:  kill {pid}")
    open(lock, 'w').write(str(os.getpid()))

    plan = {y: [m for m in months if m not in (set() if args.overwrite else _months_in_file(f'data/imerg_hourly_{y}.nc'))]
            for y in args.years}
    total = sum(len(v) for v in plan.values())
    print(f"IMERG download (regrid: {args.regrid})\n  years {args.years[0]}–{args.years[-1]}, months {months}: "
          f"{total} month(s) to download\n  into data/imerg_hourly_<year>.nc\n"
          "  Ctrl+C stops it safely; run the same command again to continue.", flush=True)

    t_all, done, spent = time.time(), 0, 0.0
    try:
        if total:
            earthaccess.login(strategy="netrc")      # credentials from ~/.netrc, no prompt
        for year in args.years:
            year_path = f'data/imerg_hourly_{year}.nc'
            todo = plan[year]
            print(f"\n── {year} " + '─' * 60, flush=True)
            if not todo:
                print(f"  months {months} are already in {year_path} — nothing to do.", flush=True)
                continue
            print(f"  to download: months {todo}", flush=True)
            t_year, monthly = time.time(), []
            for m in todo:
                t0 = time.time()
                last = pd.Timestamp(year, m, 1).days_in_month
                mpath = f'data/imerg_hourly_{year}_{m:02d}.nc'          # one month; kept until merged
                existed = os.path.exists(mpath) and not args.overwrite
                got = download_imerg(time_range=slice(f'{year}-{m:02d}-01', f'{year}-{m:02d}-{last}'),
                                     overwrite=args.overwrite, regrid=args.regrid, out_path=mpath)
                if existed:
                    total -= 1
                elif got:
                    done += 1
                    spent += time.time() - t0
                    left = total - done
                    print(f"  {pd.Timestamp(year, m, 1):%b %Y}: done in {_dur(time.time() - t0)}  —  {done}/{total} "
                          f"months downloaded" + (f", about {_dur(left * spent / done)} left" if left else ''), flush=True)
                if got:
                    monthly.append(got)
            if monthly:
                print(f"  merging {len(monthly)} month(s) into {year_path} ...", flush=True)
                merge_into_yearly(monthly, year_path)
                for f in monthly:
                    os.remove(f)
            print(f"  {year} done in {_dur(time.time() - t_year)}", flush=True)
        print(f"\nAll done in {_dur(time.time() - t_all)}.", flush=True)
    except KeyboardInterrupt:
        print("\n\nStopped (Ctrl+C). Months that finished are kept; the yearly files are unchanged unless their "
              "merge had finished.\nRun the same command again to continue.", flush=True)
        sys.exit(130)
    finally:
        if os.path.exists(lock):
            os.remove(lock)
