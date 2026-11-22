"""
check_data.py — are there gaps in the real data that are stored as zeros instead of "missing"?

Run on EC2 from the repo folder (read-only, changes nothing):
    python check_data.py            # quick: file headers, line counts, first/last stroke per day
    python check_data.py --hourly   # also counts WWLLN strokes per hour (slower, ~10–30 min)

It checks, for Oct–Dec of every year:
  1. WWLLN .loc files: missing days, empty or cut-off files, unusually small files, duplicates.
  2. The merged hourly file: hours with IMERG exactly 0 over the whole grid, hours with no WWLLN
     data, and days where WWLLN has no lightning at all in the domain.
  3. Which IMERG regridding each year used, and whether the per-year files come from the current code.
  4. If the ILDN files are in data/: days where ILDN sees many strokes around Israel but WWLLN none
     (that is how a WWLLN gap shows up, e.g. the second half of December 2023).

Output: a summary on screen and every flagged day in data/check_data_flags.csv.
"""
import argparse, glob, os, re
import numpy as np
import pandas as pd
import xarray as xr

DATA_DIR  = 'data'
WWLN_DIR  = os.path.join(DATA_DIR, 'WWLN')
YEARS     = list(range(2013, 2026))
MONTHS    = [10, 11, 12]
BOX       = (30.0, 34.5, 32.0, 36.5)        # Israel box used by ildn_eval.py
MIN_GLOBAL_STROKES_PER_HOUR = 100           # same rule as process_wwln.py

parser = argparse.ArgumentParser()
parser.add_argument('--hourly', action='store_true', help='count WWLLN strokes per hour (slower)')
args = parser.parse_args()
flags = []

def flag(day, source, problem):
    flags.append({'day': pd.Timestamp(day).date(), 'source': source, 'problem': problem})

def head(title):
    print(f"\n{'=' * 90}\n  {title}\n{'=' * 90}")

# ── 1. WWLLN .loc files ─────────────────────────────────────────────────────────
head("1. WWLLN .loc files (Oct–Dec)")
TIME_RE = re.compile(rb'(\d{4})/(\d{2})/(\d{2}),\s*(\d{2}):(\d{2}):(\d{2})')

def first_last_count(path):
    """Line count and the times of the first and last stroke, without parsing the whole file."""
    n, first = 0, None
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            n += chunk.count(b'\n')
        f.seek(0)
        for line in f:
            if TIME_RE.search(line):
                first = line
                break
        size = f.seek(0, 2)
        f.seek(max(0, size - 8192))
        tail = [l for l in f.read().splitlines() if TIME_RE.search(l)]
    last = tail[-1] if tail else None
    to_t = lambda l: pd.Timestamp(*map(int, TIME_RE.search(l).groups())) if l is not None else pd.NaT
    return n, to_t(first), to_t(last)

rows = []
for y in YEARS:
    files = sorted(glob.glob(os.path.join(WWLN_DIR, str(y), f'*{y}', f'A{y}????.loc')))
    names = [os.path.basename(f) for f in files]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        print(f"  {y}: {len(dup)} day file(s) found twice (e.g. {dup[:3]}) — counted twice by older process_wwln.py")
        for d in dup:
            flag(pd.Timestamp(d[1:9]), 'WWLLN', 'day file appears twice')
    by_day = {}
    for f in files:
        d = pd.Timestamp(os.path.basename(f)[1:9])
        if d.month in MONTHS and d not in by_day:
            by_day[d] = f
    days = pd.date_range(f'{y}-{MONTHS[0]:02d}-01', f'{y}-{MONTHS[-1]:02d}-31', freq='D')
    for d in days:
        if d not in by_day:
            rows.append({'day': d, 'lines': 0, 'first': pd.NaT, 'last': pd.NaT, 'missing': True})
            continue
        n, t0, t1 = first_last_count(by_day[d])
        rows.append({'day': d, 'lines': n, 'first': t0, 'last': t1, 'missing': False})
loc = pd.DataFrame(rows)
if loc.empty or loc['missing'].all():
    print(f"  no .loc files found under {WWLN_DIR} — skipping this part")
else:
    loc['month'] = loc['day'].dt.to_period('M')
    med = loc[~loc.missing].groupby('month')['lines'].median()
    loc['frac_of_median'] = loc['lines'] / loc['month'].map(med)
    for _, r in loc.iterrows():
        if r.missing:
            flag(r.day, 'WWLLN', 'no .loc file (becomes NaN = missing: OK)')
        elif r.lines == 0:
            flag(r.day, 'WWLLN', 'EMPTY .loc file (older process_wwln.py counted it as "no lightning")')
        else:
            problems = []
            if pd.notna(r['last']) and r['last'].hour < 22:
                problems.append(f"last stroke at {r['last']:%H:%M} UTC (file cut off?)")
            if pd.notna(r['first']) and r['first'].hour > 1:
                problems.append(f"first stroke at {r['first']:%H:%M} UTC")
            if r.frac_of_median < 0.3:
                problems.append(f"only {r.frac_of_median:.0%} of the month's median size")
            if problems:
                flag(r.day, 'WWLLN', '; '.join(problems))
    per_year = loc.groupby(loc.day.dt.year).agg(days=('day', 'size'), missing=('missing', 'sum'))
    per_year['median_lines'] = loc[~loc.missing].groupby(loc[~loc.missing].day.dt.year)['lines'].median()
    per_year['flagged'] = [sum(1 for f in flags if f['source'] == 'WWLLN' and f['day'].year == y
                               and 'OK' not in f['problem']) for y in per_year.index]
    print(per_year.to_string())

if args.hourly and not loc.empty:
    print("\n  Counting strokes per hour (only hours with < "
          f"{MIN_GLOBAL_STROKES_PER_HOUR} strokes globally are listed)...")
    for y in YEARS:
        for f in sorted(glob.glob(os.path.join(WWLN_DIR, str(y), f'*{y}', f'A{y}????.loc'))):
            d = pd.Timestamp(os.path.basename(f)[1:9])
            if d.month not in MONTHS:
                continue
            t = pd.read_csv(f, header=None, usecols=[0, 1], names=['d', 't'], dtype=str, skipinitialspace=True)
            hrs = pd.to_datetime(t['d'].str.strip() + ' ' + t['t'].str.strip().str[:8],
                                 format='%Y/%m/%d %H:%M:%S', errors='coerce').dt.hour
            counts = hrs.value_counts().reindex(range(24), fill_value=0)
            short = counts[counts < MIN_GLOBAL_STROKES_PER_HOUR]
            if len(short):
                flag(d, 'WWLLN', f"{len(short)} hour(s) with < {MIN_GLOBAL_STROKES_PER_HOUR} strokes globally: "
                                 f"{list(short.index)}")

# ── 2. Per-year source files ────────────────────────────────────────────────────
head("2. Source files per year")
src = []
for y in YEARS:
    r = {'year': y}
    p = os.path.join(DATA_DIR, f'imerg_hourly_{y}.nc')
    if os.path.exists(p):
        with xr.open_dataset(p) as d:
            r['imerg_regrid'] = d.attrs.get('regrid', 'bilinear? (no attribute: old file)')
            r['imerg_missing_h'] = d.attrs.get('missing_hours', '?')
    p = os.path.join(DATA_DIR, f'wwln_on_era5_grid_{y}.nc')
    if os.path.exists(p):
        with xr.open_dataset(p) as d:
            r['wwln_flag'] = ('per hour (new)' if 'availability' in d.attrs else
                              'per day (old)' if 'data_available' in d else 'NONE (old)')
    p = os.path.join(DATA_DIR, f'jones_israel_{y}.nc')
    if os.path.exists(p):
        with xr.open_dataset(p) as d:
            r['year_file'] = 'current' if 'missing_data' in d.attrs else 'OLD (zeros for gaps?)'
    src.append(r)
src = pd.DataFrame(src).set_index('year')
print(src.to_string())
if 'imerg_regrid' in src and src['imerg_regrid'].nunique() > 1:
    print("  WARNING: IMERG regridding differs between years — precipitation is not comparable across years")

# ── 3. The merged hourly file ───────────────────────────────────────────────────
head("3. Merged hourly file data/jones_israel_all_years.nc (Oct–Dec)")
hp = os.path.join(DATA_DIR, 'jones_israel_all_years.nc')
ww_box = None
if os.path.exists(hp):
    ds = xr.open_dataset(hp)
    ds = ds.sel(time=ds.time.dt.month.isin(MONTHS))
    lat_m = (ds.lat >= 30) & (ds.lat <= 43); lon_m = (ds.lon >= 10) & (ds.lon <= 39)
    dom = ds[['precipitation', 'ltg']].isel(lat=np.where(lat_m)[0], lon=np.where(lon_m)[0])
    pr_nan  = dom['precipitation'].isnull().all(['lat', 'lon']).to_series()
    pr_zero = ((dom['precipitation'].fillna(0) == 0).all(['lat', 'lon'])).to_series() & ~pr_nan
    lt_nan  = dom['ltg'].isnull().all(['lat', 'lon']).to_series()
    lt_day  = dom['ltg'].sum(['lat', 'lon'], min_count=1).to_series().resample('D').sum(min_count=1)
    lt_day  = lt_day[lt_day.index.month.isin(MONTHS)]
    summ = pd.DataFrame({
        'precip_missing_h': pr_nan.groupby(pr_nan.index.year).sum(),
        'precip_all_zero_h': pr_zero.groupby(pr_zero.index.year).sum(),
        'ltg_missing_h': lt_nan.groupby(lt_nan.index.year).sum(),
        'days_no_lightning_in_domain': (lt_day == 0).groupby(lt_day.index.year).sum(),
    })
    print(summ.to_string())
    for t in pr_zero[pr_zero].index.normalize().unique():
        flag(t, 'IMERG', 'precipitation exactly 0 over the whole domain for some hours (zero-filled gap?)')
    for t in lt_day[lt_day == 0].index:
        flag(t, 'WWLLN', 'no lightning anywhere in 30–43N 10–39E all day (possible, but check)')
    bb = ds['ltg'].sel(lat=slice(BOX[1], BOX[0]) if ds.lat[0] > ds.lat[-1] else slice(BOX[0], BOX[1]),
                       lon=slice(BOX[2], BOX[3]))
    ww_box = bb.sum(['lat', 'lon'], min_count=1).to_series().resample('D').sum(min_count=1)
else:
    print(f"  {hp} not found")

# ── 4. WWLLN vs ILDN around Israel ──────────────────────────────────────────────
head("4. WWLLN vs ILDN in the Israel box (days where ILDN sees ≥ 300 strokes)")
ildn_files = sorted(glob.glob(os.path.join(DATA_DIR, '*_for_yoav_including_cloud.txt')))
if ildn_files and ww_box is not None:
    il = pd.concat([pd.read_csv(f, sep=r'\s+', header=None, usecols=[0, 2, 3], names=['d', 'lat', 'lon'])
                    for f in ildn_files])
    il = il[(il.lat >= BOX[0]) & (il.lat <= BOX[1]) & (il.lon >= BOX[2]) & (il.lon <= BOX[3])]
    il_day = il.groupby(pd.to_datetime(il['d'])).size()
    il_day = il_day[il_day.index.month.isin(MONTHS)]
    both = pd.DataFrame({'ILDN': il_day, 'WWLLN': ww_box}).dropna(subset=['ILDN'])
    big = both[both.ILDN >= 300]
    sus = big[big.WWLLN.fillna(0) == 0]
    print(f"  {len(big)} days with ≥ 300 ILDN strokes; WWLLN shows 0 strokes on {len(sus)} of them:")
    print(sus.to_string() if len(sus) else "  none")
    for t in sus.index:
        flag(t, 'WWLLN', f"ILDN {int(sus.loc[t, 'ILDN'])} strokes in the Israel box, WWLLN 0 (WWLLN gap?)")
    ratio = both.groupby(both.index.year)[['ILDN', 'WWLLN']].sum()
    ratio['WWLLN/ILDN'] = (ratio.WWLLN / ratio.ILDN).round(2)
    print("\n  Yearly totals in the box:\n" + ratio.to_string())
else:
    print("  ILDN files or merged file not found — skipped")

# ── Summary ─────────────────────────────────────────────────────────────────────
head("SUMMARY")
out = os.path.join(DATA_DIR, 'check_data_flags.csv')
fl = pd.DataFrame(flags, columns=['day', 'source', 'problem']).sort_values(['day', 'source'])
fl.to_csv(out, index=False)
serious = fl[~fl.problem.str.contains('OK|possible, but check')]
print(f"  {len(fl)} flagged days in total, {len(serious)} that need attention → {out}")
if len(serious):
    print(serious.groupby([pd.to_datetime(serious.day).dt.year, 'source']).size().to_string())
    print("\n  If WWLLN days are flagged: rerun process_wwln.py for those years (it now marks such hours as\n"
          "  missing), then build_jones_netcdf.py --years <years> --overwrite, then retrain.")
