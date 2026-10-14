"""
checks.py — is it the model or the inputs?

For the biggest test-set lightning events, shows what the ERA5 inputs looked like, and checks
whether any year (e.g. 2025) has missing / anomalous input data.

Usage:
    python checks.py                                        # the run currently set in train_unet.py
    python checks.py unet_12h_log_mse_OND_predictions.nc
    python checks.py unet_6h_mse_OND_predictions.nc        # any window length / loss
Outputs: printed tables + case_check_inputs.png in the run's figure folder (same folder as jones_figures.py)
"""
import os, sys, ast, warnings
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

DATA_DIR  = '/home/ec2-user/ML_Thesis_with_DL/data'
FIG_DIR   = '/home/ec2-user/ML_Thesis_with_DL/figures'
from run_info import default_pred_file, run_folder
PRED_FILE = sys.argv[1] if len(sys.argv) > 1 else default_pred_file()    # default: run set in train_unet.py
PRED_PATH = PRED_FILE if os.path.isabs(PRED_FILE) else os.path.join(DATA_DIR, PRED_FILE)
if not os.path.exists(PRED_PATH):
    sys.exit(f"{PRED_PATH} not found — run train_unet.py with these settings first, or pass a file")

SHOW      = ['precipitation', 'total_column_cloud_ice_water', 'cape', 'total_totals_index']   # maps (if present)
N_CASES   = 4
STORM_MIN = 10.0      # strokes per window: pixels counted as "inside a storm"

# ── Load ─────────────────────────────────────────────────────────────────────
pred = xr.open_dataset(PRED_PATH)
FIG_DIR = os.path.join(FIG_DIR, run_folder(pred.attrs, PRED_PATH))   # same folder as jones_figures.py uses
os.makedirs(FIG_DIR, exist_ok=True)
AGG  = int(pred.attrs.get('agg_hours', 12))
PREP_PATH = os.path.join(DATA_DIR, f'jones_israel_{AGG}h_OND.nc')     # cache written by train_unet.py
ds   = xr.open_dataset(PREP_PATH).sel(lat=pred.lat, lon=pred.lon).load()
# inputs = the ones this run was trained with (recorded in the predictions file), else whatever the cache holds
try:
    import json as _json
    FEATURES = [v for v in _json.loads(pred.attrs['features']) if v in ds]
except Exception:
    FEATURES = [v for v in ds.data_vars if v != 'ltg']
SHOW = [v for v in SHOW if v in ds] or FEATURES[:4]
# short, unique column names (cutting names to N letters made the ice and liquid water columns collide)
SHORT = {'precipitation': 'precip', 'total_totals_index': 'TT', 'total_column_cloud_ice_water': 'tciw',
         'total_column_cloud_liquid_water': 'tclw', 'k_index': 'K', 'relative_humidity': 'rh'}
short = lambda v: SHORT.get(v, v if len(v) <= 12 else v[:12])
assert len({short(v) for v in FEATURES}) == len(FEATURES), 'short column names collide — extend SHORT'
print(f"Predictions: {PRED_PATH}  (loss={pred.attrs.get('loss', '?')}, "
      f"calibration={pred.attrs.get('calibration', '?')})\n")

years = ds.time.dt.year.values

# Training years = everything not used for validation/test/excluded (read from the run that made
# the predictions, so this stays correct when the split changes).
def _years_attr(name, default):
    try:
        return list(ast.literal_eval(pred.attrs[name]))
    except (KeyError, ValueError, SyntaxError):
        return default
held_out    = _years_attr('val_years', [2021]) + _years_attr('test_years', [2022, 2024]) \
              + _years_attr('excluded_years', [2023, 2025])
TRAIN_YEARS = [int(y) for y in np.unique(years) if y not in held_out]
print(f"Training years used as the reference: {TRAIN_YEARS}\n")
ltg   = ds['ltg'].fillna(0)

# ── 1. Input sanity per year ─────────────────────────────────────────────────
print("=" * 100)
print("1. INPUTS PER YEAR — NaN fraction / fraction exactly zero (any year standing out = data problem)")
print("=" * 100)
rows = []
for yr in np.unique(years):
    d = ds.sel(time=ds.time.dt.year == yr)
    r = {'year': yr}
    for v in FEATURES:
        a = d[v].values
        r[f'{short(v)}_nan'] = np.isnan(a).mean()
        r[f'{short(v)}_zero'] = (a == 0).mean()
    rows.append(r)
t1 = pd.DataFrame(rows).set_index('year')
with pd.option_context('display.width', 250, 'display.max_columns', 40, 'display.float_format', '{:.3f}'.format):
    print(t1[[c for c in t1.columns if c.endswith('_nan')]])
    print()
    print(t1[[c for c in t1.columns if c.endswith('_zero')]])

# ── 2. What do inputs look like INSIDE storms, per year? ─────────────────────
print("\n" + "=" * 100)
print(f"2. MEAN INPUT VALUE AT STORM PIXELS (obs ≥ {STORM_MIN:g} strokes/{AGG}h), per year")
print("   If test years are far below training years, the model never 'sees' those storms.")
print("=" * 100)
storm = (ltg >= STORM_MIN)
rows = []
for yr in np.unique(years):
    sel = (ds.time.dt.year == yr)
    m = storm.sel(time=sel)
    r = {'year': yr, 'n_storm_px': int(m.sum())}
    for v in FEATURES:
        r[short(v)] = float(ds[v].sel(time=sel).where(m).mean())
    rows.append(r)
t2 = pd.DataFrame(rows).set_index('year')
train = t2.loc[t2.index.isin(TRAIN_YEARS)]
z = (t2 - train.mean()) / train.std()
with pd.option_context('display.width', 250, 'display.max_columns', 40, 'display.float_format', '{:.3g}'.format):
    print(t2)
    print(f"\n   z-score vs training years {TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]} (|z| > 3 is suspicious):")
    print(z.drop(columns='n_storm_px').round(1))

# ── 3. Skill per test year ───────────────────────────────────────────────────
def fss(o, p, w=3, thr=1.0):
    of = uniform_filter((o >= thr).astype(float), w, mode='constant')
    pf = uniform_filter((p >= thr).astype(float), w, mode='constant')
    den = np.sum(of ** 2 + pf ** 2)
    return np.nan if den == 0 else 1 - np.sum((of - pf) ** 2) / den

obs_np  = np.nan_to_num(pred['obs_count'].values)
pred_np = np.nan_to_num(pred['pred_count'].values)
tyears  = pd.DatetimeIndex(pred.time.values).year
print("\n" + "=" * 100)
print("3. TEST SKILL PER YEAR")
print("=" * 100)
for yr in np.unique(tyears):
    i = tyears == yr
    f = np.array([fss(obs_np[k], pred_np[k]) for k in np.where(i)[0]])
    print(f"  {yr}: mean obs {obs_np[i].mean():.3f}  mean pred {pred_np[i].mean():.3f}  "
          f"event rate obs/pred {(obs_np[i] >= 1).mean():.3f}/{(pred_np[i] >= 1).mean():.3f}  "
          f"mean FSS {np.nanmean(f):.3f}")

# ── 4. Figure: inputs for the biggest events ─────────────────────────────────
totals = obs_np.sum(axis=(1, 2))
cases  = np.argsort(totals)[::-1][:N_CASES]
lats, lons = pred.lat.values, pred.lon.values

# colour limits from TRAINING storms, so an "empty" test field is obvious
lims = {}
for v in SHOW:
    a = ds[v].sel(time=ds.time.dt.year.isin(TRAIN_YEARS)).values
    lims[v] = (np.nanpercentile(a, 1), np.nanpercentile(a, 99.5))

try:
    import cartopy.crs as ccrs, cartopy.feature as cfeature
    kw = {'subplot_kw': {'projection': ccrs.PlateCarree()}}
except ImportError:
    ccrs = None; kw = {}

ncol = 2 + len(SHOW)
fig, axs = plt.subplots(len(cases), ncol, figsize=(3.6 * ncol, 2.1 * len(cases) + 0.8),
                        squeeze=False, **kw)
norm_l = mcolors.LogNorm(vmin=1, vmax=max(1.1, np.percentile(obs_np[cases][obs_np[cases] > 0], 99)))
cm = plt.cm.RdYlBu_r.copy(); cm.set_bad('white')

for r, t in enumerate(cases):
    when = pd.Timestamp(pred.time.values[t])
    tsel = ds.sel(time=pred.time.values[t])
    panels = [('Observed ltg', obs_np[t], norm_l, cm),
              ('Predicted ltg', pred_np[t], norm_l, cm)] + \
             [(v, tsel[v].values, mcolors.Normalize(*lims[v]), 'viridis') for v in SHOW]
    for c, (name, field, norm, cmap) in enumerate(panels):
        ax = axs[r, c]
        f = np.where(field >= 1, field, np.nan) if c < 2 else field
        kwp = {'transform': ccrs.PlateCarree()} if ccrs else {}
        im = ax.pcolormesh(lons, lats, f, norm=norm, cmap=cmap, shading='auto', **kwp)
        if ccrs:
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{name}\n{when:%Y-%m-%d %HZ}" if r == 0 or c == 0 else name, fontsize=8)
        if r == len(cases) - 1:
            fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.06, pad=0.04)

fig.suptitle('Biggest test events: lightning vs ERA5 inputs (feature colours fixed to training-period range)',
             fontsize=11)
fig.tight_layout()
out = os.path.join(FIG_DIR, 'case_check_inputs.png')
fig.savefig(out, dpi=130, bbox_inches='tight')
print(f"\nSaved {out}")