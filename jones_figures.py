"""
jones_figures.py
----------------
Evaluation figures for the Israel-domain U-Net (OND season, 12-hourly, log-target MSE model).

Reads the output of unet_12h.py (pred_count [+ pred_count_raw] + obs_count, strokes per 12 h).

Figures produced:
  Fig 1 — Monthly mean maps: Observed vs Predicted (Oct / Nov / Dec)
  Fig 2 — Monthly FSS bar chart by window size (Oct / Nov / Dec)
  Fig 3 — Metrics summary table
  Fig 4 — Jones-style: test-period mean lightning per cell, modelled vs observed
           (rows all / land / ocean; columns R14 baseline, U-Net raw, U-Net calibrated)
  Fig 4b — Observed vs predicted for every pixel × 12-h window
  Fig 5 — Pearson & Spearman correlation of each ERA5 variable vs 12-h lightning (TRAINING years,
           so it can be used to choose inputs without looking at the test set)
  Fig 6 — Case studies: the most active 12-h windows in the test set

Usage (on EC2):
    cd /home/ec2-user/ML_Thesis_with_DL
    python jones_figures.py                                                  # poisson, calibrated
    python jones_figures.py unet_12h_poisson_OND_predictions.nc pred_count_raw
    python jones_figures.py unet_12h_log_mse_OND_predictions.nc

Outputs saved to:  figures/<run name>/
"""

import xarray as xr
import numpy as np
import pandas as pd
import os, warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import uniform_filter, gaussian_filter
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import precision_score, recall_score, f1_score, r2_score

# ── Paths ────────────────────────────────────────────────────────────────────
import sys
DATA_DIR    = '/home/ec2-user/ML_Thesis_with_DL/data'
# Usage: python jones_figures.py [predictions_file] [pred_variable]
#   e.g. python jones_figures.py unet_12h_poisson_OND_predictions.nc pred_count_raw
PRED_FILE   = sys.argv[1] if len(sys.argv) > 1 else 'unet_12h_poisson_OND_predictions.nc'
PRED_VAR    = sys.argv[2] if len(sys.argv) > 2 else 'pred_count'      # 'pred_count' or 'pred_count_raw'
PRED_PATH   = PRED_FILE if os.path.isabs(PRED_FILE) else os.path.join(DATA_DIR, PRED_FILE)
RUN_NAME    = os.path.basename(PRED_PATH).replace('_OND_predictions.nc', '').replace('.nc', '') \
              + ('' if PRED_VAR == 'pred_count' else f'_{PRED_VAR}')
FIG_DIR     = os.path.join('/home/ec2-user/ML_Thesis_with_DL/figures', RUN_NAME)   # one folder per run
HOURLY_PATH = os.path.join(DATA_DIR, 'jones_israel_all_years.nc')         # only needed for Fig 5
os.makedirs(FIG_DIR, exist_ok=True)

# ── Settings ─────────────────────────────────────────────────────────────────
AGG_HOURS       = 12
EVENT_THRESHOLD = 1.0        # a pixel "has lightning" if ≥ 1 stroke in the 12-h window (obs AND pred)
WINDOW_SIZE     = 3
WINDOW_SIZES    = [1, 3, 5]
SMOOTH_SIGMA    = 0.0        # extra Gaussian smoothing on Fig 1 maps (0 = show the real fields)
MAP_COLOR_SCALE = 'linear'   # 'linear' or 'log' for Fig 1
N_CASES         = 4          # number of case-study windows in Fig 6
OND_MONTHS      = [10, 11, 12]
OND_LABELS      = ['October', 'November', 'December']

# ── Load predictions + observations (both strokes per 12 h) ─────────────────
print("Loading predictions...", flush=True)
ds_pred    = xr.open_dataset(PRED_PATH)
count_da   = ds_pred[PRED_VAR]
print(f"  file: {PRED_PATH}\n  variable: {PRED_VAR}  →  figures in {FIG_DIR}", flush=True)
test_times = ds_pred['time'].values
lats, lons = ds_pred['lat'].values, ds_pred['lon'].values
n_lat, n_lon = len(lats), len(lons)

if 'obs_count' in ds_pred:
    obs_da = ds_pred['obs_count']
else:
    # Fallback: build 12-h sums from the hourly file (window starts at each test timestamp)
    print("  obs_count not in prediction file — summing hourly observations...", flush=True)
    ltg_h = xr.open_dataset(HOURLY_PATH)['ltg'].sel(lat=lats, lon=lons, method='nearest')
    obs_da = xr.concat([ltg_h.sel(time=slice(t, t + np.timedelta64(AGG_HOURS - 1, 'h'))).sum('time')
                        for t in test_times], dim=pd.Index(test_times, name='time'))

count_np = np.nan_to_num(count_da.values.astype(float))   # (T, lat, lon) predicted strokes / 12 h
obs_np   = np.nan_to_num(obs_da.values.astype(float))     # (T, lat, lon) observed strokes / 12 h

obs_binary_np = (obs_np   >= EVENT_THRESHOLD).astype(float)
pred_np       = (count_np >= EVENT_THRESHOLD).astype(float)

months_arr = pd.DatetimeIndex(test_times).month
TEST_LABEL = ', '.join(str(y) for y in sorted(set(pd.DatetimeIndex(test_times).year)))   # e.g. '2022, 2024'
step = pd.Series(np.diff(test_times)).mode()[0]
print(f"Test period : {pd.Timestamp(test_times[0]).date()} → {pd.Timestamp(test_times[-1]).date()}")
print(f"Grid        : {n_lat}×{n_lon}, Timesteps: {len(test_times)}, step {step}", flush=True)
print(f"Event rate  : obs {obs_binary_np.mean():.4f}  pred {pred_np.mean():.4f}", flush=True)

# ── FSS ──────────────────────────────────────────────────────────────────────
def compute_fss(obs_bin, pred_bin, window_size):
    """FSS for one timestep; inputs are already binary (0/1)."""
    obs_frac  = uniform_filter(obs_bin,  size=window_size, mode='constant', cval=0)
    pred_frac = uniform_filter(pred_bin, size=window_size, mode='constant', cval=0)
    num = np.sum((pred_frac - obs_frac) ** 2)
    den = np.sum(pred_frac ** 2 + obs_frac ** 2)
    return np.nan if den == 0 else 1 - num / den

print("Computing FSS...", flush=True)
fss_by_window = {w: np.array([compute_fss(obs_binary_np[i], pred_np[i], w)
                              for i in range(len(test_times))]) for w in WINDOW_SIZES}
fss_arr   = fss_by_window[WINDOW_SIZE]
fss_valid = fss_arr[np.isfinite(fss_arr)]

def month_fss(m, w):
    s = fss_by_window[w][months_arr == m]
    s = s[np.isfinite(s)]
    return (s.mean() if len(s) else np.nan,
            s.std() / np.sqrt(len(s)) if len(s) > 1 else 0.0)

# ── Metrics ──────────────────────────────────────────────────────────────────
obs_flat, pred_flat = obs_binary_np.ravel().astype(int), pred_np.ravel().astype(int)
precision = precision_score(obs_flat, pred_flat, zero_division=0)
recall    = recall_score(obs_flat, pred_flat, zero_division=0)
f1        = f1_score(obs_flat, pred_flat, zero_division=0)
tp = np.sum((obs_flat == 1) & (pred_flat == 1))
csi = tp / max(1, np.sum((obs_flat == 1) | (pred_flat == 1)))

o_all, p_all = obs_np.ravel(), count_np.ravel()
rmse   = float(np.sqrt(np.mean((p_all - o_all) ** 2)))
bias   = float(p_all.mean() - o_all.mean())
r2_all = r2_score(o_all, p_all)
r2_log = r2_score(np.log1p(o_all), np.log1p(p_all))

obs_mean_2d  = obs_np.mean(axis=0)            # per-cell mean over test period
pred_mean_2d = count_np.mean(axis=0)
r2_clim = r2_score(obs_mean_2d.ravel(), pred_mean_2d.ravel())
r_clim  = pearsonr(obs_mean_2d.ravel(), pred_mean_2d.ravel())[0]

print(f"Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}  CSI={csi:.3f}", flush=True)
print(f"Mean FSS(3×3)={np.nanmean(fss_valid):.3f}  R²(clim)={r2_clim:.3f}  R²(log1p)={r2_log:.3f}", flush=True)

# ── Map helpers ───────────────────────────────────────────────────────────────
lon_min, lon_max = float(lons.min()), float(lons.max())
lat_min, lat_max = float(lats.min()), float(lats.max())
disp_extent = [lon_min - 0.3, lon_max + 0.3, lat_min - 0.3, lat_max + 0.3]

cmap_ltg = plt.cm.RdYlBu_r.copy()
cmap_ltg.set_bad(color='white')
cmap_ltg.set_under(color='white')

def fmt_deg(x, pos):
    return f"{x:.1f}°"

def add_colorbar(fig, ax, im, label):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=9)
    return cb

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY, HAS_GPD = True, False
    print("  Using cartopy for borders", flush=True)
except ImportError:
    HAS_CARTOPY = False
    try:
        import geopandas as gpd
        try:
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        except Exception:
            import geodatasets
            world = gpd.read_file(geodatasets.get_path('naturalearth.land'))
        HAS_GPD = True
        print("  cartopy not found — using geopandas for borders", flush=True)
    except Exception:
        HAS_GPD = False
        print("  Neither cartopy nor geopandas — plain axes", flush=True)

PROJ = ccrs.PlateCarree() if HAS_CARTOPY else None

def new_map_figure(nrows, ncols, figsize):
    kw = {'subplot_kw': {'projection': PROJ}} if HAS_CARTOPY else {}
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, **kw)
    return fig, axs

def draw_map(fig, ax, data, cmap, norm):
    """pcolormesh + borders; returns the mappable."""
    if HAS_CARTOPY:
        im = ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm, shading='auto', transform=PROJ)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor='black')
        ax.add_feature(cfeature.BORDERS,   linewidth=0.4, edgecolor='gray')
        ax.set_extent(disp_extent, crs=PROJ)
    else:
        im = ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm, shading='auto')
        if HAS_GPD:
            world.boundary.plot(ax=ax, linewidth=0.6, color='black', zorder=2)
        ax.set_xlim(disp_extent[0], disp_extent[1])
        ax.set_ylim(disp_extent[2], disp_extent[3])
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_deg))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_deg))
    return im

domain_aspect = (disp_extent[1] - disp_extent[0]) / (disp_extent[3] - disp_extent[2])


# ============================================================
# Fig 1 — Monthly mean maps: Observed vs Predicted
# ============================================================
print("Fig 1: Monthly maps...", flush=True)

nrows = len(OND_MONTHS)
map_w = 7.5
map_h = map_w / domain_aspect
fig, axs = new_map_figure(nrows, 2, (map_w * 2 + 0.6, map_h * nrows + 0.8))
fig.subplots_adjust(left=0.04, right=0.88, top=0.93, bottom=0.03, wspace=0.04,
                    hspace=0.12 if HAS_CARTOPY else 0.22)

# One colour scale shared by ALL panels so months and obs/pred are directly comparable
monthly = {}
for m in OND_MONTHS:
    idx_m = months_arr == m
    o = obs_np[idx_m].mean(axis=0)
    p = count_np[idx_m].mean(axis=0)
    if SMOOTH_SIGMA > 0:
        o, p = gaussian_filter(o, SMOOTH_SIGMA), gaussian_filter(p, SMOOTH_SIGMA)
    monthly[m] = (o, p)
all_vals = np.concatenate([np.ravel(v) for pair in monthly.values() for v in pair])
pos_vals = all_vals[all_vals > 0]
vmax = float(np.percentile(pos_vals, 99)) if pos_vals.size else 1.0
if MAP_COLOR_SCALE == 'log':
    norm1 = mcolors.LogNorm(vmin=max(vmax / 1e3, 1e-3), vmax=vmax)
else:
    norm1 = mcolors.Normalize(vmin=0, vmax=vmax)

last_im = None
for row, (m, mlabel) in enumerate(zip(OND_MONTHS, OND_LABELS)):
    o, p = monthly[m]
    n_steps = int((months_arr == m).sum())
    last_im = draw_map(fig, axs[row, 0], np.where(o > 0, o, np.nan), cmap_ltg, norm1)
    axs[row, 0].set_title(f'{mlabel} — Observed (WWLLN)  [n={n_steps}]', fontsize=11, pad=3)
    draw_map(fig, axs[row, 1], np.where(p > 0, p, np.nan), cmap_ltg, norm1)
    axs[row, 1].set_title(f'{mlabel} — U-Net Predicted', fontsize=11, pad=3)

cbar_ax = fig.add_axes([0.90, 0.05, 0.018, 0.85])
cb = fig.colorbar(last_im, cax=cbar_ax, extend='max')
cb.set_label(f'Mean lightning strokes per {AGG_HOURS} h', fontsize=10)
yrs = pd.DatetimeIndex(test_times).year
fig.suptitle(f'Monthly Mean Lightning — Observed vs Predicted ({RUN_NAME}) — Test {TEST_LABEL}',
             fontsize=13, y=0.97)
fig.savefig(os.path.join(FIG_DIR, 'fig1_monthly_maps.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved fig1_monthly_maps.png", flush=True)


# ============================================================
# Fig 2 — Monthly FSS bar chart
# ============================================================
print("Fig 2: Monthly FSS...", flush=True)
colors_w = {1: 'steelblue', 3: 'darkorange', 5: 'forestgreen'}
fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
x, width = np.arange(len(OND_MONTHS)), 0.25
for i, w in enumerate(WINDOW_SIZES):
    vals, errs = zip(*[month_fss(m, w) for m in OND_MONTHS])
    ax.bar(x + (i - 1) * width, vals, width, yerr=errs, capsize=4,
           color=colors_w.get(w, None), alpha=0.85, label=f'{w}×{w} window')
ax.set_xticks(x)
ax.set_xticklabels(['Oct', 'Nov', 'Dec'], fontsize=12)
ax.set_ylabel('Mean FSS', fontsize=11)
ax.set_xlabel('Month', fontsize=11)
ax.set_title(f'Monthly FSS by Window Size (event: ≥{EVENT_THRESHOLD:g} stroke / {AGG_HOURS} h) — '
             f'Test {TEST_LABEL}', fontsize=11)
ax.set_ylim(0, 1)
ax.axhline(0.5, color='gray', ls=':', lw=1.2, label='FSS = 0.5')
ax.legend(fontsize=10)
ax.grid(axis='y', ls='--', alpha=0.4)
fig.savefig(os.path.join(FIG_DIR, 'fig2_monthly_fss.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved fig2_monthly_fss.png", flush=True)


# ============================================================
# Fig 3 — Metrics summary table
# ============================================================
print("Fig 3: Metrics table...", flush=True)
a = ds_pred.attrs
def fss_mean(w):
    s = fss_by_window[w]; s = s[np.isfinite(s)]
    return np.mean(s) if len(s) else np.nan

row_data = [
    ['Loss',                    a.get('loss_desc', 'MSE on z-scored log1p(strokes)' if int(a.get('log_target', 1)) else 'MSE (z-scored counts)')],
    ['Calibration',             a.get('calibration', 'none') if PRED_VAR == 'pred_count' else 'none (raw output)'],
    ['Time resolution',         f'{AGG_HOURS} h'],
    ['Validation years',        a.get('val_years', '?')],
    ['Test years',              a.get('test_years', TEST_LABEL)],
    ['Excluded years',          a.get('excluded_years', 'none')],
    ['Event threshold',         f'≥ {EVENT_THRESHOLD:g} stroke / {AGG_HOURS} h'],
    ['Event rate obs / pred',   f'{obs_binary_np.mean():.3f} / {pred_np.mean():.3f}'],
    ['Precision',               f'{precision:.3f}'],
    ['Recall',                  f'{recall:.3f}'],
    ['F1 score',                f'{f1:.3f}'],
    ['CSI',                     f'{csi:.3f}'],
    ['Mean FSS (1×1)',          f'{fss_mean(1):.3f}'],
    ['Mean FSS (3×3)',          f'{fss_mean(3):.3f}'],
    ['Mean FSS (5×5)',          f'{fss_mean(5):.3f}'],
    ['Median FSS (3×3)',        f'{np.nanmedian(fss_valid):.3f}'],
] + [[f'FSS {lbl[:3]} (3×3)', f'{month_fss(m, 3)[0]:.3f}'] for m, lbl in zip(OND_MONTHS, OND_LABELS)] + [
    ['Mean strokes obs / pred', f'{o_all.mean():.3f} / {p_all.mean():.3f}  (bias {bias:+.3f})'],
    ['RMSE (strokes)',          f'{rmse:.3f}'],
    ['R² all pixels (counts)',  f'{r2_all:.3f}'],
    ['R² all pixels (log1p)',   f'{r2_log:.3f}'],
    ['R² per-cell climatology', f'{r2_clim:.3f}  (r = {r_clim:.3f})'],
]

fig, ax = plt.subplots(figsize=(7, 0.3 * len(row_data) + 0.8), constrained_layout=True)
ax.axis('off')
tbl = ax.table(cellText=row_data, colLabels=['Metric', 'Value'], loc='center', cellLoc='left')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10.5)
tbl.scale(1.3, 1.5)
for j in range(2):
    tbl[0, j].set_facecolor('#2c5f8a')
    tbl[0, j].set_text_props(color='white', fontweight='bold')
for i in range(1, len(row_data) + 1):
    for j in range(2):
        tbl[i, j].set_facecolor('#f0f4f8' if i % 2 == 0 else 'white')
ax.set_title(f'U-Net — Test Set Metrics ({RUN_NAME})', fontsize=12, fontweight='bold', pad=10)
fig.savefig(os.path.join(FIG_DIR, 'fig3_metrics_table.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved fig3_metrics_table.png", flush=True)


# ============================================================
# Fig 4 — Jones et al. style: test-period mean lightning per grid cell, modelled vs observed
#   rows: all cells / land (lsm ≥ 0.9) / ocean (lsm ≤ 0.1)
#   columns: R14 baseline (c·CAPE·P, c fitted on training years), U-Net raw, U-Net calibrated
#   filled contours = number of grid cells; white = no cells; 1:1 line with R² in the legend
# ============================================================
print("Fig 4: Jones-style climatology comparison...", flush=True)
FIG4_CMAP  = 'turbo'     # Jones-like rainbow look. 'viridis' is safer for colour-blind readers and print.
FIG4_UNITS = 'density'   # 'density' = strokes km⁻² day⁻¹ (comparable to Jones) | 'cell' = strokes per cell per 12 h
FIG4_BINS  = 40          # histogram bins per axis (log10 space)
PREP_PATH  = os.path.join(DATA_DIR, 'jones_israel_12h_OND.nc')   # 12-hourly cache written by train_unet.py

def _attr_years(name):
    import json as _j, ast as _a
    try:
        v = ds_pred.attrs[name]
        try:
            return [int(y) for y in _j.loads(v)]
        except Exception:
            return [int(y) for y in _a.literal_eval(v)]
    except Exception:
        return None

# unit conversion: mean strokes per cell per 12 h → strokes km⁻² day⁻¹
if FIG4_UNITS == 'density':
    _R = 6371.0
    _dlat, _dlon = abs(float(np.diff(lats).mean())), abs(float(np.diff(lons).mean()))
    cell_km2 = (_R * np.deg2rad(_dlat)) * (_R * np.deg2rad(_dlon) * np.cos(np.deg2rad(lats)))[:, None] \
               * np.ones((1, n_lon))
    to_units = lambda m: m * (24.0 / AGG_HOURS) / cell_km2
    unit_label = 'strokes km$^{-2}$ day$^{-1}$'
else:
    to_units = lambda m: m
    unit_label = f'strokes per cell per {AGG_HOURS} h'

fig4_models = {}
lsm2d = None
try:
    cache = xr.open_dataset(PREP_PATH).sel(lat=lats, lon=lons, method='nearest')
    if 'lsm' in cache:
        lsm2d = (cache['lsm'].isel(time=0) if 'time' in cache['lsm'].dims else cache['lsm']).values
    if all(v in cache for v in ('cape', 'precipitation', 'ltg')):
        cyears = pd.DatetimeIndex(cache.time.values).year
        tr_years = _attr_years('train_years')
        if tr_years is None:
            held = sum([_attr_years(k) or [] for k in ('val_years', 'test_years', 'excluded_years')], [])
            tr_years = [y for y in sorted(set(cyears)) if y not in held]
        tr = np.where(np.isin(cyears, tr_years))[0]
        cp = (cache['cape'].clip(min=0) * cache['precipitation'].clip(min=0)).fillna(0)
        c_r14 = float(cache['ltg'].isel(time=tr).fillna(0).sum()) / max(float(cp.isel(time=tr).sum()), 1e-12)
        fig4_models['R14 (c·CAPE·P)'] = c_r14 * cp.sel(time=test_times).mean('time').values
        print(f"  R14 constant fitted on training years {tr_years[0]}–{tr_years[-1]}: c = {c_r14:.3e}", flush=True)
    else:
        print("  cache has no cape/precipitation — R14 column skipped", flush=True)
except Exception as e:
    print(f"  R14 baseline skipped ({e})", flush=True)

if lsm2d is None and os.path.exists(HOURLY_PATH):
    try:
        _l = xr.open_dataset(HOURLY_PATH)['lsm']
        lsm2d = (_l.isel(time=0) if 'time' in _l.dims else _l).sel(lat=lats, lon=lons, method='nearest').values
    except Exception:
        pass

if PRED_VAR == 'pred_count' and 'pred_count_raw' in ds_pred:
    fig4_models['U-Net (raw)'] = np.nan_to_num(ds_pred['pred_count_raw'].values).mean(0)
    fig4_models['U-Net (calibrated)'] = pred_mean_2d
else:
    fig4_models[f'U-Net ({PRED_VAR})'] = pred_mean_2d

rows4 = [('', np.ones((n_lat, n_lon), bool))]
if lsm2d is not None:
    rows4 += [('Land', lsm2d >= 0.9), ('Ocean', lsm2d <= 0.1)]
else:
    print("  no land-sea mask found — only the 'all cells' row is drawn", flush=True)

obs_u = to_units(obs_mean_2d)
mod_u = {k: to_units(v) for k, v in fig4_models.items()}

# common axis limits and histogram bins for every panel (so panels are directly comparable)
_allpos = np.concatenate([a[a > 0].ravel() for a in [obs_u] + list(mod_u.values())])
_hi = 10 ** np.ceil(np.log10(_allpos.max()))
_lo = max(10 ** np.floor(np.log10(np.percentile(_allpos, 0.5))), _hi / 1e7)
edges = np.linspace(np.log10(_lo), np.log10(_hi), FIG4_BINS + 1)
centres = 10 ** ((edges[:-1] + edges[1:]) / 2)

panels, hmax = {}, 1
for rname, rmask in rows4:
    for mname, m in mod_u.items():
        x, y = obs_u[rmask].ravel(), m[rmask].ravel()
        ok = np.isfinite(x) & np.isfinite(y)
        pos = ok & (x > 0) & (y > 0)
        H, _, _ = np.histogram2d(np.log10(x[pos]), np.log10(y[pos]), bins=[edges, edges])
        r2 = r2_score(x[ok], y[ok]) if ok.sum() > 2 else np.nan
        r2l = r2_score(np.log10(x[pos]), np.log10(y[pos])) if pos.sum() > 2 else np.nan
        panels[(rname, mname)] = (H.T, r2, r2l, int(ok.sum() - pos.sum()))
        hmax = max(hmax, H.max())
levels = [l for l in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000] if l <= hmax * 1.01]
levels = levels + [levels[-1] * 2] if len(levels) > 1 else [1, 2]
norm4 = mcolors.LogNorm(vmin=1, vmax=levels[-1])

ncol4, nrow4 = len(mod_u), len(rows4)
fig, axs = plt.subplots(nrow4, ncol4, figsize=(4.4 * ncol4, 4.2 * nrow4), squeeze=False)
for i, (rname, _) in enumerate(rows4):
    for j, mname in enumerate(mod_u):
        ax = axs[i, j]
        H, r2, r2l, n_zero = panels[(rname, mname)]
        cf = ax.contourf(centres, centres, np.ma.masked_less(H, 1), levels=levels, norm=norm4,
                         cmap=FIG4_CMAP, extend='max')
        ax.plot([_lo, _hi], [_lo, _hi], 'k-', lw=1,
                label=f'1:1 line (R² = {r2:.2f}; log R² = {r2l:.2f})')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(_lo, _hi); ax.set_ylim(_lo, _hi); ax.set_aspect('equal')
        ax.legend(loc='upper left', fontsize=7.5, handlelength=1.5)
        ax.set_title(f'{mname}: {rname}' if rname else mname, fontsize=11)
        if i == nrow4 - 1:
            ax.set_xlabel(f'Observed lightning ({unit_label})', fontsize=9)
        if j == 0:
            ax.set_ylabel(f'Modelled lightning ({unit_label})', fontsize=9)
        cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.06)
        cb = fig.colorbar(cf, cax=cax)
        cb.ax.tick_params(labelsize=7)
        if n_zero:
            print(f"  {mname} {rname or 'all'}: {n_zero} cells with zero observed or modelled mean "
                  f"left out of the log-log density (still in R²)", flush=True)
fig.suptitle(f'Test-period mean lightning per grid cell, modelled vs observed — Test {TEST_LABEL}, '
             f'{"–".join(pd.Timestamp(2001, m, 1).strftime("%b") for m in (min(months_arr), max(months_arr)))}'
             f'   (colours: number of grid cells)', fontsize=12, y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig4_r2_scatter.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved fig4_r2_scatter.png", flush=True)


# ============================================================
# Fig 4b — Observed vs predicted (log scales), the earlier two-panel version
#   (a) per-cell mean over the test period
#   (b) every pixel × 12-h window, log1p space, density hexbin
# ============================================================
print("Fig 4b: Obs vs pred scatter...", flush=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)

xc, yc = obs_mean_2d.ravel(), pred_mean_2d.ravel()
floor = max(1e-3, np.percentile(np.concatenate([xc, yc])[np.concatenate([xc, yc]) > 0], 1)) \
        if np.any(np.concatenate([xc, yc]) > 0) else 1e-3
xcp, ycp = np.clip(xc, floor, None), np.clip(yc, floor, None)
hb1 = ax1.hexbin(xcp, ycp, xscale='log', yscale='log', gridsize=40, bins='log', cmap='viridis', mincnt=1)
lo, hi = floor, max(xcp.max(), ycp.max()) * 1.2
ax1.plot([lo, hi], [lo, hi], 'k--', lw=1.3, label='1:1')
ax1.set_xlim(lo, hi); ax1.set_ylim(lo, hi)
ax1.set_xlabel(f'Observed mean strokes / {AGG_HOURS} h (per cell)')
ax1.set_ylabel(f'Predicted mean strokes / {AGG_HOURS} h (per cell)')
ax1.set_title(f'(a) Per-cell climatology\nR² = {r2_clim:.3f}, r = {r_clim:.3f}', fontsize=11)
for axis in (ax1.xaxis, ax1.yaxis):
    axis.set_major_formatter(mticker.FormatStrFormatter('%g'))
    axis.set_minor_formatter(mticker.NullFormatter())
ax1.legend(loc='upper left')
plt.colorbar(hb1, ax=ax1, label='log10(# cells)')

xl, yl = np.log1p(o_all), np.log1p(p_all)
hb2 = ax2.hexbin(xl, yl, gridsize=50, bins='log', cmap='viridis', mincnt=1)
hi2 = max(xl.max(), yl.max()) * 1.05
ax2.plot([0, hi2], [0, hi2], 'k--', lw=1.3, label='1:1')
ax2.set_xlim(0, hi2); ax2.set_ylim(0, hi2)
ticks = [0, 1, 10, 100, 1000, 10000]
tick_pos = [np.log1p(t) for t in ticks if np.log1p(t) <= hi2]
ax2.set_xticks(tick_pos); ax2.set_yticks(tick_pos)
ax2.set_xticklabels([str(t) for t in ticks][:len(tick_pos)])
ax2.set_yticklabels([str(t) for t in ticks][:len(tick_pos)])
ax2.set_xlabel(f'Observed strokes / {AGG_HOURS} h (log1p axis)')
ax2.set_ylabel(f'Predicted strokes / {AGG_HOURS} h (log1p axis)')
ax2.set_title(f'(b) All pixels × windows\nR² (log1p) = {r2_log:.3f}', fontsize=11)
ax2.legend(loc='upper left')
plt.colorbar(hb2, ax=ax2, label='log10(# pixel-windows)')

fig.suptitle(f'Observed vs Predicted Lightning — Test {TEST_LABEL}', fontsize=12)
fig.savefig(os.path.join(FIG_DIR, 'fig4b_pixel_scatter.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved fig4b_pixel_scatter.png", flush=True)


# ============================================================
# Fig 5 — Correlation of ERA5 variables with 12-h lightning (TRAINING years)
# ============================================================
# Computed on the training years only: ranking inputs on the test years and then choosing inputs from
# that ranking would leak test information into the model design.
print("Fig 5: Correlations (training years)...", flush=True)
import json as _json, ast as _ast
SKIP_VARS = {'ltg', 'lsm'}
corr_results = {}

def _years_from_attr(name):
    try:
        v = ds_pred.attrs[name]
        return list(_json.loads(v)) if v.strip().startswith('[') and '"' not in v else list(_ast.literal_eval(v))
    except Exception:
        return None

if os.path.exists(HOURLY_PATH):
    ds_h = xr.open_dataset(HOURLY_PATH).sel(lat=lats, lon=lons, method='nearest')
    all_years = sorted(set(pd.DatetimeIndex(ds_h.time.values).year))
    CORR_YEARS = _years_from_attr('train_years')
    if CORR_YEARS is None:      # older prediction file: everything that is not val/test/excluded
        held = sum([_years_from_attr(k) or [] for k in ('val_years', 'test_years', 'excluded_years')], [])
        CORR_YEARS = [y for y in all_years if y not in held]
    CORR_MONTHS = sorted(set(pd.DatetimeIndex(test_times).month))
    print(f"  years {CORR_YEARS}, months {CORR_MONTHS}", flush=True)

    def _windows(var, how):
        parts = []
        for y in CORR_YEARS:
            d = ds_h[var].sel(time=(ds_h.time.dt.year == y) & ds_h.time.dt.month.isin(CORR_MONTHS))
            if d.sizes['time'] == 0:
                continue
            r = d.resample(time=f'{AGG_HOURS}h')
            parts.append((r.sum(min_count=1) if how == 'sum' else r.mean()).values.astype('float32'))
        return np.concatenate(parts).ravel() if parts else np.array([])

    ltg_flat = _windows('ltg', 'sum')
    rng_c = np.random.default_rng(0)
    samp = rng_c.choice(ltg_flat.size, size=min(2_000_000, ltg_flat.size), replace=False)
    feature_vars = [v for v in ds_h.data_vars if v not in SKIP_VARS and 'time' in ds_h[v].dims]
    print(f"  Variables: {feature_vars}", flush=True)
    for var in feature_vars:
        try:
            arr = _windows(var, 'mean')[samp]
            lt = ltg_flat[samp]
            m = np.isfinite(arr) & np.isfinite(lt)
            if m.sum() < 100 or np.nanstd(arr[m]) == 0:
                continue
            r_p = pearsonr(arr[m], lt[m])[0]
            r_s = spearmanr(arr[m], lt[m])[0]
            corr_results[var] = (r_p, r_s)
            print(f"    {var:32s} Pearson={r_p:+.3f}  Spearman={r_s:+.3f}", flush=True)
        except Exception as e:
            print(f"    {var}: skipped ({e})", flush=True)
else:
    print(f"  {HOURLY_PATH} not found — skipping Fig 5", flush=True)

if corr_results:
    names  = sorted(corr_results, key=lambda v: abs(corr_results[v][1]), reverse=True)
    r_pear = [corr_results[v][0] for v in names]
    r_spea = [corr_results[v][1] for v in names]
    xb = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.55), 5), constrained_layout=True)
    ax.bar(xb - 0.2, r_pear, 0.4, color='#2c5f8a', label='Pearson (linear, raw counts)')
    ax.bar(xb + 0.2, r_spea, 0.4, color='#e67e22', label='Spearman (rank, monotonic)')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(xb)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(f'Correlation with {AGG_HOURS}-h lightning')
    ax.set_title(f'ERA5 variables vs lightning — training years {CORR_YEARS[0]}–{CORR_YEARS[-1]} '
                 f'(sorted by |Spearman|)', fontsize=11)
    ax.set_ylim(-1, 1)
    ax.legend(fontsize=9)
    ax.grid(axis='y', ls='--', alpha=0.4)
    fig.savefig(os.path.join(FIG_DIR, 'fig5_correlation.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig5_correlation.png", flush=True)


# ============================================================
# Fig 6 — Case studies: most active 12-h windows
# ============================================================
print("Fig 6: Case studies...", flush=True)
totals   = obs_np.sum(axis=(1, 2))
case_idx = np.argsort(totals)[::-1][:N_CASES]
case_idx = case_idx[totals[case_idx] > 0]

if len(case_idx):
    map_w6 = 6.0
    map_h6 = map_w6 / domain_aspect
    fig, axs = new_map_figure(len(case_idx), 2, (map_w6 * 2 + 1.2, map_h6 * len(case_idx) + 0.8))
    fig.subplots_adjust(left=0.04, right=0.86, top=0.94, bottom=0.03, wspace=0.05,
                    hspace=0.18 if HAS_CARTOPY else 0.28)
    vmax6 = max(np.percentile(obs_np[case_idx][obs_np[case_idx] > 0], 99), EVENT_THRESHOLD * 2)
    norm6 = mcolors.LogNorm(vmin=EVENT_THRESHOLD, vmax=vmax6)
    for row, t in enumerate(case_idx):
        label = pd.Timestamp(test_times[t]).strftime('%Y-%m-%d %H UTC')
        im6 = draw_map(fig, axs[row, 0], np.where(obs_np[t] >= EVENT_THRESHOLD, obs_np[t], np.nan),
                       cmap_ltg, norm6)
        axs[row, 0].set_title(f'{label} — Observed', fontsize=10)
        draw_map(fig, axs[row, 1], np.where(count_np[t] >= EVENT_THRESHOLD, count_np[t], np.nan),
                 cmap_ltg, norm6)
        axs[row, 1].set_title(f'{label} — Predicted  (FSS 3×3 = {fss_arr[t]:.2f})', fontsize=10)
    cax6 = fig.add_axes([0.88, 0.05, 0.018, 0.85])
    cb6 = fig.colorbar(im6, cax=cax6, extend='max')
    cb6.set_label(f'Strokes per {AGG_HOURS} h (log scale; white = < {EVENT_THRESHOLD:g})')
    fig.suptitle(f'Case studies ({RUN_NAME}) — {len(case_idx)} most active {AGG_HOURS}-h windows in the test set',
                 fontsize=12, y=0.98)
    fig.savefig(os.path.join(FIG_DIR, 'fig6_case_studies.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig6_case_studies.png", flush=True)


# ============================================================
# Summary
# ============================================================
print(f"\n{'='*58}")
print(f"  TEST SET SUMMARY")
print(f"{'='*58}")
print(f"  Period          : {pd.Timestamp(test_times[0]).date()} → {pd.Timestamp(test_times[-1]).date()}")
print(f"  Timesteps       : {len(test_times)} × {AGG_HOURS}h")
print(f"  Event rate o/p  : {obs_binary_np.mean():.4f} / {pred_np.mean():.4f}")
print(f"  Precision       : {precision:.4f}")
print(f"  Recall          : {recall:.4f}")
print(f"  F1              : {f1:.4f}")
print(f"  CSI             : {csi:.4f}")
print(f"  Mean FSS(3×3)   : {np.nanmean(fss_valid):.4f}")
print(f"  Median FSS      : {np.nanmedian(fss_valid):.4f}")
print(f"  Bias (strokes)  : {bias:+.4f}")
print(f"  R² (clim)       : {r2_clim:.4f}")
print(f"  R² (log1p, all) : {r2_log:.4f}")
print(f"{'='*58}")
print(f"\nFigures saved → {FIG_DIR}", flush=True)