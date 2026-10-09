"""
jones_figures.py
----------------
Evaluation figures for the Israel-domain U-Net (OND season binary model).

Figures produced:
  Fig 1 — Monthly probability maps: Observed vs Predicted (Oct / Nov / Dec)
  Fig 2 — Monthly FSS bar chart (Oct / Nov / Dec)
  Fig 3 — Metrics summary table
  Fig 4 — Pearson correlation of each ERA5 feature vs lightning

Usage (on EC2):
    cd /home/ec2-user/ML_Thesis_with_DL
    python jones_figures.py

Outputs saved to:  data/figures/
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
from matplotlib.dates import DateFormatter, MonthLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import uniform_filter, gaussian_filter
from scipy.stats import pearsonr
from sklearn.metrics import precision_score, recall_score, f1_score, r2_score

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = '/home/ec2-user/ML_Thesis_with_DL/data'
FIG_DIR  = os.path.join(DATA_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load predictions (binary OND model, test set 2024-2025) ─────────────────
print("Loading predictions...", flush=True)
ds_pred    = xr.open_dataset(os.path.join(DATA_DIR, 'unet_binary_OND_predictions.nc'))
prob_da    = ds_pred['prob']   # (T, lat, lon)  predicted probability 0–1
pred_da    = ds_pred['pred']   # (T, lat, lon)  binary prediction 0/1
test_times = ds_pred['time'].values
lats = ds_pred['lat'].values
lons = ds_pred['lon'].values
n_lat, n_lon = len(lats), len(lons)

# ── Load observations — resample hourly → 12-hourly binary ──────────────────
print("Loading observations...", flush=True)
ds_obs_h = xr.open_dataset(os.path.join(DATA_DIR, 'jones_israel_all_years.nc'))
ds_obs_h = ds_obs_h.isel(lat=slice(0, n_lat), lon=slice(0, n_lon))
obs_12h  = (ds_obs_h['ltg'].resample(time='12h').sum() > 0).astype(float)
obs_da   = obs_12h.sel(time=test_times, method='nearest')

print(f"Test period : {pd.Timestamp(test_times[0]).date()} → {pd.Timestamp(test_times[-1]).date()}")
print(f"Grid        : {n_lat}×{n_lon}, Timesteps: {len(test_times)}", flush=True)

prob_np = prob_da.values   # (T, lat, lon)  probability
pred_np = pred_da.values   # (T, lat, lon)  binary prediction
obs_np  = obs_da.values    # (T, lat, lon)  binary observation (0/1)

# ── FSS ──────────────────────────────────────────────────────────────────────
def compute_fss(obs, pred, threshold, window_size):
    obs_bin   = (obs  >= threshold).astype(int)
    pred_bin  = (pred >= threshold).astype(int)
    obs_frac  = uniform_filter(obs_bin.astype(float),  size=window_size, mode='constant', cval=0)
    pred_frac = uniform_filter(pred_bin.astype(float), size=window_size, mode='constant', cval=0)
    num = np.sum((pred_frac - obs_frac) ** 2)
    den = np.sum(pred_frac**2 + obs_frac**2)
    return np.nan if den == 0 else 1 - num / den

THRESHOLD    = 0.5
WINDOW_SIZE  = 3
WINDOW_SIZES = [1, 3, 5]

print("Computing FSS...", flush=True)
fss_by_window = {}
for w in WINDOW_SIZES:
    scores = [compute_fss(obs_np[i], prob_np[i], THRESHOLD, w)
              for i in range(len(test_times))]
    fss_by_window[w] = np.array(scores)

fss_arr   = fss_by_window[WINDOW_SIZE]
fss_valid = fss_arr[np.isfinite(fss_arr)]

# ── Classification metrics ────────────────────────────────────────────────────
obs_flat  = obs_np.ravel().astype(int)
pred_flat = pred_np.ravel().astype(int)
fin = np.isfinite(obs_flat) & np.isfinite(pred_flat)
precision = precision_score(obs_flat[fin], pred_flat[fin], zero_division=0)
recall    = recall_score(obs_flat[fin], pred_flat[fin], zero_division=0)
f1        = f1_score(obs_flat[fin], pred_flat[fin], zero_division=0)
obs_freq  = obs_np.mean(axis=0).ravel()
pred_freq = pred_np.mean(axis=0).ravel()
fin2 = np.isfinite(obs_freq) & np.isfinite(pred_freq)
r2_clim = r2_score(obs_freq[fin2], pred_freq[fin2])

print(f"Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}", flush=True)
print(f"Mean FSS(3×3)={np.nanmean(fss_valid):.3f}  R²={r2_clim:.3f}", flush=True)

# ── Map helpers ───────────────────────────────────────────────────────────────
lon_min, lon_max = float(lons.min()), float(lons.max())
lat_min, lat_max = float(lats.min()), float(lats.max())

# Display extent = data extent (domain now covers full Israel including Negev)
# Add only a tiny margin so borders aren't clipped at the edge
disp_extent = [lon_min - 0.3, lon_max + 0.3,
               lat_min - 0.3, lat_max + 0.3]

cmap_ltg = plt.cm.RdYlBu_r.copy()
cmap_ltg.set_bad(color='white')     # NaN → white
cmap_ltg.set_under(color='white')   # below vmin → white

# Spatial smoothing sigma (pixels) applied to both obs and prob for a fair visual comparison.
# The U-Net's convolutional layers produce naturally smooth output; we match that here.
SMOOTH_SIGMA = 1.5

def fmt_deg(x, pos):
    return f"{x:.1f}°"

def add_colorbar(fig, ax, im, label):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=9)
    return cb

# ── Try cartopy → geopandas → plain ──────────────────────────────────────────
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
    HAS_GPD = False
    print("  Using cartopy for borders", flush=True)
except ImportError:
    HAS_CARTOPY = False
    try:
        import geopandas as gpd
        HAS_GPD = True
        try:
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        except Exception:
            import geodatasets
            world = gpd.read_file(geodatasets.get_path('naturalearth.land'))
        print("  cartopy not found — using geopandas for borders", flush=True)
    except ImportError:
        HAS_GPD = False
        print("  Neither cartopy nor geopandas — plain axes", flush=True)


def draw_map(fig, ax, data, vmin, vmax, cmap, proj=None, add_cb=True, clabel=''):
    """Draw pcolormesh + borders. Returns the mappable so caller can share a colorbar."""
    if HAS_CARTOPY:
        im = ax.pcolormesh(lons, lats, data, cmap=cmap, vmin=vmin, vmax=vmax,
                           shading='auto', transform=proj)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor='black')
        ax.add_feature(cfeature.BORDERS,   linewidth=0.4, edgecolor='gray')
        ax.set_extent(disp_extent, crs=proj)   # extended to show full Israel coast
        if add_cb:
            plt.colorbar(im, ax=ax, label=clabel, shrink=0.85)
    elif HAS_GPD:
        im = ax.pcolormesh(lons, lats, data, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        world.boundary.plot(ax=ax, linewidth=0.6, color='black', zorder=2)
        ax.set_xlim(disp_extent[0], disp_extent[1])
        ax.set_ylim(disp_extent[2], disp_extent[3])
        ax.set_facecolor('#d0e8f0')
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_deg))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_deg))
        if add_cb:
            add_colorbar(fig, ax, im, clabel)
    else:
        im = ax.pcolormesh(lons, lats, data, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        ax.set_xlim(disp_extent[0], disp_extent[1])
        ax.set_ylim(disp_extent[2], disp_extent[3])
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_deg))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_deg))
        if add_cb:
            add_colorbar(fig, ax, im, clabel)
    return im


# ============================================================
# Fig 1 — Monthly Probability Maps: Observed vs Predicted
# ============================================================
print("Fig 1: Monthly probability maps...", flush=True)

OND_MONTHS = [10, 11, 12]
OND_LABELS = ['October', 'November', 'December']
months_arr = pd.DatetimeIndex(test_times).month

# Shared normalisation: global max of monthly mean obs freq and monthly mean prob
all_monthly_obs  = np.stack([obs_np[months_arr == m].mean(axis=0)  for m in OND_MONTHS])
all_monthly_prob = np.stack([prob_np[months_arr == m].mean(axis=0) for m in OND_MONTHS])
global_max = max(np.nanmax(all_monthly_obs), np.nanmax(all_monthly_prob))
print(f"  Monthly maps global_max = {global_max:.4f}", flush=True)

VMIN_DISPLAY = 0.02   # values below this render as white (effectively "no signal")
nrows = len(OND_MONTHS)

# Domain aspect ratio based on display extent (includes padding)
domain_aspect = (disp_extent[1] - disp_extent[0]) / (disp_extent[3] - disp_extent[2])
map_w  = 7.5                        # width per map panel (inches)
map_h  = map_w / domain_aspect      # height that preserves true aspect
fig_w  = map_w * 2 + 0.6            # two panels + thin colorbar strip
fig_h  = map_h * nrows + 0.8        # rows + suptitle space

if HAS_CARTOPY:
    proj = ccrs.PlateCarree()
    fig, axs = plt.subplots(nrows, 2, figsize=(fig_w, fig_h),
                             subplot_kw={'projection': proj})
else:
    fig, axs = plt.subplots(nrows, 2, figsize=(fig_w, fig_h))
    proj = None

# Tight spacing: minimal gaps between panels
fig.subplots_adjust(left=0.04, right=0.88, top=0.93, bottom=0.03,
                    wspace=0.04, hspace=0.12)

last_im = None
for row, (m, mlabel) in enumerate(zip(OND_MONTHS, OND_LABELS)):
    idx_m = np.where(months_arr == m)[0]

    raw_obs  = obs_np[idx_m].mean(axis=0)
    raw_prob = prob_np[idx_m].mean(axis=0)

    # Same Gaussian smoothing on both panels for fair visual comparison
    raw_obs_sm  = gaussian_filter(raw_obs,  sigma=SMOOTH_SIGMA)
    raw_prob_sm = gaussian_filter(raw_prob, sigma=SMOOTH_SIGMA)

    obs_plot  = np.where(raw_obs_sm  > 0, raw_obs_sm  / global_max, np.nan)
    prob_plot = np.where(raw_prob_sm > 0, raw_prob_sm / global_max, np.nan)

    last_im = draw_map(fig, axs[row, 0], obs_plot,  VMIN_DISPLAY, 1, cmap_ltg,
                       proj=proj, add_cb=False)
    axs[row, 0].set_title(f'{mlabel} — Observed (WWLN)', fontsize=11, pad=3)

    draw_map(fig, axs[row, 1], prob_plot, VMIN_DISPLAY, 1, cmap_ltg,
             proj=proj, add_cb=False)
    axs[row, 1].set_title(f'{mlabel} — U-Net Probability', fontsize=11, pad=3)

# Single shared colorbar on the far right
cbar_ax = fig.add_axes([0.90, 0.05, 0.018, 0.85])
cb = fig.colorbar(last_im, cax=cbar_ax)
cb.set_label('Relative lightning freq / probability  (0–1)', fontsize=10)

fig.suptitle('Monthly Lightning Frequency vs Predicted Probability — Test 2024–2025',
             fontsize=13, y=0.97)
fig.savefig(os.path.join(FIG_DIR, 'fig1_monthly_maps.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved fig1_monthly_maps.png", flush=True)


# ============================================================
# Fig 2 — Monthly FSS Bar Chart (OND)
# ============================================================
print("Fig 2: Monthly FSS...", flush=True)

month_labels_short = {10: 'Oct', 11: 'Nov', 12: 'Dec'}
colors_w = {1: 'steelblue', 3: 'darkorange', 5: 'forestgreen'}

fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)

x     = np.arange(len(OND_MONTHS))
width = 0.25

for i, w in enumerate(WINDOW_SIZES):
    vals = []
    errs = []
    for m in OND_MONTHS:
        idx_m  = np.where(months_arr == m)[0]
        scores = [compute_fss(obs_np[t], prob_np[t], THRESHOLD, w) for t in idx_m]
        valid  = [s for s in scores if np.isfinite(s)]
        vals.append(np.mean(valid) if valid else np.nan)
        errs.append(np.std(valid) / np.sqrt(len(valid)) if len(valid) > 1 else 0)
    ax.bar(x + (i - 1) * width, vals, width, yerr=errs, capsize=4,
           color=colors_w[w], alpha=0.85, label=f'{w}×{w} window')

ax.set_xticks(x)
ax.set_xticklabels([month_labels_short[m] for m in OND_MONTHS], fontsize=12)
ax.set_ylabel('Mean FSS', fontsize=11)
ax.set_xlabel('Month', fontsize=11)
ax.set_title('Monthly FSS by Kernel Size — Test 2024–2025', fontsize=12)
ax.set_ylim(0, 1)
ax.axhline(0.5, color='gray', ls=':', lw=1.2, label='FSS = 0.5 (skillful)')
ax.legend(fontsize=10)
ax.grid(axis='y', ls='--', alpha=0.4)
fig.savefig(os.path.join(FIG_DIR, 'fig2_monthly_fss.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved fig2_monthly_fss.png", flush=True)


# ============================================================
# Fig 3 — Metrics Summary Table
# ============================================================
print("Fig 3: Metrics table...", flush=True)

# Per-month FSS (3×3)
month_fss_vals = {}
for m in OND_MONTHS:
    idx_m  = np.where(months_arr == m)[0]
    scores = [compute_fss(obs_np[t], prob_np[t], THRESHOLD, WINDOW_SIZE) for t in idx_m]
    valid  = [s for s in scores if np.isfinite(s)]
    month_fss_vals[m] = np.mean(valid) if valid else np.nan

TRAIN_START = 'Oct 2013'
TRAIN_END   = 'Dec 2023'

col_labels = ['Metric', 'Value']
row_data = [
    ['Decision Threshold',         f'{THRESHOLD:.2f}'],
    ['Precision',                  f'{precision:.3f}'],
    ['Recall',                     f'{recall:.3f}'],
    ['F1 Score',                   f'{f1:.3f}'],
    ['Mean FSS (1×1)',              f'{np.nanmean(fss_by_window[1][np.isfinite(fss_by_window[1])]):.3f}'],
    ['Mean FSS (3×3)',              f'{np.nanmean(fss_valid):.3f}'],
    ['Mean FSS (5×5)',              f'{np.nanmean(fss_by_window[5][np.isfinite(fss_by_window[5])]):.3f}'],
    ['Median FSS (3×3)',            f'{np.nanmedian(fss_valid):.3f}'],
    ['FSS Oct (3×3)',               f'{month_fss_vals[10]:.3f}'],
    ['FSS Nov (3×3)',               f'{month_fss_vals[11]:.3f}'],
    ['FSS Dec (3×3)',               f'{month_fss_vals[12]:.3f}'],
    ['Training period',             f'{TRAIN_START} – {TRAIN_END}  (12h intervals)'],
    ['Test period',                 f'{pd.Timestamp(test_times[0]).strftime("%b %Y")} – {pd.Timestamp(test_times[-1]).strftime("%b %Y")}  (12h intervals)'],
]

fig, ax = plt.subplots(figsize=(6, 0.45 * len(row_data) + 1.2), constrained_layout=True)
ax.axis('off')
tbl = ax.table(
    cellText=row_data,
    colLabels=col_labels,
    loc='center',
    cellLoc='left',
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.3, 1.6)

for j in range(len(col_labels)):
    tbl[0, j].set_facecolor('#2c5f8a')
    tbl[0, j].set_text_props(color='white', fontweight='bold')

for i in range(1, len(row_data) + 1):
    fc = '#f0f4f8' if i % 2 == 0 else 'white'
    for j in range(len(col_labels)):
        tbl[i, j].set_facecolor(fc)

ax.set_title('U-Net OND Model — Test Set Metrics (2024–2025)', fontsize=12,
             fontweight='bold', pad=10)
fig.savefig(os.path.join(FIG_DIR, 'fig3_metrics_table.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved fig3_metrics_table.png", flush=True)


# ============================================================
# Fig 4 — R² Scatter: Observed vs Predicted Lightning Frequency (KDE contourf, log-log)
# ============================================================
print("Fig 4: R² scatter (obs vs pred frequency, KDE contourf)...", flush=True)

from scipy.stats import gaussian_kde as _kde

# Per-cell mean frequency over the test period
obs_freq_2d  = obs_np.mean(axis=0)   # (lat, lon)
pred_freq_2d = pred_np.mean(axis=0)

x = obs_freq_2d.ravel()
y = pred_freq_2d.ravel()

# Keep only cells where BOTH obs and pred are strictly positive (needed for log scale)
mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
x, y = x[mask], y[mask]

lx, ly = np.log10(x), np.log10(y)

# KDE on log-space grid
MAX_KDE = 6000
if len(lx) > MAX_KDE:
    idx_s = np.random.default_rng(42).choice(len(lx), MAX_KDE, replace=False)
    lxk, lyk = lx[idx_s], ly[idx_s]
else:
    lxk, lyk = lx, ly

kde = _kde(np.vstack([lxk, lyk]))

lo = min(lxk.min(), lyk.min()) - 0.2
hi = max(lxk.max(), lyk.max()) + 0.2
xi = np.linspace(lo, hi, 100)
yi = np.linspace(lo, hi, 100)
Xi, Yi = np.meshgrid(xi, yi)
Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)

# Log-spaced contour levels (low=blue → high=red, matching Jones style)
lev_min = max(Zi[Zi > 0].min(), Zi.max() * 1e-4)
levels  = np.logspace(np.log10(lev_min), np.log10(Zi.max()), 20)

fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
cf = ax.contourf(10**Xi, 10**Yi, Zi, levels=levels,
                 cmap='jet', norm=mcolors.LogNorm())
cb = plt.colorbar(cf, ax=ax)
cb.set_label('Point density', fontsize=10)

# 1:1 line
lim_lo = 10**lo
lim_hi = 10**hi
ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], 'k--', lw=1.5,
        label=f'1:1 line  (r² = {r2_clim:.2f})')

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(lim_lo, lim_hi); ax.set_ylim(lim_lo, lim_hi)
ax.set_xlabel('Observed lightning frequency (per grid cell)', fontsize=11)
ax.set_ylabel('Predicted lightning frequency (per grid cell)', fontsize=11)
ax.set_title('Observed vs Predicted Lightning Frequency\n'
             f'R² = {r2_clim:.3f}  —  Test 2024–2025', fontsize=12)
ax.legend(fontsize=10, loc='upper left')
ax.grid(ls='--', alpha=0.3)
fig.savefig(os.path.join(FIG_DIR, 'fig4_r2_scatter.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved fig4_r2_scatter.png", flush=True)


# ============================================================
# Fig 5 — Pearson Correlation: ERA5 Features vs Lightning
# ============================================================
print("Fig 5: Pearson correlations...", flush=True)

# Load ERA5 features for test timesteps only
# Skip non-feature variables
SKIP_VARS = {'ltg', 'lsm', 'lat', 'lon', 'time', 'latitude', 'longitude'}

# Lightning observations as flat array for correlation (use raw hourly counts)
# Resample ltg to 12h sum (not binary) for a richer signal
ltg_12h  = ds_obs_h['ltg'].resample(time='12h').sum()
ltg_test = ltg_12h.sel(time=test_times, method='nearest').values  # (T, lat, lon)
ltg_flat = ltg_test.ravel()

feature_vars = [v for v in ds_obs_h.data_vars if v not in SKIP_VARS]
print(f"  Feature variables found: {feature_vars}", flush=True)

pearson_results = {}
for var in feature_vars:
    try:
        da = ds_obs_h[var]
        # Resample to 12h mean to match prediction timesteps
        if 'time' in da.dims:
            da_12h = da.resample(time='12h').mean()
            arr = da_12h.sel(time=test_times, method='nearest').values
        else:
            # Static variable (e.g. lsm) — tile over time
            arr = np.broadcast_to(da.values[:n_lat, :n_lon][np.newaxis],
                                  (len(test_times), n_lat, n_lon)).copy()
        arr_flat = arr.ravel()
        # Only finite values
        mask = np.isfinite(arr_flat) & np.isfinite(ltg_flat)
        if mask.sum() < 100:
            continue
        r, p = pearsonr(arr_flat[mask], ltg_flat[mask])
        pearson_results[var] = (r, p)
        print(f"    {var:20s}  r={r:+.3f}  p={p:.2e}", flush=True)
    except Exception as e:
        print(f"    {var}: skipped ({e})", flush=True)

if pearson_results:
    var_names = list(pearson_results.keys())
    r_vals    = [pearson_results[v][0] for v in var_names]
    p_vals    = [pearson_results[v][1] for v in var_names]

    # Sort by absolute correlation
    order     = np.argsort(np.abs(r_vals))[::-1]
    var_names = [var_names[i] for i in order]
    r_vals    = [r_vals[i]    for i in order]
    p_vals    = [p_vals[i]    for i in order]

    colors = ['#c0392b' if r > 0 else '#2980b9' for r in r_vals]
    # Mark significant (p < 0.05) with full opacity, non-significant with 40%
    alphas = [0.9 if p < 0.05 else 0.35 for p in p_vals]

    fig, ax = plt.subplots(figsize=(max(7, len(var_names) * 0.9), 5), constrained_layout=True)
    bars = ax.bar(var_names, r_vals, color=colors,
                  alpha=0.85, edgecolor='white', linewidth=0.5)
    for bar, a in zip(bars, alphas):
        bar.set_alpha(a)

    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylabel('Pearson r  (vs. 12h lightning count)', fontsize=11)
    ax.set_xlabel('ERA5 Feature Variable', fontsize=11)
    ax.set_title('Pearson Correlation of ERA5 Features with Lightning\n'
                 '(Test period 2024–2025, OND; faded = p ≥ 0.05)', fontsize=12)
    ax.set_ylim(-1, 1)
    plt.xticks(rotation=30, ha='right', fontsize=10)
    ax.grid(axis='y', ls='--', alpha=0.4)

    # Add r value labels on bars
    for bar, r in zip(bars, r_vals):
        ypos = bar.get_height() + 0.02 if r >= 0 else bar.get_height() - 0.05
        ax.text(bar.get_x() + bar.get_width() / 2, ypos, f'{r:+.2f}',
                ha='center', va='bottom', fontsize=8)

    fig.savefig(os.path.join(FIG_DIR, 'fig5_pearson_correlation.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig5_pearson_correlation.png", flush=True)
else:
    print("  No feature variables found for Pearson — skipping Fig 4", flush=True)


# ============================================================
# Summary
# ============================================================
print(f"\n{'='*55}")
print(f"  TEST SET SUMMARY")
print(f"{'='*55}")
print(f"  Period        : {pd.Timestamp(test_times[0]).date()} → {pd.Timestamp(test_times[-1]).date()}")
print(f"  Timesteps     : {len(test_times)} × 12h")
print(f"  Precision     : {precision:.4f}")
print(f"  Recall        : {recall:.4f}")
print(f"  F1            : {f1:.4f}")
print(f"  Mean FSS(3×3) : {np.nanmean(fss_valid):.4f}")
print(f"  Median FSS    : {np.nanmedian(fss_valid):.4f}")
print(f"  R² (clim)     : {r2_clim:.4f}")
print(f"{'='*55}")
print(f"\nFigures saved → {FIG_DIR}", flush=True)
