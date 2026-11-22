"""
thesis_figures.py — every figure and table in the Figure Plan, in one script.

Run on EC2 from the repo folder, after train_unet.py has written its predictions:

    python thesis_figures.py                      # everything whose inputs exist
    python thesis_figures.py --only F4 F5 T3      # some items only
    python thesis_figures.py --list               # every item and whether its inputs exist
    python thesis_figures.py --audit-log audit_log.txt
        audit_log.txt : the printed output of audit.py         (python audit.py | tee audit_log.txt)
        --train-log train_log.txt only for models trained before the loss history was saved

    python thesis_figures.py --months 10 11 12    # the Oct–Dec run (default: ACTIVE_MONTHS of train_unet.py)
    python thesis_figures.py --eval-months 10 11 12   # whole-year model, scored on Oct–Dec only (to compare
                                                      # it with the Oct–Dec model on the same windows)

Outputs (folder figures/thesis/ for the Oct–Dec run; figures/thesis_ALL/ for the whole year;
figures/thesis_ALL_evalOND/ with --eval-months 10 11 12):
    F01_…pdf / .png     main figures            (PDF for LaTeX, PNG for a quick look)
    A1_…, O2_…          appendix and optional figures
    tables/T1_….tex     tables for LaTeX (booktabs), plus .csv
    captions.md         a draft caption with the key numbers for every item made
    thesis_numbers.json all numbers in one place (handy for the abstract and the text)

Items whose inputs don't exist yet (new experiments) are skipped with a message saying what is missing.
Nothing in the data folder is changed.

Conventions used in every figure
  • one colour per model (COL below), colour-blind safe (Okabe–Ito)
  • lightning amounts on maps / scatter plots in strokes km⁻² yr⁻¹ (for a season: the rate during that
    season, as in Jones et al. 2026); case studies in strokes per cell per window
  • more than 4 months (whole year): "by month" panels are grouped by season (DJF, MAM, JJA, SON)
  • skill = FSS for "≥ 1 stroke per window" in a 3 × 3-cell neighbourhood unless stated
  • 95 % confidence intervals from a block bootstrap over BLOCK_DAYS-day blocks (storms last days,
    so single windows are not independent)
  • two versions of every model, each where it fits:
      - calibrated (quantile-mapped on the validation year) for yes/no event scores: FSS, CSI, POD, case studies.
        Quantile mapping onto whole stroke counts (~96 % zeros) turns most low values into exact 0 and the rest
        into values near 1, 2, 3 … — right for "is there lightning?", wrong for averages and rankings.
      - expected strokes (U-Net: the Poisson rate before calibration; CAPE × P: its fitted constant) for
        everything that averages or ranks amounts: mean maps, bias, land/sea, domain totals, ILDN, all-pixels.
  • differences between two models are tested on the per-window difference (paired bootstrap), because all
    models share the same easy and hard days; the interval of one model alone is much wider
  • no titles inside figures and no run-id footers — the caption says what a figure shows

Inputs for figures that need extra runs — train_unet.py makes them with command-line options and names
the files itself (the settings in train_unet.py stay at the main run):
  F13 storm history:   python train_unet.py --lags 3                 (+ --seed 14, --seed 15 for more seeds)
                       python train_unet.py --lags 3 --lag-control   (control: same inputs, no storm history)
                       python train_unet.py --seed 14 / --seed 15    (more seeds of the normal model)
  O2  learning curve:  python train_unet.py --train-years 2 / 4 / 6
  A4  training curves: read from the predictions file (models trained with the current train_unet.py),
                       otherwise from --train-log
  A5  robustness:      python audit.py | tee audit_log.txt   →   --audit-log audit_log.txt
File names and patterns ({season} = OND, ALL …, the season of the run; {stag} = '' for OND, else _ALL …):
  F3  ablation:        ABLATION below. Each step reads its own predictions file if it exists; otherwise
                       it falls back to the FSS stored in figures/_old/<run folder>/run_summary.json
                       (shown as open markers, without a confidence interval).
  F13 storm history:   LAG_RUNS below (predictions of the 3-window model, extra seeds, control run) and
                       LAG_CKPT for the per-time-slot importance panel.
  O1  per-cell model:  PERCELL_FILE. When it exists it is added to F4, F7 and T3 automatically.
  O2  learning curve:  LEARNING_CURVE_FILES.
"""
import argparse, ast, glob, json, os, re, sys, traceback, warnings
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter
from scipy.stats import pearsonr, spearmanr, ttest_1samp

warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

HERE = os.path.dirname(os.path.abspath(__file__))

# ═════════════════════════════════════════════════════════════════════════════════════════════
# Settings
# ═════════════════════════════════════════════════════════════════════════════════════════════
OUT_DIR      = os.path.join(HERE, 'figures', 'thesis')
OLD_FIG_DIR  = os.path.join(HERE, 'figures', '_old')       # earlier figure folders (run_summary.json)
TRAIN_SCRIPT = os.path.join(HERE, 'train_unet.py')

THR, WIN     = 1.0, 3                    # event = ≥ 1 stroke per window; FSS neighbourhood 3 × 3 cells
SCALES       = [1, 3, 5, 7, 9, 13, 17, 25]   # neighbourhood sizes (cells) for F4 / F13
INTENSITY_BINS = [(1, 10), (10, 100), (100, np.inf)]   # strokes per cell per window, F7 (Jones-style MFSS)
TAIL_PCT     = 95                        # F7b: the most active (100 − TAIL_PCT) % of windows
BLOCK_DAYS   = 7                         # bootstrap block length (days)
N_BOOT       = 1000
N_CASES      = 3                         # F10 case studies (A6 shows the next N_MORE_CASES)
N_MORE_CASES = 6
N_PERM       = 5                         # F11 repeats of each permutation
ILDN_BOX     = (30.0, 34.5, 32.0, 36.5)  # lat_min, lat_max, lon_min, lon_max (same as ildn_eval.py)
ILDN_GLOB    = '*_for_yoav_including_cloud.txt'
ILDN_TIME_OFFSET_H = 0                   # hours to subtract from ILDN times (same as ildn_eval.py)
LAND_LSM     = 0.5                       # land: lsm ≥ 0.5, sea: lsm < 0.5

# Okabe–Ito colours; one per model in every figure
COL = {'Observed': '#000000', 'ILDN': '#000000', 'WWLLN': '#E69F00',
       'U-Net': '#D55E00', 'U-Net (raw)': '#F0A27A', 'CAPE × P': '#0072B2', 'Climatology': '#8C8C8C',
       'Persistence': '#56B4E9', 'Per-cell model': '#009E73',
       '1 window': '#D55E00', '3 windows': '#CC79A7', 'Control': '#8C8C8C'}

# F3 — ablation steps (label, predictions file in the data folder, variable, earlier run folder)
ABLATION = [
    ('Jones et al. (2026) set-up',            'ablation_1_jones_setup_{season}_predictions.nc', 'pred_count_raw', 'unet_12h_mse_jonesnorm_raw_866fb6'),
    ('+ eight inputs of this study',           'ablation_2_our_inputs_{season}_predictions.nc',  'pred_count_raw', 'unet_12h_mse_jonesnorm_raw_f7c7a9'),
    ('+ log-transformed inputs',       'ablation_3_log_inputs_{season}_predictions.nc',  'pred_count_raw', 'unet_12h_mse_jonesnorm_raw_3a89cb'),
    ('+ quantile-mapping calibration', 'ablation_3_log_inputs_{season}_predictions.nc',  'pred_count',     'unet_12h_mse_jonesnorm_cal_112405'),
    ('+ Poisson loss (final model)',   None,                                        'pred_count',     'unet_12h_poisson_cal_6124ab'),
]   # None = the main predictions file. The run folders (earlier Oct–Dec runs) are used for Oct–Dec only.

# F13 — storm history (glob patterns in the data folder; several files = several random seeds)
#   '1 window'  : the normal model (inputs of the current window only)
#   '3 windows' : inputs of the current window + the two previous windows (24 channels, oldest first)
#   'Control'   : the same 3-window model, but the two earlier slots hold inputs from random other days —
#                 same number of inputs, no storm history. If '3 windows' beats 'Control', the gain comes
#                 from the history and not just from a bigger model.
LAG_RUNS = {
    '1 window':  ['{main}', 'unet_12h_poisson_seed*_{season}_predictions.nc'],
    '3 windows': ['unet_12h_poisson_lags3_{season}_predictions.nc', 'unet_12h_poisson_lags3_seed*_{season}_predictions.nc'],
    'Control':   ['unet_12h_poisson_lags3ctrl*_{season}_predictions.nc'],
}
LAG_CKPT = 'unet_12h_poisson{stag}_lags3_best.pt'     # + .json passport with "n_lags" (oldest slot first)

PERCELL_FILE = 'percell_poisson_{season}_predictions.nc'                 # O1
LEARNING_CURVE_FILES = {2: 'unet_12h_poisson_train2y_{season}_predictions.nc',
                        4: 'unet_12h_poisson_train4y_{season}_predictions.nc',
                        6: 'unet_12h_poisson_train6y_{season}_predictions.nc',
                        'all': '{main}'}                             # O2 ('all' = every training year)

NICE = {'precipitation': 'Precipitation (IMERG)', 'total_totals_index': 'Total Totals index',
        'ciwc500': 'Cloud ice at 500 hPa', 'total_column_cloud_ice_water': 'Total column cloud ice',
        'cape': 'CAPE', 'k_index': 'K index', 'total_column_cloud_liquid_water': 'Total column cloud liquid',
        'lsm': 'Land–sea mask', 'rh': 'Relative humidity', 'shear': 'Wind shear', 't2m': '2-m temperature',
        'wcd': 'Warm cloud depth', 'relative_humidity': 'Relative humidity',
        'total_column_water_vapour': 'Total column water vapour'}
SOURCE = {   # T1: dataset, units, native resolution
    'precipitation': ('IMERG (NASA GPM)', 'mm h⁻¹', '0.1°, 30 min'),
    'total_totals_index': ('ERA5', 'K', '0.25°, 1 h'),
    'ciwc500': ('ERA5', 'kg kg⁻¹', '0.25°, 1 h'),
    'total_column_cloud_ice_water': ('ERA5', 'kg m⁻²', '0.25°, 1 h'),
    'cape': ('ERA5', 'J kg⁻¹', '0.25°, 1 h'),
    'k_index': ('ERA5', 'K', '0.25°, 1 h'),
    'total_column_cloud_liquid_water': ('ERA5', 'kg m⁻²', '0.25°, 1 h'),
    'lsm': ('ERA5', '0–1', '0.25°'),
}

# figure sizes (inches): AGU/AMS single column ≈ 3.5, full width ≈ 7.2
W1, W2 = 3.5, 7.2
plt.rcParams.update({
    'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300, 'figure.dpi': 100,
    'axes.grid': False, 'legend.frameon': False,
})
LTG_CMAP = plt.cm.YlGnBu.copy(); LTG_CMAP.set_bad('white'); LTG_CMAP.set_under('white')
CASE_CMAP = plt.cm.magma_r.copy(); CASE_CMAP.set_bad('white'); CASE_CMAP.set_under('white')


# ═════════════════════════════════════════════════════════════════════════════════════════════
# Small helpers
# ═════════════════════════════════════════════════════════════════════════════════════════════
def read_train_config(path, overrides=None):
    """Top-level settings of train_unet.py (plain values and f-strings), read without running it.
    `overrides` replaces settings the way train_unet.py's command-line options do (e.g. ACTIVE_MONTHS)."""
    overrides = overrides or {}
    if not os.path.exists(path):
        return {}
    src = open(path).read()
    cfg, tree = {}, ast.parse(src)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            pairs = [(tgt, node.value)] if isinstance(tgt, ast.Name) else (
                list(zip(tgt.elts, node.value.elts)) if isinstance(tgt, ast.Tuple) and isinstance(node.value, ast.Tuple) else [])
            for t, v in pairs:
                if not isinstance(t, ast.Name):
                    continue
                if t.id in overrides:
                    cfg[t.id] = overrides[t.id]
                    continue
                try:
                    cfg[t.id] = ast.literal_eval(v)
                except Exception:
                    try:
                        cfg[t.id] = eval(compile(ast.Expression(v), path, 'eval'), {'__builtins__': {}}, dict(cfg))
                    except Exception:
                        pass
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SimpleCNN'), None)
    cfg['_model_src'] = ast.get_source_segment(src, cls) if cls else None
    return cfg


def _load(v):
    if not isinstance(v, str):
        return v.item() if hasattr(v, 'item') else v
    for parse in (json.loads, ast.literal_eval):
        try:
            return parse(v)
        except Exception:
            pass
    return v


SEASON_MONTHS = {'OND': [10, 11, 12], 'ALL': list(range(1, 13))}
MET_SEASONS = [('DJF', (12, 1, 2)), ('MAM', (3, 4, 5)), ('JJA', (6, 7, 8)), ('SON', (9, 10, 11))]


def season_name(months):
    """Same rule as SEASON in train_unet.py: 'OND', 'ALL', else e.g. 'M060708'."""
    ms = sorted(set(months))
    return next((k for k, v in SEASON_MONTHS.items() if v == ms), 'M' + ''.join(f'{m:02d}' for m in ms))


def months_label(months):
    """'Oct–Dec', 'Jan–Dec', 'Jun', 'Jan, Mar' …"""
    ms = sorted(set(months))
    ab = lambda m: pd.Timestamp(2000, m, 1).strftime('%b')
    if len(ms) == 1:
        return ab(ms[0])
    return f'{ab(ms[0])}–{ab(ms[-1])}' if ms == list(range(ms[0], ms[-1] + 1)) else ', '.join(map(ab, ms))


def whole_year():
    return sorted(MONTHS) == list(range(1, 13))


def rate_note():
    """Unit note for strokes km⁻² yr⁻¹: a season's rate, or nothing for the whole year."""
    return '' if whole_year() else f' ({months_label(MONTHS)} rate)'


def period_groups(months):
    """(label, mask) for 'by month' panels: each month, or each season (DJF …) when there are more than 4."""
    months = np.asarray(months)
    present = sorted(set(months.tolist()))
    if len(present) <= 4:
        return [(pd.Timestamp(2000, m, 1).strftime('%b'), months == m) for m in present]
    return [(lab, np.isin(months, ms)) for lab, ms in MET_SEASONS if np.isin(months, ms).any()]


class Missing(Exception):
    """An input of an item does not exist (yet): the item is skipped, not failed."""


CAPTIONS, NUMBERS, MADE = {}, {}, []


def out(name):
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(OUT_DIR, name)


def save(fig, name, caption=None):
    fig.savefig(out(name + '.pdf'), bbox_inches='tight')
    fig.savefig(out(name + '.png'), bbox_inches='tight', dpi=200)
    plt.close(fig)
    MADE.append(name)
    if caption:
        CAPTIONS[name] = caption
    print(f"  saved {name}.pdf / .png", flush=True)


def letter(ax, s):
    """Panel letter just above the top-left corner (offset in points, so it never hits the tick labels)."""
    ax.annotate(f'({s})', xy=(0, 1), xycoords='axes fraction', xytext=(-4, 5), textcoords='offset points',
                fontsize=9, fontweight='bold', va='bottom', ha='right')


def fmt(x, nd=2):
    return '–' if x is None or not np.isfinite(x) else f'{x:.{nd}f}'


def fmt_ci(m, lo, hi, nd=2):
    return f"{fmt(m, nd)} [{fmt(lo, nd)}, {fmt(hi, nd)}]"


TEX = [('\\', r'\textbackslash{}'), ('%', r'\%'), ('&', r'\&'), ('_', r'\_'), ('#', r'\#'),
       ('×', r'$\times$'), ('≥', r'$\geq$'), ('≤', r'$\leq$'), ('−', r'$-$'), ('λ', r'$\lambda$'),
       ('±', r'$\pm$'), ('⁻¹', r'$^{-1}$'), ('⁻²', r'$^{-2}$'), ('²', r'$^{2}$'), ('°', r'\textdegree{}'),
       ('₀', r'$_0$')]


def tex(v):
    s = str(v)
    for a, b in TEX:
        s = s.replace(a, b)
    return s


def write_table(name, df, caption, latex_cols=None, col_format=None):
    """df → tables/<name>.csv and a LaTeX table tables/<name>.tex (\\input{} it in the thesis)."""
    tdir = out('tables'); os.makedirs(tdir, exist_ok=True)
    df.to_csv(os.path.join(tdir, name + '.csv'), index=False)
    cols = latex_cols or list(df.columns)
    col_format = col_format or ('l' + 'c' * (len(cols) - 1))
    wide = len(cols) > 4                     # scaled to the text width
    lines = [r'% made by thesis_figures.py — \input{tables/' + name + '} in the thesis',
             r'\providecommand{\toprule}{\hline}\providecommand{\midrule}{\hline}\providecommand{\bottomrule}{\hline}'
             r'  % \usepackage{booktabs} gives nicer lines',
             r'\begin{table}[ht]', r'\centering', r'\small', rf'\caption{{{tex(caption)}}}', rf'\label{{tab:{name}}}',
             (r'\resizebox{\textwidth}{!}{' if wide else '') + rf'\begin{{tabular}}{{{col_format}}}', r'\toprule',
             ' & '.join(tex(c) for c in cols) + r' \\', r'\midrule']
    for _, row in df.iterrows():
        lines.append(' & '.join(tex(v) for v in row.values) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}' + ('}' if wide else ''), r'\end{table}']
    with open(os.path.join(tdir, name + '.tex'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    MADE.append(name)
    CAPTIONS[name] = caption
    print(f"  saved tables/{name}.tex / .csv", flush=True)


def block_ids(times):
    t = pd.DatetimeIndex(times)
    start = pd.to_datetime([f'{y}-{m:02d}-01' for y, m in zip(t.year, np.full(len(t), MONTHS[0]))])
    return np.asarray(t.year * 1000 + ((t - start).days // BLOCK_DAYS))


def boot_ci(vals, blocks, n=N_BOOT, seed=0):
    """95 % CI of the mean of `vals` (NaNs ignored) by resampling whole blocks."""
    v = np.asarray(vals, float)
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return np.nan, np.nan
    ub, inv = np.unique(np.asarray(blocks)[ok], return_inverse=True)
    s, c = np.bincount(inv, v[ok]), np.bincount(inv).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(ub), (n, len(ub)))
    b = s[idx].sum(1) / np.maximum(c[idx].sum(1), 1)
    return tuple(np.percentile(b, [2.5, 97.5]))


def paired_ci(a, b, blocks):
    """Mean of a − b (per-window scores of two models on the same windows) with its 95 % block-bootstrap CI."""
    return mean_ci(np.asarray(a, float) - np.asarray(b, float), blocks)


def mean_ci(vals, blocks, seed=0):
    v = np.asarray(vals, float)
    lo, hi = boot_ci(v, blocks, seed=seed)
    return (np.nanmean(v) if np.isfinite(v).any() else np.nan), lo, hi


# ═════════════════════════════════════════════════════════════════════════════════════════════
# Scores
# ═════════════════════════════════════════════════════════════════════════════════════════════
def fss_series(obs, pred, thr=THR, w=WIN, hi=None, valid=None):
    """FSS of every window (NaN where neither field has an event). `hi` turns the threshold into a bin."""
    ob = obs >= thr if hi is None else (obs >= thr) & (obs < hi)
    pb = pred >= thr if hi is None else (pred >= thr) & (pred < hi)
    of = uniform_filter(ob.astype('float32'), size=(1, w, w), mode='constant')
    pf = uniform_filter(pb.astype('float32'), size=(1, w, w), mode='constant')
    num = ((pf - of) ** 2).sum((1, 2)); den = (pf ** 2 + of ** 2).sum((1, 2))
    res = np.full(len(obs), np.nan)
    ok = den > 0
    res[ok] = 1 - num[ok] / den[ok]
    if valid is not None:
        res[~valid] = np.nan
    return res


def categorical(obs, pred, thr=THR, valid=None):
    if valid is not None:
        obs, pred = obs[valid], pred[valid]
    o, p = (obs >= thr).ravel(), (pred >= thr).ravel()
    h, m, f = np.sum(o & p), np.sum(o & ~p), np.sum(~o & p)
    pod = h / max(h + m, 1); sr = h / max(h + f, 1)
    return {'POD': pod, 'SR': sr, 'CSI': h / max(h + m + f, 1), 'F1': 2 * pod * sr / max(pod + sr, 1e-12),
            'freq_bias': (h + f) / max(h + m, 1), 'event_rate': p.mean()}


def fit_quantile_map(pred, obs):
    probs = np.unique(np.concatenate([np.linspace(0, 0.9, 200), 1 - np.logspace(-6, -1, 2000)]))
    qp = np.quantile(np.nan_to_num(pred).ravel(), probs)
    qo = np.quantile(np.nan_to_num(obs).ravel(), probs)
    return np.maximum.accumulate(qp) + np.arange(len(qp)) * 1e-12, qo


def calibrate(pred_val, obs_val, pred):
    qp, qo = fit_quantile_map(pred_val, obs_val)
    return np.interp(pred, qp, qo).reshape(pred.shape)


# ═════════════════════════════════════════════════════════════════════════════════════════════
# Data (loaded once, only when an item needs it)
# ═════════════════════════════════════════════════════════════════════════════════════════════
class Data:
    def __init__(self, args):
        self.args = args
        months = getattr(args, 'months', None)
        self.cfg = read_train_config(args.train_script, {'ACTIVE_MONTHS': sorted(set(months))} if months else None)
        m = re.search(r'_(OND|ALL|M\d+)_predictions\.nc$', args.pred or '')
        if m and m.group(1) in SEASON_MONTHS and m.group(1) != self.cfg.get('SEASON', 'OND'):
            # --pred of another season: read the settings of that season (aggregated file, checkpoint …)
            self.cfg = read_train_config(args.train_script, {'ACTIVE_MONTHS': SEASON_MONTHS[m.group(1)]})
        self.season = m.group(1) if m else (self.cfg.get('SEASON') or 'OND')
        self.stag = '' if self.season == 'OND' else '_' + self.season
        self.eval_months = sorted(set(getattr(args, 'eval_months', None) or [])) or None
        self.data_dir = args.data_dir or self.cfg.get('DATA_DIR') or os.path.join(HERE, 'data')
        rel = lambda p: os.path.join(self.data_dir, os.path.basename(p)) if p else None
        tag = self.cfg.get('RUN_TAG') or ''
        if tag:      # a variant (--lags, --seed …) set in the file itself: figures are still about the main run
            print(f"NOTE: train_unet.py has RUN_TAG '{tag}' — using the main run's files (without '{tag}')", flush=True)
            for k in ('OUT_PATH', 'CKPT_PATH'):
                if self.cfg.get(k):
                    self.cfg[k] = (self.cfg[k].replace(f'{tag}_{self.season}_', f'_{self.season}_')
                                   .replace(tag + '_best', '_best'))
        self.pred_path = args.pred if args.pred and os.path.isabs(args.pred) else (
            os.path.join(self.data_dir, args.pred) if args.pred else rel(self.cfg.get('OUT_PATH')))
        self.prep_path = rel(self.cfg.get('PREP_PATH'))
        self.hourly_path = rel(self.cfg.get('DATA_PATH'))
        self.ckpt_path = rel(self.cfg.get('CKPT_PATH'))
        self._cache = {}

    def path(self, name):
        if name is None or name == '{main}':
            return self.pred_path
        name = name.replace('{season}', self.season).replace('{stag}', self.stag)
        return name if os.path.isabs(name) else os.path.join(self.data_dir, name)

    @property
    def period_word(self):
        return "year" if whole_year() else "season"

    # ── predictions of the final model ──
    def pred(self):
        if 'pred' not in self._cache:
            if not self.pred_path or not os.path.exists(self.pred_path):
                raise Missing(f"predictions file {self.pred_path} (run train_unet.py first)")
            p = xr.open_dataset(self.pred_path).load()
            self.A = {k: _load(v) for k, v in p.attrs.items()}
            self.agg = int(self.A.get('agg_hours', 12))
            global MONTHS
            MONTHS = list(self.A.get('months', [10, 11, 12]))
            if self.eval_months:                     # score on some months only (e.g. Oct–Dec of a whole-year model)
                em = [m for m in MONTHS if m in self.eval_months]
                if not em:
                    raise Missing(f'test windows in months {self.eval_months} (the model has months {MONTHS})')
                p = p.sel(time=p.time.dt.month.isin(em))
                print(f"  scoring on {months_label(em)} only: {p.sizes['time']} test windows "
                      f"(model trained on {months_label(MONTHS)})", flush=True)
                MONTHS = em
            self._cache['pred'] = p
        return self._cache['pred']

    # ── aggregated data file (inputs + WWLLN on the model grid, all years) ──
    def prep(self):
        if 'prep' not in self._cache:
            p = self.pred()
            if not self.prep_path or not os.path.exists(self.prep_path):
                raise Missing(f"aggregated data file {self.prep_path} (train_unet.py builds it)")
            ds = xr.open_dataset(self.prep_path)
            ds = ds.sel(lat=p.lat.values, lon=p.lon.values, method='nearest').load()
            ds = ds.assign_coords(lat=p.lat.values, lon=p.lon.values)
            self._cache['prep'] = ds
        return self._cache['prep']

    # ── everything derived: split, observations, baselines ──
    def core(self):
        if 'core' in self._cache:
            return self._cache['core']
        p, ds = self.pred(), self.prep()
        A = self.A
        years = pd.DatetimeIndex(ds.time.values).year
        tr_years = A.get('train_years') or []
        va_years = A.get('val_years') or []
        te_times = p.time.values
        idx_tr = np.where(np.isin(years, tr_years))[0]
        idx_va = np.where(np.isin(years, va_years))[0]
        t_index = {t: i for i, t in enumerate(ds.time.values)}
        idx_te = np.array([t_index[t] for t in te_times])
        ltg = np.nan_to_num(ds['ltg'].values.astype('float32'))
        mon = pd.DatetimeIndex(ds.time.values).month
        obs_te, obs_va = np.nan_to_num(p['obs_count'].values.astype('float32')), ltg[idx_va]
        c = {'times': te_times, 'blocks': block_ids(te_times), 'months': pd.DatetimeIndex(te_times).month,
             'obs': obs_te, 'obs_val': obs_va, 'idx_tr': idx_tr, 'idx_va': idx_va, 'idx_te': idx_te,
             'tr_years': tr_years, 'va_years': va_years, 'te_years': sorted(set(pd.DatetimeIndex(te_times).year)),
             'lats': p.lat.values, 'lons': p.lon.values}
        lsm = ds['lsm'] if 'lsm' in ds else None
        c['lsm'] = None if lsm is None else (lsm.isel(time=0) if 'time' in lsm.dims else lsm).values
        # models on the test windows (strokes per window); 'cal' = quantile-mapped on the validation year
        M, M_val = {}, {}
        M['U-Net'] = np.nan_to_num(p['pred_count'].values.astype('float32'))
        if 'pred_count_raw' in p:
            M['U-Net (raw)'] = np.nan_to_num(p['pred_count_raw'].values.astype('float32'))
        # training-year mean per month (a month without training windows gets the all-month mean)
        clim = {m: ltg[idx_tr][mon[idx_tr] == m].mean(0) if (mon[idx_tr] == m).any() else ltg[idx_tr].mean(0)
                for m in sorted(set(mon))}
        clim_va = np.stack([clim[m] for m in mon[idx_va]]); clim_te = np.stack([clim[m] for m in mon[idx_te]])
        M['Climatology'] = calibrate(clim_va, obs_va, clim_te)
        RAW = {'U-Net': M.get('U-Net (raw)', M['U-Net']), 'Climatology': clim_te}
        if 'cape' in ds and 'precipitation' in ds:
            cp = (np.nan_to_num(ds['cape'].values).clip(min=0) * np.nan_to_num(ds['precipitation'].values).clip(min=0))
            c_r14 = ltg[idx_tr].sum() / max(cp[idx_tr].sum(), 1e-12)
            RAW['CAPE × P'] = (c_r14 * cp[idx_te]).astype('float32')
            M['CAPE × P'] = calibrate(c_r14 * cp[idx_va], obs_va, RAW['CAPE × P'])
            c['r14_c'] = float(c_r14)
            c['r14_rate'] = RAW['CAPE × P']
            c['r14_rate_val'] = (c_r14 * cp[idx_va]).astype('float32')
        prev = np.full_like(obs_te, np.nan)
        step = np.timedelta64(self.agg, 'h')
        for j, i in enumerate(idx_te):
            k = t_index.get(ds.time.values[i] - step)
            if k is not None:
                prev[j] = ltg[k]
        c['valid_pers'] = np.isfinite(prev).all(axis=(1, 2))
        M['Persistence'] = np.nan_to_num(prev)
        if os.path.exists(self.path(PERCELL_FILE)):
            pc = xr.open_dataset(self.path(PERCELL_FILE)).sel(time=te_times)
            M['Per-cell model'] = np.nan_to_num(pc['pred_count'].values.astype('float32'))
        c['M'], c['RAW'] = M, RAW
        c['clim_by_month'] = clim
        # land / sea, cell area (km²) and the unit conversion strokes/window → strokes km⁻² yr⁻¹
        R = 6371.0
        lats, lons = c['lats'], c['lons']
        dlat, dlon = abs(np.diff(lats).mean()), abs(np.diff(lons).mean())
        area = (R * np.deg2rad(dlat)) * (R * np.deg2rad(dlon) * np.cos(np.deg2rad(lats)))[:, None] * np.ones((1, len(lons)))
        c['area'] = area
        c['to_density'] = lambda m: m * (365.25 * 24 / self.agg) / area
        c['ltg_all'], c['mon_all'], c['ds_times'] = ltg, mon, ds.time.values
        self._cache['core'] = c
        return c

    def valid(self, name, c):
        return c['valid_pers'] if name == 'Persistence' else None


# ═════════════════════════════════════════════════════════════════════════════════════════════
# Maps
# ═════════════════════════════════════════════════════════════════════════════════════════════
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    PROJ = ccrs.PlateCarree()
except Exception:
    PROJ = None
COAST = None      # coastlines from Natural Earth (downloaded once by cartopy); else the land–sea mask contour


def _coast_ok():
    global COAST
    if COAST is None:
        try:
            next(iter(cfeature.COASTLINE.with_scale('50m').geometries()))
            next(iter(cfeature.BORDERS.with_scale('50m').geometries()))
            COAST = True
        except Exception:
            print("  (cartopy could not get coastlines — drawing the land–sea mask outline instead)", flush=True)
            COAST = False
    return COAST


def map_axes(nrows, ncols, width, extra_h=0.0, hspace=0.08, wspace=0.04, extent=None):
    ext = extent
    aspect = (ext[1] - ext[0]) / (ext[3] - ext[2])
    h = width * 0.86 / ncols / aspect * nrows * (1 + hspace) + extra_h
    kw = {'subplot_kw': {'projection': PROJ}} if PROJ is not None else {}
    fig, axs = plt.subplots(nrows, ncols, figsize=(width, h), squeeze=False, **kw)
    fig.subplots_adjust(hspace=hspace, wspace=wspace)
    return fig, axs


def draw_map(ax, c, field, cmap, norm, extent=None, coast=True, ticks=(True, True), lats=None, lons=None):
    lats = c['lats'] if lats is None else lats
    lons = c['lons'] if lons is None else lons
    ext = extent or [lons.min() - 0.125, lons.max() + 0.125, lats.min() - 0.125, lats.max() + 0.125]
    if PROJ is not None:
        im = ax.pcolormesh(lons, lats, field, cmap=cmap, norm=norm, shading='nearest', transform=PROJ, rasterized=True)
        ax.set_extent(ext, crs=PROJ)
        if coast and _coast_ok():
            ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5, edgecolor='0.2')
            ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.3, edgecolor='0.5')
        elif coast:
            _lsm_coast(ax, c, lats, lons)
        st = 5 if ext[1] - ext[0] > 12 else 2
        xt = [x for x in range(int(np.ceil(ext[0] / st) * st), int(np.floor(ext[1])) + 1, st)]
        yt = [y for y in range(int(np.ceil(ext[2] / st) * st), int(np.floor(ext[3])) + 1, st)]
        if ticks[0]:
            ax.set_xticks(xt, crs=PROJ); ax.xaxis.set_major_formatter(LongitudeFormatter())
        if ticks[1]:
            ax.set_yticks(yt, crs=PROJ); ax.yaxis.set_major_formatter(LatitudeFormatter())
    else:
        im = ax.pcolormesh(lons, lats, field, cmap=cmap, norm=norm, shading='nearest', rasterized=True)
        if coast:
            _lsm_coast(ax, c, lats, lons)
        ax.set_xlim(ext[:2]); ax.set_ylim(ext[2:]); ax.set_aspect('equal')
        f = mticker.FuncFormatter(lambda v, _: f'{v:g}°')
        ax.xaxis.set_major_formatter(f); ax.yaxis.set_major_formatter(f)
        st = 5 if ext[1] - ext[0] > 12 else 2
        ax.xaxis.set_major_locator(mticker.MultipleLocator(st)); ax.yaxis.set_major_locator(mticker.MultipleLocator(st))
        if not ticks[0]:
            ax.set_xticklabels([])
        if not ticks[1]:
            ax.set_yticklabels([])
    ax.tick_params(labelsize=6.5, length=2)
    return im


def _lsm_coast(ax, c, lats, lons):
    if c.get('lsm') is None:
        return
    li = np.abs(c['lats'][None, :] - np.asarray(lats)[:, None]).argmin(1)     # works for either lat order
    lj = np.abs(c['lons'][None, :] - np.asarray(lons)[:, None]).argmin(1)
    l2 = c['lsm'][np.ix_(li, lj)]
    kw = {'transform': PROJ} if PROJ is not None else {}
    ax.contour(lons, lats, l2, levels=[0.5], colors='0.2', linewidths=0.5, **kw)


def box_outline(ax, box, **kw):
    lat0, lat1, lon0, lon1 = box
    t = {'transform': PROJ} if PROJ is not None else {}
    ax.plot([lon0, lon1, lon1, lon0, lon0], [lat0, lat0, lat1, lat1, lat0], **kw, **t)


def colorbar(fig, im, axs, label, **kw):
    cb = fig.colorbar(im, ax=axs, shrink=kw.pop('shrink', 0.85), pad=kw.pop('pad', 0.02), **kw)
    cb.set_label(label, fontsize=7); cb.ax.tick_params(labelsize=6.5)
    return cb


# ═════════════════════════════════════════════════════════════════════════════════════════════
# Main figures
# ═════════════════════════════════════════════════════════════════════════════════════════════
def F1_study_area(D):
    c = D.core()
    tr_m = c['idx_tr'][np.isin(c['mon_all'][c['idx_tr']], MONTHS)]     # training windows of the months shown
    clim = c['to_density'](c['ltg_all'][tr_m].mean(0))
    lats, lons = c['lats'], c['lons']
    ext = [lons.min() - 0.125, lons.max() + 0.125, lats.min() - 0.125, lats.max() + 0.125]
    fig, axs = map_axes(1, 1, W2, extra_h=0.3, extent=ext)
    ax = axs[0, 0]
    pos = clim[clim > 0]
    norm = mcolors.LogNorm(vmin=max(np.percentile(pos, 2), 1e-3), vmax=np.percentile(pos, 99.5))
    im = draw_map(ax, c, np.where(clim > 0, clim, np.nan), LTG_CMAP, norm)
    box_outline(ax, ILDN_BOX, color='#D55E00', lw=1.2)
    tkw = {'transform': PROJ} if PROJ is not None else {}
    ax.text(ILDN_BOX[3] - 0.1, ILDN_BOX[1] + 0.2, 'ILDN box', color='#D55E00', fontsize=7, ha='right', **tkw)
    colorbar(fig, im, ax, 'WWLLN strokes km$^{-2}$ yr$^{-1}$' + rate_note(), extend='both')
    yrs = c['tr_years']
    NUMBERS['F1'] = {'domain_mean_density': float(np.mean(clim)), 'max_density': float(np.max(clim))}
    save(fig, 'F01_study_area',
         f"Study domain ({lats.min():.2f}–{lats.max():.2f}°N, {lons.min():.2f}–{lons.max():.2f}°E, 0.25° grid) and mean "
         f"WWLLN lightning stroke density over the training years ({min(yrs)}–{max(yrs)}, "
         f"{'all months' if whole_year() else months_label(MONTHS)}), "
         f"in strokes km⁻² yr⁻¹. The orange box marks the area used for the comparison with the Israeli "
         f"network (ILDN).")


def F2_pipeline(D):
    p = D.pred(); A = D.A
    feats = A.get('features') or []
    ny, nx = p.sizes['lat'], p.sizes['lon']
    agg = D.agg
    n_imerg = sum(SOURCE.get(v, ('',))[0].startswith('IMERG') for v in feats)
    src = f"ERA5 ({len(feats) - n_imerg})" + (' + IMERG rain' if n_imerg else '')
    fig, ax = plt.subplots(figsize=(W2, 2.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 36); ax.axis('off')
    boxes = [
        (1, 'Inputs', f"{len(feats)} fields, {agg}-h means\n{src}\n{ny} × {nx} cells, 0.25°", '#E8F1F8'),
        (21, 'U-Net', 'encoder–decoder\nchannels 32, 16, 8,\n16, 32', '#FBE9E1'),
        (41, 'Poisson rate', 'λ = expected strokes\nper cell per window\nP(≥1) = 1 − e$^{-λ}$', '#FBE9E1'),
        (61, 'Calibration', f"quantile mapping\nfitted on the\nvalidation year {', '.join(map(str, D.A.get('val_years') or []))}", '#EEEEEE'),
        (81, 'Lightning map', f"strokes per cell\nper {agg} h", '#E8F1F8'),
    ]
    for x, title, body, fc in boxes:
        ax.add_patch(mpatches.FancyBboxPatch((x, 14), 17, 20, boxstyle='round,pad=0.4,rounding_size=1.5',
                                             fc=fc, ec='0.35', lw=0.8))
        ax.text(x + 8.5, 30.5, title, ha='center', va='center', fontsize=8, fontweight='bold')
        ax.text(x + 8.5, 22, body, ha='center', va='center', fontsize=6.6, linespacing=1.3)
    for x in (18.5, 38.5, 58.5, 78.5):
        ax.annotate('', xy=(x + 2.2, 24), xytext=(x, 24), arrowprops=dict(arrowstyle='-|>', color='0.25', lw=1))
    ax.add_patch(mpatches.FancyBboxPatch((33, 1), 25, 7, boxstyle='round,pad=0.4,rounding_size=1.5',
                                         fc='white', ec='0.35', lw=0.8, ls='--'))
    ax.text(45.5, 4.5, 'WWLLN strokes (target)\n→ Poisson loss during training', ha='center', va='center', fontsize=6.6)
    ax.annotate('', xy=(49.5, 13.6), xytext=(49.5, 8.8), arrowprops=dict(arrowstyle='-|>', color='0.35', lw=0.9, ls='--'))
    save(fig, 'F02_pipeline',
         f"Data and model pipeline. {len(feats)} input fields averaged over {agg}-h windows enter a U-Net-style "
         f"encoder–decoder that outputs the expected number of strokes per cell (a Poisson rate λ). The network is "
         f"trained with a Poisson loss against WWLLN strokes; its output is then calibrated by quantile mapping "
         f"fitted on the validation year.")


def _pred_file_fss(D, path, var, c):
    if not os.path.exists(path):
        return None
    f = xr.open_dataset(path)
    if var not in f:
        return None
    f = f.sel(time=c['times'])
    return fss_series(c['obs'], np.nan_to_num(f[var].values))


def F3_ablation(D):
    c = D.core()
    rows, series_of = [], []
    for label, fname, var, old in ABLATION:
        path = D.path(fname)
        series = _pred_file_fss(D, path, var, c) if fname else fss_series(c['obs'], c['M']['U-Net'] if var == 'pred_count'
                                                                         else c['M']['U-Net (raw)'])
        series_of.append(series)
        if series is not None:
            m, lo, hi = mean_ci(series, c['blocks'])
            rows.append((label, m, lo, hi, 'file'))
            continue
        if D.season != 'OND' or D.eval_months:     # the earlier runs were Oct–Dec runs scored on Oct–Dec
            rows.append((label, np.nan, np.nan, np.nan, 'missing'))
            continue
        js = os.path.join(OLD_FIG_DIR, old, 'run_summary.json')
        if not os.path.exists(js):
            js = os.path.join(HERE, 'figures', old, 'run_summary.json')
        if os.path.exists(js):
            m = json.load(open(js)).get('metrics', {}).get('fss_3x3')
            rows.append((label, m if m is not None else np.nan, np.nan, np.nan, 'earlier run'))
        else:
            rows.append((label, np.nan, np.nan, np.nan, 'missing'))
    if sum(r[4] != 'missing' for r in rows) < 2:
        raise Missing('at least two ablation steps (ABLATION files or figures/_old/<run>/run_summary.json)')
    r14 = mean_ci(fss_series(c['obs'], c['M']['CAPE × P']), c['blocks']) if 'CAPE × P' in c['M'] else None
    fig, ax = plt.subplots(figsize=(W1 + 0.6, 2.6))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, m, lo, hi, kind) in zip(y, rows):
        if not np.isfinite(m):
            continue
        filled = kind == 'file'
        if filled:
            ax.errorbar(m, yi, xerr=[[m - lo], [hi - m]], fmt='o', color=COL['U-Net'], ms=4, capsize=2, lw=1)
        else:
            ax.plot(m, yi, 'o', mfc='white', mec=COL['U-Net'], ms=4)
        ax.text(m + 0.012, yi + 0.12, f'{m:.2f}', fontsize=6.5, va='bottom')
    gains = {}
    for i in range(1, len(rows)):                   # gain over the previous step, paired on the same windows
        if series_of[i] is not None and series_of[i - 1] is not None:
            g = paired_ci(series_of[i], series_of[i - 1], c['blocks'])
            gains[rows[i][0]] = g
            ax.text(1.02, (y[i] + y[i - 1]) / 2, f"{g[0]:+.3f}\n[{g[1]:+.3f}, {g[2]:+.3f}]", transform=ax.get_yaxis_transform(),
                    fontsize=5.8, va='center', color='0.3')
    if gains:
        ax.text(1.02, y.max() + 0.6, 'gain over the\nstep above [95 % CI]', transform=ax.get_yaxis_transform(),
                fontsize=5.8, va='center', color='0.3', fontweight='bold')
    if r14:
        ax.axvline(r14[0], color=COL['CAPE × P'], ls='--', lw=0.9)
        ax.text(r14[0], y.max() + 0.6, 'CAPE × P\n(calibrated)', color=COL['CAPE × P'], fontsize=6.3, ha='center')
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel('FSS (3 × 3 cells, ≥ 1 stroke)')
    vals = [r[1] for r in rows if np.isfinite(r[1])] + ([r14[0]] if r14 else [])
    ax.set_xlim(max(0, min(vals) - 0.08), min(1, max(vals) + 0.1)); ax.set_ylim(-0.6, len(rows) - 0.2 + 0.8)
    if any(r[4] == 'earlier run' for r in rows):
        ax.plot([], [], 'o', mfc='white', mec=COL['U-Net'], ms=4, label='earlier run (no CI yet)')
        ax.errorbar([], [], xerr=[], fmt='o', color=COL['U-Net'], ms=4, label='this run, 95 % CI')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.3)
    NUMBERS['F3'] = {r[0]: {'fss': r[1], 'ci': [r[2], r[3]], 'source': r[4],
                            'gain_vs_previous': list(gains[r[0]]) if r[0] in gains else None} for r in rows}
    gain = rows[-1][1] - rows[0][1] if np.isfinite(rows[-1][1]) and np.isfinite(rows[0][1]) else np.nan
    vs_r14 = (paired_ci(fss_series(c['obs'], c['M']['U-Net']), fss_series(c['obs'], c['M']['CAPE × P']), c['blocks'])
              if 'CAPE × P' in c['M'] else None)
    NUMBERS['F3']['final_minus_cape_p'] = list(vs_r14) if vs_r14 else None
    save(fig, 'F03_ablation',
         f"Ablation from the set-up of Jones et al. (2026) to the final model. Each row adds one change; points show "
         f"the mean FSS (3 × 3 cells, ≥ 1 stroke per window) on the test years with 95 % block-bootstrap confidence "
         f"intervals. These intervals are wide because some days are harder than others for every model; whether a "
         f"step helps is tested on the per-window difference between consecutive steps (numbers on the right, "
         f"paired 95 % CI), where both have predictions. Total gain: {fmt(gain)}. The dashed line is the calibrated "
         f"CAPE × P baseline" + (f"; final model minus CAPE × P: {fmt_ci(*vs_r14, nd=3)} (paired)." if vs_r14 else '.')
         + (" Open circles: earlier runs, shown without intervals until their predictions are regenerated."
            if any(r[4] == 'earlier run' for r in rows) else ''))


def _models_for_skill(c):
    return [k for k in ['U-Net', 'Per-cell model', 'CAPE × P', 'Climatology', 'Persistence'] if k in c['M']]


def F4_fss_vs_scale(D):
    c = D.core()
    models = _models_for_skill(c)
    f0 = (c['obs'] >= THR).mean()
    fig, (ax, dx, bx) = plt.subplots(1, 3, figsize=(W2, 2.6), gridspec_kw={'width_ratios': [1.25, 1, 1]})
    res, per_w = {}, {}
    for name in models:
        mm, lo, hi = [], [], []
        per_w[name] = [fss_series(c['obs'], c['M'][name], w=s, valid=D.valid(name, c)) for s in SCALES]
        for sr in per_w[name]:
            m, l, h = mean_ci(sr, c['blocks'])
            mm.append(m); lo.append(l); hi.append(h)
        res[name] = (np.array(mm), np.array(lo), np.array(hi))
        ax.plot(SCALES, mm, '-o', color=COL[name], ms=2.5, lw=1.2, label=name)
        ax.fill_between(SCALES, lo, hi, color=COL[name], alpha=0.18, lw=0)
    diffs = {}                                         # (b) U-Net − model, paired on the same windows
    for name in [n for n in models if n not in ('U-Net', 'Climatology')]:
        d = np.array([paired_ci(per_w['U-Net'][k], per_w[name][k], c['blocks']) for k in range(len(SCALES))])
        diffs[name] = d
        dx.plot(SCALES, d[:, 0], '-o', color=COL[name], ms=2.5, lw=1.2)
        dx.fill_between(SCALES, d[:, 1], d[:, 2], color=COL[name], alpha=0.2, lw=0)
    dx.axhline(0, color='0.3', lw=0.7)
    dx.set_xticks(SCALES); dx.set_xticklabels([str(s) if s in (1, 3, 5, 9, 17, 25) else '' for s in SCALES])
    dx.set_xlabel('neighbourhood size (cells)'); dx.set_ylabel('FSS of U-Net − FSS of model')
    letter(dx, 'b')
    ax.axhline(0.5 + f0 / 2, color='0.4', ls=':', lw=0.8)
    ax.text(SCALES[-1], 0.5 + f0 / 2 + 0.01, 'useful skill', fontsize=6, ha='right', va='bottom', color='0.4')
    ax.set_xlabel('neighbourhood size (cells; 1 cell = 0.25°)'); ax.set_ylabel('FSS (≥ 1 stroke)')
    ax.set_xticks(SCALES); ax.set_ylim(0, 1); ax.legend(loc='lower right')
    letter(ax, 'a')
    # (c) by month (or by season for the whole year) at 3 × 3
    groups = period_groups(c['months'])
    width = 0.8 / len(models)
    for k, name in enumerate(models):
        s = fss_series(c['obs'], c['M'][name], valid=D.valid(name, c))
        for j, (_, sel) in enumerate(groups):
            mu, lo, hi = mean_ci(s[sel], c['blocks'][sel])
            bx.bar(j + (k - (len(models) - 1) / 2) * width, mu, width * 0.92, color=COL[name],
                   yerr=[[mu - lo], [hi - mu]] if np.isfinite(lo) else None, error_kw={'lw': 0.7, 'capsize': 1.5})
    bx.set_xticks(range(len(groups))); bx.set_xticklabels([g[0] for g in groups])
    bx.set_ylabel('FSS (3 × 3 cells)'); bx.set_ylim(0, 1)
    letter(bx, 'c')
    fig.tight_layout()
    i3 = SCALES.index(3) if 3 in SCALES else 1
    NUMBERS['F4'] = {n: {'fss_by_scale': dict(zip(map(str, SCALES), np.round(v[0], 4).tolist())),
                         'fss_3x3_ci': [float(v[1][i3]), float(v[2][i3])],
                         'unet_minus_model_3x3': diffs[n][i3].tolist() if n in diffs else None} for n, v in res.items()}
    u, b = res['U-Net'], res.get('CAPE × P')
    dr = diffs.get('CAPE × P')
    save(fig, 'F04_fss_vs_scale',
         f"(a) Fractions Skill Score for ≥ 1 stroke per {D.agg} h as a function of neighbourhood size, test years "
         f"{', '.join(map(str, c['te_years']))}. Shading: 95 % block-bootstrap confidence intervals. Dotted line: "
         f"useful-skill level 0.5 + f₀/2 (f₀ = {f0:.3f}). At 3 × 3 cells the U-Net reaches "
         f"{fmt_ci(u[0][i3], u[1][i3], u[2][i3])}"
         + (f" against {fmt_ci(b[0][i3], b[1][i3], b[2][i3])} for calibrated CAPE × P" if b else '') +
         ". The bands overlap because some days are harder than others for every model. (b) The difference that "
         "matters: FSS of the U-Net minus FSS of each baseline, computed window by window, with its 95 % "
         "block-bootstrap interval; above zero = U-Net better"
         + (f" (CAPE × P at 3 × 3 cells: {fmt_ci(*dr[i3], nd=3)})" if dr is not None else '') +
         ". Climatology is left out of (b) because its difference is off the scale. (c) FSS at 3 × 3 cells by "
         + ("season (DJF = Dec–Feb …)." if len(set(c['months'])) > 4 else "month."))


def _block_ttest(diff, blocks):
    """p-value per cell of a zero-mean test on block means of `diff` (time, lat, lon)."""
    ub, inv = np.unique(blocks, return_inverse=True)
    sums = np.zeros((len(ub),) + diff.shape[1:]); cnt = np.bincount(inv).astype(float)
    np.add.at(sums, inv, diff)
    bm = sums / cnt[:, None, None]
    return ttest_1samp(bm, 0.0, axis=0).pvalue


def F5_bias_maps(D):
    c = D.core()
    obs_d = c['to_density'](c['obs'].mean(0))
    models = [('U-Net', c['RAW']['U-Net'])] + ([('CAPE × P', c['RAW']['CAPE × P'])] if 'CAPE × P' in c['RAW'] else [])
    lats, lons = c['lats'], c['lons']
    ext = [lons.min() - 0.125, lons.max() + 0.125, lats.min() - 0.125, lats.max() + 0.125]
    fig, axs = map_axes(1 + len(models), 1, W2 * 0.78, extra_h=0.2, hspace=0.12, extent=ext)
    pos = obs_d[obs_d > 0]
    im0 = draw_map(axs[0, 0], c, np.where(obs_d > 0, obs_d, np.nan), LTG_CMAP,
                   mcolors.LogNorm(vmin=max(np.percentile(pos, 2), 1e-3), vmax=np.percentile(pos, 99.5)),
                   ticks=(False, True))
    colorbar(fig, im0, axs[0, 0], 'observed\nstrokes km$^{-2}$ yr$^{-1}$', extend='both')
    letter(axs[0, 0], 'a')
    diffs = {n: c['to_density'](m.mean(0)) - obs_d for n, m in models}
    vmax = np.nanpercentile(np.abs(np.concatenate([d.ravel() for d in diffs.values()])), 98)
    lev = mticker.MaxNLocator(nbins=10, symmetric=True).tick_values(-vmax, vmax)
    norm = mcolors.BoundaryNorm(lev, plt.cm.RdBu_r.N, extend='both')
    stats = {}
    for k, (name, m) in enumerate(models, start=1):
        d = diffs[name]
        pmap = _block_ttest(m - c['obs'], c['blocks'])
        ax = axs[k, 0]
        im = draw_map(ax, c, d, plt.cm.RdBu_r, norm, ticks=(k == len(models), True))
        ns = (pmap >= 0.05) | ~np.isfinite(pmap)
        yy, xx = np.where(~ns)                                      # dots = significant
        sk = {'transform': PROJ} if PROJ is not None else {}
        ax.scatter(lons[xx], lats[yy], s=1.2, c='k', marker='.', lw=0, **sk)
        ax.text(0.01, 0.03, f'{name} − observed', transform=ax.transAxes, fontsize=7,
                bbox=dict(fc='white', ec='none', alpha=0.8, pad=1))
        letter(ax, 'abc'[k])
        stats[name] = {'mean_bias': float(np.mean(d)), 'frac_significant': float(1 - ns.mean())}
        last = im
    colorbar(fig, last, axs[1:, 0], 'difference, strokes km$^{-2}$ yr$^{-1}$', extend='both')
    NUMBERS['F5'] = stats
    save(fig, 'F05_bias_maps',
         f"(a) Observed mean WWLLN stroke density on the test years (strokes km⁻² yr⁻¹"
         f"{'' if whole_year() else ', ' + months_label(MONTHS) + ' rate'}). "
         "(b) U-Net (expected strokes = Poisson rate, before calibration) minus observed and (c) CAPE × P (constant "
         "fitted on the training years) minus observed. "
         f"Dots mark cells where the difference is significant (p < 0.05, t-test on {BLOCK_DAYS}-day block "
         "means). " + ' '.join(f"{n}: mean bias {fmt(s['mean_bias'], 3)}, significant in {100 * s['frac_significant']:.0f} % of cells."
                               for n, s in stats.items()))


def F6_cell_scatter(D):
    c = D.core()
    if c['lsm'] is None:
        raise Missing('land–sea mask (lsm) in the aggregated data file')
    obs_d = c['to_density'](c['obs'].mean(0))
    models = [('U-Net', c['to_density'](c['RAW']['U-Net'].mean(0)))]
    if 'CAPE × P' in c['RAW']:
        models.append(('CAPE × P', c['to_density'](c['RAW']['CAPE × P'].mean(0))))
    rows = [('land', c['lsm'] >= LAND_LSM), ('sea', c['lsm'] < LAND_LSM)]
    allv = np.concatenate([obs_d.ravel()] + [m.ravel() for _, m in models])
    pos = allv[allv > 0]
    lo, hi = 10 ** np.floor(np.log10(np.percentile(pos, 0.5))), 10 ** np.ceil(np.log10(pos.max()))
    edges = np.linspace(np.log10(lo), np.log10(hi), 41)
    fig, axs = plt.subplots(2, len(models), figsize=(W1 * len(models) * 0.85 + 0.5, 2 * W1 * 0.8), squeeze=False)
    stats, hmax, panels = {}, 1, {}
    for i, (rname, rm) in enumerate(rows):
        for j, (mname, m) in enumerate(models):
            x, y = obs_d[rm], m[rm]
            ok = (x > 0) & (y > 0)
            H, _, _ = np.histogram2d(np.log10(x[ok]), np.log10(y[ok]), bins=[edges, edges])
            panels[(i, j)] = H.T; hmax = max(hmax, H.max())
            lx, ly = np.log10(x[ok]), np.log10(y[ok])
            r2 = 1 - np.sum((ly - lx) ** 2) / np.sum((lx - lx.mean()) ** 2) if ok.sum() > 2 else np.nan
            stats[f'{mname} {rname}'] = {'r': float(pearsonr(lx, ly)[0]) if ok.sum() > 2 else np.nan,
                                         'r2_log_vs_1to1': float(r2), 'n_cells': int(rm.sum())}
    norm = mcolors.LogNorm(vmin=1, vmax=hmax)
    for (i, j), H in panels.items():
        ax = axs[i, j]
        im = ax.pcolormesh(10 ** edges, 10 ** edges, np.ma.masked_less(H, 1), cmap='viridis', norm=norm, rasterized=True)
        ax.plot([lo, hi], [lo, hi], 'k-', lw=0.8)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
        st = stats[f'{models[j][0]} {rows[i][0]}']
        ax.text(0.04, 0.96, f"{models[j][0]}, {rows[i][0]}\nr = {fmt(st['r'])}, R² (1:1) = {fmt(st['r2_log_vs_1to1'])}",
                transform=ax.transAxes, va='top', fontsize=6.5)
        if i == 1:
            ax.set_xlabel('observed, strokes km$^{-2}$ yr$^{-1}$')
        if j == 0:
            ax.set_ylabel('modelled, strokes km$^{-2}$ yr$^{-1}$')
        letter(ax, 'abcd'[i * len(models) + j])
    fig.tight_layout()
    colorbar(fig, im, axs, 'number of cells', shrink=0.6)
    NUMBERS['F6'] = stats
    save(fig, 'F06_cell_scatter_land_sea',
         "Mean stroke density per grid cell on the test years, modelled against observed (log–log), over land "
         f"(top, lsm ≥ {LAND_LSM}) and sea (bottom). Modelled = expected strokes: the U-Net Poisson rate before "
         "calibration, and CAPE × P with its constant fitted on the training years. Colours: number of cells; black line: 1:1. r is the correlation "
         "of the log values; R² measures agreement with the 1:1 line.")


def F7_intensity(D):
    c = D.core()
    models = _models_for_skill(c)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(W2, 2.6), gridspec_kw={'width_ratios': [1.5, 1]})
    width = 0.8 / len(models)
    res = {}
    labels = [f'{lo:g}–{hi:g}' if np.isfinite(hi) else f'≥ {lo:g}' for lo, hi in INTENSITY_BINS]
    for k, name in enumerate(models):
        for j, (lo, hi) in enumerate(INTENSITY_BINS):
            s = fss_series(c['obs'], c['M'][name], thr=lo, hi=None if not np.isfinite(hi) else hi, valid=D.valid(name, c))
            mu, l, h = mean_ci(s, c['blocks'])
            res[(name, labels[j])] = (mu, l, h)
            ax.bar(j + (k - (len(models) - 1) / 2) * width, mu, width * 0.92, color=COL[name], label=name if j == 0 else None,
                   yerr=[[mu - l], [h - mu]] if np.isfinite(l) else None, error_kw={'lw': 0.7, 'capsize': 1.5})
    ax.set_xticks(range(len(INTENSITY_BINS))); ax.set_xticklabels(labels)
    ax.set_xlabel(f'strokes per cell per {D.agg} h'); ax.set_ylabel('FSS (3 × 3 cells)'); ax.set_ylim(0, 1)
    fig.legend(*ax.get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(0.5, 0.98), ncol=len(models))
    letter(ax, 'a')
    tot = c['obs'].sum((1, 2))
    tail = tot >= np.percentile(tot, TAIL_PCT)
    for k, name in enumerate(models):
        s = fss_series(c['obs'], c['M'][name], valid=D.valid(name, c))
        for j, sel in enumerate([np.ones_like(tail), tail]):
            mu, l, h = mean_ci(s[sel], c['blocks'][sel])
            res[(name, ['all windows', 'most active'][j])] = (mu, l, h)
            bx.bar(j + (k - (len(models) - 1) / 2) * width, mu, width * 0.92, color=COL[name],
                   yerr=[[mu - l], [h - mu]] if np.isfinite(l) else None, error_kw={'lw': 0.7, 'capsize': 1.5})
    bx.set_xticks([0, 1]); bx.set_xticklabels(['all windows', f'most active {100 - TAIL_PCT} %'])
    bx.set_ylabel('FSS (3 × 3 cells, ≥ 1 stroke)'); bx.set_ylim(0, 1); letter(bx, 'b')
    fig.tight_layout()
    NUMBERS['F7'] = {f'{a} | {b}': [float(x) for x in v] for (a, b), v in res.items()}
    save(fig, 'F07_skill_by_intensity',
         f"(a) FSS (3 × 3 cells) for lightning-intensity bins, in strokes per cell per {D.agg} h (an event is a cell "
         f"inside the bin, as in the MFSS of Jones et al. 2026). (b) FSS for all test windows and for the "
         f"{100 - TAIL_PCT} % of windows with the most observed strokes. Error bars: 95 % block-bootstrap intervals.")


def _prob_models(D, c):
    p = D.pred()
    P = {}
    if 'pred_prob' in p:
        P['U-Net'] = np.nan_to_num(p['pred_prob'].values)
    elif 'pred_count_raw' in p:
        P['U-Net'] = 1 - np.exp(-np.nan_to_num(p['pred_count_raw'].values))
    else:
        raise Missing('pred_prob / pred_count_raw in the predictions file')
    if 'r14_rate' in c:
        P['CAPE × P'] = 1 - np.exp(-c['r14_rate'])
    ltg, mon, tr = c['ltg_all'], c['mon_all'], c['idx_tr']
    freq = {m: (ltg[tr][mon[tr] == m] >= THR).mean(0) if (mon[tr] == m).any() else (ltg[tr] >= THR).mean(0)
            for m in sorted(set(mon))}
    P['Climatology'] = np.stack([freq[m] for m in c['months']])
    return P


def _val_rate(D, c):
    """U-Net Poisson rate on the validation windows (rebuilt from the checkpoint)."""
    if 'val_rate' not in D._cache:
        mc = ModelCtx(D)
        iv, Xv = mc.inputs(c['idx_va'])
        D._cache['val_rate'] = (iv, mc.rate(Xv))
    return D._cache['val_rate']


def _calibrate_probs(D, c, P):
    """Isotonic regression of the observed event on the forecast probability, fitted on the validation year."""
    from sklearn.isotonic import IsotonicRegression
    out_, notes = {}, []
    val = {}
    try:
        iv, rv = _val_rate(D, c)
        val['U-Net'] = (1 - np.exp(-rv), c['ltg_all'][iv] >= THR)
    except Missing as e:
        notes.append(f'U-Net probabilities not calibrated (needs {e})')
    if 'r14_rate_val' in c:
        val['CAPE × P'] = (1 - np.exp(-c['r14_rate_val']), c['obs_val'] >= THR)
    for name, (pv, ov) in val.items():
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(pv.ravel(), ov.ravel().astype(float))
        out_[name] = iso.predict(P[name].ravel()).reshape(P[name].shape)
    return out_, notes


def F8_reliability(D):
    c = D.core()
    P_raw = _prob_models(D, c)
    P_cal, notes = _calibrate_probs(D, c, P_raw)
    for n in notes:
        print('   ', n, flush=True)
    P = {**P_raw, **P_cal}                       # calibrated where possible
    o = (c['obs'] >= THR).ravel()
    edges = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(W2, 3.2))
    clim_bs = None
    stats = {}
    for name in ['Climatology', 'CAPE × P', 'U-Net']:
        if name not in P:
            continue
        f = P[name].ravel()
        bs = np.mean((f - o) ** 2)
        if name == 'Climatology':
            clim_bs = bs
        idx = np.digitize(f, edges) - 1
        mf = np.array([f[idx == i].mean() if np.any(idx == i) else np.nan for i in range(len(edges) - 1)])
        of = np.array([o[idx == i].mean() if np.any(idx == i) else np.nan for i in range(len(edges) - 1)])
        n = np.array([np.sum(idx == i) for i in range(len(edges) - 1)])
        ok = n >= 30
        bss = 1 - bs / clim_bs if clim_bs else np.nan
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            rng = np.random.default_rng(0)
            sub = rng.choice(f.size, size=min(f.size, 3_000_000), replace=False)
            auc, ap = roc_auc_score(o[sub], f[sub]), average_precision_score(o[sub], f[sub])
        except Exception:
            auc = ap = np.nan
        stats[name] = {'brier': float(bs), 'bss_vs_clim': float(bss), 'roc_auc': float(auc), 'pr_auc': float(ap),
                       'calibrated': name in P_cal}
        ax.plot(mf[ok], of[ok], '-o', ms=2.5, lw=1.2, color=COL[name],
                label=f"{name} (BSS {fmt(bss)})" if name != 'Climatology' else name)
        if name in P_cal:                        # before calibration: thin dashed line
            fr = P_raw[name].ravel()
            ir = np.digitize(fr, edges) - 1
            mr = np.array([fr[ir == i].mean() if np.sum(ir == i) >= 30 else np.nan for i in range(len(edges) - 1)])
            orr = np.array([o[ir == i].mean() if np.sum(ir == i) >= 30 else np.nan for i in range(len(edges) - 1)])
            stats[name]['bss_before_calibration'] = float(1 - np.mean((fr - o) ** 2) / clim_bs) if clim_bs else np.nan
            ax.plot(mr, orr, '--', lw=0.8, color=COL[name], alpha=0.7)
        thr = np.linspace(0.02, 0.98, 49)
        pod, sr = [], []
        for t in thr:
            pb = f >= t
            h = np.sum(pb & o); m_ = np.sum(~pb & o); fa = np.sum(pb & ~o)
            pod.append(h / max(h + m_, 1)); sr.append(h / max(h + fa, 1) if h + fa else np.nan)
        bx.plot(sr, pod, '-', lw=1.2, color=COL[name], label=f'{name} (AUC {fmt(auc)})')
    ax.plot([0, 1], [0, 1], 'k:', lw=0.7)
    ax.set_xlabel('forecast probability of ≥ 1 stroke'); ax.set_ylabel('observed frequency')
    if P_cal:
        ax.plot([], [], '--', lw=0.8, color='0.4', label='before calibration')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal'); ax.legend(loc='upper left'); letter(ax, 'a')
    ins = ax.inset_axes([0.6, 0.08, 0.36, 0.25])
    ins.hist(P['U-Net'].ravel(), bins=edges, color=COL['U-Net'], log=True)
    ins.set_xticks([0, 0.5, 1]); ins.tick_params(labelsize=5); ins.set_title('U-Net forecasts', fontsize=5.5)
    ins.yaxis.set_major_locator(mticker.LogLocator(numticks=3)); ins.yaxis.set_minor_formatter(mticker.NullFormatter())
    # performance diagram background: CSI contours and frequency-bias lines
    srg, podg = np.meshgrid(np.linspace(0.01, 1, 200), np.linspace(0.01, 1, 200))
    csi = 1 / (1 / srg + 1 / podg - 1)
    cs = bx.contour(srg, podg, csi, levels=np.arange(0.1, 1, 0.1), colors='0.75', linewidths=0.5)
    bx.clabel(cs, fmt='%.1f', fontsize=5)
    for b in (0.5, 1, 2):                         # frequency bias = POD / SR
        xs = np.linspace(0, min(1, 1 / b), 50)
        bx.plot(xs, b * xs, ls='--', lw=0.5, color='0.6')
        bx.text(xs[-1], b * xs[-1], f' bias {b:g}', fontsize=5, color='0.5', va='bottom' if b <= 1 else 'top',
                ha='left' if b < 1 else 'center')
    cat = categorical(c['obs'], c['M']['U-Net'])
    bx.plot(cat['SR'], cat['POD'], '*', color=COL['U-Net'], ms=8, mec='k', mew=0.4, label='U-Net map (≥ 1 stroke)')
    bx.set_xlim(0, 1); bx.set_ylim(0, 1); bx.set_aspect('equal')
    bx.set_xlabel('success ratio (1 − false alarm ratio)'); bx.set_ylabel('probability of detection')
    bx.legend(loc='lower left', fontsize=6); letter(bx, 'b')
    fig.tight_layout()
    NUMBERS['F8'] = stats
    u = stats.get('U-Net', {})
    save(fig, 'F08_reliability',
         "(a) Reliability diagram for the probability of at least one stroke per cell and window. For the U-Net and "
         "CAPE × P the probability starts as 1 − e^(−λ), with λ the predicted Poisson rate. Because lightning is "
         "clustered (overdispersed) and the predicted rate is spread over a wider area than the strokes, this "
         "Poisson probability is far too confident (dashed lines"
         + (f"; U-Net BSS {fmt(u.get('bss_before_calibration'))}" if 'bss_before_calibration' in u else '') +
         "). Solid lines: after calibrating it on the validation year (isotonic regression). For climatology it is "
         "the training-year frequency per cell and month. BSS: Brier skill score relative to climatology. Inset: "
         "number of calibrated U-Net forecasts per probability bin (log scale). (b) Performance diagram: probability of detection against "
         "success ratio for probability thresholds 0.02–0.98; grey curves are CSI, dashed lines frequency bias. "
         f"U-Net: BSS {fmt(u.get('bss_vs_clim'))}, ROC AUC {fmt(u.get('roc_auc'))}, PR AUC {fmt(u.get('pr_auc'))}.")


def F9_season(D):
    c = D.core()
    years = c['te_years']
    t = pd.DatetimeIndex(c['times'])
    fig, axs = plt.subplots(2 * len(years), 1, figsize=(W2, 2.0 * len(years) + 0.4), sharex=False,
                            gridspec_kw={'height_ratios': [1.3, 1] * len(years)})
    series = {'Observed': c['obs'].sum((1, 2)), 'U-Net': c['RAW']['U-Net'].sum((1, 2))}
    if 'CAPE × P' in c['RAW']:
        series['CAPE × P'] = c['RAW']['CAPE × P'].sum((1, 2))
    fss = {n: fss_series(c['obs'], c['M'][n], valid=D.valid(n, c)) for n in ['U-Net', 'CAPE × P', 'Climatology'] if n in c['M']}
    stats = {}
    for k, y in enumerate(years):
        sel = t.year == y
        ax, bx = axs[2 * k], axs[2 * k + 1]
        for n, s in series.items():
            ax.plot(t[sel], s[sel] + 1, color=COL[n], lw=1.3 if n == 'Observed' else 1.0, label=n)
        ax.set_yscale('log'); ax.set_ylabel(f'strokes / {D.agg} h\n(domain, +1)')
        ax.text(0.005, 0.97, str(y), transform=ax.transAxes, fontweight='bold', va='top')
        ax.set_xticklabels([])
        for n, s in fss.items():
            r = pd.Series(s[sel]).rolling(10, min_periods=4, center=True).mean()
            bx.plot(t[sel], r, color=COL[n], lw=1.0)
        bx.set_ylim(0, 1); bx.set_ylabel('FSS, 5-day\nrunning mean')
        if (t[sel].max() - t[sel].min()).days > 120:          # whole year: one tick per month
            bx.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
            bx.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b'))
        else:
            bx.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d %b'))
        rr = spearmanr(series['Observed'][sel], series['U-Net'][sel])[0]
        stats[str(y)] = {'spearman_domain_total_unet': float(rr)}
        letter(ax, 'abcdef'[2 * k]); letter(bx, 'abcdef'[2 * k + 1])
    fig.tight_layout()
    fig.legend(*axs[0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3)
    NUMBERS['F9'] = stats
    save(fig, 'F09_season_timeseries',
         f"Lightning through each test {D.period_word}. Top of each pair: strokes per {D.agg}-h window summed over the domain, "
         "observed and modelled as expected strokes (U-Net Poisson rate; CAPE × P with its fitted constant), log scale. "
         "Bottom: FSS (3 × 3 cells, ≥ 1 stroke) as a 5-day running mean for the calibrated U-Net and baselines. " +
         ' '.join(f"{y}: Spearman r of domain totals {fmt(v['spearman_domain_total_unet'])}." for y, v in stats.items()))


def _cases_figure(D, c, idx, name, caption):
    cols = [('WWLLN (observed)', c['obs'])] + [(n, c['M'][n]) for n in ['U-Net', 'CAPE × P'] if n in c['M']]
    lats, lons = c['lats'], c['lons']
    ext = [lons.min() - 0.125, lons.max() + 0.125, lats.min() - 0.125, lats.max() + 0.125]
    fig, axs = map_axes(len(idx), len(cols), W2, extra_h=0.3, hspace=0.18, wspace=0.03, extent=ext)
    vals = c['obs'][idx]
    vmax = max(np.percentile(vals[vals > 0], 99) if (vals > 0).any() else 10, THR * 2)
    norm = mcolors.LogNorm(vmin=THR, vmax=vmax)
    fss = {n: fss_series(c['obs'], m) for n, m in cols[1:]}
    for r, t in enumerate(idx):
        for j, (n, arr) in enumerate(cols):
            ax = axs[r, j]
            im = draw_map(ax, c, np.where(arr[t] >= THR, arr[t], np.nan), CASE_CMAP, norm,
                          ticks=(r == len(idx) - 1, j == 0))
            lab = pd.Timestamp(c['times'][t]).strftime('%d %b %Y %H UTC') if j == 0 else f"FSS {fmt(fss[n][t])}"
            ax.set_title(f"{n}: {lab}" if j else f"{lab}", fontsize=6.8, loc='left', pad=2)
    colorbar(fig, im, axs, f'strokes per cell per {D.agg} h', extend='max', shrink=0.7)
    save(fig, name, caption)


def F10_cases(D):
    c = D.core()
    tot = c['obs'].sum((1, 2))
    idx = np.argsort(tot)[::-1][:N_CASES]
    idx = idx[tot[idx] > 0]
    NUMBERS['F10'] = {pd.Timestamp(c['times'][t]).strftime('%Y-%m-%d %H'): {
        'obs_strokes': float(tot[t]), 'fss_unet': float(fss_series(c['obs'][t:t + 1], c['M']['U-Net'][t:t + 1])[0])}
        for t in idx}
    _cases_figure(D, c, idx, 'F10_case_studies',
                  f"Case studies: the {len(idx)} test windows with the most observed strokes. Columns: WWLLN, U-Net and "
                  f"calibrated CAPE × P, in strokes per cell per {D.agg} h (white: < {THR:g}). FSS (3 × 3 cells) in "
                  "each panel title.")


# ── model-based items (F11, F13c, O3) ────────────────────────────────────────────────────────
class ModelCtx:
    """Rebuilds the model inputs exactly as train_unet.py does and loads a checkpoint."""
    def __init__(self, D, ckpt_path=None, n_lags=1):
        try:
            import torch, torch.nn as nn
        except ImportError:
            raise Missing('PyTorch (run this item where train_unet.py runs)')
        self.torch = torch
        cfg, c = D.cfg, D.core()
        if cfg.get('_model_src') is None:
            raise Missing(f'SimpleCNN class in {D.args.train_script}')
        ckpt = ckpt_path or D.ckpt_path
        if not ckpt or not os.path.exists(ckpt):
            raise Missing(f'checkpoint {ckpt}')
        if (D.A.get('loss') or 'poisson') != 'poisson':
            raise Missing('a Poisson model (this item reads the model output as a Poisson rate)')
        ns = {'nn': nn, 'torch': torch}
        exec(cfg['_model_src'], ns)
        ds = D.prep()
        feats = list(D.A.get('features') or cfg['feature_name'])
        logf = list(D.A.get('log_features') or cfg.get('LOG_FEATURES', []))
        dsm = ds[feats].copy()
        for v in logf:
            if v in dsm:
                dsm[v] = np.log1p(dsm[v].clip(min=0))
        tr = dsm.isel(time=c['idx_tr'])
        xm = tr.groupby('time.month').mean(dim=['time', 'lat', 'lon'])
        xs = tr.groupby('time.month').std(dim=['time', 'lat', 'lon'])
        Z = ((dsm.groupby('time.month') - xm).groupby('time.month') / xs).fillna(0)
        self.Z = np.nan_to_num(np.stack([Z[v].values for v in feats], axis=1)).astype('float32')   # (T, F, y, x)
        self.feats, self.n_lags = feats, n_lags
        self.model = ns['SimpleCNN'](len(feats) * n_lags, len(c['lats']), len(c['lons']))
        self.model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        self.model.eval()
        self.t_index = {t: i for i, t in enumerate(ds.time.values)}
        self.step = np.timedelta64(D.agg, 'h')
        self.ds_times = ds.time.values
        self.c = c

    def inputs(self, idx):
        """(n, F·n_lags, y, x); slots ordered oldest → newest. Windows without history are dropped."""
        keep, X = [], []
        for i in idx:
            src = [self.t_index.get(self.ds_times[i] - k * self.step) for k in range(self.n_lags - 1, -1, -1)]
            if any(s is None for s in src):
                continue
            keep.append(i); X.append(np.concatenate([self.Z[s] for s in src], axis=0))
        return np.array(keep, dtype=int), np.stack(X)

    def rate(self, X, bs=32):
        torch = self.torch
        with torch.no_grad():
            out = [self.model(torch.from_numpy(X[i:i + bs])).numpy()[:, 0] for i in range(0, len(X), bs)]
        return np.exp(np.clip(np.concatenate(out), -20, 12))


def F11_importance(D):
    c = D.core()
    mc = ModelCtx(D)
    iv, Xv = mc.inputs(c['idx_va']); it, Xt = mc.inputs(c['idx_te'])
    ltg = c['ltg_all']
    pv = mc.rate(Xv)
    obs_t, obs_v = ltg[it], ltg[iv]
    base_pred = calibrate(pv, obs_v, mc.rate(Xt))
    blocks = block_ids(c['ds_times'][it])
    base = np.nanmean(fss_series(obs_t, base_pred))
    saved = np.nanmean(fss_series(c['obs'], c['M']['U-Net']))
    print(f"    re-computed FSS {base:.3f} (predictions file: {saved:.3f})", flush=True)
    if abs(base - saved) > 0.02:
        print("    WARNING: the checkpoint does not reproduce the predictions file closely — is it the same run?", flush=True)
    rng = np.random.default_rng(7)
    drops = {}
    for k, v in enumerate(mc.feats):
        d = []
        for r in range(N_PERM):
            Xp = Xt.copy()
            if np.allclose(Xt[:, k].std(axis=0), 0):        # time-constant input (lsm): shuffle cells instead
                flat = Xp[:, k].reshape(len(Xp), -1)
                Xp[:, k] = flat[:, rng.permutation(flat.shape[1])].reshape(Xp[:, k].shape)
            else:
                Xp[:, k] = Xt[rng.permutation(len(Xt)), k]
            d.append(base - np.nanmean(fss_series(obs_t, calibrate(pv, obs_v, mc.rate(Xp)))))
        drops[v] = (np.mean(d), np.std(d))
    order = sorted(drops, key=lambda v: drops[v][0])
    fig, ax = plt.subplots(figsize=(W1 + 0.4, 2.6))
    ax.barh(range(len(order)), [drops[v][0] for v in order], xerr=[drops[v][1] for v in order],
            color=COL['U-Net'], height=0.6, error_kw={'lw': 0.7, 'capsize': 1.5})
    ax.set_yticks(range(len(order))); ax.set_yticklabels([NICE.get(v, v) for v in order])
    ax.set_xlabel('drop in FSS when the input is shuffled'); ax.axvline(0, color='0.3', lw=0.6)
    NUMBERS['F11'] = {'base_fss': float(base), 'drops': {v: [float(a), float(b)] for v, (a, b) in drops.items()}}
    top = order[-1]
    save(fig, 'F11_input_importance',
         f"Permutation importance: the drop in mean FSS (3 × 3 cells, ≥ 1 stroke; {fmt(base)} without shuffling) when "
         f"one input is shuffled across the test windows (land–sea mask: shuffled across cells), mean ± s.d. of "
         f"{N_PERM} repeats. The most important input is {NICE.get(top, top)}.")


def F12_ildn(D):
    c, r = D.core(), _ildn(D)
    names = ['ILDN', 'WWLLN', 'U-Net'] + (['CAPE × P'] if 'CAPE × P' in r['fields'] else [])
    means = {n: (r['I'] if n == 'ILDN' else r['fields'][n]).mean(0) for n in names}
    rel = {n: v / max(v.mean(), 1e-12) for n, v in means.items()}
    pos = np.concatenate([v[v > 0] for v in rel.values()])
    norm = mcolors.LogNorm(vmin=max(np.percentile(pos, 2), 1e-2), vmax=np.percentile(pos, 99.5))
    years = r['years']
    fig = plt.figure(figsize=(W2, 2.1 + 1.5 * len(years)))
    gs = fig.add_gridspec(1 + len(years), len(names), height_ratios=[1.35] + [1] * len(years), hspace=0.45, wspace=0.08)
    blat, blon = r['blat'], r['blon']
    ext = [blon.min() - 0.125, blon.max() + 0.125, blat.min() - 0.125, blat.max() + 0.125]
    for j, n in enumerate(names):
        kw = {'projection': PROJ} if PROJ is not None else {}
        ax = fig.add_subplot(gs[0, j], **kw)
        im = draw_map(ax, c, np.ma.masked_less_equal(rel[n], 0), LTG_CMAP, norm, extent=ext, lats=blat, lons=blon,
                      ticks=(True, j == 0))
        sub = '' if n == 'ILDN' else f"  r = {fmt(r['scores']['all'][n]['map_spearman'])}"
        ax.set_title(f'{n}{sub}', fontsize=7, loc='left', pad=2)
        letter(ax, 'abcd'[j])
    cb = fig.colorbar(im, ax=fig.axes[:len(names)], shrink=0.8, pad=0.02, extend='both')
    cb.set_label('mean ÷ box mean', fontsize=6.5); cb.ax.tick_params(labelsize=6)
    t = r['times']
    for k, y in enumerate(years):
        ax = fig.add_subplot(gs[1 + k, :])
        m = t.year == y
        for n, s, lw in [('ILDN', r['I'][m].sum((1, 2)), 1.2), ('WWLLN', r['fields']['WWLLN'][m].sum((1, 2)), 1.0),
                         ('U-Net', r['fields']['U-Net'][m].sum((1, 2)), 1.0)]:
            ax.plot(t[m], s + 1, color=COL[n], lw=lw, label=n)
        ax.set_yscale('log'); ax.set_ylabel(f'strokes / {D.agg} h\nin box (+1)')
        sc = r['scores'][str(y)]
        ax.text(0.005, 0.95, f"{y}   Spearman r with ILDN: U-Net {fmt(sc['U-Net']['spearman_box_total'])}, "
                f"WWLLN {fmt(sc['WWLLN']['spearman_box_total'])}", transform=ax.transAxes, va='top', fontsize=6.5)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d %b'))
        if k == 0:
            ax.legend(loc='upper right', ncol=3)
        letter(ax, 'efgh'[k])
    NUMBERS['F12'] = r['scores']
    save(fig, 'F12_ildn',
         f"Independent comparison with the Israeli Lightning Detection Network (ILDN) in the box "
         f"{ILDN_BOX[0]}–{ILDN_BOX[1]}°N, {ILDN_BOX[2]}–{ILDN_BOX[3]}°E, {r['period']}. "
         "(a–d) Mean lightning divided by its box mean, so patterns are comparable despite different detection "
         "efficiencies; r is the Spearman correlation with the ILDN map. Bottom: strokes per window in the box. "
         "U-Net: expected strokes (Poisson rate before calibration); CAPE × P with its fitted constant. ILDN detects "
         "best near its sensors in Israel, so its map is weighted toward the land and coast.")


def _ildn(D):
    if 'ildn' in D._cache:
        return D._cache['ildn']
    files = sorted(glob.glob(os.path.join(D.data_dir, ILDN_GLOB)))
    if not files:
        raise Missing(f'ILDN files {ILDN_GLOB} in {D.data_dir}')
    c = D.core()
    p = D.pred()
    uvar = 'pred_count_raw' if 'pred_count_raw' in p else 'pred_count'     # expected strokes (Poisson rate)
    parts = [p[[uvar, 'obs_count']]]
    ex = re.sub(r'_predictions\.nc$', '_predictions_extra.nc', D.pred_path)
    if os.path.exists(ex):
        parts.append(xr.open_dataset(ex)[[uvar, 'obs_count']].load())
    pr = xr.concat(parts, dim='time').sortby('time')
    cols = ['date', 'time', 'lat', 'lon', 'peak_ka', 'c6', 'n_sensors', 'c8', 'c9', 'type', 'c11', 'c12', 'c13']
    dfs = []
    for f in files:
        df = pd.read_csv(f, sep=r'\s+', header=None, names=cols, usecols=['date', 'time', 'lat', 'lon', 'type'])
        df['t'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
        dfs.append(df.dropna(subset=['t']))
    il = pd.concat(dfs, ignore_index=True)
    il['t'] = il['t'] - pd.Timedelta(hours=ILDN_TIME_OFFSET_H)
    il = il[il.t.dt.month.isin(MONTHS)]
    years = sorted(set(il.t.dt.year) & set(pd.DatetimeIndex(pr.time.values).year))
    if not years:
        raise Missing('a year with both ILDN data and model predictions')
    covered = set(zip(il.t.dt.year, il.t.dt.month))          # (year, month) with ILDN data; a month without
    tt = pd.DatetimeIndex(pr.time.values)                     # a single ILDN stroke counts as not covered
    pr = pr.sel(time=np.array([(y, m) in covered for y, m in zip(tt.year, tt.month)]))
    times = pd.DatetimeIndex(pr.time.values)
    lats, lons = c['lats'], c['lons']
    dlat, dlon = lats[1] - lats[0], lons[1] - lons[0]
    il['i'] = np.round((il['lat'] - lats[0]) / dlat).astype(int)
    il['j'] = np.round((il['lon'] - lons[0]) / dlon).astype(int)
    il = il[(il.i >= 0) & (il.i < len(lats)) & (il.j >= 0) & (il.j < len(lons))]
    s0 = pd.to_datetime(il.t.dt.year.astype(str) + f'-{min(MONTHS):02d}-01')
    step = pd.Timedelta(hours=D.agg)
    il['k'] = times.get_indexer(s0 + ((il.t - s0) // step) * step)
    il = il[il.k >= 0]
    I_full = np.zeros((len(times), len(lats), len(lons)), dtype='float32')
    np.add.at(I_full, (il.k.values, il.i.values, il.j.values), 1)
    bi = np.where((lats >= ILDN_BOX[0]) & (lats <= ILDN_BOX[1]))[0]
    bj = np.where((lons >= ILDN_BOX[2]) & (lons <= ILDN_BOX[3]))[0]
    crop = lambda a: a[:, bi][:, :, bj]
    fields = {'U-Net': crop(np.nan_to_num(pr[uvar].values)), 'WWLLN': crop(np.nan_to_num(pr['obs_count'].values))}
    ds = D.prep()
    ti = {t: i for i, t in enumerate(ds.time.values)}
    ii = np.array([ti[t] for t in pr.time.values])
    if 'r14_c' in c:
        cp = np.nan_to_num(ds['cape'].values[ii]).clip(min=0) * np.nan_to_num(ds['precipitation'].values[ii]).clip(min=0)
        fields['CAPE × P'] = crop(c['r14_c'] * cp)
    mon = pd.DatetimeIndex(ds.time.values).month
    fields['Climatology'] = crop(np.stack([c['clim_by_month'][m] for m in mon[ii]]))
    I = crop(I_full)

    def fss_mean(ob, eb):
        s = fss_series(ob.astype('float32'), eb.astype('float32'), thr=0.5)
        return float(np.nanmean(s)) if np.isfinite(s).any() else np.nan

    def scores(Io, est):
        from sklearn.metrics import roc_auc_score
        ti_, te_ = Io.sum((1, 2)), est.sum((1, 2))
        act = ti_ >= 1
        ob = Io >= 1
        f0 = ob.mean()
        thr = np.quantile(est, 1 - f0) if 0 < f0 < 1 else np.inf
        efm = est >= thr if thr > 0 else est > 0
        return {'spearman_box_total': float(spearmanr(te_, ti_)[0]),
                'auc_active_window': float(roc_auc_score(act, te_)) if 0 < act.sum() < len(act) else np.nan,
                'fss_freq_matched': fss_mean(ob, efm), 'fss_ge1': fss_mean(ob, est >= 1),
                'map_spearman': float(spearmanr(est.mean(0).ravel(), Io.mean(0).ravel())[0]),
                'ratio_to_ildn': float(te_.sum() / max(ti_.sum(), 1e-9))}
    groups = {str(y): times.year == y for y in years}
    groups['all'] = np.ones(len(times), bool)
    res = {g: {n: scores(I[m], f[m]) for n, f in fields.items()} for g, m in groups.items()}
    for g, m in groups.items():
        res[g]['_ildn'] = {'strokes_in_box': int(I[m].sum()), 'windows': int(m.sum())}
    per_year = {y: sorted({m for (yy, m) in covered if yy == y}) for y in years}
    period = '; '.join(f"{months_label(ms)} {y}" for y, ms in per_year.items() if ms)
    print(f"  ILDN comparison: {period}", flush=True)
    D._cache['ildn'] = {'I': I, 'fields': fields, 'times': times, 'years': years, 'scores': res,
                        'blat': lats[bi], 'blon': lons[bj], 'il': il, 'period': period}
    return D._cache['ildn']


def _lag_group(D, patterns, c):
    files = []
    for pat in patterns:
        files += sorted(glob.glob(D.path(pat))) if pat != '{main}' else [D.pred_path]
    files = [f for f in dict.fromkeys(files) if os.path.exists(f)]
    out_ = []
    for f in files:
        x = xr.open_dataset(f)
        out_.append(x.reindex(time=c['times'])['pred_count'].values)
    return files, out_


def F13_storm_history(D):
    c = D.core()
    groups = {}
    for name, pats in LAG_RUNS.items():
        files, preds = _lag_group(D, pats, c)
        print(f"    {name}: {len(files)} file(s) " + ', '.join(os.path.basename(f) for f in files), flush=True)
        if preds:
            groups[name] = preds
    if '3 windows' not in groups:
        raise Missing("predictions of the 3-window model (LAG_RUNS['3 windows'])")
    common = np.ones(len(c['times']), bool)
    for preds in groups.values():
        for pr in preds:
            common &= np.isfinite(pr).all(axis=(1, 2))
    fig = plt.figure(figsize=(W2, 2.7))
    has_c = os.path.exists(D.path(LAG_CKPT)) and os.path.exists(D.path(LAG_CKPT) + '.json')
    gs = fig.add_gridspec(1, 3 if has_c else 2, width_ratios=[1.3, 1, 1] if has_c else [1.3, 1], wspace=0.6)
    ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    stats = {}
    for name, preds in groups.items():
        mm, lo, hi = [], [], []
        for s in SCALES:
            per_seed = np.array([fss_series(c['obs'], np.nan_to_num(pr), w=s, valid=common) for pr in preds])
            v = np.nanmean(per_seed, axis=0)
            m, l, h = mean_ci(v, c['blocks'])
            mm.append(m); lo.append(l); hi.append(h)
        ax.plot(SCALES, mm, '-o', ms=2.5, lw=1.2, color=COL[name], label=f'{name} ({len(preds)} seed{"s" if len(preds) > 1 else ""})')
        ax.fill_between(SCALES, lo, hi, color=COL[name], alpha=0.18, lw=0)
        stats[name] = {'fss_by_scale': dict(zip(map(str, SCALES), np.round(mm, 4).tolist())), 'n_seeds': len(preds)}
    ax.set_xticks(SCALES); ax.set_ylim(0, 1); ax.set_xlabel('neighbourhood size (cells)'); ax.set_ylabel('FSS (≥ 1 stroke)')
    ax.legend(loc='lower right'); letter(ax, 'a')
    one = np.nanmean([fss_series(c['obs'], np.nan_to_num(p), valid=common) for p in groups['1 window']], axis=0)
    three = np.nanmean([fss_series(c['obs'], np.nan_to_num(p), valid=common) for p in groups['3 windows']], axis=0)
    diff = three - one
    pg = period_groups(c['months'])
    labs = [g[0] for g in pg] + ['all']
    for j, sel in enumerate([g[1] for g in pg] + [np.ones(len(diff), bool)]):
        mu, l, h = mean_ci(diff[sel], c['blocks'][sel])
        bx.errorbar(j, mu, yerr=[[mu - l], [h - mu]] if np.isfinite(l) else None, fmt='o', color=COL['3 windows'],
                    ms=4, capsize=2, lw=1)
        stats[f'gain_{labs[j]}'] = [float(mu), float(l), float(h)]
    bx.axhline(0, color='0.3', lw=0.7)
    bx.set_xticks(range(len(labs))); bx.set_xticklabels(labs)
    bx.set_ylabel('FSS gain: 3 windows − 1 window'); letter(bx, 'b')
    if has_c:
        cx = fig.add_subplot(gs[2])
        meta = json.load(open(D.path(LAG_CKPT) + '.json'))
        n_l = int(meta.get('n_lags', 3))
        mc = ModelCtx(D, ckpt_path=D.path(LAG_CKPT), n_lags=n_l)
        iv, Xv = mc.inputs(c['idx_va']); it, Xt = mc.inputs(c['idx_te'])
        ltg = c['ltg_all']; pv = mc.rate(Xv)
        base = np.nanmean(fss_series(ltg[it], calibrate(pv, ltg[iv], mc.rate(Xt))))
        nf = len(mc.feats)
        rng = np.random.default_rng(11)
        slot_names = [f'−{D.agg * (n_l - 1 - s)} h' if s < n_l - 1 else '0 h' for s in range(n_l)]
        drops = []
        for s in range(n_l):
            d = []
            for _ in range(N_PERM):
                Xp = Xt.copy()
                Xp[:, s * nf:(s + 1) * nf] = Xt[rng.permutation(len(Xt)), s * nf:(s + 1) * nf]
                d.append(base - np.nanmean(fss_series(ltg[it], calibrate(pv, ltg[iv], mc.rate(Xp)))))
            drops.append((np.mean(d), np.std(d)))
        cx.bar(range(n_l), [d[0] for d in drops], yerr=[d[1] for d in drops], color=COL['3 windows'],
               error_kw={'lw': 0.7, 'capsize': 1.5})
        cx.set_xticks(range(n_l)); cx.set_xticklabels(slot_names); cx.set_xlabel('input window (0 h = the predicted one)')
        cx.set_ylabel('drop in FSS when shuffled'); letter(cx, 'c')
        stats['slot_importance'] = dict(zip(slot_names, [list(map(float, d)) for d in drops]))
    NUMBERS['F13'] = stats
    g = stats.get('gain_all', [np.nan] * 3)
    save(fig, 'F13_storm_history',
         f"Does storm history help? (a) FSS against neighbourhood size for the model that sees only the current "
         f"window and for the model that also sees the two previous {D.agg}-h windows (mean over random seeds; "
         f"shading: 95 % block-bootstrap interval). (b) FSS gain (3 × 3 cells) of the three-window model by "
         f"{'season' if len(set(c['months'])) > 4 else 'month'}; "
         f"all windows: {fmt_ci(*g, nd=3)}. An interval that crosses zero means no significant gain."
         + (" Control: the three-window model with the two earlier slots filled with inputs from random other "
            "days (same number of inputs, no storm history)." if 'Control' in groups else '')
         + (" (c) Drop in FSS when the inputs of one time slot are shuffled." if has_c else ''))


# ═════════════════════════════════════════════════════════════════════════════════════════════
# Tables
# ═════════════════════════════════════════════════════════════════════════════════════════════
def T1_data(D):
    D.pred()
    feats = D.A.get('features') or D.cfg.get('feature_name', [])
    logf = set(D.A.get('log_features') or [])
    rows = [[NICE.get(v, v), SOURCE.get(v, ('?', '?', '?'))[0], SOURCE.get(v, ('?', '?', '?'))[1],
             SOURCE.get(v, ('?', '?', '?'))[2], 'yes' if v in logf else 'no'] for v in feats]
    rows.append(['Lightning strokes (target)', 'WWLLN', f'strokes per cell per {D.agg} h', 'stroke locations', '–'])
    rows.append(['Lightning strokes (independent check)', 'ILDN', 'strokes', 'stroke locations', '–'])
    df = pd.DataFrame(rows, columns=['Variable', 'Source', 'Units', 'Native resolution', 'Log-transformed'])
    write_table('T1_data_sources', df,
                f"Data used in this study. All inputs are averaged over {D.agg}-h windows on the 0.25° ERA5 grid; "
                "IMERG precipitation is regridded to that grid.")


def T2_model(D):
    D.pred(); A = D.A
    hp = A.get('hyperparams') if isinstance(A.get('hyperparams'), dict) else {}
    md = A.get('model') if isinstance(A.get('model'), dict) else {}
    tr = A.get('training') if isinstance(A.get('training'), dict) else {}
    yrs = lambda v: (f"{min(v)}–{max(v)}" if isinstance(v, list) and len(v) > 2 and max(v) - min(v) == len(v) - 1
                     else ', '.join(map(str, v)) if isinstance(v, list) else str(v))
    rows = [
        ['Architecture', 'U-Net-style encoder–decoder without skip connections (Jones et al. 2026)'],
        ['Channels', 'encoder 32, 16; centre 8; decoder 16, 32; 1 × 1 output'],
        ['Parameters', f"{md.get('n_params', '?'):,}" if isinstance(md.get('n_params'), int) else '?'],
        ['Inputs', f"{len(A.get('features') or [])} fields, {D.agg}-h means, standardised per month"],
        ['Loss', A.get('loss_desc', A.get('loss', '?'))],
        ['Optimiser', f"Adam, learning rate {hp.get('lr', '?')}, gradient clipping {hp.get('grad_clip', '?')}"],
        ['Batch size / max. epochs', f"{hp.get('batch_size', '?')} / {hp.get('max_epochs', '?')}"],
        ['Early stopping', f"patience {hp.get('patience', '?')} epochs on validation loss"
                           + (f"; best epoch {tr.get('best_epoch')} of {tr.get('epochs_run')}" if tr else '')],
        ['Random seed', str(hp.get('seed', '?'))],
        ['Calibration', str(A.get('calibration', '?')).replace('[', '').replace(']', '')],
        ['Training / validation / test years', f"{yrs(A.get('train_years'))} / {yrs(A.get('val_years'))} / {yrs(A.get('test_years'))}"],
        ['Months', ('all (Jan–Dec)' if sorted(A.get('months') or []) == list(range(1, 13))
                    else months_label(A.get('months') or [10, 11, 12]))
                   + (f"; scored on {months_label(MONTHS)} only" if D.eval_months else '')],
    ]
    write_table('T2_model_training', pd.DataFrame(rows, columns=['Setting', 'Value']),
                'Model and training settings.', col_format=r'lp{0.62\textwidth}')


def T3_scores(D):
    c = D.core()
    rows = []
    names = [n for n in ['U-Net', 'U-Net (raw)', 'Per-cell model', 'CAPE × P', 'Climatology', 'Persistence'] if n in c['M']]
    s_unet = fss_series(c['obs'], c['M']['U-Net'])
    for n in names:
        v = D.valid(n, c)
        s3 = fss_series(c['obs'], c['M'][n], valid=v)
        m3, lo, hi = mean_ci(s3, c['blocks'])
        dif = '–' if n == 'U-Net' else fmt_ci(*paired_ci(s_unet, s3, c['blocks']), nd=3)
        cat = categorical(c['obs'], c['M'][n], valid=v)
        o = c['obs'] if v is None else c['obs'][v]
        pm = c['M'][n] if v is None else c['M'][n][v]
        rows.append([n, fmt(np.nanmean(fss_series(c['obs'], c['M'][n], w=1, valid=v))), fmt_ci(m3, lo, hi),
                     fmt(np.nanmean(fss_series(c['obs'], c['M'][n], w=5, valid=v))), dif, fmt(cat['CSI']),
                     fmt(cat['POD']), fmt(cat['SR']), fmt(cat['F1']), fmt(cat['freq_bias']), fmt(pm.mean() / max(o.mean(), 1e-12))])
        NUMBERS.setdefault('T3', {})[n] = {'fss3': [m3, lo, hi], **cat, 'mean_ratio': float(pm.mean() / max(o.mean(), 1e-12))}
    df = pd.DataFrame(rows, columns=['Model', 'FSS 1×1', 'FSS 3×3 [95% CI]', 'FSS 5×5', 'U-Net gain (FSS 3×3) [95% CI]',
                                     'CSI', 'POD', 'Success ratio', 'F1', 'Frequency bias', 'Mean ratio'])
    write_table('T3_scores', df,
                f"Scores on the test years {', '.join(map(str, c['te_years']))} for ≥ 1 stroke per {D.agg} h. "
                "Baselines are calibrated with the same quantile mapping as the U-Net; persistence uses the observed "
                f"lightning of the previous window. Event rate observed: {(c['obs'] >= THR).mean():.3f}. "
                "U-Net gain: FSS of the U-Net minus FSS of the model, computed window by window (paired), with its "
                "95 % block-bootstrap interval. Mean ratio: mean predicted / mean observed strokes.",
                col_format='l' + 'c' * 10)


def T4_bias(D):
    c = D.core()
    if c['lsm'] is None:
        raise Missing('land–sea mask')
    obs = c['to_density'](c['obs'].mean(0))
    regions = {'Whole domain': np.ones_like(obs, bool), 'Land': c['lsm'] >= LAND_LSM, 'Sea': c['lsm'] < LAND_LSM}
    models = {'U-Net (Poisson rate)': c['RAW']['U-Net'], 'U-Net (calibrated)': c['M']['U-Net']}
    models.update({k + ' (fitted constant)' if k == 'CAPE × P' else k: v for k, v in c['RAW'].items() if k != 'U-Net'})
    rows = []
    for rname, rm in regions.items():
        o = obs[rm].mean()
        row = [rname, f'{o:.3g}']
        for n, m in models.items():
            if m is None:
                continue
            d = c['to_density'](m.mean(0))[rm].mean() - o
            row.append(f'{d:+.3g} ({100 * d / max(o, 1e-12):+.0f}%)')
        rows.append(row)
    cols = ['Region', 'Observed'] + [n for n, m in models.items() if m is not None]
    write_table('T4_bias_land_sea', pd.DataFrame(rows, columns=cols),
                f"Mean bias (model − observed) in strokes km⁻² yr⁻¹{rate_note()} on the test years, with the relative "
                f"bias in brackets. Land: lsm ≥ {LAND_LSM}. The Poisson rate (expected strokes) is the U-Net output "
                "for amounts; the calibrated output sets most weak values to zero and is meant for event scores.")


def T5_ildn(D):
    r = _ildn(D)
    labels = {'spearman_box_total': 'Spearman r of box totals (when)',
              'auc_active_window': 'AUC, active window (≥ 1 ILDN stroke)',
              'fss_freq_matched': 'FSS 3×3, frequency-matched (where)', 'fss_ge1': 'FSS 3×3, ≥ 1 stroke in both',
              'map_spearman': 'Spearman r of mean maps', 'ratio_to_ildn': 'Total / ILDN total'}
    order = [k for k in ['U-Net', 'CAPE × P', 'Climatology', 'WWLLN'] if k in r['fields']]
    s = r['scores']['all']
    rows = [[lab] + [fmt(s[k][key]) for k in order] for key, lab in labels.items()]
    write_table('T5_ildn_scores', pd.DataFrame(rows, columns=['Score against ILDN'] + order),
                f"Scores against ILDN in the box {ILDN_BOX[0]}–{ILDN_BOX[1]}°N, {ILDN_BOX[2]}–{ILDN_BOX[3]}°E, "
                f"{r['period']} ({s['_ildn']['strokes_in_box']:,} ILDN strokes). WWLLN is "
                "the reference: the model was trained to reproduce it. U-Net: expected strokes (Poisson rate before "
                "calibration).", col_format='l' + 'c' * len(order))


# ═════════════════════════════════════════════════════════════════════════════════════════════
# Appendix
# ═════════════════════════════════════════════════════════════════════════════════════════════
def A1_monthly(D):
    c = D.core()
    groups = period_groups(c['months'])                   # months, or DJF/MAM/JJA/SON for the whole year
    by_season = len(set(c['months'])) > 4
    lats, lons = c['lats'], c['lons']
    ext = [lons.min() - 0.125, lons.max() + 0.125, lats.min() - 0.125, lats.max() + 0.125]
    fig, axs = map_axes(len(groups), 2, W2, extra_h=0.3, hspace=0.12, extent=ext)
    pairs = {g: (c['to_density'](c['obs'][sel].mean(0)), c['to_density'](c['RAW']['U-Net'][sel].mean(0)))
             for g, sel in groups}
    pos = np.concatenate([v[v > 0] for pr in pairs.values() for v in pr])
    norm = mcolors.LogNorm(vmin=max(np.percentile(pos, 2), 1e-3), vmax=np.percentile(pos, 99.5))
    for r, (g, _) in enumerate(groups):
        name = g if by_season else pd.to_datetime(g, format='%b').strftime('%B')
        for j, (lab, f) in enumerate(zip(['Observed', 'U-Net'], pairs[g])):
            im = draw_map(axs[r, j], c, np.where(f > 0, f, np.nan), LTG_CMAP, norm, ticks=(r == len(groups) - 1, j == 0))
            axs[r, j].set_title(f"{name}, {lab}", fontsize=7, loc='left', pad=2)
    colorbar(fig, im, axs, 'strokes km$^{-2}$ yr$^{-1}$', extend='both', shrink=0.7)
    save(fig, 'A1_monthly_maps', "Observed (left) and U-Net (right; expected strokes = Poisson rate) mean stroke "
                                 "density for each " + ("season (DJF = Dec–Feb, MAM = Mar–May, JJA = Jun–Aug, "
                                 "SON = Sep–Nov)" if by_season else "month") + " of the test years.")


def A2_correlations(D):
    c = D.core(); ds = D.prep()
    tr = c['idx_tr']
    y = c['ltg_all'][tr].ravel()
    rng = np.random.default_rng(0)
    samp = rng.choice(y.size, size=min(2_000_000, y.size), replace=False)
    feats = [v for v in (D.A.get('features') or []) if v in ds and v != 'lsm']
    res = {}
    for v in feats:
        a = np.nan_to_num(ds[v].values[tr]).ravel()[samp]
        if a.std() > 0:
            res[v] = (pearsonr(a, y[samp])[0], spearmanr(a, y[samp])[0])
    extra = []
    if D.args.hourly_corr and D.hourly_path and os.path.exists(D.hourly_path):
        h = xr.open_dataset(D.hourly_path).sel(lat=c['lats'], lon=c['lons'], method='nearest')
        extra = [v for v in h.data_vars if v not in res and v not in ('ltg', 'lsm') and 'time' in h[v].dims]
        tt = pd.DatetimeIndex(h.time.values)
        sel = np.isin(tt.year, c['tr_years']) & np.isin(tt.month, MONTHS)
        lt = h['ltg'].isel(time=np.where(sel)[0]).resample(time=f'{D.agg}h').sum(min_count=1).values.ravel()
        s2 = rng.choice(lt.size, size=min(2_000_000, lt.size), replace=False)
        for v in extra:
            a = h[v].isel(time=np.where(sel)[0]).resample(time=f'{D.agg}h').mean().values.ravel()[s2]
            ok = np.isfinite(a) & np.isfinite(lt[s2])
            if ok.sum() > 100 and np.std(a[ok]) > 0:
                res[v] = (pearsonr(a[ok], lt[s2][ok])[0], spearmanr(a[ok], lt[s2][ok])[0])
    names = sorted(res, key=lambda v: -abs(res[v][1]))
    fig, ax = plt.subplots(figsize=(W2, 2.6))
    x = np.arange(len(names))
    ax.bar(x - 0.2, [res[v][0] for v in names], 0.4, color='#0072B2', label='Pearson')
    ax.bar(x + 0.2, [res[v][1] for v in names], 0.4, color='#E69F00', label='Spearman')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([NICE.get(v, v) + ('' if v in feats else ' *') for v in names], rotation=40, ha='right')
    ax.set_ylabel(f'correlation with strokes per {D.agg} h'); ax.legend()
    save(fig, 'A2_input_correlations',
         f"Correlation of each variable with lightning on the training years ({min(c['tr_years'])}–{max(c['tr_years'])})"
         + (", sorted by |Spearman|. * = tested but not used as input." if extra else ", sorted by |Spearman|."))


def A3_all_pixels(D):
    c = D.core()
    o = np.log1p(c['obs'].ravel())
    versions = [('expected strokes (Poisson rate)', np.log1p(c['RAW']['U-Net'].ravel())),
                ('calibrated', np.log1p(c['M']['U-Net'].ravel()))]
    hi = max(o.max(), *(p.max() for _, p in versions)) * 1.05
    ticks = [t for t in [0, 1, 10, 100, 1000, 10000] if np.log1p(t) <= hi]
    fig, axs = plt.subplots(1, 2, figsize=(W2 * 0.85, W1 * 0.9), sharey=True)
    r2s = {}
    for k, (ax, (lab, p)) in enumerate(zip(axs, versions)):
        hb = ax.hexbin(o, p, gridsize=50, bins='log', cmap='viridis', mincnt=1, extent=(0, hi, 0, hi), vmin=1, vmax=o.size)
        ax.plot([0, hi], [0, hi], 'k--', lw=0.8)
        ax.set_xticks(np.log1p(ticks)); ax.set_xticklabels(ticks); ax.set_yticks(np.log1p(ticks)); ax.set_yticklabels(ticks)
        ax.set_xlabel(f'observed strokes per cell per {D.agg} h')
        r2s[lab] = 1 - np.sum((p - o) ** 2) / np.sum((o - o.mean()) ** 2)
        ax.set_title(f'U-Net, {lab}\nR² of log(1 + strokes) = {fmt(r2s[lab])}', fontsize=7, loc='left')
        letter(ax, 'ab'[k])
    axs[0].set_ylabel(f'U-Net strokes per cell per {D.agg} h')
    colorbar(fig, hb, axs, 'number of cell-windows (log)')
    save(fig, 'A3_all_pixels', "Observed against U-Net strokes for every cell and test window (log(1 + x) axes). "
                               "(a) Expected strokes (Poisson rate): smooth, it never predicts exactly zero. (b) After "
                               "quantile mapping: most weak values become 0 and the rest cluster near whole numbers "
                               "(horizontal lines), because the observed values it is mapped onto are whole counts.")


def A4_training(D):
    D.pred()
    hist = D.A.get('training', {}).get('history') if isinstance(D.A.get('training'), dict) else None
    if hist:
        ep, trl, val = (list(x) for x in zip(*hist))
    else:
        ep, trl, val = _loss_from_log(D)
    _plot_training(D, ep, trl, val)


def _loss_from_log(D):
    if not D.args.train_log or not os.path.exists(D.args.train_log):
        raise Missing('the loss of every epoch: retrain the main model with the current train_unet.py '
                      '(python train_unet.py --retrain), or pass --train-log with its printed output')
    rx = re.compile(r'Epoch (\d+)/\d+, Train loss: ([-0-9.eE+]+), Val loss: ([-0-9.eE+]+)')
    ep, trl, val = [], [], []
    for line in open(D.args.train_log):
        m = rx.search(line)
        if m:
            ep.append(int(m.group(1))); trl.append(float(m.group(2))); val.append(float(m.group(3)))
    if not ep:
        raise Missing(f'epoch lines in {D.args.train_log}')
    if len(set(ep)) < len(ep):        # several trainings in one log: keep the last
        start = max(i for i, e in enumerate(ep) if e == 1)
        ep, trl, val = ep[start:], trl[start:], val[start:]
    return ep, trl, val


def _plot_training(D, ep, trl, val):
    best = ep[int(np.argmin(val))]
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    ax.plot(ep, trl, color='0.3', lw=1.2, label='training')
    ax.plot(ep, val, color=COL['U-Net'], lw=1.2, label='validation')
    ax.axvline(best, color='0.5', ls=':', lw=0.8)
    ax.text(best, max(max(trl), max(val)), f' best epoch {best}', fontsize=6.5, va='top')
    ax.set_xlabel('epoch'); ax.set_ylabel('Poisson loss'); ax.legend()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    save(fig, 'A4_training_curves', f"Training and validation loss per epoch; the model of the best validation epoch "
                                    f"({best}) is kept (early stopping).")


def A5_robustness(D):
    if not D.args.audit_log or not os.path.exists(D.args.audit_log):
        raise Missing('--audit-log (save it with: python audit.py | tee audit_log.txt)')
    txt = open(D.args.audit_log).read()
    seeds = [float(x) for x in re.findall(r'real model, seed \d+: calibrated FSS ([0-9.]+)', txt)]
    sh = re.search(r'with inputs ([0-9.]+) days earlier FSS drops ([0-9.]+) → ([0-9.]+)', txt)
    shuf = re.search(r'shuffled-target model FSS ([0-9.]+)', txt)
    if not seeds:
        raise Missing(f'seed results in {D.args.audit_log} (run audit.py without --quick)')
    labels, vals, errs = [f'real model\n({len(seeds)} seeds)'], [np.mean(seeds)], [np.std(seeds)]
    if sh:
        labels.append(f'inputs from\n{float(sh.group(1)):g} days earlier'); vals.append(float(sh.group(3))); errs.append(0)
    if shuf:
        labels.append('shuffled\ntargets'); vals.append(float(shuf.group(1))); errs.append(0)
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    ax.bar(range(len(vals)), vals, color=[COL['U-Net']] + ['0.6'] * (len(vals) - 1), width=0.6)
    ax.errorbar(0, vals[0], yerr=errs[0], color='k', lw=0.8, capsize=2)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=6.5)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels); ax.set_ylim(0, 1); ax.set_ylabel('FSS (3 × 3 cells)')
    NUMBERS['A5'] = {'seed_fss': seeds}
    save(fig, 'A5_robustness', f"Robustness: FSS of the real model over {len(seeds)} random seeds "
                               f"({np.mean(seeds):.3f} ± {np.std(seeds):.3f}) and of control models that cannot have "
                               "skill: inputs taken from other days, and shuffled targets.")


def A6_more_cases(D):
    c = D.core()
    tot = c['obs'].sum((1, 2))
    idx = np.argsort(tot)[::-1][N_CASES:N_CASES + N_MORE_CASES]
    idx = idx[tot[idx] > 0]
    if not len(idx):
        raise Missing('further active windows')
    _cases_figure(D, c, idx, 'A6_more_cases', "Further case studies: the next most active test windows, as in Fig. 10.")


def A7_ildn_time(D):
    r = _ildn(D)
    if not D.hourly_path or not os.path.exists(D.hourly_path):
        raise Missing(f'hourly data file {D.hourly_path}')
    h = xr.open_dataset(D.hourly_path)['ltg']
    hb = h.isel(lat=np.where((h.lat >= ILDN_BOX[0] - 1e-6) & (h.lat <= ILDN_BOX[1] + 1e-6))[0],
                lon=np.where((h.lon >= ILDN_BOX[2] - 1e-6) & (h.lon <= ILDN_BOX[3] + 1e-6))[0])
    hb = hb.sel(time=hb.time.dt.year.isin(r['years']) & hb.time.dt.month.isin(MONTHS))
    w = hb.sum(['lat', 'lon'], min_count=1).to_series()
    il = r['il']
    ib = il[(il.lat >= ILDN_BOX[0]) & (il.lat <= ILDN_BOX[1]) & (il.lon >= ILDN_BOX[2]) & (il.lon <= ILDN_BOX[3])]
    i_h = ib.groupby(ib.t.dt.floor('h')).size().reindex(w.index, fill_value=0)
    lags = np.arange(-6, 7)
    cors = []
    for L in lags:
        a, b = w.values, i_h.shift(-L).values
        ok = np.isfinite(a) & np.isfinite(b)
        cors.append(pearsonr(np.log1p(a[ok]), np.log1p(b[ok]))[0] if ok.sum() > 10 else np.nan)
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    ax.plot(lags, cors, '-o', color='0.2', ms=3)
    ax.axvline(0, color='0.5', ls=':', lw=0.8)
    ax.set_xlabel('lag L (h): WWLLN(t) vs ILDN(t + L)'); ax.set_ylabel('r (hourly box totals, log)')
    best = int(lags[np.nanargmax(cors)])
    save(fig, 'A7_ildn_time_alignment', f"Time alignment check: correlation of hourly WWLLN and ILDN stroke counts "
                                        f"in the ILDN box at different lags; best lag {best:+d} h.")


# ═════════════════════════════════════════════════════════════════════════════════════════════
# Optional analyses
# ═════════════════════════════════════════════════════════════════════════════════════════════
def O2_learning_curve(D):
    c = D.core()
    pts = []
    for n, f in LEARNING_CURVE_FILES.items():
        n = len(c['tr_years']) if n == 'all' else n
        series = _pred_file_fss(D, D.path(f), 'pred_count', c)
        if series is not None:
            pts.append((n, *mean_ci(series, c['blocks'])))
    pts.sort()
    if len(pts) < 3:
        raise Missing('at least 3 of LEARNING_CURVE_FILES')
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    n, m, lo, hi = map(np.array, zip(*pts))
    ax.errorbar(n, m, yerr=[m - lo, hi - m], fmt='-o', color=COL['U-Net'], ms=3, capsize=2, lw=1)
    if 'CAPE × P' in c['M']:
        ax.axhline(np.nanmean(fss_series(c['obs'], c['M']['CAPE × P'])), color=COL['CAPE × P'], ls='--', lw=0.9, label='CAPE × P')
        ax.legend()
    w = D.period_word
    ax.set_xlabel(f'training {w}s'); ax.set_ylabel('FSS (3 × 3 cells)'); ax.set_xticks(n)
    save(fig, 'O2_learning_curve', f"FSS on the test years against the number of training {w}s (the most recent "
                                   f"{w}s before the validation year; 95 % CI). The dashed line is calibrated CAPE × P.")


def O3_receptive_field(D):
    c = D.core()
    mc = ModelCtx(D)
    torch = mc.torch
    it, Xt = mc.inputs(c['idx_te'])
    obs = c['ltg_all'][it]
    tot = obs.sum((1, 2))
    top = np.argsort(tot)[::-1][:20]
    rng = np.random.default_rng(3)
    lats, lons = c['lats'], c['lons']
    dy = 111.2 * abs(lats[1] - lats[0])
    prof_bins = np.arange(0, 1050, 25.0)
    profs = []
    for t in top:
        cand = np.argwhere(obs[t] >= THR)
        if not len(cand):
            continue
        for (i, j) in cand[rng.choice(len(cand), size=min(5, len(cand)), replace=False)]:
            x = torch.from_numpy(Xt[t:t + 1]).requires_grad_(True)
            mc.model(x)[0, 0, i, j].backward()
            g = x.grad.abs().sum(1)[0].numpy()
            yy, xx = np.meshgrid(np.arange(len(lats)), np.arange(len(lons)), indexing='ij')
            dist = np.sqrt(((yy - i) * dy) ** 2 + ((xx - j) * dy * np.cos(np.deg2rad(lats[i]))) ** 2)
            k = np.digitize(dist.ravel(), prof_bins) - 1
            s = np.bincount(k, g.ravel(), minlength=len(prof_bins))[:len(prof_bins) - 1]
            n = np.bincount(k, minlength=len(prof_bins))[:len(prof_bins) - 1]
            profs.append(np.where(n > 0, s / np.maximum(n, 1), np.nan))
    if not profs:
        raise Missing('active cells in the test windows')
    P = np.nanmean(profs, axis=0); P = P / np.nanmax(P)
    centres = (prof_bins[:-1] + prof_bins[1:]) / 2
    r10 = centres[np.argmax(P < 0.1)] if np.any(P < 0.1) else np.nan
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    ax.plot(centres, P, color=COL['U-Net'], lw=1.3)
    ax.axhline(0.1, color='0.5', ls=':', lw=0.8)
    if np.isfinite(r10):
        ax.axvline(r10, color='0.5', ls='--', lw=0.8)
        ax.text(r10, 0.9, f' {r10:.0f} km', fontsize=6.5)
    ax.set_xlabel('distance from the predicted cell (km)'); ax.set_ylabel('mean |gradient| (normalised)')
    NUMBERS['O3'] = {'radius_10pct_km': float(r10), 'n_targets': len(profs)}
    save(fig, 'O3_receptive_field', f"How far the U-Net looks: mean absolute gradient of the predicted rate at a "
                                    f"lightning cell with respect to the inputs, against distance ({len(profs)} cells "
                                    f"in the 20 most active test windows). It falls below 10 % of its peak at "
                                    f"about {fmt(r10, 0)} km.")


def O4_day_night(D):
    c = D.core()
    hours = pd.DatetimeIndex(c['times']).hour
    hs = sorted(set(hours))
    if len(hs) < 2:
        raise Missing('windows starting at different hours')
    models = _models_for_skill(c)
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    width = 0.8 / len(models)
    for k, n in enumerate(models):
        s = fss_series(c['obs'], c['M'][n], valid=D.valid(n, c))
        for j, h in enumerate(hs):
            sel = hours == h
            mu, lo, hi = mean_ci(s[sel], c['blocks'][sel])
            ax.bar(j + (k - (len(models) - 1) / 2) * width, mu, width * 0.92, color=COL[n], label=n if j == 0 else None,
                   yerr=[[mu - lo], [hi - mu]] if np.isfinite(lo) else None, error_kw={'lw': 0.7, 'capsize': 1.5})
    ax.set_xticks(range(len(hs))); ax.set_xticklabels([f'{h:02d}–{(h + D.agg) % 24:02d} UTC' for h in hs])
    ax.set_ylabel('FSS (3 × 3 cells)'); ax.set_ylim(0, 1)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6)
    save(fig, 'O4_day_night', "FSS for windows starting at each hour of the day (95 % CI).")


# ═════════════════════════════════════════════════════════════════════════════════════════════
ITEMS = {
    'F1': F1_study_area, 'F2': F2_pipeline, 'F3': F3_ablation, 'F4': F4_fss_vs_scale, 'F5': F5_bias_maps,
    'F6': F6_cell_scatter, 'F7': F7_intensity, 'F8': F8_reliability, 'F9': F9_season, 'F10': F10_cases,
    'F11': F11_importance, 'F12': F12_ildn, 'F13': F13_storm_history,
    'T1': T1_data, 'T2': T2_model, 'T3': T3_scores, 'T4': T4_bias, 'T5': T5_ildn,
    'A1': A1_monthly, 'A2': A2_correlations, 'A3': A3_all_pixels, 'A4': A4_training, 'A5': A5_robustness,
    'A6': A6_more_cases, 'A7': A7_ildn_time,
    'O2': O2_learning_curve, 'O3': O3_receptive_field, 'O4': O4_day_night,
}
MONTHS = [10, 11, 12]


def main():
    global OUT_DIR
    ap = argparse.ArgumentParser(description='All thesis figures and tables.')
    ap.add_argument('--only', nargs='*', help='items to make, e.g. F4 F5 T3 (default: all)')
    ap.add_argument('--list', action='store_true', help='list the items and stop')
    ap.add_argument('--pred', help='predictions file (default: OUT_PATH of train_unet.py)')
    ap.add_argument('--data-dir', help='data folder (default: DATA_DIR of train_unet.py)')
    ap.add_argument('--train-script', default=TRAIN_SCRIPT)
    ap.add_argument('--train-log', help='printed output of train_unet.py, for A4')
    ap.add_argument('--audit-log', help='printed output of audit.py, for A5')
    ap.add_argument('--hourly-corr', action='store_true', help='A2: also correlate the other hourly variables (slow)')
    ap.add_argument('--months', type=int, nargs='+',
                    help='which run: its months, like train_unet.py --months (default: ACTIVE_MONTHS in train_unet.py)')
    ap.add_argument('--eval-months', type=int, nargs='+',
                    help='score on these months only, e.g. 10 11 12 (whole-year model on Oct–Dec)')
    ap.add_argument('--out', help=f'output folder (default {OUT_DIR}, + _<season> other than Oct–Dec, '
                                  f'+ _eval<months> with --eval-months)')
    args = ap.parse_args()
    if args.list:
        for k, f in ITEMS.items():
            print(f"  {k:4s} {f.__name__}")
        return
    D = Data(args)
    if args.out:
        OUT_DIR = args.out
    elif D.season != 'OND' or D.eval_months:
        OUT_DIR = OUT_DIR + ('' if D.season == 'OND' else f'_{D.season}') + (
            f'_eval{season_name(D.eval_months)}' if D.eval_months else '')
    todo = [k.upper() for k in args.only] if args.only else list(ITEMS)
    unknown = [k for k in todo if k not in ITEMS]
    if unknown:
        sys.exit(f"unknown items {unknown}; use --list")
    print(f"predictions: {D.pred_path}\ndata folder: {D.data_dir}\noutput:      {OUT_DIR}\n"
          f"season:      {D.season}" + (f", scored on months {D.eval_months}" if D.eval_months else ''), flush=True)
    status = {}
    for k in todo:
        print(f"\n{k} — {ITEMS[k].__name__}", flush=True)
        try:
            ITEMS[k](D)
            status[k] = 'made'
        except Missing as e:
            status[k] = f'skipped: needs {e}'
            print(f"  skipped — needs {e}", flush=True)
        except Exception as e:
            status[k] = f'FAILED: {type(e).__name__}: {e}'
            traceback.print_exc()
    os.makedirs(OUT_DIR, exist_ok=True)
    caps, nums, order = {}, {}, []
    if args.only:            # keep the captions and numbers of the items not made this time
        if os.path.exists(out('captions.md')):
            for part in open(out('captions.md')).read().split('\n## ')[1:]:
                name, _, text = part.partition('\n\n')
                caps[name.strip()] = text.strip()
                order.append(name.strip())
        if os.path.exists(out('thesis_numbers.json')):
            try:
                nums = json.load(open(out('thesis_numbers.json')))
            except Exception:
                nums = {}
    for name in MADE:
        caps[name] = CAPTIONS.get(name, '')
        if name not in order:
            order.append(name)
    nums.update(NUMBERS)
    with open(out('captions.md'), 'w') as fh:
        fh.write('# Draft captions\n\nGenerated by thesis_figures.py from '
                 f'{os.path.basename(D.pred_path or "?")}. Numbers are from this run; edit the wording freely.\n\n')
        for name in order:
            fh.write(f"## {name}\n\n{caps[name]}\n\n")
    with open(out('thesis_numbers.json'), 'w') as fh:
        json.dump(nums, fh, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x))
    print('\n' + '=' * 70)
    for k, s in status.items():
        print(f"  {k:4s} {s}")
    print(f"\n  figures, tables/, captions.md, thesis_numbers.json → {OUT_DIR}")
    if any(s.startswith('FAILED') for s in status.values()):
        sys.exit(1)


if __name__ == '__main__':
    main()
