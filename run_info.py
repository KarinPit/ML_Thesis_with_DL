"""
run_info.py — which training configuration a predictions file came from, and where its figures go.

Shared by jones_figures.py, checks.py and audit.py, so all three put a run's figures in the same
folder, and a different configuration always gets a different folder:

    figures/unet_<window>_<loss>[_jonesnorm]_<cal|raw>_<config id>/

The config id is a short fingerprint of everything that changes the results (loss, window, inputs,
years, hyper-parameters, model code, calibration). Same configuration → same id → same folder
(re-running overwrites it). Any change → new id → new folder. The full configuration is written to
run_summary.json in that folder, so you can always see what an id stands for.
"""
import ast
import hashlib
import json
import os

HERE         = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(HERE, 'train_unet.py')


def read_train_config(path=TRAIN_SCRIPT):
    """Top-level settings of train_unet.py (plain values and f-strings), read without running it."""
    if not os.path.exists(path):
        return {}
    src = open(path).read()
    cfg = {}
    for node in ast.parse(src).body:
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                pairs = [(tgt, node.value)]
            elif isinstance(tgt, ast.Tuple) and isinstance(node.value, ast.Tuple):
                pairs = list(zip(tgt.elts, node.value.elts))
            else:
                continue
            for t, v in pairs:
                if not isinstance(t, ast.Name):
                    continue
                try:
                    cfg[t.id] = ast.literal_eval(v)
                except Exception:
                    try:   # f-strings / simple expressions built from earlier settings
                        cfg[t.id] = eval(compile(ast.Expression(v), path, 'eval'), {'__builtins__': {}}, dict(cfg))
                    except Exception:
                        pass
    return cfg


def default_pred_file(path=TRAIN_SCRIPT):
    """Predictions file of the configuration currently set in train_unet.py (None if unknown)."""
    return read_train_config(path).get('OUT_PATH')


def _load(v):
    """Attributes are stored as text; turn JSON / Python literals back into values."""
    if not isinstance(v, str):
        return v.item() if hasattr(v, 'item') else v
    for parse in (json.loads, ast.literal_eval):
        try:
            return parse(v)
        except Exception:
            pass
    return v


def run_config(attrs, pred_var='pred_count'):
    """Everything that can change the results, from the attributes of a predictions file."""
    a = {k: _load(v) for k, v in dict(attrs).items()}
    calibrated = (pred_var == 'pred_count') and a.get('calibration', 'none') != 'none'
    srt = lambda k: sorted(a[k]) if isinstance(a.get(k), (list, tuple)) else a.get(k)
    model = a.get('model') if isinstance(a.get('model'), dict) else {}
    cfg = {
        'agg_hours':      a.get('agg_hours'),
        'loss':           a.get('loss'),
        'mse_unnorm':     a.get('mse_unnorm') if a.get('loss') == 'mse' else None,
        'output':         'calibrated' if calibrated else 'raw',
        'calibration':    a.get('calibration') if calibrated else 'none',
        'features':       srt('features'),
        'log_features':   srt('log_features'),
        'train_years':    srt('train_years'),
        'val_years':      srt('val_years'),
        'test_years':     srt('test_years'),
        'excluded_years': srt('excluded_years'),
        'months':         srt('months'),
        'domain':         a.get('domain'),
        'wcd_min':        a.get('wcd_min'),
        'hyperparams':    a.get('hyperparams') if isinstance(a.get('hyperparams'), dict) else None,
        'model_code':     model.get('source_md5'),
        'event_threshold': a.get('fss_threshold'),
    }
    return {k: v for k, v in cfg.items() if v is not None}      # older files simply have fewer keys


def config_id(cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:6]


def run_name(attrs, pred_path):
    """e.g. '12h_mse_jonesnorm' — from the file attributes, else from the file name."""
    if 'run_name' in attrs:
        return str(attrs['run_name'])
    stem = os.path.basename(pred_path).replace('_OND_predictions.nc', '').replace('.nc', '')
    return stem[5:] if stem.startswith('unet_') else stem


def run_folder(attrs, pred_path, pred_var='pred_count'):
    """Folder name for this run's figures, e.g. 'unet_12h_mse_jonesnorm_cal_3f9a2c'."""
    cfg = run_config(attrs, pred_var)
    return f"unet_{run_name(attrs, pred_path)}_{'cal' if cfg['output'] == 'calibrated' else 'raw'}_{config_id(cfg)}"


SHORT = {'precipitation': 'precip', 'total_totals_index': 'TT', 'total_column_cloud_ice_water': 'tciw',
         'total_column_cloud_liquid_water': 'tclw', 'k_index': 'K', 'relative_humidity': 'rh',
         'total_column_water_vapour': 'tcwv'}

def short_name(v):
    """Short, unique label for an input variable (tables and figure annotations)."""
    return SHORT.get(v, v)


def run_label(attrs, pred_var='pred_count'):
    """Human-readable label for figure titles, e.g. 'U-Net · MSE, Jones un-norm. · 12 h · calibrated'."""
    cfg = run_config(attrs, pred_var)
    loss = {'poisson': 'Poisson', 'log_mse': 'log1p-MSE',
            'mse': f"MSE, {'Jones' if cfg.get('mse_unnorm') == 'jones' else 'direct'} un-norm."}.get(cfg.get('loss'), str(cfg.get('loss')))
    return f"U-Net · {loss} · {cfg.get('agg_hours', '?')} h · {cfg['output']}"
