"""
plot_lag_comparison.py
----------------------
Load saved LightGBM and XGBoost models for lags 0-6, predict on the test set,
and plot all ROC + PR curves on one figure.

Each lag gets a distinct color. AUC, PR-AUC, and the chosen threshold are shown
in the legend. A no-skill line is drawn on both plots.

Output: data/lag_comparison_{balance_str}_{ts}.png
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, roc_curve)

# ── Configuration ─────────────────────────────────────────────────────────────
LAGS        = list(range(7))        # 0 .. 6
DATA_DIR    = 'data'
OUT_DIR     = 'data'
PRED_CHUNK  = 100_000
MIN_RECALL  = 0.30

TRAIN_YEARS  = '2004_2005_2006_2008_2009_2023_2024'
TEST_YEAR    = '2025'
MASK_STR     = '_convmask'
BALANCE_STR  = '_balanced'          # '_balanced' for 1:1, '_balanced50to1' for 50:1
# ──────────────────────────────────────────────────────────────────────────────

COLORS = plt.cm.plasma(np.linspace(0.05, 0.90, len(LAGS)))


def _predict_in_chunks(model, parquet_path, feature_cols):
    pf = pq.ParquetFile(parquet_path)
    y_true_list, y_proba_list = [], []
    total = pf.metadata.num_rows
    processed = 0
    for batch in pf.iter_batches(batch_size=PRED_CHUNK):
        df = batch.to_pandas()
        df['lightning_binary'] = (df['lightning_count'] > 0).astype(int)
        y_true_list.append(df['lightning_binary'].values)
        y_proba_list.append(model.predict_proba(df[feature_cols])[:, 1])
        processed += len(df)
        print(f"  {processed:,}/{total:,}", end='\r')
    print()
    return np.concatenate(y_true_list), np.concatenate(y_proba_list)


def _threshold_at_min_recall(y_test, y_proba):
    """Return (precision, recall) at the threshold that maximises precision
    while keeping recall >= MIN_RECALL."""
    prec, rec, thr = precision_recall_curve(y_test, y_proba)
    prec, rec = prec[:-1], rec[:-1]
    valid = rec >= MIN_RECALL
    if valid.any():
        best = np.argmax(prec[valid])
        return float(prec[valid][best]), float(rec[valid][best])
    else:
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        best = np.argmax(f1)
        return float(prec[best]), float(rec[best])


def collect_curves(model_type):
    """For each lag, load the saved model and compute curve data.
    model_type: 'lgb' or 'xgb'
    """
    results = []
    for lag in LAGS:
        lag_str  = f"_lag{lag}" if lag > 0 else ""
        ts       = f"train{TRAIN_YEARS[:4]}_test{TEST_YEAR}_{lag}"

        # model path
        if model_type == 'lgb':
            model_path = f"{DATA_DIR}/lightgbm_model_{ts}.txt"
        else:
            model_path = f"{DATA_DIR}/xgboost_model_{ts}.json"

        # test parquet for this lag (always unbalanced)
        test_path = f"{DATA_DIR}/tabular_dataset_{TEST_YEAR}{lag_str}{MASK_STR}.parquet"

        try:
            print(f"\nLag {lag} — loading {model_type} from {model_path}")
            if model_type == 'lgb':
                booster  = lgb.Booster(model_file=model_path)
                # wrap in a sklearn-compatible predictor
                class _LGBWrapper:
                    def __init__(self, b): self.b = b
                    def predict_proba(self, X):
                        p = self.b.predict(X)
                        return np.column_stack([1 - p, p])
                model = _LGBWrapper(booster)
            else:
                model = xgb.XGBClassifier()
                model.load_model(model_path)

            # infer feature columns from model
            pf = pq.ParquetFile(test_path)
            sample = next(pf.iter_batches(batch_size=1000)).to_pandas()
            feature_cols = [c for c in sample.columns
                            if c not in ['time', 'lightning_count',
                                         'lightning_binary', 'lat', 'lon']]

            y_test, y_proba = _predict_in_chunks(model, test_path, feature_cols)

            fpr, tpr, _      = roc_curve(y_test, y_proba)
            pr_prec, pr_rec, _ = precision_recall_curve(y_test, y_proba)
            roc_auc          = roc_auc_score(y_test, y_proba)
            pr_auc           = average_precision_score(y_test, y_proba)
            best_p, best_r   = _threshold_at_min_recall(y_test, y_proba)
            base_rate        = float(y_test.mean())

            results.append({
                'lag': lag, 'fpr': fpr, 'tpr': tpr,
                'pr_rec': pr_rec, 'pr_prec': pr_prec,
                'roc_auc': roc_auc, 'pr_auc': pr_auc,
                'best_p': best_p, 'best_r': best_r,
                'base_rate': base_rate,
            })
        except FileNotFoundError as e:
            print(f"  Skipping lag {lag}: {e}")

    return results


def plot_lag_figure(lgb_results, xgb_results):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f'Lag Comparison: T to T-6  |  train={BALANCE_STR.strip("_")}  |  test=unbalanced',
        fontsize=13)

    base_rate = lgb_results[0]['base_rate'] if lgb_results else xgb_results[0]['base_rate']

    for row, (model_name, results) in enumerate([('LightGBM', lgb_results),
                                                  ('XGBoost',  xgb_results)]):
        ax_roc, ax_pr = axes[row, 0], axes[row, 1]

        for d, color in zip(results, COLORS[:len(results)]):
            lag_label = f"lag {d['lag']}h"
            ax_roc.plot(d['fpr'], d['tpr'], color=color, lw=1.8,
                        label=f"{lag_label}  AUC={d['roc_auc']:.4f}")
            ax_pr.plot(d['pr_rec'], d['pr_prec'], color=color, lw=1.8,
                       label=f"{lag_label}  PR-AUC={d['pr_auc']:.4f}")
            ax_pr.scatter(d['best_r'], d['best_p'], color=color,
                          marker='*', s=100, zorder=5)

        # no-skill lines
        ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, label='No skill')
        ax_pr.axhline(base_rate, color='k', lw=1, linestyle='--',
                      label=f'No skill (base rate={base_rate:.4f})')
        ax_pr.axvline(MIN_RECALL, color='gray', linestyle=':', lw=1,
                      label=f'Min recall={MIN_RECALL}')

        ax_roc.set_title(f'ROC — {model_name}')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate (Recall)')
        ax_roc.set_xlim([0, 1]); ax_roc.set_ylim([0, 1])
        ax_roc.legend(fontsize=7, loc='lower right')
        ax_roc.grid(True, alpha=0.3)

        ax_pr.set_title(f'Precision-Recall — {model_name}')
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_xlim([0, 1]); ax_pr.set_ylim([0, 1])
        ax_pr.legend(fontsize=7, loc='upper right')
        ax_pr.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"{OUT_DIR}/lag_comparison{BALANCE_STR}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    print("Collecting LightGBM curves...")
    lgb_results = collect_curves('lgb')

    print("\nCollecting XGBoost curves...")
    xgb_results = collect_curves('xgb')

    print("\nPlotting...")
    plot_lag_figure(lgb_results, xgb_results)
