"""
plot_lag_results.py
-------------------
Reads metrics JSON files saved by train_lightgbm_xgboost.py and plots
ROC-AUC vs lag for both LightGBM and XGBoost.

Run after all lag experiments are complete.
"""

import json
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = 'data'
OUT_DIR  = 'data'


def load_metrics(data_dir):
    """Load all metrics JSON files and return a dict keyed by lag."""
    results = {}  # lag → {LightGBM: {...}, XGBoost: {...}}

    for path in glob.glob(os.path.join(data_dir, 'metrics_train2004_test2025*.json')):
        with open(path) as f:
            m = json.load(f)

        # extract lag from filename: metrics_train2004_test2025_lag1.json → 1
        # or metrics_train2004_test2025.json → 0
        fname = os.path.basename(path)
        if '_lag' in fname:
            lag = int(fname.split('_lag')[1].replace('.json', ''))
        else:
            lag = 0

        results[lag] = m

    return dict(sorted(results.items()))


def plot_auc_vs_lag(results, out_dir):
    lags   = sorted(results.keys())
    models = ['LightGBM', 'XGBoost']
    colors = {'LightGBM': 'steelblue', 'XGBoost': 'darkorange'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model Performance vs Forecast Lead Time', fontsize=14)

    # ── Left: ROC-AUC vs lag ─────────────────────────────────────────────────
    ax = axes[0]
    for model in models:
        aucs = [results[lag][model]['roc_auc'] for lag in lags if model in results[lag]]
        ax.plot(lags, aucs, marker='o', color=colors[model], label=model, linewidth=2)
        for lag, auc in zip(lags, aucs):
            ax.annotate(f'{auc:.4f}', (lag, auc), textcoords='offset points',
                        xytext=(0, 8), ha='center', fontsize=8)

    ax.set_xlabel('Lag (hours)')
    ax.set_ylabel('ROC-AUC')
    ax.set_title('ROC-AUC vs Lead Time')
    ax.set_xticks(lags)
    ax.set_xticklabels([f'T-{l}' if l > 0 else 'T (sync)' for l in lags])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.0)

    # ── Right: Precision & Recall at optimised threshold vs lag ──────────────
    ax = axes[1]
    for model in models:
        precs   = [results[lag][model]['opt_precision'] for lag in lags if model in results[lag]]
        recalls = [results[lag][model]['opt_recall']    for lag in lags if model in results[lag]]
        ax.plot(lags, precs,   marker='o',  color=colors[model], label=f'{model} precision', linewidth=2)
        ax.plot(lags, recalls, marker='s', color=colors[model], label=f'{model} recall',
                linestyle='--', linewidth=2)

    ax.set_xlabel('Lag (hours)')
    ax.set_ylabel('Score')
    ax.set_title('Precision & Recall vs Lead Time\n(optimised threshold, min recall=0.30)')
    ax.set_xticks(lags)
    ax.set_xticklabels([f'T-{l}' if l > 0 else 'T (sync)' for l in lags])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'lag_comparison.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    results = load_metrics(DATA_DIR)
    if not results:
        print("No metrics JSON files found. Run the lag experiments first.")
    else:
        print(f"Found results for lags: {sorted(results.keys())}")
        for lag, m in results.items():
            label = f"T-{lag}" if lag > 0 else "T (sync)"
            for model in ['LightGBM', 'XGBoost']:
                if model in m:
                    print(f"  {label}  {model}: AUC={m[model]['roc_auc']:.4f}")
        plot_auc_vs_lag(results, OUT_DIR)
