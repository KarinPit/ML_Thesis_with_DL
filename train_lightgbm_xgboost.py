import json
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import pyarrow.parquet as pq
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, roc_curve
import matplotlib
matplotlib.use('Agg')  # no display needed — saves to file
import matplotlib.pyplot as plt
import re

# Minimum recall we're willing to accept when optimizing the threshold.
MIN_RECALL   = 0.30
EVAL_SAMPLE  = 200_000   # rows sampled from test parquet for early stopping
PRED_CHUNK   = 100_000   # rows per chunk during full test prediction


def _load_eval_sample(test_parquet_path, feature_cols):
    """Load first EVAL_SAMPLE rows from test parquet for early stopping."""
    pf = pq.ParquetFile(test_parquet_path)
    batch = next(pf.iter_batches(batch_size=EVAL_SAMPLE))
    df = batch.to_pandas()
    df['lightning_binary'] = (df['lightning_count'] > 0).astype(int)
    return df[feature_cols], df['lightning_binary']


def _predict_in_chunks(model, test_parquet_path, feature_cols):
    """Predict on the full test set in chunks — avoids loading it all into RAM."""
    pf = pq.ParquetFile(test_parquet_path)
    y_true_list = []
    y_proba_list = []
    total = pf.metadata.num_rows
    processed = 0
    for batch in pf.iter_batches(batch_size=PRED_CHUNK):
        df = batch.to_pandas()
        df['lightning_binary'] = (df['lightning_count'] > 0).astype(int)
        y_true_list.append(df['lightning_binary'].values)
        y_proba_list.append(model.predict_proba(df[feature_cols])[:, 1])
        processed += len(df)
        print(f"  Predicting... {processed:,} / {total:,}", end='\r')
    print()
    return np.concatenate(y_true_list), np.concatenate(y_proba_list)


def evaluate_model(model_name, y_test, y_proba, feature_cols, feature_importances, ts, out_dir, metrics_dict=None):
    """Evaluate a trained model: classification report, threshold optimization, plots, feature importance."""

    y_pred = (y_proba >= 0.5).astype(int)

    print(f"\n{'='*60}")
    print(f"{model_name} — Classification Report (default threshold=0.50)")
    print('='*60)
    print(classification_report(y_test, y_pred, target_names=['No Lightning', 'Lightning']))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    # threshold optimization: maximize precision while keeping recall >= MIN_RECALL
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    precisions, recalls = precisions[:-1], recalls[:-1]

    valid = recalls >= MIN_RECALL
    if valid.any():
        best_idx       = np.argmax(precisions[valid])
        best_threshold = thresholds[valid][best_idx]
        best_precision = precisions[valid][best_idx]
        best_recall    = recalls[valid][best_idx]
    else:
        f1_scores      = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        best_idx       = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_precision = precisions[best_idx]
        best_recall    = recalls[best_idx]
        print(f"  (No threshold achieved recall ≥ {MIN_RECALL}, falling back to max-F1)")

    print(f"\n── Optimized threshold (min recall={MIN_RECALL}) ──")
    print(f"  Threshold: {best_threshold:.4f}")
    print(f"  Precision: {best_precision:.4f}  |  Recall: {best_recall:.4f}")

    y_pred_opt = (y_proba >= best_threshold).astype(int)
    print(f"\n── Classification Report (optimized threshold) ──")
    print(classification_report(y_test, y_pred_opt, target_names=['No Lightning', 'Lightning']))

    # plots: ROC + PR curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(model_name)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    axes[0].plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {auc:.4f}')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate (Recall)')
    axes[0].set_title('ROC Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    pr_precisions, pr_recalls, _ = precision_recall_curve(y_test, y_proba)
    axes[1].plot(pr_recalls, pr_precisions, color='darkorange', lw=2)
    axes[1].scatter(best_recall, best_precision, color='red', zorder=5,
                    label=f'Chosen threshold\nP={best_precision:.3f}, R={best_recall:.3f}')
    axes[1].axvline(MIN_RECALL, color='gray', linestyle='--', lw=1, label=f'Min recall={MIN_RECALL}')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    name_slug = model_name.lower().replace(' ', '_')
    plot_path = f'{out_dir}/{name_slug}_curves_{ts}.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Curves saved to {plot_path}")

    # feature importance (top 20)
    importance = pd.Series(feature_importances, index=feature_cols)
    print(f"\n── Top 20 most important features ──")
    print(importance.nlargest(20).to_string())

    # save metrics for lag comparison plots
    if metrics_dict is not None:
        auc = roc_auc_score(y_test, y_proba)
        metrics_dict[model_name] = {
            'roc_auc':        auc,
            'opt_threshold':  float(best_threshold),
            'opt_precision':  float(best_precision),
            'opt_recall':     float(best_recall),
        }


def train_lightgbm_and_xgboost(train_parquet_path, test_parquet_path, out_dir='data', lag=0):
    # load train data (balanced, small — fits in memory)
    train_df = pd.read_parquet(train_parquet_path)
    print(f"Train dataset: {train_df.shape[0]:,} rows × {train_df.shape[1]} columns")

    # test set: only metadata for reporting; full predictions done in chunks
    test_pf   = pq.ParquetFile(test_parquet_path)
    test_rows = test_pf.metadata.num_rows
    print(f"Test dataset:  {test_rows:,} rows (will predict in chunks of {PRED_CHUNK:,})")

    # binary target for train
    train_df['lightning_binary'] = (train_df['lightning_count'] > 0).astype(int)
    print(f"\nTrain lightning: {train_df['lightning_binary'].sum():,} "
          f"({train_df['lightning_binary'].mean()*100:.2f}%)")

    # features
    feature_cols = [c for c in train_df.columns
                    if c not in ['time', 'lightning_count', 'lightning_binary', 'lat', 'lon']]
    print(f"Number of features: {len(feature_cols)}")

    X_train, y_train = train_df[feature_cols], train_df['lightning_binary']

    # small eval sample from test for early stopping
    X_eval, y_eval = _load_eval_sample(test_parquet_path, feature_cols)
    print(f"Early-stopping eval sample: {len(X_eval):,} rows "
          f"({y_eval.sum():,} lightning, {(y_eval==0).sum():,} no-lightning)")

    # timestamp for file naming
    train_year = pd.to_datetime(train_df['time']).dt.year.min()
    match      = re.search(r'(\d{4})(?!.*\d{4})', test_parquet_path)
    test_year  = int(match.group(1)) if match else 'unknown'
    ts = f"train{train_year}_test{test_year}_{lag}"

    # ── LightGBM ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Training LightGBM...")
    print("="*60)
    lgb_model = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=64,
        min_child_samples=20,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_eval, y_eval)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
    )

    metrics = {}

    print("Running full test set prediction in chunks...")
    lgb_y_test, lgb_proba = _predict_in_chunks(lgb_model, test_parquet_path, feature_cols)
    evaluate_model('LightGBM', lgb_y_test, lgb_proba, feature_cols,
                   lgb_model.booster_.feature_importance(importance_type='gain'), ts, out_dir, metrics)

    lgb_model_path = f'{out_dir}/lightgbm_model_{ts}.txt'
    lgb_model.booster_.save_model(lgb_model_path)
    print(f"LightGBM model saved to {lgb_model_path}")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Training XGBoost...")
    print("="*60)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=20,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        early_stopping_rounds=50,
        verbosity=1,
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_eval, y_eval)],
        verbose=50,
    )

    print("Running full test set prediction in chunks...")
    xgb_y_test, xgb_proba = _predict_in_chunks(xgb_model, test_parquet_path, feature_cols)
    evaluate_model('XGBoost', xgb_y_test, xgb_proba, feature_cols,
                   xgb_model.feature_importances_, ts, out_dir, metrics)

    xgb_model_path = f'{out_dir}/xgboost_model_{ts}.json'
    xgb_model.save_model(xgb_model_path)
    print(f"XGBoost model saved to {xgb_model_path}")

    # save metrics JSON for lag comparison plots
    metrics_path = f'{out_dir}/metrics_{ts}.json'
    with open(metrics_path, 'w') as f:
        json.dump({'ts': ts, **metrics}, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    return lgb_model, xgb_model


if __name__ == "__main__":
    lags = [1, 2, 3, 4, 5, 6]

    for lag in lags:
        lag_str = f"_lag{lag}" if lag > 0 else ""
        train_lightgbm_and_xgboost(
            train_parquet_path=f'data/tabular_dataset_2004_2005_2006_2008_2009_2023_2024{lag_str}_balanced.parquet',
            test_parquet_path=f'data/tabular_dataset_2025{lag_str}.parquet', lag=lag
        )
