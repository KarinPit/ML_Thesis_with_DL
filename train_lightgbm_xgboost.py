import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, roc_curve
import matplotlib
import matplotlib.pyplot as plt

# Minimum recall we're willing to accept when optimizing the threshold.
# The threshold that maximizes precision above this recall floor will be selected.
MIN_RECALL = 0.30


def evaluate_model(model_name, y_test, y_proba, feature_cols, feature_importances, ts, out_dir):
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


def train_lightgbm_and_xgboost(train_parquet_path, test_parquet_path, out_dir='data'):
    # load train and test data
    train_df = pd.read_parquet(train_parquet_path)
    test_df  = pd.read_parquet(test_parquet_path)
    print(f"Train dataset: {train_df.shape[0]} rows × {train_df.shape[1]} columns")
    print(f"Test dataset:  {test_df.shape[0]} rows × {test_df.shape[1]} columns")

    # binary target
    train_df['lightning_binary'] = (train_df['lightning_count'] > 0).astype(int)
    test_df['lightning_binary']  = (test_df['lightning_count']  > 0).astype(int)
    print(f"\nTrain lightning: {train_df['lightning_binary'].sum()} "
          f"({train_df['lightning_binary'].mean()*100:.2f}%)")
    print(f"Test  lightning: {test_df['lightning_binary'].sum()} "
          f"({test_df['lightning_binary'].mean()*100:.2f}%)")

    # features
    feature_cols = [c for c in train_df.columns if c not in ['time', 'lightning_count', 'lightning_binary', 'lat', 'lon']]
    print(f"\nNumber of features: {len(feature_cols)}")

    X_train, y_train = train_df[feature_cols], train_df['lightning_binary']
    X_test,  y_test  = test_df[feature_cols],  test_df['lightning_binary']

    # timestamp for file naming
    train_year = pd.to_datetime(train_df['time']).dt.year.min()
    test_year  = pd.to_datetime(test_df['time']).dt.year.iloc[0]
    ts = f"train{train_year}_test{test_year}"

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
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
    )
    lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
    evaluate_model('LightGBM', y_test, lgb_proba, feature_cols, lgb_model.feature_importances_, ts, out_dir)

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
        scale_pos_weight=scale_pos_weight,  # equivalent to class_weight='balanced'
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        early_stopping_rounds=50,
        verbosity=1,
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    evaluate_model('XGBoost', y_test, xgb_proba, feature_cols, xgb_model.feature_importances_, ts, out_dir)

    xgb_model_path = f'{out_dir}/xgboost_model_{ts}.json'
    xgb_model.save_model(xgb_model_path)
    print(f"XGBoost model saved to {xgb_model_path}")

    return lgb_model, xgb_model


if __name__ == "__main__":
    train_lightgbm_and_xgboost(
        train_parquet_path='data/tabular_dataset_2023_2024_balanced.parquet',
        test_parquet_path='data/tabular_dataset_2025.parquet',
    )
