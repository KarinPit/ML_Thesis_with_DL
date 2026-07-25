import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from datetime import datetime

# Minimum recall we're willing to accept when optimizing the threshold.
# The threshold that maximizes precision above this recall floor will be selected.
MIN_RECALL = 0.30

def train_lightgbm(train_parquet_path, test_parquet_path, out_dir='data'):
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

    # train LightGBM
    model = lgb.LGBMClassifier(
    objective='binary',
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=64,
    min_child_samples=20,
    class_weight='balanced',  # handles class imbalance (few lightning cells)
    random_state=42,
    n_jobs=-1,
    )

    model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
    )

    # evaluate at default threshold (0.5)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = model.predict(X_test)

    print("\n── Classification Report (default threshold=0.50) ──")
    print(classification_report(y_test, y_pred, target_names=['No Lightning', 'Lightning']))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    # threshold optimization: maximize precision while keeping recall >= MIN_RECALL
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    # precision_recall_curve returns one more value than thresholds, align them
    precisions, recalls = precisions[:-1], recalls[:-1]

    valid = recalls >= MIN_RECALL
    if valid.any():
        best_idx       = np.argmax(precisions[valid])
        best_threshold = thresholds[valid][best_idx]
        best_precision = precisions[valid][best_idx]
        best_recall    = recalls[valid][best_idx]
    else:
        # fallback: just maximize F1
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
    print("\n── Classification Report (optimized threshold) ──")
    print(classification_report(y_test, y_pred_opt, target_names=['No Lightning', 'Lightning']))

    # feature importance (top 20)
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    print("\n── Top 20 most important features ──")
    print(importance.nlargest(20).to_string())

    # save model
    train_year = pd.to_datetime(train_df['time']).dt.year.iloc[0]
    test_year  = pd.to_datetime(test_df['time']).dt.year.iloc[0]
    ts = f"train{train_year}_test{test_year}"
    model_path = f'{out_dir}/lightgbm_model_{ts}.txt'
    model.booster_.save_model(model_path)
    print(f"\nModel saved to {model_path}")

    return model

if __name__ == "__main__":
    train_lightgbm(
        train_parquet_path='data/tabular_dataset_2024_balanced.parquet',
        test_parquet_path='data/tabular_dataset_2025.parquet',
    )
