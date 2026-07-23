import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score

def train_lightgbm():
    # load tabular data
    df = pd.read_parquet('data/case6/tabular_dataset.parquet')
    print(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

    # define features and target. Binary target: 1 if any lightning in the cell, 0 otherwise
    df['lightning_binary'] = (df['lightning_count'] > 0).astype(int)
    print(f"Lightning distribution:\n{df['lightning_binary'].value_counts()}")
    print(f"  → {df['lightning_binary'].mean()*100:.2f}% of cells have lightning")

    # Features: drop identifiers and target columns
    feature_cols = [c for c in df.columns if c not in ['time', 'lightning_count', 'lightning_binary', 'lat', 'lon']]
    print(f"\nNumber of features: {len(feature_cols)}")

    X = df[feature_cols]
    y = df['lightning_binary']

    # time-based train/test split
    unique_times = np.sort(df['time'].unique())
    split_idx = int(len(unique_times) * 0.8)
    train_times = unique_times[:split_idx]
    test_times  = unique_times[split_idx:]

    train_mask = df['time'].isin(train_times)
    test_mask  = df['time'].isin(test_times)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    print(f"\nTrain: {len(train_times)} hours ({train_mask.sum()} rows)")
    print(f"Test:  {len(test_times)} hours ({test_mask.sum()} rows)")

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

    # evaluate
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n── Classification Report ──")
    print(classification_report(y_test, y_pred, target_names=['No Lightning', 'Lightning']))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    # feature importance (top 20)
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    print("\n── Top 20 most important features ──")
    print(importance.nlargest(20).to_string())

    # save model
    model.booster_.save_model('data/case6/lightgbm_model.txt')
    print("\nModel saved to data/case6/lightgbm_model.txt")

    return model

if __name__ == "__main__":
    train_lightgbm()
