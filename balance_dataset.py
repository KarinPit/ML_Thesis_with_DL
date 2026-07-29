import pandas as pd

# ── configuration ─────────────────────────────────────────────────────────────
INPUT_PARQUET  = 'data/tabular_dataset_2023.parquet'
OUTPUT_PARQUET = 'data/tabular_dataset_2023_balanced.parquet'
RATIO = 1  # no-lightning rows per lightning row (1 = 50/50)
RANDOM_STATE = 42
# ──────────────────────────────────────────────────────────────────────────────

def balance_dataset(input_path=INPUT_PARQUET, output_path=OUTPUT_PARQUET,
                    ratio=RATIO, random_state=RANDOM_STATE):
    df = pd.read_parquet(input_path)
    print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    df['lightning_binary'] = (df['lightning_count'] > 0).astype(int)
    print(f"Lightning: {df['lightning_binary'].sum()} rows "
          f"({df['lightning_binary'].mean()*100:.2f}%)")

    pos = df[df['lightning_binary'] == 1]
    neg = df[df['lightning_binary'] == 0].sample(n=len(pos) * ratio, random_state=random_state)

    df_balanced = pd.concat([pos, neg]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    print(f"\nBalanced dataset: {len(df_balanced)} rows "
          f"({df_balanced['lightning_binary'].sum()} lightning, "
          f"{(df_balanced['lightning_binary'] == 0).sum()} no-lightning)")

    df_balanced.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")

    return df_balanced

if __name__ == "__main__":
    balance_dataset()
