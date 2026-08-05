import pandas as pd
import pyarrow.parquet as pq

# ── configuration ─────────────────────────────────────────────────────────────
RATIO        = 1     # no-lightning rows per lightning row (1 = 50/50, 50 = 50:1)
CHUNK_SIZE   = 50000  # rows per chunk when scanning the parquet
RANDOM_STATE = 42
# ──────────────────────────────────────────────────────────────────────────────


def balance_dataset(input_path, output_path,
                    ratio=RATIO, random_state=RANDOM_STATE):

    parquet_file = pq.ParquetFile(input_path)
    total_rows = parquet_file.metadata.num_rows
    print(f"Total rows in parquet: {total_rows:,}")

    # ── Pass 1: collect all lightning rows + count no-lightning rows ──────────
    print("Pass 1: scanning for lightning rows...")
    lightning_chunks = []
    n_neg = 0

    for batch in parquet_file.iter_batches(batch_size=CHUNK_SIZE):
        df_chunk = batch.to_pandas()
        is_lightning = df_chunk['lightning_count'] > 0
        lightning_chunks.append(df_chunk[is_lightning])
        n_neg += (~is_lightning).sum()

    pos = pd.concat(lightning_chunks, ignore_index=True)
    n_pos = len(pos)
    n_neg_target = n_pos * ratio
    print(f"Lightning rows: {n_pos:,}  |  No-lightning rows: {n_neg:,}")
    print(f"Sampling {n_neg_target:,} no-lightning rows ({ratio}:1 ratio)")

    # ── Pass 2: proportionally sample no-lightning rows ───────────────────────
    print("Pass 2: sampling no-lightning rows...")
    sample_fraction = n_neg_target / n_neg
    neg_chunks = []

    for batch in parquet_file.iter_batches(batch_size=CHUNK_SIZE):
        df_chunk = batch.to_pandas()
        neg_chunk = df_chunk[df_chunk['lightning_count'] == 0]
        if len(neg_chunk) > 0:
            n_sample = max(1, round(len(neg_chunk) * sample_fraction))
            neg_chunks.append(neg_chunk.sample(n=min(n_sample, len(neg_chunk)),
                                                random_state=random_state))

    neg = pd.concat(neg_chunks, ignore_index=True)

    # ── Combine, shuffle, save ────────────────────────────────────────────────
    df_balanced = pd.concat([pos, neg]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    df_balanced['lightning_binary'] = (df_balanced['lightning_count'] > 0).astype(int)

    print(f"\nBalanced dataset: {len(df_balanced):,} rows "
          f"({df_balanced['lightning_binary'].sum():,} lightning, "
          f"{(df_balanced['lightning_binary'] == 0).sum():,} no-lightning)")

    df_balanced.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")

    return df_balanced


if __name__ == "__main__":
    LAGS            = [12, 24, 48]
    CONVECTIVE_MASK = True            # must match build_tabular_dataset.py setting

    LPATS_YEARS = [2004, 2005, 2006, 2008, 2009]
    ILDN_YEARS  = [2023, 2024, 2025]        # 2025 is test set — never balanced

    mask_str  = "_convmask" if CONVECTIVE_MASK else ""
    ratio_str = "_balanced" if RATIO == 1 else f"_balanced{RATIO}to1"

    for LAG in LAGS:
        lag_str = f"_lag{LAG}" if LAG > 0 else ""
        print(f"\n{'='*60}\nBalancing datasets for LAG={LAG}\n{'='*60}")
        for year in LPATS_YEARS + ILDN_YEARS:
            input_parquet  = f'data/tabular_dataset_{year}{lag_str}{mask_str}.parquet'
            output_parquet = f'data/tabular_dataset_{year}{lag_str}{mask_str}{ratio_str}.parquet'
            balance_dataset(input_path=input_parquet, output_path=output_parquet, ratio=RATIO)
