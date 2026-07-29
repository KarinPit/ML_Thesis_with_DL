import pandas as pd


if __name__ == "__main__":
    df = pd.concat([
        pd.read_parquet('data/tabular_dataset_2023_balanced.parquet'),
        pd.read_parquet('data/tabular_dataset_2024_balanced.parquet'),
    ]).sort_values('time').reset_index(drop=True)

    df.to_parquet('data/tabular_dataset_2023_2024_balanced.parquet', index=False)
    print(df.shape)