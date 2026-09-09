import pandas as pd
parquets = ['data/combined_tabular_dataset_2004.parquet',
            'data/combined_tabular_dataset_2005.parquet',
            'data/combined_tabular_dataset_2006.parquet']
active_months = [10, 11, 12, 1, 2]
pos = neg = 0
for p in parquets:
    df = pd.read_parquet(p, columns=['time', 'lightning_count'])
    df['time'] = pd.to_datetime(df['time'])
    df = df[df['time'].dt.month.isin(active_months)]
    pos += (df['lightning_count'] > 0).sum()
    neg += (df['lightning_count'] == 0).sum()
print(f"pos_weight should be ~{neg/pos:.0f}")