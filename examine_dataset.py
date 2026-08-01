import os
import pandas as pd
import pyarrow.parquet as pq
from data_profiling import ProfileReport


if __name__ == '__main__':
    parquet_path = 'data/tabular_dataset_2004_2005_2006_2008_2009_2023_2024_balanced.parquet'
    out_dir      = 'data/examine_preprocessing'
    os.makedirs(out_dir, exist_ok=True)

    name     = os.path.splitext(os.path.basename(parquet_path))[0]
    out_path = f'{out_dir}/report_{name}.html'

    is_sample = True  # set False only if you have enough RAM (balanced parquet is small)

    if is_sample:
        # read first 500K rows — safe on any instance size
        pf = pq.ParquetFile(parquet_path)
        df = next(pf.iter_batches(batch_size=500_000)).to_pandas()
    else:
        df = pd.read_parquet(parquet_path)

    print(f"Loaded {len(df):,} rows. Generating report...")
    ProfileReport(df, minimal=True).to_file(out_path)
    print(f"Saved to {out_path}")

    