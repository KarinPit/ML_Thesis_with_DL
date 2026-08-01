import pandas as pd
import pyarrow.parquet as pq
from data_profiling import ProfileReport


if __name__ == '__main__':
    is_sample = False

    if is_sample:
        # read first 500K rows if the analysis get and OOM error "killed"
        pf = pq.ParquetFile('data/tabular_dataset_2004_2005_2006_2008_2009_2023_2024_balanced.parquet')
        df = next(pf.iter_batches(batch_size=500_000)).to_pandas()
        ProfileReport(df, minimal=True).to_file('data/examine preprocessing/report_2004_2005_2006_2008_2009_2023_2024_balanced.html')
    else:
        df = pd.read_parquet('data/tabular_dataset_2004_2005_2006_2008_2009_2023_2024_balanced.parquet')
        ProfileReport(df, minimal=True).to_file('data/examine preprocessing/report_2004_2005_2006_2008_2009_2023_2024_balanced.html')

    