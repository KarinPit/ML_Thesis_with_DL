from download_era5_arco import download_era5
from ildn_to_era5_grid import build_lightning_grid
from build_tabular_dataset import build_tabular_dataset
from train_lightgbm import train_lightgbm

# configuration
# Time range for ERA5 download
TIME_START = '2025-01-01T00:00'
TIME_END   = '2025-12-31T23:00'

# Path(s) to ILDN lightning data file(s)
ILDN_PATH = 'data/2025_for_yoav_including_cloud.txt'

# Output directory for all generated files
OUT_DIR = 'data'


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1: Downloading ERA5 data")
    print("=" * 60)
    era5_single_path, era5_pressure_path = download_era5(
        time_range=slice(TIME_START, TIME_END),
        out_dir=OUT_DIR,
    )

    print("\n" + "=" * 60)
    print("STEP 2: Building ILDN lightning grid")
    print("=" * 60)
    lightning_path = build_lightning_grid(
        ildn_path=ILDN_PATH,
        era5_single_path=era5_single_path,
        out_dir=OUT_DIR,
    )

    print("\n" + "=" * 60)
    print("STEP 3: Building tabular dataset")
    print("=" * 60)
    _, parquet_path = build_tabular_dataset(
        era5_single_path=era5_single_path,
        era5_pressure_path=era5_pressure_path,
        lightning_path=lightning_path,
        out_dir=OUT_DIR,
    )

    print("\n" + "=" * 60)
    print("STEP 4: Training LightGBM")
    print("=" * 60)
    train_lightgbm(
        parquet_path=parquet_path,
        out_dir=OUT_DIR,
    )

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
