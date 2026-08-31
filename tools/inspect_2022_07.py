from pathlib import Path
import pandas as pd

run_dir = Path("./sa_btc_um")
input_subdir = "aft_data_liquidity"
cache_subdir = "metrics_cache"
cache_name = "BTCUSDT_um_liquidity_metrics.csv.gz"

aug_path = run_dir / input_subdir / "spot_2022-07.parquet"
cache_path = run_dir / cache_subdir / cache_name

def show_augmented():
    print(f"Augmented file: {aug_path}")
    if not aug_path.exists():
        print(" Augmented file not found")
        return
    df = pd.read_parquet(aug_path)
    print(f" Rows: {len(df)} Columns: {list(df.columns)}")
    col = "trader_long_short"
    if col in df.columns:
        nonnull = int(df[col].notna().sum())
        print(f" {col} non-null: {nonnull}/{len(df)}")
        print(df[["start_ts", col]].head(10))
    else:
        print(f" {col} not in augmented file")

def show_cache():
    print(f"Metrics cache: {cache_path}")
    if not cache_path.exists():
        print(" Metrics cache not found")
        return
    metrics = pd.read_csv(cache_path, compression="infer")
    metrics["create_time"] = pd.to_datetime(metrics["create_time"], utc=True, errors="coerce")
    july = metrics[metrics["create_time"].dt.to_period("M") == pd.Period("2022-07")]
    print(f" Metrics rows total: {len(metrics)}, July-2022 rows: {len(july)}")
    col = "trader_long_short"
    if col in metrics.columns:
        nonnull = int(july[col].notna().sum())
        print(f" {col} non-null in July: {nonnull}/{len(july)}")
        print(july[["create_time", col]].head(10))
    else:
        print(f" {col} not present in metrics cache columns: {list(metrics.columns)}")

if __name__ == '__main__':
    show_augmented()
    print()
    show_cache()
