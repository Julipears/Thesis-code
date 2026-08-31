import zipfile
from io import BytesIO
from pathlib import Path
import requests
import pandas as pd

run_dir = Path("./sa_btc_um")
cache_path = run_dir / "metrics_cache" / "BTCUSDT_um_liquidity_metrics.csv.gz"

def check_cache():
    print(f"Checking cache: {cache_path}")
    if not cache_path.exists():
        print(" Cache not found")
        return
    metrics = pd.read_csv(cache_path, compression='infer')
    print(f" Cache rows: {len(metrics)} columns: {list(metrics.columns)}")
    col = 'sum_toptrader_long_short_ratio'
    outcol = 'trader_long_short'
    if col in metrics.columns:
        nonnull = int(metrics[col].notna().sum())
        print(f" Source column {col} non-null: {nonnull}/{len(metrics)}")
    if outcol in metrics.columns:
        nonnull = int(metrics[outcol].notna().sum())
        print(f" Output column {outcol} non-null: {nonnull}/{len(metrics)}")
    # summarize by month
    try:
        metrics['create_time'] = pd.to_datetime(metrics['create_time'], utc=True, errors='coerce')
        metrics['month'] = metrics['create_time'].dt.to_period('M')
        if outcol in metrics.columns:
            grp = metrics.groupby('month')[outcol].apply(lambda s: int(s.notna().sum())).reset_index()
            print('\nNon-null counts per month for trader_long_short:')
            print(grp.tail(12))
    except Exception:
        pass

def fetch_day(day='2022-07-01'):
    symbol = 'BTCUSDT'
    cm_um = 'um'
    day_string = pd.Timestamp(day).strftime('%Y-%m-%d')
    url = f"https://data.binance.vision/data/futures/{cm_um}/daily/metrics/{symbol}/{symbol}-metrics-{day_string}.zip"
    print(f"Downloading {url}")
    r = requests.get(url, timeout=30)
    print(f"Status: {r.status_code}")
    if r.status_code != 200:
        print("Failed to download")
        return
    with zipfile.ZipFile(BytesIO(r.content)) as z:
        names = [n for n in z.namelist() if not n.endswith('/')]
        print('ZIP contains:', names)
        with z.open(names[0]) as f:
            df = pd.read_csv(f, dtype=str, low_memory=False)
    print('Columns in CSV:', df.columns.tolist())
    src = 'sum_toptrader_long_short_ratio'
    if src in df.columns:
        ser = pd.to_numeric(df[src], errors='coerce')
        print(f"{src} non-null: {int(ser.notna().sum())}/{len(ser)}")
        print(ser.head(10))
    else:
        print(f"{src} not present in daily CSV")


def fetch_days(days):
    for day in days:
        print('\n==', day, '==')
        symbol = 'BTCUSDT'
        cm_um = 'um'
        day_string = pd.Timestamp(day).strftime('%Y-%m-%d')
        url = f"https://data.binance.vision/data/futures/{cm_um}/daily/metrics/{symbol}/{symbol}-metrics-{day_string}.zip"
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            print(' status', r.status_code)
            continue
        with zipfile.ZipFile(BytesIO(r.content)) as z:
            names = [n for n in z.namelist() if not n.endswith('/')]
            with z.open(names[0]) as f:
                df = pd.read_csv(f, dtype=str, low_memory=False)
        for col in ['sum_toptrader_long_short_ratio', 'count_toptrader_long_short_ratio']:
            if col in df.columns:
                ser = pd.to_numeric(df[col], errors='coerce')
                print(f" {col} non-null: {int(ser.notna().sum())}/{len(ser)} mean: {ser.mean():.6f}")
            else:
                print(f" {col} missing in CSV")

if __name__ == '__main__':
    check_cache()
    print('\n---\n')
    fetch_day('2022-07-01')
    fetch_days(pd.date_range('2022-07-01', periods=7, freq='D').strftime('%Y-%m-%d'))
