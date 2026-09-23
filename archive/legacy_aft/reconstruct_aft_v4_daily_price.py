"""Add a leakage-safe daily spot-price level to the unit-agreeing-OI AFT inputs."""

from __future__ import annotations

import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from survival_analysis_utils_final import atomic_pandas_csv, atomic_pandas_parquet


MARKETS = ("btc_um", "btc_cm", "eth_um", "eth_cm")
ASSETS = ("btc", "eth")
SOURCE_DIR_NAME = "aft_data_liquidity_unit_agreeing_oi"
OUTPUT_DIR_NAME = "aft_data_liquidity_v4_daily_price"
RAW_ROOT = Path("open_interest_figures/native/daily_spot_price/raw_monthly_klines")
START_MONTH = pd.Period("2020-12", freq="M")
END_MONTH = pd.Period("2025-12", freq="M")


def archive_path(asset: str, month: pd.Period) -> Path:
    symbol = f"{asset.upper()}USDT"
    return RAW_ROOT / asset / f"{symbol}-1d-{str(month)}.zip"


def archive_url(asset: str, month: pd.Period) -> str:
    symbol = f"{asset.upper()}USDT"
    name = f"{symbol}-1d-{str(month)}.zip"
    return f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/1d/{name}"


def fetch_month(asset: str, month: pd.Period) -> pd.DataFrame:
    path = archive_path(asset, month)
    url = archive_url(asset, month)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        last_error = None
        for attempt in range(4):
            try:
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                    if archive.testzip() is not None:
                        raise ValueError("corrupt ZIP archive")
                path.write_bytes(response.content)
                break
            except Exception as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(attempt + 1)
        else:
            raise last_error

    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"Unexpected archive contents: {path}")
        with archive.open(names[0]) as handle:
            raw = pd.read_csv(handle, header=None, dtype=str)
    raw = raw.loc[pd.to_numeric(raw[0], errors="coerce").notna()].copy()
    if raw.shape[1] != 12:
        raise ValueError(f"Unexpected kline schema: {path}")
    raw = raw.apply(pd.to_numeric, errors="raise")
    times = raw[0].to_numpy(dtype=float)
    # Binance archives have used milliseconds and, in some periods, microseconds.
    times_ms = np.where(times > 1e14, times / 1000.0, times)
    dates = pd.to_datetime(times_ms, unit="ms", utc=True).tz_localize(None).normalize()
    close = pd.to_numeric(raw[4], errors="raise").to_numpy(dtype=float)
    out = pd.DataFrame({
        "date": dates,
        "daily_price": close,
        "asset": asset,
        "symbol": f"{asset.upper()}USDT",
        "source_url": url,
    })
    if out["date"].duplicated().any() or not out["daily_price"].gt(0).all():
        raise ValueError(f"Invalid daily price data: {path}")
    return out


def load_price_maps() -> dict[str, pd.DataFrame]:
    months = list(pd.period_range(START_MONTH, END_MONTH, freq="M"))
    maps = {}
    for asset in ASSETS:
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(fetch_month, asset, month): month for month in months}
            frames = [future.result() for future in as_completed(futures)]
        frame = pd.concat(frames, ignore_index=True).sort_values("date")
        frame = frame.drop_duplicates("date", keep="last")
        expected = pd.date_range(
            START_MONTH.start_time.normalize(), END_MONTH.end_time.normalize(), freq="D"
        )
        missing = expected.difference(pd.DatetimeIndex(frame["date"]))
        if len(missing):
            raise ValueError(f"Missing {asset} spot daily prices: {list(missing[:10])}")
        # Standardize the log price separately for BTC and ETH.  The price used
        # for an event is still the prior completed UTC day's close; these
        # asset-level constants only put the two price series on a comparable
        # scale.
        log_price = np.log(frame["daily_price"].to_numpy(dtype=float))
        log_mean = float(log_price.mean())
        log_std = float(log_price.std(ddof=0))
        if not np.isfinite(log_std) or log_std <= 0:
            raise ValueError(f"Invalid log-price standard deviation for {asset}: {log_std}")
        frame["daily_log_price"] = log_price
        frame["log_price_standardized"] = (log_price - log_mean) / log_std
        maps[asset] = frame.set_index("date")
        maps[asset].to_csv(Path("open_interest_figures/native/daily_spot_price") / f"{asset}_daily_spot_close.csv")
    return maps


def reconstruct_file(source_path: Path, output_path: Path, price_map: pd.DataFrame) -> dict:
    existed = output_path.exists()
    frame = pd.read_parquet(source_path)
    # Some early event files have no create_time.  Use the event start time
    # for those rows; it is the timestamp that defines the event's calendar
    # day and is available throughout the sample.
    create_time = pd.to_datetime(frame["create_time"], utc=True, errors="coerce")
    start_time = pd.to_datetime(frame["start_ts"], utc=True, errors="coerce")
    event_time = create_time.fillna(start_time)
    if event_time.isna().any():
        raise ValueError(f"Missing both create_time and start_ts in {source_path.name}")
    event_day = event_time.dt.floor("D").dt.tz_localize(None)
    price_date = event_day - pd.Timedelta(days=1)
    aligned = price_map.reindex(price_date.to_numpy())
    prices = pd.to_numeric(aligned["daily_price"], errors="coerce").to_numpy()
    standardized_log_prices = pd.to_numeric(
        aligned["log_price_standardized"], errors="coerce"
    ).to_numpy()
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError(f"Missing or invalid prior-day spot price in {source_path.name}")
    if not np.isfinite(standardized_log_prices).all():
        raise ValueError(f"Missing or invalid standardized log price in {source_path.name}")
    frame = frame.copy()
    frame["daily_price"] = prices
    frame["daily_log_price"] = np.log(prices)
    frame["log_price_standardized"] = standardized_log_prices
    frame["daily_price_date"] = price_date.to_numpy()
    atomic_pandas_parquet(frame, output_path)
    status = "updated_existing" if existed else "written"
    return {"source_file": source_path.name, "status": status, "output_file": str(output_path), "rows": len(frame)}


def main() -> None:
    price_maps = load_price_maps()
    manifests = []
    for market in MARKETS:
        asset = market.split("_", 1)[0]
        source_dir = Path(f"sa_{market}") / SOURCE_DIR_NAME
        output_dir = Path(f"sa_{market}") / OUTPUT_DIR_NAME
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        files = sorted(source_dir.glob("*.parquet"))
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {
                pool.submit(reconstruct_file, path, output_dir / path.name, price_maps[asset]): path
                for path in files
            }
            for index, future in enumerate(as_completed(futures), 1):
                path = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {"source_file": path.name, "status": "failed", "error": repr(exc)}
                row["market"] = market
                rows.append(row)
                if index % 20 == 0 or index == len(files):
                    print(f"[reconstruct {market}] {index}/{len(files)} {row['status']}", flush=True)
        manifest = pd.DataFrame(rows).sort_values("source_file")
        atomic_pandas_csv(manifest, output_dir / "reconstruction_manifest.csv")
        manifests.append(manifest)
    summary = pd.concat(manifests, ignore_index=True)
    summary.to_csv(Path("open_interest_figures/native/aft_v4_daily_price_reconstruction_summary.csv"), index=False)
    print(summary.groupby(["market", "status"]).size().to_string())


if __name__ == "__main__":
    main()
