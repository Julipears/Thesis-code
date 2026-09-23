"""Backfill missing Binance futures metrics days into existing caches."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading

import pandas as pd
import requests

from survival_analysis_data_pull_final import fetch_one_metric_day


MARKETS = {
    "btc_um": ("BTCUSDT", "um", Path("sa_btc_um/metrics_cache/BTCUSDT_um_liquidity_metrics.csv.gz")),
    "btc_cm": ("BTCUSDT", "cm", Path("sa_btc_cm/metrics_cache/BTCUSD_PERP_cm_liquidity_metrics.csv.gz")),
    "eth_um": ("ETHUSDT", "um", Path("sa_eth_um/metrics_cache/ETHUSDT_um_liquidity_metrics.csv.gz")),
    "eth_cm": ("ETHUSDT", "cm", Path("sa_eth_cm/metrics_cache/ETHUSD_PERP_cm_liquidity_metrics.csv.gz")),
}
START = pd.Timestamp("2021-01-01", tz="UTC")
END = pd.Timestamp("2025-12-31", tz="UTC")
THREAD_LOCAL = threading.local()


def session() -> requests.Session:
    if not hasattr(THREAD_LOCAL, "session"):
        THREAD_LOCAL.session = requests.Session()
        THREAD_LOCAL.session.headers.update({"User-Agent": "Mozilla/5.0"})
    return THREAD_LOCAL.session


def fetch(symbol: str, contract: str, day: pd.Timestamp):
    frame = fetch_one_metric_day(
        session(), symbol, contract, day,
        retries=3, timeout_seconds=45,
    )
    return day, frame


def backfill_market(market: str, workers: int) -> None:
    symbol, contract, cache_path = MARKETS[market]
    cached = pd.read_csv(cache_path, compression="infer")
    cached["create_time"] = pd.to_datetime(cached["create_time"], utc=True, errors="coerce")
    present = pd.DatetimeIndex(cached["create_time"].dropna().dt.floor("D").unique())
    expected = pd.date_range(START, END, freq="D")
    missing = expected.difference(present)
    print(f"[{market}] {len(missing)} missing days to request", flush=True)
    if missing.empty:
        return

    fetched_frames = []
    status_rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, symbol, contract, day): day for day in missing}
        for number, future in enumerate(as_completed(futures), start=1):
            day = futures[future]
            try:
                _, frame = future.result()
                if frame is not None and not frame.empty:
                    fetched_frames.append(frame)
                    status = "fetched"
                    rows = len(frame)
                else:
                    status = "unavailable"
                    rows = 0
                error = ""
            except Exception as exc:
                status = "error"
                rows = 0
                error = f"{type(exc).__name__}: {exc}"
            status_rows.append({
                "market": market,
                "day": day.strftime("%Y-%m-%d"),
                "status": status,
                "rows": rows,
                "error": error,
            })
            if number == 1 or number % 25 == 0 or number == len(missing):
                print(f"[{market}] requested {number}/{len(missing)}", flush=True)

    if fetched_frames:
        combined = pd.concat([cached, *fetched_frames], ignore_index=True, sort=False)
        combined["create_time"] = pd.to_datetime(combined["create_time"], utc=True, errors="coerce")
        combined = (
            combined.dropna(subset=["create_time"])
            .sort_values("create_time")
            .drop_duplicates("create_time", keep="last")
            .reset_index(drop=True)
        )
        temporary = cache_path.with_name(cache_path.name + ".tmp.gz")
        combined.to_csv(temporary, index=False, compression="gzip")
        temporary.replace(cache_path)
        print(f"[{market}] added {len(combined) - len(cached)} rows", flush=True)

    status_dir = Path("open_interest_figures/backfill_status")
    status_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(status_rows).sort_values("day").to_csv(
        status_dir / f"{market}_missing_day_backfill.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", nargs="+", choices=MARKETS, default=list(MARKETS))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    for market in args.markets:
        backfill_market(market, args.workers)


if __name__ == "__main__":
    main()
