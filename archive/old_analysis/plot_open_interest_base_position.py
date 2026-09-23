"""Pull and plot daily mean perpetual open interest in base-asset units."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
from pathlib import Path
import threading
import time
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import requests

OUT = Path("open_interest_figures/base_position")
START = pd.Timestamp("2021-01-01", tz="UTC")
END = pd.Timestamp("2026-01-01", tz="UTC")
LOCAL = threading.local()


def symbol_for(asset, margin):
    return f"{asset.upper()}USD_PERP" if margin == "cm" else f"{asset.upper()}USDT"


def metrics_url(asset, margin, day):
    symbol = symbol_for(asset, margin)
    return (f"https://data.binance.vision/data/futures/{margin}/daily/metrics/"
            f"{symbol}/{symbol}-metrics-{day}.zip")


def price_url(asset, day):
    symbol = f"{asset.upper()}USDT"
    return (f"https://data.binance.vision/data/spot/daily/klines/{symbol}/5m/"
            f"{symbol}-5m-{day}.zip")


def get_session():
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
    return LOCAL.session


def download_csv(url, columns, header=None):
    session = get_session()
    for attempt in range(3):
        try:
            response = session.get(url, timeout=30)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                names = [name for name in archive.namelist() if name.endswith(".csv")]
                if len(names) != 1:
                    raise ValueError(f"Unexpected ZIP contents: {url}")
                with archive.open(names[0]) as handle:
                    return pd.read_csv(handle, usecols=columns, header=header)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def fetch_day(asset, margin, day):
    cache = OUT / "raw_metrics" / f"{asset}_{margin}" / f"{day}.csv.gz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        frame = pd.read_csv(cache, compression="gzip")
    else:
        frame = download_csv(metrics_url(asset, margin, day),
                             ["create_time", "sum_open_interest", "sum_open_interest_value"])
        if frame is None:
            return None
        frame.to_csv(cache, index=False, compression="gzip")
    frame["create_time"] = pd.to_datetime(frame.create_time, utc=True, errors="raise").astype("datetime64[ns, UTC]")
    for column in ["sum_open_interest", "sum_open_interest_value"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if margin == "cm":
        frame["base_position"] = frame.sum_open_interest_value
    else:
        prices = download_csv(price_url(asset, day), [0, 4], header=None)
        if prices is None:
            return None
        prices.columns = ["open_time", "close"]
        prices["price_time"] = pd.to_datetime(prices.open_time, unit="ms", utc=True).astype("datetime64[ns, UTC]")
        prices["close"] = pd.to_numeric(prices.close, errors="raise")
        frame = pd.merge_asof(frame.sort_values("create_time"),
                              prices[["price_time", "close"]].sort_values("price_time"),
                              left_on="create_time", right_on="price_time", direction="backward")
        frame["base_position"] = frame.sum_open_interest_value / frame.close
    if not np.isfinite(frame.base_position).all() or not frame.base_position.ge(0).all():
        raise ValueError(f"Invalid base position: {asset}_{margin} {day}")
    frame["market"] = f"{asset}_{margin}"
    frame["date"] = frame.create_time.dt.floor("D").dt.tz_localize(None)
    return frame[["create_time", "date", "market", "base_position"]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    days = [str(day.date()) for day in pd.date_range(START, END - pd.Timedelta(days=1), freq="D")]
    tasks = [(asset, margin, day) for asset in ["btc", "eth"]
             for margin in ["um", "cm"] for day in days]
    frames = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch_day, *task): task for task in tasks}
        for count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result is not None:
                frames.append(result)
            if count % 100 == 0 or count == len(tasks):
                print(f"Processed {count}/{len(tasks)} metric days", flush=True)
    snapshots = pd.concat(frames, ignore_index=True).sort_values(["market", "create_time"])
    snapshots.to_csv(OUT / "open_interest_base_position_5min_2021_2025.csv", index=False)
    daily = (snapshots.groupby(["market", "date"], as_index=False)
             .base_position.agg(daily_mean="mean", observations="size"))
    daily.to_csv(OUT / "open_interest_base_position_daily_2021_2025.csv", index=False)
    draw(daily)
    daily.groupby("market").agg(first_date=("date", "min"), last_date=("date", "max"),
                                 days=("date", "nunique")).to_csv(OUT / "coverage.csv")
    print(f"Saved base-position snapshots, daily means, and figures to {OUT}", flush=True)


def draw(daily):
    grid = pd.date_range("2021-01-01", "2025-12-31", freq="D")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=False)
    colors = {"um": "tab:blue", "cm": "tab:orange"}
    for ax, (asset, margin) in zip(axes.flat, [("btc", "um"), ("btc", "cm"),
                                                ("eth", "um"), ("eth", "cm")]):
        data = daily.loc[daily.market.eq(f"{asset}_{margin}")].set_index("date")
        data = data.reindex(grid)
        ax.plot(data.index, data.daily_mean, color=colors[margin], linewidth=.8)
        ax.set_title(f"{asset.upper()} {'Linear' if margin == 'um' else 'Inverse'}")
        ax.set_ylabel(f"Daily mean open interest ({asset.upper()})")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
        ax.set_xlim(grid[0], grid[-1])
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Perpetual Futures Open Interest")
    fig.supxlabel("Date (UTC)")
    fig.tight_layout(rect=(0, 0, 1, .96))
    for extension in ["png", "pdf"]:
        fig.savefig(OUT / f"open_interest_base_position_2x2.{extension}", dpi=250,
                    bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()