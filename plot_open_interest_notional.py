"""Comparable daily notional OI; preserve legacy model caches unchanged."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
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

OUT = Path("open_interest_figures/notional")
START = pd.Timestamp("2021-01-01", tz="UTC")
END = pd.Timestamp("2026-01-01", tz="UTC")
LOCAL = threading.local()


def cache_path(asset, margin):
    symbol = f"{asset.upper()}USD_PERP" if margin == "cm" else f"{asset.upper()}USDT"
    return Path(f"sa_{asset}_{margin}/metrics_cache/{symbol}_{margin}_liquidity_metrics.csv.gz")


def read_legacy(asset, margin):
    frame = pd.read_csv(cache_path(asset, margin), usecols=["create_time", "open_interest"])
    frame["create_time"] = pd.to_datetime(frame.create_time, utc=True, errors="raise")
    frame = frame.loc[frame.create_time.ge(START) & frame.create_time.lt(END)]
    if frame.create_time.duplicated().any():
        raise ValueError(f"Duplicate cached timestamps: {asset} {margin}")
    return frame


def fetch_counts(asset, day):
    market = f"{asset}_cm"
    symbol = f"{asset.upper()}USD_PERP"
    cache = OUT / "raw_inverse_metrics" / market / f"{day}.csv.gz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://data.binance.vision/data/futures/cm/daily/metrics/{symbol}/{symbol}-metrics-{day}.zip"
    if cache.exists():
        frame = pd.read_csv(cache)
    else:
        if not hasattr(LOCAL, "session"):
            LOCAL.session = requests.Session()
        for attempt in range(3):
            try:
                response = LOCAL.session.get(url, timeout=25)
                if response.status_code == 404:
                    return None, dict(market=market, date=day, status="unavailable_404", url=url)
                response.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                    names = [name for name in archive.namelist() if name.endswith(".csv")]
                    if len(names) != 1:
                        raise ValueError("Expected one CSV in metrics archive")
                    with archive.open(names[0]) as handle:
                        frame = pd.read_csv(handle, usecols=["create_time", "sum_open_interest", "sum_open_interest_value"])
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        frame.to_csv(cache, index=False, compression="gzip")
    frame["create_time"] = pd.to_datetime(frame.create_time, utc=True, errors="raise")
    for col in ["sum_open_interest", "sum_open_interest_value"]:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    if frame.create_time.duplicated().any() or not frame.create_time.dt.strftime("%Y-%m-%d").eq(day).all():
        raise ValueError(f"Unexpected timestamps: {market} {day}")
    quantities = frame[["sum_open_interest", "sum_open_interest_value"]]
    if not np.isfinite(quantities).all().all() or not quantities.ge(0).all().all():
        raise ValueError(f"Invalid OI values: {market} {day}")
    face = 100 if asset == "btc" else 10
    return frame, dict(market=market, date=day, status="ok", observations=len(frame),
        native_contracts_mean=frame.sum_open_interest.mean(),
        notional_daily_mean=frame.sum_open_interest.mean()*face,
        face_value_usd=face, unit="USD", url=url)


def draw(daily):
    grid = pd.date_range("2021-01-01", "2025-12-31", freq="D")
    maximum = daily.notional_daily_mean.max()/1e9*1.08
    def panel(ax, asset):
        for margin, label, color in [("um", "Linear (USDT)", "tab:blue"), ("cm", "Inverse (USD)", "tab:orange")]:
            data = daily.loc[daily.market.eq(f"{asset}_{margin}")].set_index("date").notional_daily_mean.reindex(grid)
            ax.plot(data.index, data/1e9, color=color, linewidth=1, label=label)
        ax.set_title(asset.upper())
        ax.set_ylabel("Daily mean notional OI (USD-equivalent billions)")
        ax.set_ylim(0, maximum)
        ax.set_xlim(grid[0], grid[-1])
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    note = "Daily means of available 5-minute observations; gaps are not filled. USDT valued at USD 1."
    for asset in ["btc", "eth"]:
        fig, ax = plt.subplots(figsize=(12, 5))
        panel(ax, asset)
        ax.set_title(f"{asset.upper()} perpetual open interest: linear versus inverse")
        ax.set_xlabel("Date (UTC)")
        fig.text(.5, .01, note, ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, 1))
        for ext in ["png", "pdf"]:
            fig.savefig(OUT / f"{asset}_open_interest_notional.{ext}", dpi=220, bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
    for ax, asset in zip(axes, ["btc", "eth"]):
        panel(ax, asset)
    axes[-1].set_xlabel("Date (UTC)")
    fig.suptitle("Perpetual open interest in comparable notional units")
    fig.text(.5, .01, note, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .96))
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"btc_eth_open_interest_notional.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(sample=False):
    OUT.mkdir(parents=True, exist_ok=True)
    if sample:
        for asset in ["btc", "eth"]:
            _, record = fetch_counts(asset, "2023-07-01")
            print(record, flush=True)
        return
    rows, tasks = [], []
    for asset in ["btc", "eth"]:
        linear = read_legacy(asset, "um")
        linear["date"] = linear.create_time.dt.floor("D")
        values = linear.groupby("date").open_interest.agg(["mean", "count"])
        for day, row in values.iterrows():
            rows.append(dict(market=f"{asset}_um", date=day.strftime("%Y-%m-%d"), status="ok",
                notional_daily_mean=row["mean"], observations=int(row["count"]), unit="USDT"))
        inverse = read_legacy(asset, "cm")
        tasks.extend((asset, day) for day in sorted(inverse.create_time.dt.strftime("%Y-%m-%d").unique()))
    print(f"Retrieving native inverse OI for {len(tasks)} previously available contract-days", flush=True)
    statuses = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch_counts, asset, day): (asset, day) for asset, day in tasks}
        for count, future in enumerate(as_completed(futures), 1):
            asset, day = futures[future]
            try:
                _, record = future.result()
                statuses.append(record)
                if record["status"] == "ok":
                    rows.append(record)
            except Exception as exc:
                statuses.append(dict(market=f"{asset}_cm", date=day, status="error", error=str(exc)))
            if count % 100 == 0 or count == len(tasks):
                print(f"{count}/{len(tasks)} inverse contract-days; errors={sum(r['status'] != 'ok' for r in statuses)}", flush=True)
                pd.DataFrame(statuses).to_csv(OUT / "retrieval_status.csv", index=False)
    daily = pd.DataFrame(rows)
    daily["date"] = pd.to_datetime(daily.date)
    daily = daily.sort_values(["market", "date"])
    assert not daily.duplicated(["market", "date"]).any()
    assert np.isfinite(daily.notional_daily_mean).all() and daily.notional_daily_mean.ge(0).all()
    daily.to_csv(OUT / "daily_notional_open_interest.csv", index=False)
    coverage = daily.groupby("market").agg(first_date=("date", "min"), last_date=("date", "max"), days=("date", "size"))
    coverage.to_csv(OUT / "coverage.csv")
    draw(daily)
    (OUT / "README.md").write_text(
        "# Notional open-interest figures\n\nDaily arithmetic means of available five-minute OI snapshots, 2021--2025. "
        "Linear: saved sum_open_interest_value (USDT). Inverse: newly retrieved sum_open_interest (contracts) "
        "times USD 100 for BTCUSD_PERP or USD 10 for ETHUSD_PERP. "
        "Sources: https://www.binance.com/en/blog/futures/421499824684901012 and Binance daily metrics archives "
        "(source URLs in retrieval_status.csv). USDT is treated as USD 1 for descriptive comparison; no depeg adjustment. "
        "All plots use the same y-axis range. Missing dates are gaps, not zeroes. Raw source observations are retained "
        "under raw_inverse_metrics. The requested inverse dates match available dates in the original caches; "
        "this run does not fill their historical coverage gaps. See coverage.csv and retrieval_status.csv. "
        "Legacy covariate caches and fitted models are unchanged.\n", encoding="utf-8")
    print(coverage.to_string(), flush=True)
    print(f"Saved figures to {OUT}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    main(parser.parse_args().sample)
