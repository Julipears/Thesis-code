"""Download and plot open-interest positions in matching base-asset units.

For linear (USD-M) contracts the plotted field is ``sum_open_interest``.  For
inverse (COIN-M) contracts it is ``sum_open_interest_value``.  Both are kept
in each raw archive and in the daily output so the definition can be changed
without another data pull.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
from pathlib import Path
import threading
import time
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

OUT = Path("open_interest_figures/native")
START = pd.Timestamp("2021-01-01", tz="UTC")
END = pd.Timestamp("2026-01-01", tz="UTC")
LOCAL = threading.local()


def symbol_for(asset, margin):
    return f"{asset.upper()}USD_PERP" if margin == "cm" else f"{asset.upper()}USDT"


def metrics_url(asset, margin, day):
    symbol = symbol_for(asset, margin)
    return (f"https://data.binance.vision/data/futures/{margin}/daily/metrics/"
            f"{symbol}/{symbol}-metrics-{day}.zip")


def session():
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
    return LOCAL.session


def source_days(asset, margin):
    """Use dates already represented in the thesis metric caches.

    This preserves their historical coverage (e.g. inverse markets begin when
    Binance began publishing those archives) and avoids thousands of known 404s.
    """
    path = Path(f"sa_{asset}_{margin}/metrics_cache/{symbol_for(asset, margin)}_{margin}_liquidity_metrics.csv.gz")
    frame = pd.read_csv(path, usecols=["create_time"])
    dates = pd.to_datetime(frame.create_time, utc=True, errors="raise")
    dates = dates.loc[dates.ge(START) & dates.lt(END)]
    return sorted(dates.dt.strftime("%Y-%m-%d").unique())


def fetch_day(asset, margin, day):
    market = f"{asset}_{margin}"
    cache = OUT / "raw_metrics" / market / f"{day}.csv.gz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    url = metrics_url(asset, margin, day)
    frame = None
    if cache.exists():
        candidate = pd.read_csv(cache, compression="gzip")
        if {"create_time", "sum_open_interest", "sum_open_interest_value"}.issubset(candidate.columns):
            frame = candidate
    if frame is None:
        for attempt in range(3):
            try:
                response = session().get(url, timeout=30)
                if response.status_code == 404:
                    return None, dict(market=market, date=day, status="unavailable_404", url=url)
                response.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                    names = [name for name in archive.namelist() if name.endswith(".csv")]
                    if len(names) != 1:
                        raise ValueError(f"Unexpected ZIP contents: {url}")
                    with archive.open(names[0]) as handle:
                        frame = pd.read_csv(handle, usecols=["create_time", "sum_open_interest", "sum_open_interest_value"])
                frame.to_csv(cache, index=False, compression="gzip")
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
    frame["create_time"] = pd.to_datetime(frame.create_time, utc=True, errors="raise")
    frame["sum_open_interest"] = pd.to_numeric(frame.sum_open_interest, errors="raise")
    frame["sum_open_interest_value"] = pd.to_numeric(frame.sum_open_interest_value, errors="raise")
    if not frame.create_time.dt.strftime("%Y-%m-%d").eq(day).all():
        raise ValueError(f"Unexpected timestamps: {market} {day}")
    # Some early Binance archives contain each five-minute observation twice.
    # They are exact duplicates, so retain one observation per timestamp.
    frame = frame.drop_duplicates(subset=["create_time"], keep="last")
    values = frame[["sum_open_interest", "sum_open_interest_value"]]
    if not np.isfinite(values).all().all() or not values.ge(0).all().all():
        raise ValueError(f"Invalid native OI values: {market} {day}")
    selected = frame.sum_open_interest if margin == "um" else frame.sum_open_interest_value
    return dict(market=market, date=day, status="ok", observations=len(frame),
                sum_open_interest_daily_mean=float(frame.sum_open_interest.mean()),
                sum_open_interest_value_daily_mean=float(frame.sum_open_interest_value.mean()),
                daily_mean=float(selected.mean()), daily_min=float(selected.min()),
                daily_max=float(selected.max()), selected_field=("sum_open_interest" if margin == "um" else "sum_open_interest_value"),
                unit=f"{asset.upper()}",
                url=url), None


def draw(daily):
    grid = pd.date_range("2021-01-01", "2025-12-31", freq="D")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=False)
    order = [("btc", "um"), ("btc", "cm"), ("eth", "um"), ("eth", "cm")]
    colors = {"um": "tab:blue", "cm": "tab:orange"}
    for ax, (asset, margin) in zip(axes.flat, order):
        market = f"{asset}_{margin}"
        series = (daily.loc[daily.market.eq(market)]
                  .set_index("date").daily_mean.reindex(grid))
        ax.plot(series.index, series, color=colors[margin], linewidth=.8)
        contract = "Linear" if margin == "um" else "Inverse"
        unit = asset.upper()
        ax.set_title(f"{asset.upper()} {contract}")
        ax.set_ylabel(f"Daily mean OI ({unit})")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
        ax.set_xlim(grid[0], grid[-1])
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Perpetual Futures Open Interest (base-asset positions)")
    fig.supxlabel("Date (UTC)", y=.045)
    fig.text(.5, .012, "Daily arithmetic means of available 5-minute observations; missing days are not filled.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .96))
    for extension in ["png", "pdf"]:
        fig.savefig(OUT / f"open_interest_native_2x2.{extension}", dpi=250, bbox_inches="tight")
        fig.savefig(OUT / f"open_interest_positions_2x2.{extension}", dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [(asset, margin, day)
             for asset in ["btc", "eth"] for margin in ["um", "cm"]
             for day in source_days(asset, margin)]
    print(f"Retrieving native open-interest archives for {len(tasks)} contract-days", flush=True)
    rows, statuses = [], []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(fetch_day, *task): task for task in tasks}
        for count, future in enumerate(as_completed(futures), 1):
            asset, margin, day = futures[future]
            try:
                record, status = future.result()
                if record is not None:
                    rows.append(record)
                    statuses.append(record)
                elif status is not None:
                    statuses.append(status)
            except Exception as exc:
                statuses.append(dict(market=f"{asset}_{margin}", date=day,
                                     status="error", error=str(exc), url=metrics_url(asset, margin, day)))
            if count % 200 == 0 or count == len(tasks):
                pd.DataFrame(statuses).to_csv(OUT / "retrieval_status.csv", index=False)
                print(f"Processed {count}/{len(tasks)}; successful={sum(r.get('status') == 'ok' for r in statuses)}; errors={sum(r.get('status') == 'error' for r in statuses)}", flush=True)
    daily = pd.DataFrame(rows)
    daily["date"] = pd.to_datetime(daily.date)
    daily = daily.sort_values(["market", "date"])
    if daily.duplicated(["market", "date"]).any():
        raise ValueError("Duplicate market-day observations")
    daily.to_csv(OUT / "daily_open_interest_base_units.csv", index=False)
    coverage = daily.groupby("market").agg(first_date=("date", "min"), last_date=("date", "max"), days=("date", "size"))
    coverage.to_csv(OUT / "coverage.csv")
    draw(daily)
    (OUT / "README.md").write_text(
        "# Open-interest positions in base-asset units\n\n"
        "The plotted definition uses `sum_open_interest` for USD-M linear markets and "
        "`sum_open_interest_value` for COIN-M inverse markets. Both source fields are retained "
        "in every raw archive and their daily means are saved in `daily_open_interest_base_units.csv`. "
        "This puts BTC panels in BTC and ETH panels in ETH. Missing dates are gaps, not zeroes. "
        "Dates follow the coverage of the thesis metric caches. Source URLs and retrieval statuses are in `retrieval_status.csv`.\n", encoding="utf-8")
    print(coverage.to_string(), flush=True)
    print(f"Saved open-interest positions plot to {OUT / 'open_interest_native_2x2.png'}", flush=True)


if __name__ == "__main__":
    main()
