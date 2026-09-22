"""Download daily perpetual volumes from monthly Binance kline archives and plot."""
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

OUT = Path("daily_perpetual_volume_figures")
LOCAL = threading.local()


def fetch(market, month):
    asset, margin = market.split("_")
    symbol = f"{asset.upper()}USD_PERP" if margin == "cm" else f"{asset.upper()}USDT"
    name = f"{symbol}-1d-{month}.zip"
    path = OUT / "raw_monthly_klines" / market / name
    path.parent.mkdir(parents=True, exist_ok=True)
    cadence = "daily" if len(month) == 10 else "monthly"
    url = f"https://data.binance.vision/data/futures/{margin}/{cadence}/klines/{symbol}/1d/{name}"
    if not path.exists():
        if not hasattr(LOCAL, "session"):
            LOCAL.session = requests.Session()
        for attempt in range(3):
            try:
                response = LOCAL.session.get(url, timeout=30)
                response.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                    if archive.testzip() is not None:
                        raise ValueError("Corrupt archive")
                path.write_bytes(response.content)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt+1)
    with zipfile.ZipFile(path) as archive:
        names = [x for x in archive.namelist() if x.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"Unexpected ZIP contents: {path}")
        with archive.open(names[0]) as handle:
            raw = pd.read_csv(handle, header=None, dtype=str)
    # Futures archive headers vary across publication dates.
    raw = raw.loc[pd.to_numeric(raw[0], errors="coerce").notna()].copy()
    if raw.shape[1] != 12:
        raise ValueError(f"Unexpected kline schema: {path}")
    raw = raw.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(raw.to_numpy()).all():
        raise ValueError(f"Nonfinite kline data: {path}")
    times = raw[0].to_numpy()
    date = pd.to_datetime(np.where(times > 1e14, times/1000, times), unit="ms", utc=True).tz_localize(None)
    frame = pd.DataFrame(dict(date=date, market=market, symbol=symbol,
        base_volume=(raw[7] if margin == "cm" else raw[5]).to_numpy(),
        native_volume=raw[5].to_numpy(),
        notional_volume=(raw[5]*(100 if asset == "btc" else 10) if margin == "cm" else raw[7]).to_numpy(),
        base_unit=asset.upper(), native_unit="contracts" if margin == "cm" else asset.upper(),
        notional_unit="USD" if margin == "cm" else "USDT", source_url=url))
    expected = (pd.DatetimeIndex([pd.Timestamp(month)]) if cadence == "daily" else
                pd.date_range(f"{month}-01", periods=pd.Period(month).days_in_month, freq="D"))
    missing = expected.difference(pd.DatetimeIndex(frame.date))
    if len(missing) and cadence == "monthly":
        print(f"Recovering {market} missing daily bars: {list(missing.strftime('%Y-%m-%d'))}", flush=True)
        frame = pd.concat([frame, *[fetch(market, day.strftime("%Y-%m-%d")) for day in missing]], ignore_index=True).sort_values("date")
    if not pd.DatetimeIndex(frame.date).equals(expected):
        raise ValueError(f"Missing, duplicate, or out-of-order days in {path}")
    if not frame[["base_volume", "native_volume", "notional_volume"]].ge(0).all().all():
        raise ValueError(f"Negative volume in {path}")
    return frame


def draw(data, grid_only=False):
    def panel(ax, asset, quantity=False):
        for margin, name, color in [("um", "Linear", "tab:blue"), ("cm", "Inverse", "tab:orange")]:
            frame = data.loc[data.market.eq(f"{asset}_{margin}")]
            column, scale = ("base_volume", 1) if quantity else ("notional_volume", 1e9)
            ax.plot(frame.date, frame[column]/scale, color=color, linewidth=.8, label=name)
        ax.set_title(asset.upper())
        ax.set_ylabel(f"Daily traded volume ({asset.upper()})" if quantity else "Daily traded volume (USD-equivalent billions)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_xlim(pd.Timestamp("2021-01-01"), pd.Timestamp("2026-01-01"))
        ax.legend(frameon=False)
    for quantity in [False, True]:
        kind = "base_quantity" if quantity else "notional"
        for asset in ([] if grid_only else ["btc", "eth"]):
            fig, ax = plt.subplots(figsize=(12, 5))
            panel(ax, asset, quantity)
            ax.set_title(f"{asset.upper()} Perpetual Futures Volumes")
            ax.set_xlabel("Year")
            fig.tight_layout()
            for ext in ["png", "pdf"]:
                fig.savefig(OUT / f"{asset}_daily_volume_{kind}.{ext}", dpi=200, bbox_inches="tight")
            plt.close(fig)
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=not quantity)
        for ax, asset in zip(axes, ["btc", "eth"]):
            panel(ax, asset, quantity)
        fig.suptitle("Perpetual Futures Volumes")
        axes[-1].set_xlabel("Year")
        fig.tight_layout(rect=(0, 0, 1, .96))
        for ext in ([] if grid_only else ["png", "pdf"]):
            fig.savefig(OUT / f"all_contracts_daily_volume_{kind}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True,
                     sharey="row" if quantity else True)
        for row, asset in enumerate(["btc", "eth"]):
            for col, (margin, label, color) in enumerate([
                ("um", "Linear", "tab:blue"), ("cm", "Inverse", "tab:orange")
            ]):
                ax = axes[row, col]
                frame = data.loc[data.market.eq(f"{asset}_{margin}")]
                column, scale = ("base_volume", 1) if quantity else ("notional_volume", 1e9)
                ax.plot(frame.date, frame[column]/scale, color=color, linewidth=.8)
                ax.set_title(f"{asset.upper()} — {label}")
                ax.set_ylim(bottom=0)
                if col == 0:
                    ax.set_ylabel(f"Daily volume ({asset.upper()})" if quantity else "Daily volume (USD-equivalent billions)")
                if row == 1:
                    ax.set_xlabel("Year")
                ax.grid(alpha=.2)
                ax.xaxis.set_major_locator(mdates.YearLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
                ax.set_xlim(pd.Timestamp("2021-01-01"), pd.Timestamp("2026-01-01"))
        fig.suptitle("Perpetual Futures Volumes")
        fig.tight_layout(rect=(0, 0, 1, .96))
        for ext in ["png", "pdf"]:
            fig.savefig(OUT / f"all_contracts_daily_volume_{kind}_2x2.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=False)
        for row, asset in enumerate(["btc", "eth"]):
            for col, (margin, label, color) in enumerate([
                ("um", "Linear", "tab:blue"), ("cm", "Inverse", "tab:orange")
            ]):
                ax = axes[row, col]
                frame = data.loc[data.market.eq(f"{asset}_{margin}")]
                column, scale = ("base_volume", 1) if quantity else ("notional_volume", 1e9)
                ax.plot(frame.date, frame[column]/scale, color=color, linewidth=.8)
                ax.set_title(f"{asset.upper()} — {label}")
                ax.set_ylabel(f"Daily volume ({asset.upper()})" if quantity else
                              "Daily volume (USD-equivalent billions)")
                ax.set_ylim(bottom=0)
                if row == 1:
                    ax.set_xlabel("Year")
                ax.grid(alpha=.2)
                ax.xaxis.set_major_locator(mdates.YearLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
                ax.set_xlim(pd.Timestamp("2021-01-01"), pd.Timestamp("2026-01-01"))
        fig.suptitle("Perpetual Futures Volumes")
        fig.tight_layout(rect=(0, 0, 1, .96))
        for ext in ["png", "pdf"]:
            fig.savefig(OUT / f"all_contracts_daily_volume_{kind}_separate_yaxes_2x2.{ext}",
                        dpi=200, bbox_inches="tight")
        plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    tasks = [(market, str(month)) for market in ["btc_um", "btc_cm", "eth_um", "eth_cm"]
             for month in pd.period_range("2021-01", "2025-12", freq="M")]
    frames = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch, *task): task for task in tasks}
        for count, future in enumerate(as_completed(futures), 1):
            frames.append(future.result())
            if count % 30 == 0:
                print(f"Validated {count}/{len(tasks)} monthly archives", flush=True)
    data = pd.concat(frames, ignore_index=True).sort_values(["market", "date"])
    assert not data.duplicated(["market", "date"]).any()
    assert data.groupby("market").size().eq(1826).all()
    data.to_csv(OUT / "daily_perpetual_volumes_2021_2025.csv", index=False)
    data.groupby("market").agg(days=("date", "size"), first_date=("date", "min"), last_date=("date", "max")).to_csv(OUT / "coverage.csv")
    draw(data)
    (OUT / "README.md").write_text(
        "# Daily perpetual traded volumes\n\nAll four Binance perpetual contracts, 2021--2025. "
        "Daily UTC kline volumes from official monthly archives, with every day checked. "
        "No smoothing or daily averaging: each point is a daily traded-volume total. "
        "Linear notional is quote volume (USDT); inverse notional is contract volume times "
        "USD 100 for BTC or USD 10 for ETH. USD-equivalent plots assume USDT = USD 1. "
        "Base-quantity plots use linear Volume and inverse Base asset volume, in BTC or ETH. "
        "The combined notional figure shares a y-axis scale across assets. "
        "Separate-y-axis 2x2 figures are also saved for comparing each contract on its own scale. "
        "Raw archives and source URLs are retained. Old AFT caches and models are unchanged.\n\n"
        "Sources: https://github.com/binance/binance-public-data and "
        "https://www.binance.com/en/blog/futures/421499824684901012\n", encoding="utf-8")
    print(f"Saved 7,304 daily observations and figures to {OUT}", flush=True)


if __name__ == "__main__":
    main()
