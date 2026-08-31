"""Plot daily mean open interest for all four perpetual contracts, 2021-2025."""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


SERIES = {
    "btc_um": {
        "path": Path("sa_btc_um/metrics_cache/BTCUSDT_um_liquidity_metrics.csv.gz"),
        "title": "BTC linear (USDT-margined)",
        "color": "#1f77b4",
    },
    "btc_cm": {
        "path": Path("sa_btc_cm/metrics_cache/BTCUSD_PERP_cm_liquidity_metrics.csv.gz"),
        "title": "BTC inverse (coin-margined)",
        "color": "#ff7f0e",
    },
    "eth_um": {
        "path": Path("sa_eth_um/metrics_cache/ETHUSDT_um_liquidity_metrics.csv.gz"),
        "title": "ETH linear (USDT-margined)",
        "color": "#2ca02c",
    },
    "eth_cm": {
        "path": Path("sa_eth_cm/metrics_cache/ETHUSD_PERP_cm_liquidity_metrics.csv.gz"),
        "title": "ETH inverse (coin-margined)",
        "color": "#d62728",
    },
}
OUT = Path("open_interest_figures")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for market, config in SERIES.items():
        raw = pd.read_csv(config["path"], usecols=["create_time", "open_interest"])
        raw["create_time"] = pd.to_datetime(raw["create_time"], utc=True)
        raw = raw[
            raw["create_time"].between(
                pd.Timestamp("2021-01-01", tz="UTC"),
                pd.Timestamp("2025-12-31 23:59:59", tz="UTC"),
            )
        ].copy()
        raw["date"] = raw["create_time"].dt.floor("D").dt.tz_localize(None)
        daily = raw.groupby("date", as_index=False)["open_interest"].mean()
        daily["market"] = market
        frames.append(daily)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(OUT / "open_interest_daily_means_2021_2025.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.2), sharex=True)
    order = ["btc_um", "btc_cm", "eth_um", "eth_cm"]
    for ax, market in zip(axes.flat, order):
        config = SERIES[market]
        data = combined[combined["market"].eq(market)].set_index("date")
        # Reindex before plotting so genuinely unavailable archive days appear
        # as visible breaks instead of straight interpolation-like segments.
        data = data.reindex(pd.date_range("2021-01-01", "2025-12-31", freq="D"))
        ax.plot(
            data.index, data["open_interest"],
            color=config["color"], linewidth=0.8,
        )
        ax.set_title(config["title"])
        ax.set_ylabel("Daily mean open interest")
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useMathText=True)
        ax.set_xlim(pd.Timestamp("2021-01-01"), pd.Timestamp("2025-12-31"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle("Perpetual-futures open interest, 2021–2025", y=0.995)
    fig.supxlabel("Date (UTC)", y=0.045)
    fig.text(
        0.5, 0.012,
        "Daily means of available five-minute observations; source-reported units differ by contract type.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.97))
    fig.savefig(OUT / "open_interest_daily_2021_2025.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "open_interest_daily_2021_2025.pdf", bbox_inches="tight")
    plt.close(fig)

    coverage = combined.groupby("market").agg(
        first_date=("date", "min"), last_date=("date", "max"), days=("date", "nunique")
    )
    coverage.to_csv(OUT / "open_interest_coverage.csv")
    print(coverage.to_string())


if __name__ == "__main__":
    main()
