"""Generate the active VECM coefficient plots and regime heatmaps.

This module consolidates the former alpha-over-time, separate-alpha, and
April-2023 regime-heatmap scripts.  Running it generates all three figure
families and their summary CSV files.
"""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns


ALPHA_OUT = Path("vecm_hasbrouck3_alpha_plots")
HEATMAP_OUT = Path("vecm_regime_heatmaps_april2023")
LATENCIES = ["10ms", "50ms", "100ms", "200ms", "500ms", "1s"]
MONTHS = pd.date_range("2021-01-01", "2025-12-01", freq="MS")

CONFIGS = {
    "btc_um": (Path("vecm_hasbrouck3_btc_um"), "hasbrouck3_btc_alignedv2", "um", "BTC linear"),
    "btc_cm": (Path("vecm_hasbrouck3_btc_cm"), "hasbrouck3_btc_alignedv2", "cm", "BTC inverse"),
    "eth_um": (Path("vecm_hasbrouck3_eth_um"), "hasbrouck3_eth_alignedv2", "um", "ETH linear"),
    "eth_cm": (Path("vecm_hasbrouck3_eth_cm"), "hasbrouck3_eth_alignedv2", "cm", "ETH inverse"),
}
FILE_LABELS = {
    "btc_um": "btc_linear",
    "btc_cm": "btc_inverse",
    "eth_um": "eth_linear",
    "eth_cm": "eth_inverse",
}
REGIMES = [
    ("2021", "2021-01-01", "2022-01-01"),
    ("Jan-Jun 2022", "2022-01-01", "2022-07-01"),
    ("Jul 2022-Mar 2023", "2022-07-01", "2023-04-01"),
    ("Apr-Dec 2023", "2023-04-01", "2024-01-01"),
    ("2024", "2024-01-01", "2025-01-01"),
    ("2025", "2025-01-01", "2026-01-01"),
]


def _load_alpha_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and summarize saved hourly alpha coefficients."""
    summaries, coverage = [], []
    for asset in ["btc", "eth"]:
        for margin in ["um", "cm"]:
            market = f"{asset}_{margin}"
            folder = Path(f"vecm_hasbrouck3_{market}")
            for latency in LATENCIES:
                pattern = re.compile(
                    rf"hasbrouck3_{asset}_alignedv2_1h_{latency}_10_{margin}_results_"
                    rf"\d{{8}}_\d{{8}}\.csv$"
                )
                paths = sorted(p for p in folder.glob("*results*.csv") if pattern.fullmatch(p.name))
                frames = [pd.read_csv(p, usecols=["interval", "series", "alpha"]) for p in paths]
                if not frames:
                    raise RuntimeError(f"No data for {market} {latency}")
                data = pd.concat(frames, ignore_index=True)
                data["interval"] = pd.to_datetime(data["interval"], errors="coerce")
                data["alpha"] = pd.to_numeric(data["alpha"], errors="coerce")
                data = data.loc[data["interval"].ge("2021-01-01") & data["interval"].lt("2026-01-01")]
                keys = ["interval", "series"]
                stats = data.groupby(keys)["alpha"].agg(["min", "max", "count", "size"])
                scale = stats[["min", "max"]].abs().max(axis=1)
                valid = (stats["max"] - stats["min"] <= 1e-10 + 1e-6 * scale)
                valid &= stats["count"].eq(stats["size"])
                valid &= np.isfinite(stats["min"]) & np.isfinite(stats["max"])
                good_keys = stats.index[valid]
                data = (
                    data.set_index(keys)
                    .loc[lambda frame: frame.index.isin(good_keys)]
                    .reset_index()
                    .drop_duplicates(keys)
                )
                data["month"] = data["interval"].dt.to_period("M").dt.to_timestamp()
                monthly = data.groupby(["month", "series"])["alpha"].agg(
                    median_alpha="median", mean_alpha="mean", hourly_models="size"
                ).reset_index()
                monthly["market"] = market
                monthly["latency"] = latency
                summaries.append(monthly)
                coverage.append({
                    "market": market,
                    "latency": latency,
                    "files": len(paths),
                    "valid_hourly_rows": len(data),
                    "excluded_ambiguous_or_invalid_keys": int((~valid).sum()),
                })
            print(f"Loaded {market}", flush=True)
    return pd.concat(summaries, ignore_index=True), pd.DataFrame(coverage)


def generate_alpha_over_time() -> None:
    """Generate the BTC and ETH six-panel monthly-alpha figures."""
    ALPHA_OUT.mkdir(parents=True, exist_ok=True)
    summary, coverage = _load_alpha_summary()
    summary.to_csv(ALPHA_OUT / "monthly_alpha_summary.csv", index=False)
    coverage.to_csv(ALPHA_OUT / "input_coverage.csv", index=False)

    for asset in ["btc", "eth"]:
        fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True, constrained_layout=True)
        for ax, latency in zip(axes.flat, LATENCIES):
            for margin, color in [("um", "tab:blue"), ("cm", "tab:orange")]:
                for series, style in [("log_midpoint_spot", "-"), ("log_midpoint_perp", "--")]:
                    values = summary.loc[
                        summary["market"].eq(f"{asset}_{margin}")
                        & summary["latency"].eq(latency)
                        & summary["series"].eq(series)
                    ].set_index("month")["median_alpha"].reindex(MONTHS)
                    if not values.notna().any():
                        raise RuntimeError(f"No alpha summary for {asset} {margin} {latency} {series}")
                    ax.plot(values.index, values.values, color=color, linestyle=style, linewidth=1.7)
            ax.axhline(0, color="0.45", linewidth=0.7)
            ax.set_title(f"Sampling interval: {latency}")
            ax.set_ylabel(r"Adjustment coefficient $\alpha$")
            ax.grid(alpha=0.18)
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.set_xlim(pd.Timestamp("2021-01-01"), pd.Timestamp("2026-01-01"))
        handles = [
            Line2D([0], [0], color=color, linestyle=style, linewidth=1.8, label=f"{contract} - {series}")
            for contract, color in [("Linear", "tab:blue"), ("Inverse", "tab:orange")]
            for series, style in [("Spot", "-"), ("Perpetual", "--")]
        ]
        fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
        fig.suptitle(
            f"{asset.upper()}: error-correction coefficients over time\n"
            "Monthly median of hourly models; basis = log spot - log perpetual",
            fontsize=14,
        )
        fig.savefig(ALPHA_OUT / f"{asset}_alpha_over_time.png", dpi=200, bbox_inches="tight")
        fig.savefig(ALPHA_OUT / f"{asset}_alpha_over_time.pdf", bbox_inches="tight")
        plt.close(fig)

    (ALPHA_OUT / "README.md").write_text(
        "# Alpha over time\n\nMonthly medians of saved hourly alpha estimates, taken directly over "
        "hourly models (no daily averaging). Separate panels retain each sampling interval's raw "
        "coefficient units; y-axis scales vary by panel. Source: alignedv2 results with lag base 10 "
        "in the four vecm_hasbrouck3 folders. Blue: linear; orange: inverse. Solid: spot; dashed: "
        "perpetual. Basis orientation is log spot minus log perpetual.\n\n"
        "Duplicate hourly keys with conflicting alpha values are excluded; consistent duplicates "
        "are counted once. input_coverage.csv records exclusions, and monthly_alpha_summary.csv "
        "contains medians, means and observation counts.\n",
        encoding="utf-8",
    )


def generate_separate_alpha_plots() -> None:
    """Generate one figure for each asset and requested latency."""
    summary_path = ALPHA_OUT / "monthly_alpha_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Run generate_alpha_over_time first; missing {summary_path}")
    out = ALPHA_OUT / "separate"
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(summary_path, parse_dates=["month"])
    for asset in ["btc", "eth"]:
        for latency in ["10ms", "100ms", "1s"]:
            fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
            for margin, contract, color in [
                ("um", "Linear", "tab:blue"), ("cm", "Inverse", "tab:orange")
            ]:
                for series, label, style in [
                    ("log_midpoint_spot", "Spot", "-"),
                    ("log_midpoint_perp", "Perpetual", "--"),
                ]:
                    selected = summary.loc[
                        summary["market"].eq(f"{asset}_{margin}")
                        & summary["latency"].eq(latency)
                        & summary["series"].eq(series)
                    ]
                    values = selected.set_index("month")["median_alpha"].reindex(MONTHS)
                    if not values.notna().any():
                        raise RuntimeError(f"No alpha summary for {asset} {margin} {latency} {series}")
                    ax.plot(values.index, values, color=color, linestyle=style, linewidth=1.8,
                            label=f"{contract} - {label}")
            ax.axhline(0, color="0.45", linewidth=0.7)
            ax.set_title(f"{asset.upper()} adjustment coefficients - {latency}\nMonthly median of hourly estimates", fontsize=13)
            ax.set_ylabel(r"Adjustment coefficient $\alpha$")
            ax.set_xlabel("Year")
            ax.grid(alpha=0.18)
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.set_xlim(pd.Timestamp("2021-01-01"), pd.Timestamp("2026-01-01"))
            ax.legend(loc="best", ncol=2, framealpha=0.9)
            for extension in ["png", "pdf"]:
                fig.savefig(out / f"{asset}_alpha_{latency}.{extension}", dpi=200, bbox_inches="tight")
            plt.close(fig)
    print(f"Saved six separate graphs as PNG and PDF to {out}")


def load_market(market: str) -> pd.DataFrame:
    """Load saved hourly ILS estimates for one market."""
    folder, prefix, contract, title = CONFIGS[market]
    pattern = re.compile(
        rf"^{re.escape(prefix)}_1h_({'|'.join(map(re.escape, LATENCIES))})_10_"
        rf"{contract}_results_\d{{8}}_\d{{8}}\.csv$"
    )
    frames = []
    for path in folder.iterdir():
        match = pattern.match(path.name)
        if not match:
            continue
        frame = pd.read_csv(path, index_col=0)
        frame["latency"] = match.group(1)
        frames.append(frame)
    if not frames:
        raise RuntimeError(f"No hourly VECM result files found for {market}")
    data = pd.concat(frames, ignore_index=True, sort=False)
    data["interval"] = pd.to_datetime(data["interval"], errors="coerce")
    data = data.dropna(subset=["interval", "ILS_mid"])
    data = data[data["interval"].ge("2021-01-01") & data["interval"].lt("2026-01-01")]
    data = data.drop_duplicates(["interval", "series", "latency"], keep="last")
    data["market"] = market
    data["contract_title"] = title
    return data


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    spot = data[data["series"].eq("log_midpoint_spot")]
    rows = []
    for market, market_rows in spot.groupby("market"):
        title = CONFIGS[market][3]
        for regime, start, end in REGIMES:
            selected = market_rows[market_rows["interval"].ge(start) & market_rows["interval"].lt(end)]
            for latency in LATENCIES:
                values = pd.to_numeric(
                    selected.loc[selected["latency"].eq(latency), "ILS_mid"], errors="coerce"
                ).dropna()
                rows.append({
                    "market": market, "contract": title, "regime": regime,
                    "regime_start": start, "regime_end_exclusive": end, "latency": latency,
                    "median_spot_ILS": values.median() if len(values) else np.nan,
                    "mean_spot_ILS": values.mean() if len(values) else np.nan,
                    "model_intervals": len(values),
                })
    return pd.DataFrame(rows)


def summarize_perspectives(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize complementary spot and perpetual ILS perspectives."""
    rows = []
    for market, market_rows in data.groupby("market"):
        title = CONFIGS[market][3]
        for series in ("log_midpoint_spot", "log_midpoint_perp"):
            series_rows = market_rows[market_rows["series"].eq(series)]
            for regime, start, end in REGIMES:
                selected = series_rows[series_rows["interval"].ge(start) & series_rows["interval"].lt(end)]
                for latency in LATENCIES:
                    values = pd.to_numeric(
                        selected.loc[selected["latency"].eq(latency), "ILS_mid"], errors="coerce"
                    ).dropna()
                    rows.append({
                        "market": market, "contract": title, "perspective": series,
                        "series": series, "regime": regime, "regime_start": start,
                        "regime_end_exclusive": end, "latency": latency,
                        "median_ILS": values.median() if len(values) else np.nan,
                        "mean_ILS": values.mean() if len(values) else np.nan,
                        "model_intervals": len(values),
                    })
    return pd.DataFrame(rows)


def table_for(summary: pd.DataFrame, market: str) -> pd.DataFrame:
    return summary[summary["market"].eq(market)].pivot(
        index="latency", columns="regime", values="median_spot_ILS"
    ).reindex(index=LATENCIES, columns=[row[0] for row in REGIMES])


def draw(summary: pd.DataFrame, markets: list[str], filename: str, title: str) -> None:
    tables = [table_for(summary, market) for market in markets]
    finite = np.concatenate([table.to_numpy()[np.isfinite(table.to_numpy())] for table in tables])
    deviation = max(float(np.max(np.abs(finite - 0.5))), 1e-6)
    vmin, vmax = 0.5 - deviation, 0.5 + deviation
    if len(markets) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(17, 9), constrained_layout=True)
        axes = list(axes.flat)
    else:
        fig, axes = plt.subplots(1, len(markets), figsize=(17, 5), constrained_layout=True)
        axes = list(np.atleast_1d(axes).flat)
    for ax, market, table in zip(axes, markets, tables):
        sns.heatmap(table, ax=ax, cmap="coolwarm", center=0.5, vmin=vmin, vmax=vmax,
                    annot=True, fmt=".3f", linewidths=0.4, cbar=False, mask=table.isna())
        ax.set_title(CONFIGS[market][3])
        ax.set_xlabel("Regime")
        ax.set_ylabel("Latency")
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", rotation=35)
    scalar = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap="coolwarm")
    scalar.set_array([])
    fig.colorbar(scalar, ax=axes, shrink=0.82, label="Median spot information leadership share (ILS)")
    fig.suptitle(title)
    fig.savefig(HEATMAP_OUT / f"{filename}.png", dpi=220, bbox_inches="tight")
    fig.savefig(HEATMAP_OUT / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


def perspective_table(summary: pd.DataFrame, market: str, perspective: str) -> pd.DataFrame:
    return summary[
        summary["market"].eq(market) & summary["perspective"].eq(perspective)
    ].pivot(index="latency", columns="regime", values="median_ILS").reindex(
        index=LATENCIES, columns=[row[0] for row in REGIMES]
    )


def draw_spot_perp(summary: pd.DataFrame, markets: list[str], filename: str, title: str) -> None:
    tables = {
        (market, perspective): perspective_table(summary, market, perspective)
        for market in markets for perspective in ("log_midpoint_spot", "log_midpoint_perp")
    }
    finite = np.concatenate([
        table.to_numpy()[np.isfinite(table.to_numpy())] for table in tables.values()
    ])
    deviation = max(float(np.max(np.abs(finite - 0.5))), 1e-6)
    vmin, vmax = 0.5 - deviation, 0.5 + deviation
    fig, axes = plt.subplots(len(markets), 2, figsize=(17, 4.3 * len(markets)),
                             squeeze=False, constrained_layout=True)
    for row, market in enumerate(markets):
        for column, perspective in enumerate(("log_midpoint_spot", "log_midpoint_perp")):
            ax = axes[row, column]
            table = tables[(market, perspective)]
            sns.heatmap(table, ax=ax, cmap="coolwarm", center=0.5, vmin=vmin, vmax=vmax,
                        annot=True, fmt=".3f", linewidths=0.4, cbar=False, mask=table.isna())
            ax.set_title(perspective if len(markets) == 1 else f"{CONFIGS[market][3]} - {perspective}")
            ax.set_xlabel("Regime")
            ax.set_ylabel("Latency")
            ax.tick_params(axis="y", labelsize=8)
            ax.tick_params(axis="x", rotation=35)
    scalar = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap="coolwarm")
    scalar.set_array([])
    fig.colorbar(scalar, ax=axes.ravel().tolist(), shrink=0.82,
                 label="Median information leadership share (ILS)")
    fig.suptitle(title)
    fig.savefig(HEATMAP_OUT / f"{filename}.png", dpi=220, bbox_inches="tight")
    fig.savefig(HEATMAP_OUT / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


def generate_regime_heatmaps() -> None:
    """Generate regime heatmaps and their summary CSV files."""
    HEATMAP_OUT.mkdir(parents=True, exist_ok=True)
    data = pd.concat([load_market(market) for market in CONFIGS], ignore_index=True)
    summary = summarize(data)
    summary.to_csv(HEATMAP_OUT / "vecm_regime_spot_ils_summary.csv", index=False)
    perspective_summary = summarize_perspectives(data)
    perspective_summary.to_csv(HEATMAP_OUT / "vecm_regime_spot_perp_ils_summary.csv", index=False)
    data.groupby(["market", "latency"])["interval"].agg(
        first_interval="min", last_interval="max", model_intervals="nunique"
    ).reset_index().to_csv(HEATMAP_OUT / "vecm_input_coverage.csv", index=False)
    draw(summary, list(CONFIGS), "vecm_regime_heatmaps_all_contracts_april2023",
         "VECM price leadership by regime (April 2023 policy boundary)")
    draw(summary, ["btc_um", "btc_cm"], "vecm_regime_heatmaps_btc_april2023",
         "BTC VECM price leadership by regime")
    draw(summary, ["eth_um", "eth_cm"], "vecm_regime_heatmaps_eth_april2023",
         "ETH VECM price leadership by regime")
    draw_spot_perp(perspective_summary, list(CONFIGS),
                   "vecm_regime_heatmaps_spot_perp_all_contracts_april2023",
                   "VECM price leadership by regime: spot and perpetual perspectives")
    for market in CONFIGS:
        draw_spot_perp(perspective_summary, [market],
                       f"{FILE_LABELS[market]}_spot_perp_heatmap_april2023",
                       f"{CONFIGS[market][3]} VECM price leadership by regime")
    print(summary.groupby("market")["model_intervals"].sum().to_string())
    print(f"Saved outputs to {HEATMAP_OUT}")


def main() -> None:
    generate_alpha_over_time()
    generate_separate_alpha_plots()
    generate_regime_heatmaps()


if __name__ == "__main__":
    main()
