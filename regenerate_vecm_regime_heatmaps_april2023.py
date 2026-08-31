"""Regenerate VECM regime heatmaps with an April 1, 2023 boundary."""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


CONFIGS = {
    "btc_um": (Path("vecm_hasbrouck2_um"), "hasbrouck2", "um", "BTC linear"),
    "btc_cm": (Path("vecm_hasbrouck2_cm"), "hasbrouck2", "cm", "BTC inverse"),
    "eth_um": (Path("vecm_hasbrouck3_eth_um"), "hasbrouck3_eth_alignedv2", "um", "ETH linear"),
    "eth_cm": (Path("vecm_hasbrouck3_eth_cm"), "hasbrouck3_eth_alignedv2", "cm", "ETH inverse"),
}
FILE_LABELS = {
    "btc_um": "btc_linear",
    "btc_cm": "btc_inverse",
    "eth_um": "eth_linear",
    "eth_cm": "eth_inverse",
}
LATENCIES = ["10ms", "50ms", "100ms", "200ms", "500ms", "1s"]
REGIMES = [
    ("2021", "2021-01-01", "2022-01-01"),
    ("Jan–Jun 2022", "2022-01-01", "2022-07-01"),
    ("Jul 2022–Mar 2023", "2022-07-01", "2023-04-01"),
    ("Apr–Dec 2023", "2023-04-01", "2024-01-01"),
    ("2024", "2024-01-01", "2025-01-01"),
    ("2025", "2025-01-01", "2026-01-01"),
]
OUT = Path("vecm_regime_heatmaps_april2023")


def load_market(market: str) -> pd.DataFrame:
    folder, prefix, contract, title = CONFIGS[market]
    pattern = re.compile(
        rf"^{re.escape(prefix)}_1h_({ '|'.join(map(re.escape, LATENCIES)) })_10_"
        rf"{contract}_results_\d{{8}}_\d{{8}}\.csv$".replace(" ", "")
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
    data = data[
        data["interval"].ge("2021-01-01") & data["interval"].lt("2026-01-01")
    ]
    data = data.drop_duplicates(["interval", "series", "latency"], keep="last")
    data["market"] = market
    data["contract_title"] = title
    return data


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    spot = data[data["series"].eq("log_midpoint_spot")].copy()
    rows = []
    for market, market_rows in spot.groupby("market"):
        title = CONFIGS[market][3]
        for regime, start, end in REGIMES:
            selected = market_rows[
                market_rows["interval"].ge(start) & market_rows["interval"].lt(end)
            ]
            for latency in LATENCIES:
                values = pd.to_numeric(
                    selected.loc[selected["latency"].eq(latency), "ILS_mid"],
                    errors="coerce",
                ).dropna()
                rows.append({
                    "market": market,
                    "contract": title,
                    "regime": regime,
                    "regime_start": start,
                    "regime_end_exclusive": end,
                    "latency": latency,
                    "median_spot_ILS": values.median() if len(values) else np.nan,
                    "mean_spot_ILS": values.mean() if len(values) else np.nan,
                    "model_intervals": len(values),
                })
    return pd.DataFrame(rows)


def summarize_perspectives(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize the complementary spot and perpetual ILS perspectives."""
    rows = []
    series_names = {
        "log_midpoint_spot": "log_midpoint_spot",
        "log_midpoint_perp": "log_midpoint_perp",
    }
    for market, market_rows in data.groupby("market"):
        title = CONFIGS[market][3]
        for series, perspective in series_names.items():
            series_rows = market_rows[market_rows["series"].eq(series)]
            for regime, start, end in REGIMES:
                selected = series_rows[
                    series_rows["interval"].ge(start)
                    & series_rows["interval"].lt(end)
                ]
                for latency in LATENCIES:
                    values = pd.to_numeric(
                        selected.loc[selected["latency"].eq(latency), "ILS_mid"],
                        errors="coerce",
                    ).dropna()
                    rows.append({
                        "market": market,
                        "contract": title,
                        "perspective": perspective,
                        "series": series,
                        "regime": regime,
                        "regime_start": start,
                        "regime_end_exclusive": end,
                        "latency": latency,
                        "median_ILS": values.median() if len(values) else np.nan,
                        "mean_ILS": values.mean() if len(values) else np.nan,
                        "model_intervals": len(values),
                    })
    return pd.DataFrame(rows)


def table_for(summary: pd.DataFrame, market: str) -> pd.DataFrame:
    return (
        summary[summary["market"].eq(market)]
        .pivot(index="latency", columns="regime", values="median_spot_ILS")
        .reindex(index=LATENCIES, columns=[row[0] for row in REGIMES])
    )


def draw(summary: pd.DataFrame, markets: list[str], filename: str, title: str) -> None:
    tables = [table_for(summary, market) for market in markets]
    finite = np.concatenate([
        table.to_numpy()[np.isfinite(table.to_numpy())] for table in tables
    ])
    deviation = max(float(np.max(np.abs(finite - 0.5))), 1e-6)
    vmin, vmax = 0.5 - deviation, 0.5 + deviation

    if len(markets) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(17, 9), constrained_layout=True)
        axes = list(axes.flat)
    else:
        fig, axes = plt.subplots(1, len(markets), figsize=(17, 5), constrained_layout=True)
        axes = list(np.atleast_1d(axes).flat)

    image = None
    for ax, market, table in zip(axes, markets, tables):
        image = sns.heatmap(
            table,
            ax=ax,
            cmap="coolwarm",
            center=0.5,
            vmin=vmin,
            vmax=vmax,
            annot=True,
            fmt=".3f",
            linewidths=0.4,
            cbar=False,
            mask=table.isna(),
        )
        ax.set_title(CONFIGS[market][3])
        ax.set_xlabel("Regime")
        ax.set_ylabel("Latency")
        ax.tick_params(axis="x", rotation=35)

    scalar = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap="coolwarm"
    )
    scalar.set_array([])
    fig.colorbar(
        scalar, ax=axes, shrink=0.82,
        label="Median spot information leadership share (ILS)",
    )
    fig.suptitle(title)
    fig.savefig(OUT / f"{filename}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


def perspective_table(
    summary: pd.DataFrame, market: str, perspective: str
) -> pd.DataFrame:
    return (
        summary[
            summary["market"].eq(market)
            & summary["perspective"].eq(perspective)
        ]
        .pivot(index="latency", columns="regime", values="median_ILS")
        .reindex(index=LATENCIES, columns=[row[0] for row in REGIMES])
    )


def draw_spot_perp(
    summary: pd.DataFrame, markets: list[str], filename: str, title: str
) -> None:
    tables = {
        (market, perspective): perspective_table(summary, market, perspective)
        for market in markets
        for perspective in ("log_midpoint_spot", "log_midpoint_perp")
    }
    finite = np.concatenate([
        table.to_numpy()[np.isfinite(table.to_numpy())]
        for table in tables.values()
    ])
    deviation = max(float(np.max(np.abs(finite - 0.5))), 1e-6)
    vmin, vmax = 0.5 - deviation, 0.5 + deviation

    fig, axes = plt.subplots(
        len(markets), 2,
        figsize=(17, 4.3 * len(markets)),
        squeeze=False,
        constrained_layout=True,
    )
    for row, market in enumerate(markets):
        for column, perspective in enumerate(
            ("log_midpoint_spot", "log_midpoint_perp")
        ):
            ax = axes[row, column]
            table = tables[(market, perspective)]
            sns.heatmap(
                table,
                ax=ax,
                cmap="coolwarm",
                center=0.5,
                vmin=vmin,
                vmax=vmax,
                annot=True,
                fmt=".3f",
                linewidths=0.4,
                cbar=False,
                mask=table.isna(),
            )
            ax.set_title(
                perspective if len(markets) == 1
                else f"{CONFIGS[market][3]} — {perspective}"
            )
            ax.set_xlabel("Regime")
            ax.set_ylabel("Latency")
            ax.tick_params(axis="x", rotation=35)

    scalar = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap="coolwarm"
    )
    scalar.set_array([])
    fig.colorbar(
        scalar,
        ax=axes.ravel().tolist(),
        shrink=0.82,
        label="Median information leadership share (ILS)",
    )
    fig.suptitle(title)
    fig.savefig(OUT / f"{filename}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.concat([load_market(market) for market in CONFIGS], ignore_index=True)
    summary = summarize(data)
    summary.to_csv(OUT / "vecm_regime_spot_ils_summary.csv", index=False)
    perspective_summary = summarize_perspectives(data)
    perspective_summary.to_csv(
        OUT / "vecm_regime_spot_perp_ils_summary.csv", index=False
    )
    data.groupby(["market", "latency"])["interval"].agg(
        first_interval="min", last_interval="max", model_intervals="nunique"
    ).reset_index().to_csv(OUT / "vecm_input_coverage.csv", index=False)

    draw(
        summary, list(CONFIGS), "vecm_regime_heatmaps_all_contracts_april2023",
        "VECM price leadership by regime (April 2023 policy boundary)",
    )
    draw(
        summary, ["btc_um", "btc_cm"], "vecm_regime_heatmaps_btc_april2023",
        "BTC VECM price leadership by regime",
    )
    draw(
        summary, ["eth_um", "eth_cm"], "vecm_regime_heatmaps_eth_april2023",
        "ETH VECM price leadership by regime",
    )
    draw_spot_perp(
        perspective_summary,
        list(CONFIGS),
        "vecm_regime_heatmaps_spot_perp_all_contracts_april2023",
        "VECM price leadership by regime: spot and perpetual perspectives",
    )
    for market in CONFIGS:
        draw_spot_perp(
            perspective_summary,
            [market],
            f"{FILE_LABELS[market]}_spot_perp_heatmap_april2023",
            f"{CONFIGS[market][3]} VECM price leadership by regime",
        )
    print(summary.groupby("market")["model_intervals"].sum().to_string())
    print(f"Saved outputs to {OUT}")


if __name__ == "__main__":
    main()
