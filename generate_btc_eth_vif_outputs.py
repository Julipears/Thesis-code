"""Generate BTC/ETH KM panels and pooled covariate VIF diagnostics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

from km_v8_regular_lm_pilot import plot_asset_km_month_comparison


ROOT = Path("sa_results/km_v8_final_01")
OUTPUT = ROOT / "analysis_outputs"
MONTHS = ("2021-12", "2022-12", "2023-12", "2025-12")
MARKETS = ("btc_um", "btc_cm", "eth_um", "eth_cm")
COVARIATES = (
    "basis_5min",
    "fundingRate_bps",
    "open_interest",
    "shock_size_bps",
    "spread_spot_5min",
    "taker_long_short",
    "vol_spot_5min_pct",
    "volume",
)
ROWS_PER_MONTH_DIRECTION = 5000
RANDOM_SEED = 2026
VIF_THRESHOLD = 5.0
REDUCED_COVARIATES = tuple(
    column for column in COVARIATES if column != "vol_spot_5min_pct"
)


def complete_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[list(COVARIATES)].apply(pd.to_numeric, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def calculate_vif(frame: pd.DataFrame, columns=COVARIATES) -> pd.DataFrame:
    varying = [column for column in columns if frame[column].nunique() > 1]
    x = frame[varying].astype(float)
    x = ((x - x.mean()) / x.std(ddof=0)).replace([np.inf, -np.inf], np.nan).dropna()
    values = x.to_numpy()
    return pd.DataFrame(
        {
            "covariate": varying,
            "VIF": [variance_inflation_factor(values, i) for i in range(values.shape[1])],
        }
    ).sort_values("VIF", ascending=False)


def draw_lower_triangle(ax, frame: pd.DataFrame, columns, title: str) -> None:
    corr = frame[list(columns)].corr()
    # Mask the diagonal and everything above it, leaving only unique pairs.
    mask = np.triu(np.ones_like(corr, dtype=bool), k=0)
    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.4,
        cbar=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    graph_rows = []
    for asset in ("btc", "eth"):
        for horizon in (120.0, 20.0):
            path = plot_asset_km_month_comparison(
                ROOT,
                asset=asset,
                months=MONTHS,
                grid="1s",
                horizon=horizon,
                plot_step_seconds=0.25,
                require_complete=False,
            )
            graph_rows.append({"asset": asset, "horizon_seconds": horizon, "path": str(path)})
    pd.DataFrame(graph_rows).to_csv(OUTPUT / "km_graph_manifest.csv", index=False)

    vif_frames = []
    reduced_vif_frames = []
    sample_rows = []
    diagnostic_samples = {}
    for market in MARKETS:
        input_dir = ROOT / "aft_data_liquidity_monthly" / market
        for first in ("spot", "perp"):
            pieces = []
            for file_number, path in enumerate(sorted(input_dir.glob(f"{first}_*.parquet"))):
                frame = pd.read_parquet(path, columns=list(COVARIATES))
                complete = complete_numeric(frame)
                if len(complete) > ROWS_PER_MONTH_DIRECTION:
                    complete = complete.sample(
                        ROWS_PER_MONTH_DIRECTION,
                        random_state=RANDOM_SEED + file_number,
                    )
                pieces.append(complete)
            pooled = pd.concat(pieces, ignore_index=True).dropna()
            diagnostic_samples[(market, first)] = pooled
            sample_rows.append({"market": market, "first": first, "observations": len(pooled)})

            vif = calculate_vif(pooled)
            vif.insert(0, "first", first)
            vif.insert(0, "market", market)
            vif_frames.append(vif)
            reduced_vif = calculate_vif(pooled, REDUCED_COVARIATES)
            reduced_vif.insert(0, "first", first)
            reduced_vif.insert(0, "market", market)
            reduced_vif_frames.append(reduced_vif)

            for label, columns in (
                ("full", COVARIATES),
                ("without_spot_volatility", REDUCED_COVARIATES),
            ):
                fig, ax = plt.subplots(figsize=(9, 8))
                draw_lower_triangle(
                    ax,
                    pooled,
                    columns,
                    f"Covariate correlations: {market}, {first}-origin (n={len(pooled):,})",
                )
                fig.tight_layout()
                fig.savefig(
                    OUTPUT / f"covariate_correlations_lower_triangle_{label}_{market}_{first}.png",
                    dpi=150,
                )
                plt.close(fig)

    all_vif = pd.concat(vif_frames, ignore_index=True)
    all_vif.to_csv(OUTPUT / "vif_by_market_direction.csv", index=False)
    all_reduced_vif = pd.concat(reduced_vif_frames, ignore_index=True)
    all_reduced_vif.to_csv(
        OUTPUT / "vif_by_market_direction_without_spot_volatility.csv", index=False
    )
    pd.DataFrame(sample_rows).to_csv(OUTPUT / "vif_sample_sizes.csv", index=False)
    summary = (
        all_vif.groupby("covariate")["VIF"]
        .agg(mean="mean", median="median", maximum="max")
        .sort_values("maximum", ascending=False)
        .reset_index()
    )
    summary["exceeds_threshold"] = summary["maximum"] > VIF_THRESHOLD
    summary.to_csv(OUTPUT / "vif_summary.csv", index=False)

    for asset in ("btc", "eth"):
        for label, columns in (
            ("full", COVARIATES),
            ("without_spot_volatility", REDUCED_COVARIATES),
        ):
            fig, axes = plt.subplots(1, 4, figsize=(28, 7.5), constrained_layout=True)
            for contract_index, suffix in enumerate(("um", "cm")):
                for origin_index, first in enumerate(("spot", "perp")):
                    market = f"{asset}_{suffix}"
                    contract = "Linear" if suffix == "um" else "Inverse"
                    draw_lower_triangle(
                        axes[contract_index * 2 + origin_index],
                        diagnostic_samples[(market, first)],
                        columns,
                        f"{contract}, {first}-origin",
                    )
            specification = (
                "all eight covariates"
                if label == "full"
                else "seven covariates (spot volatility removed)"
            )
            fig.suptitle(
                f"{asset.upper()} lower-triangle covariate correlations: {specification}",
                fontsize=16,
            )
            fig.savefig(
                OUTPUT / f"covariate_correlations_lower_triangle_{asset}_{label}.png",
                dpi=160,
            )
            plt.close(fig)
    print(summary.to_string(index=False))
    print(f"VIF threshold: {VIF_THRESHOLD:g}")
    print(f"Any removal indicated: {summary['exceeds_threshold'].any()}")


if __name__ == "__main__":
    main()
