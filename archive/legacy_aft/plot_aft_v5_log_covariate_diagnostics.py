"""Correlation, VIF, and distribution diagnostics for the V5 AFT inputs."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor


MARKETS = ("btc_um", "btc_cm", "eth_um", "eth_cm")
COVARIATES = (
    "basis_5min",
    "fundingRate_bps",
    "log_open_interest",
    "shock_size_bps",
    "spread_spot_5min",
    "log_taker_long_short",
    "vol_spot_5min_pct",
    "log_volume",
)
INPUT_DIR_NAME = "aft_data_liquidity_v5_log_covariates"
OUTPUT = Path("open_interest_figures/native/aft_v5_log_covariate_diagnostics")
ROWS_PER_MONTH_DIRECTION = 5000
RANDOM_SEED = 2026


def load_samples() -> dict[tuple[str, str], pd.DataFrame]:
    samples = {}
    for market in MARKETS:
        input_dir = Path(f"sa_{market}") / INPUT_DIR_NAME
        for first in ("spot", "perp"):
            pieces = []
            for file_number, path in enumerate(sorted(input_dir.glob(f"{first}_*.parquet"))):
                frame = pd.read_parquet(path, columns=list(COVARIATES))
                frame = frame.apply(pd.to_numeric, errors="coerce")
                frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
                if len(frame) > ROWS_PER_MONTH_DIRECTION:
                    frame = frame.sample(
                        ROWS_PER_MONTH_DIRECTION,
                        random_state=RANDOM_SEED + file_number,
                    )
                pieces.append(frame)
            samples[(market, first)] = pd.concat(pieces, ignore_index=True)
    return samples


def calculate_vif(frame: pd.DataFrame) -> pd.DataFrame:
    x = ((frame[list(COVARIATES)] - frame[list(COVARIATES)].mean()) /
         frame[list(COVARIATES)].std(ddof=0)).dropna()
    values = x.to_numpy()
    return pd.DataFrame({
        "covariate": list(COVARIATES),
        "VIF": [variance_inflation_factor(values, i) for i in range(values.shape[1])],
    })


def draw_heatmap(ax, frame: pd.DataFrame, title: str) -> None:
    corr = frame[list(COVARIATES)].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=0)
    sns.heatmap(
        corr, mask=mask, cmap="coolwarm", center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f", square=True, linewidths=0.4, cbar=False, ax=ax,
    )
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    samples = load_samples()
    pd.DataFrame([
        {"market": market, "first": first, "observations": len(frame)}
        for (market, first), frame in samples.items()
    ]).to_csv(OUTPUT / "correlation_sample_sizes.csv", index=False)

    vif_frames = []
    dist_rows = []
    for (market, first), frame in samples.items():
        vif = calculate_vif(frame)
        vif.insert(0, "first", first)
        vif.insert(0, "market", market)
        vif_frames.append(vif)
        frame[list(COVARIATES)].corr().to_csv(OUTPUT / f"correlation_matrix_{market}_{first}.csv")
        for covariate in COVARIATES:
            values = frame[covariate]
            dist_rows.extend([
                {"market": market, "first": first, "covariate": covariate, "statistic": "mean", "value": values.mean(), "observations": len(values)},
                {"market": market, "first": first, "covariate": covariate, "statistic": "median", "value": values.median(), "observations": len(values)},
                {"market": market, "first": first, "covariate": covariate, "statistic": "std_dev", "value": values.std(ddof=1), "observations": len(values)},
            ])
    pd.concat(vif_frames, ignore_index=True).to_csv(OUTPUT / "vif_by_market_direction.csv", index=False)
    pd.DataFrame(dist_rows).to_csv(OUTPUT / "descriptive_statistics_long.csv", index=False)

    for market, first in samples:
        fig, ax = plt.subplots(figsize=(10, 9))
        draw_heatmap(ax, samples[(market, first)], f"V5 correlations: {market}, {first}-origin (n={len(samples[(market, first)]):,})")
        fig.tight_layout()
        fig.savefig(OUTPUT / f"covariate_correlations_lower_triangle_{market}_{first}.png", dpi=160)
        plt.close(fig)

    for asset in ("btc", "eth"):
        fig, axes = plt.subplots(1, 4, figsize=(31, 8.5), constrained_layout=True)
        for contract_index, suffix in enumerate(("um", "cm")):
            for origin_index, first in enumerate(("spot", "perp")):
                market = f"{asset}_{suffix}"
                contract = "Linear" if suffix == "um" else "Inverse"
                draw_heatmap(axes[contract_index * 2 + origin_index], samples[(market, first)], f"{contract}, {first}-origin")
        fig.suptitle(f"{asset.upper()} V5 covariate correlations: log liquidity variables", fontsize=16)
        fig.savefig(OUTPUT / f"covariate_correlations_lower_triangle_{asset}_v5.png", dpi=160)
        plt.close(fig)

    print("output", OUTPUT)
    print(pd.concat(vif_frames, ignore_index=True).to_string(index=False))


if __name__ == "__main__":
    main()
