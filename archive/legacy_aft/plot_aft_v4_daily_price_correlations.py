"""Plot correlations and VIFs for the nine-variable V4 AFT inputs."""

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
    "open_interest",
    "shock_size_bps",
    "spread_spot_5min",
    "taker_long_short",
    "vol_spot_5min_pct",
    "volume",
    "log_price_standardized",
)
INPUT_DIR_NAME = "aft_data_liquidity_v4_daily_price"
OUTPUT = Path("open_interest_figures/native/aft_v4_daily_price_correlations")
ROWS_PER_MONTH_DIRECTION = 5000
RANDOM_SEED = 2026


def complete_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[list(COVARIATES)].apply(pd.to_numeric, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def load_samples() -> dict[tuple[str, str], pd.DataFrame]:
    samples = {}
    for market in MARKETS:
        input_dir = Path(f"sa_{market}") / INPUT_DIR_NAME
        for first in ("spot", "perp"):
            pieces = []
            for file_number, path in enumerate(sorted(input_dir.glob(f"{first}_*.parquet"))):
                complete = complete_numeric(pd.read_parquet(path, columns=list(COVARIATES)))
                if len(complete) > ROWS_PER_MONTH_DIRECTION:
                    complete = complete.sample(
                        ROWS_PER_MONTH_DIRECTION,
                        random_state=RANDOM_SEED + file_number,
                    )
                pieces.append(complete)
            samples[(market, first)] = pd.concat(pieces, ignore_index=True).dropna()
    return samples


def draw_lower_triangle(ax, frame: pd.DataFrame, title: str) -> None:
    corr = frame[list(COVARIATES)].corr()
    corr.to_csv(OUTPUT / f"correlation_matrix_{title.lower().replace(', ', '_').replace(' ', '_')}.csv")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=0)
    sns.heatmap(
        corr, mask=mask, cmap="coolwarm", center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f", square=True, linewidths=0.4, cbar=False, ax=ax,
    )
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)


def calculate_vif(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame[list(COVARIATES)].astype(float)
    x = ((x - x.mean()) / x.std(ddof=0)).replace([np.inf, -np.inf], np.nan).dropna()
    values = x.to_numpy()
    return pd.DataFrame({
        "covariate": list(COVARIATES),
        "VIF": [variance_inflation_factor(values, i) for i in range(values.shape[1])],
    })


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    samples = load_samples()
    pd.DataFrame([
        {"market": market, "first": first, "observations": len(frame)}
        for (market, first), frame in samples.items()
    ]).to_csv(OUTPUT / "correlation_sample_sizes.csv", index=False)

    vif_frames = []
    for (market, first), frame in samples.items():
        vif = calculate_vif(frame)
        vif.insert(0, "first", first)
        vif.insert(0, "market", market)
        vif_frames.append(vif)
        corr = frame[list(COVARIATES)].corr()
        corr.to_csv(OUTPUT / f"correlation_matrix_{market}_{first}.csv")
    all_vif = pd.concat(vif_frames, ignore_index=True)
    all_vif.to_csv(OUTPUT / "vif_by_market_direction.csv", index=False)

    for market, first in samples:
        fig, ax = plt.subplots(figsize=(10, 9))
        draw_lower_triangle(
            ax, samples[(market, first)],
            f"V4 covariate correlations: {market}, {first}-origin (n={len(samples[(market, first)]):,})",
        )
        fig.tight_layout()
        fig.savefig(OUTPUT / f"covariate_correlations_lower_triangle_{market}_{first}.png", dpi=160)
        plt.close(fig)

    for asset in ("btc", "eth"):
        fig, axes = plt.subplots(1, 4, figsize=(31, 8.5), constrained_layout=True)
        for contract_index, suffix in enumerate(("um", "cm")):
            for origin_index, first in enumerate(("spot", "perp")):
                market = f"{asset}_{suffix}"
                contract = "Linear" if suffix == "um" else "Inverse"
                draw_lower_triangle(
                    axes[contract_index * 2 + origin_index], samples[(market, first)],
                    f"{contract}, {first}-origin",
                )
        fig.suptitle(f"{asset.upper()} V4 lower-triangle covariate correlations: nine variables", fontsize=16)
        fig.savefig(OUTPUT / f"covariate_correlations_lower_triangle_{asset}_nine_variable.png", dpi=160)
        plt.close(fig)

    print("output", OUTPUT)
    print(all_vif.to_string(index=False))


if __name__ == "__main__":
    main()
