"""Plot covariate correlations for the unit-agreeing-OI AFT inputs."""

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
)
REDUCED_COVARIATES = tuple(c for c in COVARIATES if c != "vol_spot_5min_pct")
INPUT_DIR_NAME = "aft_data_liquidity_unit_agreeing_oi"
OUTPUT = Path("open_interest_figures/native/aft_unit_agreeing_oi_correlations")
ROWS_PER_MONTH_DIRECTION = 5000
RANDOM_SEED = 2026


def complete_numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    out = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def load_samples() -> dict[tuple[str, str], pd.DataFrame]:
    samples = {}
    for market in MARKETS:
        input_dir = Path(f"sa_{market}") / INPUT_DIR_NAME
        for first in ("spot", "perp"):
            pieces = []
            for file_number, path in enumerate(sorted(input_dir.glob(f"{first}_*.parquet"))):
                frame = pd.read_parquet(path, columns=list(COVARIATES))
                complete = complete_numeric(frame, COVARIATES)
                if len(complete) > ROWS_PER_MONTH_DIRECTION:
                    complete = complete.sample(
                        ROWS_PER_MONTH_DIRECTION,
                        random_state=RANDOM_SEED + file_number,
                    )
                pieces.append(complete)
            samples[(market, first)] = pd.concat(pieces, ignore_index=True).dropna()
    return samples


def draw_lower_triangle(ax, frame: pd.DataFrame, columns: tuple[str, ...], title: str) -> None:
    corr = frame[list(columns)].corr()
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


def calculate_vif(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    varying = [column for column in columns if frame[column].nunique() > 1]
    x = frame[varying].astype(float)
    x = ((x - x.mean()) / x.std(ddof=0)).replace([np.inf, -np.inf], np.nan).dropna()
    values = x.to_numpy()
    return pd.DataFrame({
        "covariate": varying,
        "VIF": [variance_inflation_factor(values, i) for i in range(values.shape[1])],
    }).sort_values("VIF", ascending=False)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    samples = load_samples()
    pd.DataFrame(
        [
            {"market": market, "first": first, "observations": len(frame)}
            for (market, first), frame in samples.items()
        ]
    ).to_csv(OUTPUT / "correlation_sample_sizes.csv", index=False)

    vif_frames = []
    reduced_vif_frames = []
    for (market, first), frame in samples.items():
        for label, columns in (("full", COVARIATES), ("without_spot_volatility", REDUCED_COVARIATES)):
            vif = calculate_vif(frame, columns)
            vif.insert(0, "first", first)
            vif.insert(0, "market", market)
            vif.insert(0, "specification", label)
            if label == "full":
                vif_frames.append(vif)
            else:
                reduced_vif_frames.append(vif)
    all_vif = pd.concat(vif_frames, ignore_index=True)
    all_reduced_vif = pd.concat(reduced_vif_frames, ignore_index=True)
    all_vif.to_csv(OUTPUT / "vif_by_market_direction.csv", index=False)
    all_reduced_vif.to_csv(OUTPUT / "vif_by_market_direction_without_spot_volatility.csv", index=False)
    summary = (
        all_vif.groupby("covariate")["VIF"]
        .agg(mean="mean", median="median", maximum="max")
        .sort_values("maximum", ascending=False)
        .reset_index()
    )
    summary.to_csv(OUTPUT / "vif_summary.csv", index=False)

    for market, first in samples:
        for label, columns in (
            ("full", COVARIATES),
            ("without_spot_volatility", REDUCED_COVARIATES),
        ):
            fig, ax = plt.subplots(figsize=(9, 8))
            draw_lower_triangle(
                ax,
                samples[(market, first)],
                columns,
                f"Covariate correlations: {market}, {first}-origin (n={len(samples[(market, first)]):,})",
            )
            fig.tight_layout()
            fig.savefig(
                OUTPUT / f"covariate_correlations_lower_triangle_{label}_{market}_{first}.png",
                dpi=150,
            )
            plt.close(fig)

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
                        samples[(market, first)],
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

    print("output", OUTPUT)
    print(pd.read_csv(OUTPUT / "correlation_sample_sizes.csv").to_string(index=False))


if __name__ == "__main__":
    main()
