"""Correlation heatmaps and VIF tables for the V8-event V5 AFT inputs."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor


ROOT = Path("sa_results/km_v8_final_01")
INPUT_DIR = ROOT / "aft_data_liquidity_v5_log_covariates"
OUTPUT = ROOT / "analysis_outputs" / "aft_v5_log_covariate_diagnostics"
MARKETS = ("btc_um", "btc_cm", "eth_um", "eth_cm")
COVARIATES = (
    "basis_5min", "fundingRate_bps", "log_open_interest", "shock_size_bps",
    "spread_spot_5min", "log_taker_long_short", "vol_spot_5min_pct", "log_volume",
)
ROWS_PER_MONTH_DIRECTION = 5000
RANDOM_SEED = 2026


def complete_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[list(COVARIATES)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def calculate_vif(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [c for c in COVARIATES if frame[c].nunique() > 1]
    x = frame[columns].astype(float)
    x = ((x - x.mean()) / x.std(ddof=0)).replace([np.inf, -np.inf], np.nan).dropna()
    values = x.to_numpy()
    return pd.DataFrame({"covariate": columns, "VIF": [variance_inflation_factor(values, i) for i in range(len(columns))]}).sort_values("VIF", ascending=False)


def draw_heatmap(ax, frame: pd.DataFrame, title: str) -> None:
    corr = frame[list(COVARIATES)].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=0)
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", square=True, linewidths=0.4, cbar=False, ax=ax)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    vif_frames, sample_rows, samples = [], [], {}
    for market in MARKETS:
        for first in ("spot", "perp"):
            pieces = []
            for file_number, path in enumerate(sorted((INPUT_DIR / market).glob(f"{first}_*.parquet"))):
                frame = complete_numeric(pd.read_parquet(path, columns=list(COVARIATES)))
                if len(frame) > ROWS_PER_MONTH_DIRECTION:
                    frame = frame.sample(ROWS_PER_MONTH_DIRECTION, random_state=RANDOM_SEED + file_number)
                pieces.append(frame)
            pooled = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=COVARIATES)
            samples[(market, first)] = pooled
            sample_rows.append({"market": market, "first": first, "observations": len(pooled)})
            vif = calculate_vif(pooled)
            vif.insert(0, "first", first); vif.insert(0, "market", market); vif_frames.append(vif)
            pooled[list(COVARIATES)].corr().to_csv(OUTPUT / f"correlation_matrix_{market}_{first}.csv")
            fig, ax = plt.subplots(figsize=(9, 8))
            draw_heatmap(ax, pooled, f"V8 AFT correlations: {market}, {first}-origin (n={len(pooled):,})")
            fig.tight_layout(); fig.savefig(OUTPUT / f"covariate_correlations_lower_triangle_{market}_{first}.png", dpi=160); plt.close(fig)

    all_vif = pd.concat(vif_frames, ignore_index=True)
    all_vif.to_csv(OUTPUT / "vif_by_market_direction.csv", index=False)
    sample_table = pd.DataFrame(sample_rows); sample_table.to_csv(OUTPUT / "vif_sample_sizes.csv", index=False)
    summary = all_vif.groupby("covariate")["VIF"].agg(mean="mean", median="median", maximum="max").sort_values("maximum", ascending=False).reset_index()
    summary["exceeds_threshold"] = summary["maximum"] > 5.0
    summary.to_csv(OUTPUT / "vif_summary.csv", index=False)

    for asset in ("btc", "eth"):
        fig, axes = plt.subplots(1, 4, figsize=(28, 7.5), constrained_layout=True)
        for i, (suffix, first) in enumerate((("um", "spot"), ("um", "perp"), ("cm", "spot"), ("cm", "perp"))):
            market = f"{asset}_{suffix}"
            draw_heatmap(axes[i], samples[(market, first)], f"{'Linear' if suffix == 'um' else 'Inverse'}, {first}-origin")
        fig.suptitle(f"{asset.upper()} V8-event AFT covariate correlations", fontsize=16)
        fig.savefig(OUTPUT / f"covariate_correlations_lower_triangle_{asset}_full.png", dpi=160); plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
