"""Correlation heatmaps and VIF tables for the V8 AFT model without volatility."""

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
OUTPUT = ROOT / "analysis_outputs" / "aft_v5_without_spot_volatility_diagnostics"
MARKETS = ("btc_um", "btc_cm", "eth_um", "eth_cm")
COVARIATES = ("basis_5min", "fundingRate_bps", "log_open_interest", "shock_size_bps", "spread_spot_5min", "log_taker_long_short", "log_volume")
ROWS_PER_MONTH_DIRECTION = 5000
RANDOM_SEED = 2026


def clean(frame):
    return frame[list(COVARIATES)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def vif(frame):
    cols = [c for c in COVARIATES if frame[c].nunique() > 1]
    x = frame[cols].astype(float)
    x = ((x - x.mean()) / x.std(ddof=0)).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    return pd.DataFrame({"covariate": cols, "VIF": [variance_inflation_factor(x, i) for i in range(len(cols))]}).sort_values("VIF", ascending=False)


def draw(ax, frame, title):
    corr = frame[list(COVARIATES)].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=0)
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", square=True, linewidths=.4, cbar=False, ax=ax)
    ax.set_title(title); ax.tick_params(axis="x", rotation=45, labelsize=8); ax.tick_params(axis="y", rotation=0, labelsize=8)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frames, vifs, samples = {}, [], []
    for market in MARKETS:
        for first in ("spot", "perp"):
            pieces = []
            for i, path in enumerate(sorted((INPUT_DIR / market).glob(f"{first}_*.parquet"))):
                part = clean(pd.read_parquet(path, columns=list(COVARIATES)))
                if len(part) > ROWS_PER_MONTH_DIRECTION:
                    part = part.sample(ROWS_PER_MONTH_DIRECTION, random_state=RANDOM_SEED + i)
                pieces.append(part)
            pooled = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=COVARIATES)
            frames[(market, first)] = pooled
            samples.append({"market": market, "first": first, "observations": len(pooled)})
            out = vif(pooled); out.insert(0, "first", first); out.insert(0, "market", market); vifs.append(out)
            pooled[list(COVARIATES)].corr().to_csv(OUTPUT / f"correlation_matrix_{market}_{first}.csv")
            fig, ax = plt.subplots(figsize=(9, 8)); draw(ax, pooled, f"V8 reduced AFT correlations: {market}, {first}-origin (n={len(pooled):,})"); fig.tight_layout(); fig.savefig(OUTPUT / f"covariate_correlations_lower_triangle_{market}_{first}.png", dpi=160); plt.close(fig)
    all_vif = pd.concat(vifs, ignore_index=True); all_vif.to_csv(OUTPUT / "vif_by_market_direction.csv", index=False)
    pd.DataFrame(samples).to_csv(OUTPUT / "vif_sample_sizes.csv", index=False)
    summary = all_vif.groupby("covariate")["VIF"].agg(mean="mean", median="median", maximum="max").sort_values("maximum", ascending=False).reset_index(); summary["exceeds_threshold"] = summary.maximum > 5; summary.to_csv(OUTPUT / "vif_summary.csv", index=False)
    for asset in ("btc", "eth"):
        fig, axes = plt.subplots(1, 4, figsize=(28, 7.5), constrained_layout=True)
        for i, (suffix, first) in enumerate((("um", "spot"), ("um", "perp"), ("cm", "spot"), ("cm", "perp"))):
            draw(axes[i], frames[(f"{asset}_{suffix}", first)], f"{'Linear' if suffix == 'um' else 'Inverse'}, {first}-origin")
        fig.suptitle(f"{asset.upper()} V8-event reduced AFT correlations (volatility removed)", fontsize=16); fig.savefig(OUTPUT / f"covariate_correlations_lower_triangle_{asset}_without_spot_volatility.png", dpi=160); plt.close(fig)
    order=[("btc_um","spot"),("btc_um","perp"),("btc_cm","spot"),("btc_cm","perp"),("eth_um","spot"),("eth_um","perp"),("eth_cm","spot"),("eth_cm","perp")]
    wide=all_vif.pivot(index="covariate",columns=["market","first"],values="VIF").reindex(COVARIATES).reindex(columns=order); wide.to_csv(OUTPUT / "vif_table_reduced.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
