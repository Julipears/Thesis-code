"""Correlation heatmaps and VIF tables for the V8 AFT model without volatility."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

from final_code import FULL_RUN, PAPER_DATA, RESULTS


ROOT = FULL_RUN
INPUT_DIR = ROOT / "aft_data_liquidity_v5_log_covariates"
MARKETS = ("btc_um", "btc_cm", "eth_um", "eth_cm")
COVARIATES = ("basis_5min", "fundingRate_bps", "log_open_interest", "shock_size_bps", "spread_spot_5min", "log_taker_long_short", "log_volume")
ROWS_PER_MONTH_DIRECTION = 5000
RANDOM_SEED = 2026


def clean(frame):
    return frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def vif(frame):
    cols = [c for c in frame if frame[c].nunique() > 1]
    x = frame[cols].astype(float)
    x = ((x - x.mean()) / x.std(ddof=0)).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    return pd.DataFrame({"covariate": cols, "VIF": [variance_inflation_factor(x, i) for i in range(len(cols))]}).sort_values("VIF", ascending=False)


def draw(ax, corr, title):
    mask = np.triu(np.ones_like(corr, dtype=bool), k=0)
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", square=True, linewidths=.4, cbar=False, ax=ax)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)


def compute_diagnostics(covariates, groups):
    correlations, vifs, samples = {}, [], []
    for market, first in groups:
        pieces = []
        for i, path in enumerate(sorted((INPUT_DIR / market).glob(f"{first}_*.parquet"))):
            part = clean(pd.read_parquet(path, columns=list(covariates)))
            if len(part) > ROWS_PER_MONTH_DIRECTION:
                part = part.sample(ROWS_PER_MONTH_DIRECTION, random_state=RANDOM_SEED + i)
            pieces.append(part)
        if not pieces:
            raise FileNotFoundError(f"No V8 input files for {market} {first} under {INPUT_DIR}")
        pooled = pd.concat(pieces, ignore_index=True)
        correlations[(market, first)] = pooled.corr()
        samples.append({"market": market, "first": first, "observations": len(pooled)})
        vifs.append(vif(pooled).assign(market=market, first=first))
    return correlations, pd.concat(vifs, ignore_index=True), pd.DataFrame(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-spot-volatility", action="store_true")
    parser.add_argument("--from-run", action="store_true", help="Recompute diagnostics from the local monthly input files.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    variant = "full" if args.with_spot_volatility else "reduced"
    covariates = (*COVARIATES[:-1], "vol_spot_5min_pct", COVARIATES[-1]) if args.with_spot_volatility else COVARIATES
    groups = [(market, first) for market in MARKETS for first in ("spot", "perp")]
    output = args.output or RESULTS / "aft" / "diagnostics" / variant
    output.mkdir(parents=True, exist_ok=True)

    if args.from_run:
        correlations, all_vif, samples = compute_diagnostics(covariates, groups)
    else:
        source = PAPER_DATA / "section_5_2/diagnostics" / variant
        correlations = {
            (market, first): pd.read_csv(source / f"corr_{market}_{first}.csv", index_col=0,
                                         float_precision="round_trip")
            for market, first in groups
        }
        all_vif = pd.read_csv(source / "vif.csv", float_precision="round_trip")
        samples = pd.read_csv(source / "sample_sizes.csv")

    counts = samples.set_index(["market", "first"])["observations"]
    title = "V8 AFT correlations" if args.with_spot_volatility else "V8 reduced AFT correlations"
    for market, first in groups:
        corr = correlations[(market, first)]
        corr.to_csv(output / f"corr_{market}_{first}.csv")
        fig, ax = plt.subplots(figsize=(9, 8))
        draw(ax, corr, f"{title}: {market}, {first}-origin (n={counts.loc[(market, first)]:,})")
        fig.tight_layout()
        fig.savefig(output / f"corr_{market}_{first}.png", dpi=160)
        plt.close(fig)

    for asset in ("btc", "eth"):
        fig, axes = plt.subplots(1, 4, figsize=(28, 7.5), constrained_layout=True)
        for ax, (suffix, first) in zip(axes, (("um", "spot"), ("um", "perp"), ("cm", "spot"), ("cm", "perp"))):
            contract = "Linear" if suffix == "um" else "Inverse"
            draw(ax, correlations[(f"{asset}_{suffix}", first)], f"{contract}, {first}-origin")
        panel_title = "V8-event AFT correlations" if args.with_spot_volatility else "V8-event reduced AFT correlations (volatility removed)"
        fig.suptitle(f"{asset.upper()} {panel_title}", fontsize=16)
        fig.savefig(output / f"corr_{asset}.png", dpi=160)
        plt.close(fig)

    all_vif.to_csv(output / "vif.csv", index=False)
    samples.to_csv(output / "sample_sizes.csv", index=False)
    summary = all_vif.groupby("covariate")["VIF"].agg(
        mean="mean", median="median", maximum="max"
    ).sort_values("maximum", ascending=False).reset_index()
    summary["exceeds_threshold"] = summary.maximum > 5
    summary.to_csv(output / "vif_summary.csv", index=False)
    wide = all_vif.pivot(index="covariate", columns=["market", "first"], values="VIF")
    wide.reindex(index=covariates, columns=groups).to_csv(output / "vif_table.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
