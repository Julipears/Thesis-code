"""Generate V5-style coefficient, concordance, and significance figures for V8 AFT fits."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path("sa_results/km_v8_final_01")
MARKETS = ("btc_um", "btc_cm", "eth_um", "eth_cm")
COVARIATES = [
    "basis_5min", "fundingRate_bps", "log_open_interest", "shock_size_bps",
    "spread_spot_5min", "log_taker_long_short", "vol_spot_5min_pct", "log_volume",
]
INPUT_DIR = ROOT / "aft_data_liquidity_v5_log_covariates"
RESULTS_DIR = ROOT / "aft_results_v5_log_covariates"
OUTPUT = ROOT / "analysis_outputs" / "aft_v5_log_covariate_results"


def load_manifest() -> pd.DataFrame:
    frames = []
    for market in MARKETS:
        frame = pd.read_csv(RESULTS_DIR / market / "aft_fit_manifest.csv")
        frame["market"] = market
        parsed = frame["source_file"].str.removesuffix(".parquet").str.split("_", n=1, expand=True)
        frame["first"], frame["period"] = parsed[0], parsed[1]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def coefficient_rows(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    complete = manifest[manifest.status.isin(["complete", "skipped_existing"])]
    for market in MARKETS:
        result_dir = RESULTS_DIR / market / "loglogistic" / "coefficients"
        subset = complete[complete.market.eq(market)]
        for source_file in subset.source_file:
            result_path = result_dir / f"{Path(source_file).stem}_coefficients.csv"
            input_path = INPUT_DIR / market / source_file
            if not result_path.exists() or not input_path.exists():
                continue
            coefficients = pd.read_csv(result_path)
            coefficients = coefficients[coefficients.covariate.isin(COVARIATES)].copy()
            frame = pd.read_parquet(input_path, columns=COVARIATES).apply(pd.to_numeric, errors="coerce")
            sd = frame.std(ddof=0)
            coefficients["market"] = market
            coefficients["first"], coefficients["period"] = Path(source_file).stem.split("_", 1)
            coefficients["covariate_sd"] = coefficients.covariate.map(sd)
            coefficients["standardized_coef"] = coefficients.coef * coefficients.covariate_sd
            rows.append(coefficients[["market", "first", "period", "covariate", "coef", "standardized_coef", "p"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_heatmaps(coefficients: pd.DataFrame) -> None:
    panels = [("um", "spot"), ("um", "perp"), ("cm", "spot"), ("cm", "perp")]
    for asset in ("btc", "eth"):
        tables = [coefficients[(coefficients.market == f"{asset}_{margin}") & (coefficients["first"] == first)]
                  .pivot(index="covariate", columns="period", values="standardized_coef").reindex(COVARIATES)
                  for margin, first in panels]
        finite = np.concatenate([t.to_numpy()[np.isfinite(t.to_numpy())] for t in tables if np.isfinite(t.to_numpy()).any()])
        limit = max(float(np.nanpercentile(np.abs(finite), 98)), 1e-9)
        fig, axes = plt.subplots(1, 4, figsize=(32, 6), sharey=True, constrained_layout=True)
        for i, ((margin, first), table) in enumerate(zip(panels, tables)):
            sns.heatmap(table, cmap="coolwarm", center=0, vmin=-limit, vmax=limit, ax=axes[i], cbar=i == 3,
                        cbar_kws={"label": "Change in log duration per 1-SD increase"})
            axes[i].set_title(("Linear" if margin == "um" else "Inverse") + f", {first}-origin")
            axes[i].set_xlabel("Month"); axes[i].set_ylabel("Covariate" if i == 0 else "")
            axes[i].tick_params(axis="x", labelrotation=90, labelsize=6)
        fig.suptitle(f"{asset.upper()} V8-event Log-Logistic AFT standardized coefficients")
        fig.savefig(OUTPUT / f"aft_coefficient_heatmap_{asset}_v8_loglogistic_4col.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def plot_concordance(manifest: pd.DataFrame) -> None:
    complete = manifest[manifest.status.isin(["complete", "skipped_existing"])].copy()
    complete["period_date"] = pd.to_datetime(complete.period, format="%Y-%m")
    for asset in ("btc", "eth"):
        fig, axes = plt.subplots(1, 4, figsize=(24, 5), sharex=True, sharey=True, constrained_layout=True)
        for i, (margin, first) in enumerate([("um", "spot"), ("um", "perp"), ("cm", "spot"), ("cm", "perp")]):
            rows = complete[(complete.market == f"{asset}_{margin}") & (complete["first"] == first)].sort_values("period_date")
            axes[i].plot(rows.period_date, rows.concordance, marker="o", markersize=2, linewidth=1.2,
                         color="tab:blue" if margin == "um" else "tab:orange")
            axes[i].set_title(("Linear" if margin == "um" else "Inverse") + f", {first}-origin")
            axes[i].set_ylabel("Concordance index" if i == 0 else ""); axes[i].grid(alpha=0.3); axes[i].set_ylim(0.45, 0.8)
        fig.suptitle(f"{asset.upper()} V8-event Log-Logistic AFT concordance")
        fig.savefig(OUTPUT / f"aft_concordance_{asset}_v8_loglogistic_4col.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    styles = {("um", "spot"): ("Linear: spot", "tab:blue", "-"), ("um", "perp"): ("Linear: perp", "tab:blue", "--"),
              ("cm", "spot"): ("Inverse: spot", "tab:orange", "-"), ("cm", "perp"): ("Inverse: perp", "tab:orange", "--")}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True, constrained_layout=True)
    for ax, asset in zip(axes, ("btc", "eth")):
        for (margin, first), (label, color, style) in styles.items():
            rows = complete[(complete.market == f"{asset}_{margin}") & (complete["first"] == first)].sort_values("period_date")
            ax.plot(rows.period_date, rows.concordance, label=label, color=color, linestyle=style, marker="o", markersize=2.5, linewidth=1.4)
        ax.set_title(asset.upper()); ax.set_xlabel("Month"); ax.set_ylim(0.45, 0.80); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Concordance index")
    handles, labels = axes[0].get_legend_handles_labels(); fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("V8-event monthly Log-Logistic AFT concordance", y=1.08)
    fig.savefig(OUTPUT / "aft_concordance_btc_eth_v8_loglogistic_2col.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    complete.groupby(["market", "first"])["concordance"].agg(mean="mean", median="median", minimum="min", maximum="max", months="count").reset_index().to_csv(OUTPUT / "concordance_summary.csv", index=False)


def significance_table(coefficients: pd.DataFrame) -> None:
    table = coefficients.assign(significant=coefficients.p < 0.01).groupby(["market", "first", "covariate"]).agg(
        significant_months=("significant", "sum"), fitted_months=("p", "count"),
    ).reset_index()
    table["significance_percent"] = 100 * table.significant_months / table.fitted_months
    table.to_csv(OUTPUT / "significance_1pct_by_market_origin.csv", index=False)
    table.pivot_table(index="covariate", columns=["market", "first"], values="significance_percent").to_csv(OUTPUT / "significance_1pct_wide.csv")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    coefficients = coefficient_rows(manifest)
    coefficients.to_csv(OUTPUT / "standardized_coefficients.csv", index=False)
    plot_heatmaps(coefficients)
    plot_concordance(manifest)
    significance_table(coefficients)
    print(manifest.groupby(["market", "status"]).size().to_string())
    print("coefficient rows", len(coefficients))


if __name__ == "__main__":
    main()
