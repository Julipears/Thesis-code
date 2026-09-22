"""Plot concordance and standardized-coefficient heatmaps for v2 AFT fits."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


MARKETS = ("btc_um", "btc_cm", "eth_um", "eth_cm")
COVARIATES = [
    "basis_5min",
    "fundingRate_bps",
    "shock_size_bps",
    "spread_spot_5min",
    "taker_long_short",
    "volume",
]
INPUT_DIR_NAME = "aft_data_liquidity_unit_agreeing_oi"
RESULTS_DIR_NAME = "aft_results_unit_agreeing_oi_v2_no_open_interest"
PLOTS = Path("open_interest_figures/native/aft_v2_no_oi_plots")


def load_manifest() -> pd.DataFrame:
    frames = []
    for market in MARKETS:
        path = Path(f"sa_{market}") / RESULTS_DIR_NAME / "aft_fit_manifest.csv"
        frame = pd.read_csv(path)
        frame["market"] = market
        parsed = frame["source_file"].str.removesuffix(".parquet").str.split("_", n=1, expand=True)
        frame["first"] = parsed[0]
        frame["period"] = parsed[1]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def coefficient_rows(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for market in MARKETS:
        results_dir = Path(f"sa_{market}") / RESULTS_DIR_NAME / "loglogistic" / "coefficients"
        input_dir = Path(f"sa_{market}") / INPUT_DIR_NAME
        complete = manifest[(manifest.market == market) & manifest.status.isin(["complete", "skipped_existing"])]
        for source_file in complete.source_file.unique():
            path = results_dir / f"{Path(source_file).stem}_coefficients.csv"
            if not path.exists():
                continue
            coefficients = pd.read_csv(path)
            coefficients = coefficients[coefficients.covariate.isin(COVARIATES)].copy()
            input_path = input_dir / source_file
            frame = pd.read_parquet(input_path, columns=COVARIATES)
            standard_deviations = frame[COVARIATES].apply(pd.to_numeric, errors="coerce").std(ddof=0)
            coefficients["market"] = market
            first, period = Path(source_file).stem.split("_", 1)
            coefficients["first"] = first
            coefficients["period"] = period
            coefficients["covariate_sd"] = coefficients.covariate.map(standard_deviations)
            coefficients["standardized_coef"] = coefficients.coef * coefficients.covariate_sd
            rows.append(coefficients[[
                "market", "first", "period", "covariate", "coef", "covariate_sd",
                "standardized_coef", "p", "coef lower 95%", "coef upper 95%",
            ]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_heatmaps(coefficients: pd.DataFrame) -> None:
    for asset in ("btc", "eth"):
        panels = [("um", "spot"), ("um", "perp"), ("cm", "spot"), ("cm", "perp")]
        tables = []
        for margin, first in panels:
            subset = coefficients[(coefficients["market"] == f"{asset}_{margin}") & (coefficients["first"] == first)]
            tables.append(subset.pivot(index="covariate", columns="period", values="standardized_coef").reindex(COVARIATES))
        finite = np.concatenate([table.to_numpy()[np.isfinite(table.to_numpy())] for table in tables if np.isfinite(table.to_numpy()).any()])
        limit = max(float(np.nanpercentile(np.abs(finite), 98)), 1e-9)
        fig, axes = plt.subplots(1, 4, figsize=(32, 6), sharey=True, constrained_layout=True)
        for column, ((margin, first), table) in enumerate(zip(panels, tables)):
            sns.heatmap(table, cmap="coolwarm", center=0, vmin=-limit, vmax=limit,
                        ax=axes[column], cbar=column == 3,
                        cbar_kws={"label": "Change in log duration per 1-SD increase"})
            contract = "Linear" if margin == "um" else "Inverse"
            axes[column].set_title(f"{contract}, {first}-origin")
            axes[column].set_xlabel("Month")
            axes[column].set_ylabel("Covariate" if column == 0 else "")
            axes[column].tick_params(axis="x", labelrotation=90, labelsize=6)
        fig.suptitle(f"{asset.upper()} Log-Logistic AFT standardized coefficient heatmaps", fontsize=15)
        fig.savefig(PLOTS / f"aft_coefficient_heatmap_{asset}_loglogistic_4col.png", dpi=160, bbox_inches="tight")
        fig.savefig(PLOTS / f"aft_coefficient_heatmap_{asset}_loglogistic_4col.pdf", bbox_inches="tight")
        plt.close(fig)


def plot_concordance(manifest: pd.DataFrame) -> None:
    complete = manifest[manifest.status.isin(["complete", "skipped_existing"])].copy()
    complete["period_date"] = pd.to_datetime(complete.period, format="%Y-%m")
    for asset in ("btc", "eth"):
        fig, axes = plt.subplots(1, 4, figsize=(24, 5), sharex=True, sharey=True, constrained_layout=True)
        panels = [("um", "spot"), ("um", "perp"), ("cm", "spot"), ("cm", "perp")]
        for column, (margin, first) in enumerate(panels):
            subset = complete[(complete["market"] == f"{asset}_{margin}") & (complete["first"] == first)].sort_values("period_date")
            axes[column].plot(subset.period_date, subset.concordance, marker="o", markersize=2, linewidth=1.2,
                              color="tab:blue" if margin == "um" else "tab:orange")
            contract = "Linear" if margin == "um" else "Inverse"
            axes[column].set_title(f"{contract}, {first}-origin")
            axes[column].set_ylabel("Concordance index" if column == 0 else "")
            axes[column].grid(alpha=0.3)
            axes[column].set_ylim(0.45, 0.8)
        fig.suptitle(f"{asset.upper()} monthly Log-Logistic AFT concordance", fontsize=15)
        fig.savefig(PLOTS / f"aft_concordance_{asset}_loglogistic_4col.png", dpi=160, bbox_inches="tight")
        fig.savefig(PLOTS / f"aft_concordance_{asset}_loglogistic_4col.pdf", bbox_inches="tight")
        plt.close(fig)

    styles = {
        ("um", "spot"): ("Linear: spot", "tab:blue", "-"),
        ("um", "perp"): ("Linear: perp", "tab:blue", "--"),
        ("cm", "spot"): ("Inverse: spot", "tab:orange", "-"),
        ("cm", "perp"): ("Inverse: perp", "tab:orange", "--"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True, constrained_layout=True)
    for ax, asset in zip(axes, ("btc", "eth")):
        for (margin, first), (label, color, linestyle) in styles.items():
            rows = complete[(complete["market"] == f"{asset}_{margin}") & (complete["first"] == first)].sort_values("period_date")
            ax.plot(rows.period_date, rows.concordance, label=label, color=color, linestyle=linestyle,
                    marker="o", markersize=2.5, linewidth=1.4)
        ax.set_title(asset.upper())
        ax.set_xlabel("Month")
        ax.set_ylim(0.45, 0.80)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Concordance index")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Monthly Log-Logistic AFT concordance", y=1.16)
    fig.savefig(PLOTS / "aft_concordance_btc_eth_loglogistic_2col.png", dpi=180, bbox_inches="tight")
    fig.savefig(PLOTS / "aft_concordance_btc_eth_loglogistic_2col.pdf", bbox_inches="tight")
    plt.close(fig)

    summary = complete.groupby(["market", "first"])["concordance"].agg(
        mean="mean", median="median", minimum="min", maximum="max", months="count"
    ).reset_index()
    summary.to_csv(PLOTS / "concordance_summary.csv", index=False)


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    coefficients = coefficient_rows(manifest)
    coefficients.to_csv(PLOTS / "standardized_coefficients_loglogistic.csv", index=False)
    plot_heatmaps(coefficients)
    plot_concordance(manifest)
    print(manifest.groupby(["market", "status"]).size().to_string())
    print("coefficient rows", len(coefficients))
    print("plots", sorted(path.name for path in PLOTS.iterdir()))


if __name__ == "__main__":
    main()
