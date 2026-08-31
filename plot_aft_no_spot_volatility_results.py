"""Plot coefficient heatmaps and concordance from the no-volatility AFT refit."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path("sa_results/km_v8_final_01")
RESULTS = ROOT / "aft_results_no_spot_volatility_by_market"
INPUTS = ROOT / "aft_data_liquidity_monthly"
PLOTS = RESULTS / "plots"
COVARIATES = [
    "basis_5min",
    "fundingRate_bps",
    "open_interest",
    "shock_size_bps",
    "spread_spot_5min",
    "taker_long_short",
    "volume",
]
MODELS = ["weibull", "lognormal", "loglogistic"]


def select_family(manifest: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    complete = manifest[manifest["status"].eq("complete")].copy()
    keys = ["market", "first", "period"]
    complete["delta_AIC"] = complete["AIC"] - complete.groupby(keys)["AIC"].transform("min")
    summary = (
        complete.groupby("model")
        .agg(
            mean_delta_AIC=("delta_AIC", "mean"),
            median_delta_AIC=("delta_AIC", "median"),
            complete_fits=("model", "size"),
            mean_concordance=("concordance", "mean"),
        )
        .sort_values(["mean_delta_AIC", "median_delta_AIC"])
        .reset_index()
    )
    return str(summary.iloc[0]["model"]), summary


def coefficient_rows(selected_model: str) -> pd.DataFrame:
    rows = []
    for market in ("btc_um", "btc_cm", "eth_um", "eth_cm"):
        coefficient_dir = RESULTS / market / selected_model / "coefficients"
        for path in sorted(coefficient_dir.glob("*_coefficients.csv")):
            first, period = path.stem.removesuffix("_coefficients").split("_", 1)
            coefficients = pd.read_csv(path)
            coefficients = coefficients[
                coefficients["covariate"].isin(COVARIATES)
                & ~coefficients["covariate"].eq("Intercept")
            ].copy()
            input_path = INPUTS / market / f"{first}_{period}.parquet"
            input_frame = pd.read_parquet(input_path, columns=COVARIATES)
            standard_deviations = input_frame[COVARIATES].std(ddof=0)
            coefficients["market"] = market
            coefficients["first"] = first
            coefficients["period"] = period
            coefficients["covariate_sd"] = coefficients["covariate"].map(standard_deviations)
            coefficients["standardized_coef"] = coefficients["coef"] * coefficients["covariate_sd"]
            rows.append(
                coefficients[[
                    "market", "first", "period", "covariate", "coef", "covariate_sd",
                    "standardized_coef", "p", "coef lower 95%", "coef upper 95%",
                ]]
            )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_coefficient_heatmaps(coefficients: pd.DataFrame, selected_model: str) -> None:
    for asset in ("btc", "eth"):
        panels = [("um", "spot"), ("um", "perp"), ("cm", "spot"), ("cm", "perp")]
        tables = []
        for suffix, first in panels:
            subset = coefficients[
                coefficients["market"].eq(f"{asset}_{suffix}")
                & coefficients["first"].eq(first)
            ]
            tables.append(subset.pivot(
                index="covariate", columns="period", values="standardized_coef"
            ).reindex(COVARIATES))
        finite = np.concatenate([
            table.to_numpy()[np.isfinite(table.to_numpy())] for table in tables
        ])
        limit = max(float(np.nanpercentile(np.abs(finite), 98)), 1e-9)
        fig, axes = plt.subplots(1, 4, figsize=(32, 6), sharey=True,
                                 constrained_layout=True)
        for column, ((suffix, first), table) in enumerate(zip(panels, tables)):
            sns.heatmap(
                table, cmap="coolwarm", center=0, vmin=-limit, vmax=limit,
                ax=axes[column], cbar=column == 3,
                cbar_kws={"label": "Change in log duration per 1-SD increase"},
            )
            contract = "Linear" if suffix == "um" else "Inverse"
            axes[column].set_title(f"{contract}, {first}-origin")
            axes[column].set_xlabel("Month")
            axes[column].set_ylabel("Covariate" if column == 0 else "")
            axes[column].tick_params(axis="x", labelrotation=90, labelsize=6)
        fig.suptitle(
            f"{asset.upper()} {selected_model.title()} AFT standardized coefficient heatmaps",
            fontsize=15,
        )
        fig.savefig(PLOTS / f"aft_coefficient_heatmap_{asset}_{selected_model}_4col.png",
                    dpi=160, bbox_inches="tight")
        plt.close(fig)


def plot_concordance(manifest: pd.DataFrame, selected_model: str) -> None:
    complete = manifest[manifest["status"].eq("complete")].copy()
    complete["period_date"] = pd.to_datetime(complete["period"], format="%Y-%m")
    for asset in ("btc", "eth"):
        fig, axes = plt.subplots(1, 4, figsize=(24, 5), sharex=True, sharey=True,
                                 constrained_layout=True)
        panels = [("um", "spot"), ("um", "perp"), ("cm", "spot"), ("cm", "perp")]
        for column, (suffix, first) in enumerate(panels):
            market = f"{asset}_{suffix}"
            subset = complete[
                complete["market"].eq(market) & complete["first"].eq(first)
            ]
            for model in MODELS:
                model_rows = subset[subset["model"].eq(model)].sort_values("period_date")
                axes[column].plot(
                    model_rows["period_date"], model_rows["concordance"],
                    marker="o", markersize=2, linewidth=1.2,
                    label=model + (" (selected)" if model == selected_model else ""),
                )
            contract = "Linear" if suffix == "um" else "Inverse"
            axes[column].set_title(f"{contract}, {first}-origin")
            axes[column].set_ylabel("Concordance index" if column == 0 else "")
            axes[column].grid(alpha=0.3)
            axes[column].set_ylim(0.45, 0.8)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.035),
                   ncol=3, frameon=False)
        fig.suptitle(f"{asset.upper()} monthly AFT concordance", fontsize=15, y=1.075)
        fig.savefig(PLOTS / f"aft_concordance_{asset}_all_models_4col.png", dpi=160,
                    bbox_inches="tight")
        plt.close(fig)

    summary = (
        complete.groupby(["market", "first", "model"])["concordance"]
        .agg(mean="mean", median="median", minimum="min", maximum="max", months="count")
        .reset_index()
    )
    summary.to_csv(RESULTS / "concordance_summary.csv", index=False)
    average_by_model = (
        complete.groupby("model", as_index=False)["concordance"]
        .mean()
        .rename(columns={"concordance": "average_concordance"})
    )
    average_by_model.to_csv(RESULTS / "average_concordance_by_model.csv", index=False)


def plot_selected_concordance_two_columns(
    manifest: pd.DataFrame, selected_model: str
) -> None:
    """One BTC panel and one ETH panel, using the KM contract/origin styling."""
    complete = manifest[
        manifest["status"].eq("complete") & manifest["model"].eq(selected_model)
    ].copy()
    complete["period_date"] = pd.to_datetime(complete["period"], format="%Y-%m")
    styles = {
        ("um", "spot"): ("Linear: spot", "tab:blue", "-"),
        ("um", "perp"): ("Linear: perp", "tab:blue", "--"),
        ("cm", "spot"): ("Inverse: spot", "tab:orange", "-"),
        ("cm", "perp"): ("Inverse: perp", "tab:orange", "--"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, asset in zip(axes, ("btc", "eth")):
        for (suffix, first), (label, color, linestyle) in styles.items():
            rows = complete[
                complete["market"].eq(f"{asset}_{suffix}")
                & complete["first"].eq(first)
            ].sort_values("period_date")
            ax.plot(
                rows["period_date"], rows["concordance"], label=label,
                color=color, linestyle=linestyle, marker="o", markersize=2.5,
                linewidth=1.4,
            )
        ax.set_title(asset.upper())
        ax.set_xlabel("Month")
        ax.set_ylim(0.45, 0.80)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Concordance index")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(f"Monthly {selected_model.title()} AFT concordance", y=1.16)
    fig.savefig(
        PLOTS / f"aft_concordance_btc_eth_{selected_model}_2col.png",
        dpi=180, bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(RESULTS / "aft_fit_manifest.csv")
    selected_model, model_summary = select_family(manifest)
    model_summary.to_csv(RESULTS / "aft_family_selection.csv", index=False)
    coefficients = coefficient_rows(selected_model)
    coefficients.to_csv(RESULTS / f"standardized_coefficients_{selected_model}.csv", index=False)
    plot_coefficient_heatmaps(coefficients, selected_model)
    plot_concordance(manifest, selected_model)
    plot_selected_concordance_two_columns(manifest, selected_model)
    print("selected_model", selected_model)
    print(model_summary.to_string(index=False))
    print("plots", sorted(str(path) for path in PLOTS.glob("*.png")))


if __name__ == "__main__":
    main()
