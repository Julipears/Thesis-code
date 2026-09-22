"""Plot V5 coefficient heatmaps, concordance, and 1% significance frequencies."""

from pathlib import Path

import numpy as np
import pandas as pd

import plot_aft_v3_all_covariates_results as base


MARKETS = ("btc_um", "btc_cm", "eth_um", "eth_cm")
base.MARKETS = MARKETS
base.COVARIATES = [
    "basis_5min",
    "fundingRate_bps",
    "log_open_interest",
    "shock_size_bps",
    "spread_spot_5min",
    "log_taker_long_short",
    "vol_spot_5min_pct",
    "log_volume",
]
base.INPUT_DIR_NAME = "aft_data_liquidity_v5_log_covariates"
base.RESULTS_DIR_NAME = "aft_results_v5_log_covariates"
base.PLOTS = Path("open_interest_figures/native/aft_v5_log_covariate_results_plots")


def significance_table() -> None:
    rows = []
    for market in MARKETS:
        manifest_path = Path(f"sa_{market}") / base.RESULTS_DIR_NAME / "aft_fit_manifest.csv"
        manifest = pd.read_csv(manifest_path)
        parsed = manifest["source_file"].str.removesuffix(".parquet").str.split("_", n=1, expand=True)
        manifest["first"] = parsed[0]
        complete = manifest[manifest.status.isin(["complete", "skipped_existing"])]
        coefficient_dir = Path(f"sa_{market}") / base.RESULTS_DIR_NAME / "loglogistic" / "coefficients"
        for first in ("spot", "perp"):
            subset = complete[complete["first"] == first]
            for covariate in base.COVARIATES:
                p_values = []
                for source_file in subset.source_file:
                    path = coefficient_dir / f"{Path(source_file).stem}_coefficients.csv"
                    if not path.exists():
                        continue
                    coeff = pd.read_csv(path)
                    values = pd.to_numeric(
                        coeff.loc[coeff.covariate == covariate, "p"], errors="coerce"
                    )
                    if len(values):
                        p_values.append(float(values.iloc[0]))
                p_values = np.asarray(p_values, dtype=float)
                rows.append({
                    "market": market[:3].upper(),
                    "contract_type": "linear" if market.endswith("um") else "inverse",
                    "origin": first,
                    "variable": covariate,
                    "significant_months_p_lt_0_01": int(np.isfinite(p_values).sum() and np.sum(p_values < 0.01)),
                    "fitted_months": int(len(subset)),
                    "p_value_available_months": int(np.isfinite(p_values).sum()),
                })
    table = pd.DataFrame(rows)
    table["significance_percent"] = 100 * table["significant_months_p_lt_0_01"] / table["fitted_months"]
    table.to_csv(base.PLOTS / "significance_1pct_by_currency_contract_origin.csv", index=False)

    order = [
        ("BTC", "linear", "spot"), ("BTC", "linear", "perp"),
        ("BTC", "inverse", "spot"), ("BTC", "inverse", "perp"),
        ("ETH", "linear", "spot"), ("ETH", "linear", "perp"),
        ("ETH", "inverse", "spot"), ("ETH", "inverse", "perp"),
    ]
    wide = table.pivot_table(
        index="variable", columns=["market", "contract_type", "origin"],
        values="significance_percent",
    ).reindex(columns=order)
    # Present covariates from the highest to lowest average significance
    # frequency across all eight currency/contract/origin columns.
    wide = wide.loc[wide.mean(axis=1).sort_values(ascending=False).index]
    wide.to_csv(base.PLOTS / "significance_1pct_wide.csv")

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption[One-percent significance frequency for V5 coefficients]",
        r"{\small Percentage of completed monthly V5 log-logistic AFT fits in which each coefficient is statistically significant at the 1\% level ($p<0.01$). Open interest, volume, and the taker long--short ratio are log1p-transformed; the price variable is excluded.}",
        r"\begin{tabular}{|l|cc|cc|cc|cc|}",
        r"\hline",
        r"& \multicolumn{4}{c|}{\textbf{BTC}} & \multicolumn{4}{c|}{\textbf{ETH}} \\",
        r"\cline{2-9}",
        r"& \multicolumn{2}{c|}{\textbf{Linear}} & \multicolumn{2}{c|}{\textbf{Inverse}} & \multicolumn{2}{c|}{\textbf{Linear}} & \multicolumn{2}{c|}{\textbf{Inverse}} \\",
        r"\hline",
        r"\textbf{Variable} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} \\",
        r"\hline",
    ]
    for covariate in wide.index:
        vals = []
        for key in order:
            vals.append(f"{wide.loc[covariate, key]:.1f}\\%")
        label = covariate.replace("_", r"\_")
        lines.append(label + "\n& " + " & ".join(vals) + r" \\")
        lines.append("")
    lines.extend([r"\hline", r"\end{tabular}", r"\label{tab:significance_1pct_v5}", r"\end{table}"])
    (base.PLOTS / "significance_1pct_v5_table.tex").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    base.main()
    significance_table()
