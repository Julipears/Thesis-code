"""Format all-month average likelihood-ratio statistics for V5 univariate AFT fits."""

from pathlib import Path

import pandas as pd


OUT = Path("open_interest_figures/native/aft_v5_log_covariate_results_plots")
MONTHLY = OUT / "aft_v5_univariate_loglik_improvements_monthly.csv"
VARIABLES = ["shock_size_bps", "spread_spot_5min", "vol_spot_5min_pct"]
ORDER = [
    ("BTC", "linear", "spot"), ("BTC", "linear", "perp"),
    ("BTC", "inverse", "spot"), ("BTC", "inverse", "perp"),
    ("ETH", "linear", "spot"), ("ETH", "linear", "perp"),
    ("ETH", "inverse", "spot"), ("ETH", "inverse", "perp"),
]


def main() -> None:
    monthly = pd.read_csv(MONTHLY)
    monthly["two_delta_log_likelihood"] = 2.0 * monthly["log_likelihood_difference"]
    summary = monthly.groupby(
        ["currency", "contract_type", "origin", "variable"], as_index=False
    ).agg(
        monthly_models=("two_delta_log_likelihood", "count"),
        mean_2_delta_log_likelihood=("two_delta_log_likelihood", "mean"),
        median_2_delta_log_likelihood=("two_delta_log_likelihood", "median"),
    )
    summary.to_csv(OUT / "aft_v5_univariate_2delta_loglik_all_months.csv", index=False)

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption[All-month univariate likelihood-ratio statistics]",
        r"{\small Mean monthly likelihood-ratio statistics $2\Delta\ell=2(\ell_{\mathrm{univariate}}-\ell_{\mathrm{null}})$ for the three univariate V5 log-logistic AFT models. The statistic is calculated for each monthly fit before averaging over 2021-01--2025-12.}",
        r"\begin{tabular}{|l|cc|cc|cc|cc|}",
        r"\hline",
        r" & \multicolumn{4}{c|}{\textbf{BTC}} & \multicolumn{4}{c|}{\textbf{ETH}} \\",
        r"\cline{2-9}",
        r" & \multicolumn{2}{c|}{\textbf{Linear}} & \multicolumn{2}{c|}{\textbf{Inverse}} & \multicolumn{2}{c|}{\textbf{Linear}} & \multicolumn{2}{c|}{\textbf{Inverse}} \\",
        r"\hline",
        r"\textbf{Univariate variable} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} \\",
        r"\hline",
    ]
    for variable in VARIABLES:
        values = []
        for currency, contract_type, origin in ORDER:
            row = summary[
                summary.currency.eq(currency)
                & summary.contract_type.eq(contract_type)
                & summary.origin.eq(origin)
                & summary.variable.eq(variable)
            ]
            value = row["mean_2_delta_log_likelihood"].iloc[0] if not row.empty else float("nan")
            values.append(f"{value:,.0f}" if pd.notna(value) else "")
        lines.append(variable.replace("_", r"\_") + " & " + " & ".join(values) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\label{tab:aft_v5_univariate_2delta_loglik_all_months}", r"\end{table}"])
    (OUT / "aft_v5_univariate_2delta_loglik_all_months_table.tex").write_text("\n".join(lines), encoding="utf-8")
    print("Saved", OUT / "aft_v5_univariate_2delta_loglik_all_months_table.tex")


if __name__ == "__main__":
    main()
