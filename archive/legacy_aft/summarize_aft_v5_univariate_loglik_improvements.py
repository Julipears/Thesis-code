"""Compute monthly univariate AFT log-likelihood improvements over the null model."""

from pathlib import Path

import pandas as pd


MARKETS = ("btc_um", "btc_cm", "eth_um", "eth_cm")
UNIVARIATE = {
    "shock_size_bps": "aft_results_v5_univariate_shock_size",
    "spread_spot_5min": "aft_results_v5_univariate_spread",
    "vol_spot_5min_pct": "aft_results_v5_univariate_volatility",
}
NULL_DIR = "aft_results_v5_null"
OUTPUT_DIR = Path("open_interest_figures/native/aft_v5_log_covariate_results_plots")
PERIODS = [
    ("2021-01", "2022-01", "2021"),
    ("2022-01", "2022-07", "Jan--Jun 2022"),
    ("2022-07", "2023-04", "Jul 2022--Mar 2023"),
    ("2023-04", "2024-01", "Apr--Dec 2023"),
    ("2024-01", "2025-01", "2024"),
    ("2025-01", "2026-01", "2025"),
]
TYPES = [
    ("BTC", "linear", "BTC Linear", "btc_linear"),
    ("BTC", "inverse", "BTC Inverse", "btc_inverse"),
    ("ETH", "linear", "ETH Linear", "eth_linear"),
    ("ETH", "inverse", "ETH Inverse", "eth_inverse"),
]


def period_for(month: str):
    for start, end, label in PERIODS:
        if start <= month < end:
            return start + "_" + end, label
    return None


def read_loglik(path: Path) -> float | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path, usecols=["log_likelihood"])
    values = pd.to_numeric(frame["log_likelihood"], errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else None


def main() -> None:
    rows = []
    for market in MARKETS:
        null_root = Path(f"sa_{market}") / NULL_DIR / "loglogistic" / "coefficients"
        currency = market[:3].upper()
        contract_type = "linear" if market.endswith("um") else "inverse"
        for variable, result_dir in UNIVARIATE.items():
            univ_root = Path(f"sa_{market}") / result_dir / "loglogistic" / "coefficients"
            for origin in ("spot", "perp"):
                for month in pd.period_range("2021-01", "2025-12", freq="M").astype(str):
                    period = period_for(month)
                    if period is None:
                        continue
                    stem = f"{origin}_{month}_coefficients.csv"
                    null_ll = read_loglik(null_root / stem)
                    univ_ll = read_loglik(univ_root / stem)
                    if null_ll is None or univ_ll is None or null_ll == 0:
                        continue
                    difference = univ_ll - null_ll
                    pct = 100.0 * difference / abs(null_ll)
                    rows.append({
                        "currency": currency,
                        "contract_type": contract_type,
                        "origin": origin,
                        "variable": variable,
                        "month": month,
                        "period": period[0],
                        "period_label": period[1],
                        "null_log_likelihood": null_ll,
                        "univariate_log_likelihood": univ_ll,
                        "log_likelihood_difference": difference,
                        "pct_improvement": pct,
                    })

    monthly = pd.DataFrame(rows)
    monthly.to_csv(OUTPUT_DIR / "aft_v5_univariate_loglik_improvements_monthly.csv", index=False)

    group_cols = ["period", "period_label", "currency", "contract_type", "origin", "variable"]
    summary = monthly.groupby(group_cols, as_index=False).agg(
        monthly_models=("pct_improvement", "count"),
        mean_pct_improvement=("pct_improvement", "mean"),
        median_pct_improvement=("pct_improvement", "median"),
        mean_log_likelihood_difference=("log_likelihood_difference", "mean"),
        mean_null_log_likelihood=("null_log_likelihood", "mean"),
        mean_univariate_log_likelihood=("univariate_log_likelihood", "mean"),
    )
    summary.to_csv(OUTPUT_DIR / "aft_v5_univariate_loglik_improvements_vecm_periods.csv", index=False)

    # Overall summary: calculate the percentage for each month first, then
    # average those percentages over the full 2021-01--2025-12 sample.
    overall_group_cols = ["currency", "contract_type", "origin", "variable"]
    overall = monthly.groupby(overall_group_cols, as_index=False).agg(
        monthly_models=("pct_improvement", "count"),
        mean_pct_improvement=("pct_improvement", "mean"),
        median_pct_improvement=("pct_improvement", "median"),
        mean_log_likelihood_difference=("log_likelihood_difference", "mean"),
        mean_null_log_likelihood=("null_log_likelihood", "mean"),
        mean_univariate_log_likelihood=("univariate_log_likelihood", "mean"),
    )
    overall.to_csv(OUTPUT_DIR / "aft_v5_univariate_loglik_improvements_all_months.csv", index=False)

    # Separate table for each currency/contract type, with origin-market sections.
    period_keys = [f"{start}_{end}" for start, end, _ in PERIODS]
    period_labels = [label for _, _, label in PERIODS]
    combined = []
    for currency, contract_type, title, stem in TYPES:
        sub = summary[
            summary.currency.eq(currency) & summary.contract_type.eq(contract_type)
        ]
        lines = [
            r"\begin{table}[H]",
            r"\centering",
            rf"\caption[{title} univariate log-likelihood improvements]",
            rf"{{\small Mean monthly improvement in log likelihood relative to the matching null model for the three univariate V5 log-logistic AFT models. The percentage is calculated for each monthly fit before averaging within each VECM period.}}",
            r"\begin{tabular}{|l|l|rrrrrr|}",
            r"\hline",
            r"\textbf{Origin market} & \textbf{Univariate variable} & \textbf{2021} & \textbf{Jan--Jun 2022} & \textbf{Jul 2022--Mar 2023} & \textbf{Apr--Dec 2023} & \textbf{2024} & \textbf{2025} \\",
            r"\hline",
        ]
        for origin in ("spot", "perp"):
            origin_label = "Spot" if origin == "spot" else "Perp"
            variables = list(UNIVARIATE)
            for i, variable in enumerate(variables):
                row = sub[(sub.origin.eq(origin)) & sub.variable.eq(variable)].set_index("period")
                vals = [
                    f"{row.loc[key, 'mean_pct_improvement']:.4f}\\%" if key in row.index else ""
                    for key in period_keys
                ]
                section = rf"\multirow{{{len(variables)}}}{{*}}{{{origin_label}}}" if i == 0 else ""
                lines.append(section + " & " + variable.replace("_", r"\_") + " & " + " & ".join(vals) + r" \\")
            lines.append(r"\hline")
        lines.extend([r"\end{tabular}", rf"\label{{tab:aft_v5_{currency.lower()}_{contract_type}_univariate_ll_improvement}}", r"\end{table}"])
        table = "\n".join(lines)
        (OUTPUT_DIR / f"aft_v5_{stem}_univariate_loglik_improvements_table.tex").write_text(table, encoding="utf-8")
        combined.append(table)
    (OUTPUT_DIR / "aft_v5_univariate_loglik_improvements_tables.tex").write_text("\n\n".join(combined), encoding="utf-8")

    # Separate all-months table for each currency/contract type.
    combined_overall = []
    for currency, contract_type, title, stem in TYPES:
        sub = overall[
            overall.currency.eq(currency) & overall.contract_type.eq(contract_type)
        ]
        lines = [
            r"\begin{table}[H]",
            r"\centering",
            rf"\caption[{title} all-month univariate log-likelihood improvements]",
            rf"{{\small Mean percentage improvement in log likelihood relative to the matching null model for the three univariate V5 log-logistic AFT models. The percentage is calculated for each monthly fit before averaging over all months from 2021-01 through 2025-12.}}",
            r"\begin{tabular}{|l|l|r|}",
            r"\hline",
            r"\textbf{Origin market} & \textbf{Univariate variable} & \textbf{All months (\%)} \\",
            r"\hline",
        ]
        for origin in ("spot", "perp"):
            origin_label = "Spot" if origin == "spot" else "Perp"
            variables = list(UNIVARIATE)
            for i, variable in enumerate(variables):
                row = sub[(sub.origin.eq(origin)) & sub.variable.eq(variable)]
                value = row["mean_pct_improvement"].iloc[0] if not row.empty else float("nan")
                formatted = f"{value:.4f}\\%" if pd.notna(value) else ""
                section = rf"\multirow{{{len(variables)}}}{{*}}{{{origin_label}}}" if i == 0 else ""
                lines.append(section + " & " + variable.replace("_", r"\_") + " & " + formatted + r" \\")
            lines.append(r"\hline")
        lines.extend([r"\end{tabular}", rf"\label{{tab:aft_v5_{currency.lower()}_{contract_type}_univariate_ll_improvement_all_months}}", r"\end{table}"])
        table = "\n".join(lines)
        (OUTPUT_DIR / f"aft_v5_{stem}_univariate_loglik_improvements_all_months_table.tex").write_text(table, encoding="utf-8")
        combined_overall.append(table)
    (OUTPUT_DIR / "aft_v5_univariate_loglik_improvements_all_months_tables.tex").write_text("\n\n".join(combined_overall), encoding="utf-8")

    # One combined table matching the contract/origin layout used elsewhere.
    order = [
        ("BTC", "linear", "spot"), ("BTC", "linear", "perp"),
        ("BTC", "inverse", "spot"), ("BTC", "inverse", "perp"),
        ("ETH", "linear", "spot"), ("ETH", "linear", "perp"),
        ("ETH", "inverse", "spot"), ("ETH", "inverse", "perp"),
    ]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption[All-month univariate log-likelihood improvements]",
        r"{\small Mean percentage improvement in log likelihood relative to the matching null model for the univariate V5 log-logistic AFT models. The percentage is calculated for each monthly fit before averaging over all months from 2021-01 through 2025-12.}",
        r"\begin{tabular}{|l|cc|cc|cc|cc|}",
        r"\hline",
        r" & \multicolumn{4}{c|}{\textbf{BTC}} & \multicolumn{4}{c|}{\textbf{ETH}} \\",
        r"\cline{2-9}",
        r" & \multicolumn{2}{c|}{\textbf{Linear}} & \multicolumn{2}{c|}{\textbf{Inverse}} & \multicolumn{2}{c|}{\textbf{Linear}} & \multicolumn{2}{c|}{\textbf{Inverse}} \\",
        r"\hline",
        r"\textbf{Univariate variable} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} \\",
        r"\hline",
    ]
    variables = list(UNIVARIATE)
    for variable in variables:
        vals = []
        for currency, contract_type, origin in order:
            row = overall[
                overall.currency.eq(currency)
                & overall.contract_type.eq(contract_type)
                & overall.origin.eq(origin)
                & overall.variable.eq(variable)
            ]
            value = row["mean_pct_improvement"].iloc[0] if not row.empty else float("nan")
            vals.append(f"{value:.4f}\\%" if pd.notna(value) else "")
        lines.append(variable.replace("_", r"\_") + " & " + " & ".join(vals) + r" \\")
    lines.extend([r"\end{tabular}", r"\label{tab:aft_v5_univariate_ll_improvement_all_months}", r"\end{table}"])
    (OUTPUT_DIR / "aft_v5_univariate_loglik_improvements_all_months_combined_table.tex").write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved {len(monthly)} monthly comparisons and period summaries to {OUTPUT_DIR}")
    print(summary.groupby(["currency", "contract_type", "origin", "variable"]).monthly_models.max().to_string())


if __name__ == "__main__":
    main()
