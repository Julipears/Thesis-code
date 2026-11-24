"""Summarize V8 univariate AFT coefficients and null-model likelihood gains."""

import argparse
from pathlib import Path

import pandas as pd

from final_code import RESULTS
from final_code.section_5_2.saved_fits import read_monthly_coefficients


OUT = RESULTS / "aft" / "univariate"
MARKETS = {"btc_um": ("BTC", "Linear"), "btc_cm": ("BTC", "Inverse"), "eth_um": ("ETH", "Linear"), "eth_cm": ("ETH", "Inverse")}
ORIGINS = ("spot", "perp")
PERIODS = [
    ("2021-01", "2022-01", "2021"),
    ("2022-01", "2022-07", "-June 2022"),
    ("2022-07", "2023-04", "Jul 2022-Mar 2023"),
    ("2023-04", "2024-01", "Apr- 2023"),
    ("2024-01", "2025-01", "2024"),
    ("2025-01", "2026-01", "2025"),
]
SPECS = {
    "shock_size": ("shock_size_bps", "aft_results_v8_univariate_shock_size", "shock size"),
    "spread": ("spread_spot_5min", "aft_results_v8_univariate_spread", "spread"),
    "log_volume": ("log_volume", "aft_results_v8_univariate_log_volume", "log volume"),
    "basis": ("basis_5min", "aft_results_v8_univariate_basis", "basis"),
}
NULL_DIR = "aft_results_v8_null"
PERIOD_KEYS = [f"{start}_{end}" for start, end, _ in PERIODS]
PERIOD_LABELS = [label for _, _, label in PERIODS]


def period_for(month: str):
    for start, end, label in PERIODS:
        if start <= month < end:
            return f"{start}_{end}", label
    return None


def esc(value: str) -> str:
    return value.replace("_", r"\_")


def read_results(run_dir: Path | None = None):
    metrics, coefficients = [], []
    models = {**SPECS, "intercept_only": ("null", NULL_DIR, "null model")}
    for spec, (covariate, result_dir, display) in models.items():
        table = read_monthly_coefficients(spec, result_dir, run_dir)
        table = table.rename(columns={"first": "origin", "period": "month"})
        table["currency"] = table["market"].map({market: label[0] for market, label in MARKETS.items()})
        table["contract"] = table["market"].map({market: label[1] for market, label in MARKETS.items()})
        periods = table["month"].map(period_for)
        table["period"], table["period_label"] = periods.str[0], periods.str[1]
        table = table.dropna(subset=["period"])
        keys = ["market", "currency", "contract", "origin", "month", "period", "period_label"]
        keep = table[[*keys, "param", "covariate", "coef"]].copy()
        keep["coef"] = pd.to_numeric(keep["coef"], errors="coerce")
        coefficients.append(keep.assign(model_variable=covariate, model_label=display))
        if spec == "intercept_only":
            continue
        monthly = table.drop_duplicates(keys)[[*keys, "log_likelihood", "null_log_likelihood_matching_rows"]].rename(
            columns={"null_log_likelihood_matching_rows": "null_log_likelihood"}
        )
        delta = monthly["log_likelihood"] - monthly["null_log_likelihood"]
        metrics.append(monthly.assign(
            variable=covariate, variable_label=display,
            log_likelihood_difference=delta, two_delta_log_likelihood=2.0 * delta,
            percent_improvement=100.0 * delta / monthly["null_log_likelihood"].abs().where(
                monthly["null_log_likelihood"].ne(0)
            ),
        ))
    return pd.concat(metrics, ignore_index=True), pd.concat(coefficients, ignore_index=True)


def summary_table(metrics: pd.DataFrame, value: str, filename: str, title: str, caption: str, suffix: str = ""):
    group_cols = ["period", "period_label", "currency", "contract", "origin", "variable", "variable_label"]
    summary = metrics.groupby(group_cols, as_index=False).agg(monthly_models=(value, "count"), mean_value=(value, "mean"), median_value=(value, "median"))
    summary.to_csv(OUT / f"{filename}.csv", index=False)
    combined = []
    for currency, contract in [("BTC", "Linear"), ("BTC", "Inverse"), ("ETH", "Linear"), ("ETH", "Inverse")]:
        sub = summary[(summary.currency == currency) & (summary.contract == contract)]
        lines = [r"\begin{table}[H]", r"\centering", f"\\caption[{currency} {contract.lower()} {title}]", f"{{\\small {caption}}}", r"\begin{tabular}{|l|l|rrrrrr|}", r"\hline", r"\textbf{Origin} & \textbf{Univariate variable} & " + " & ".join(rf"\textbf{{{x}}}" for x in PERIOD_LABELS) + r" \\", r"\hline"]
        for oi, origin in enumerate(ORIGINS):
            for vi, (spec, (cov, _, display)) in enumerate(SPECS.items()):
                row = sub[(sub.origin == origin) & (sub.variable == cov)].set_index("period")
                vals = [
                    (f"{row.loc[key, 'mean_value']:.4f}\\%" if value == "percent_improvement" else f"{row.loc[key, 'mean_value']:.4f}")
                    if key in row.index else ""
                    for key in PERIOD_KEYS
                ]
                cell = rf"\multirow{{{len(SPECS)}}}{{*}}{{{origin.title()}}}" if vi == 0 else ""
                lines.append(cell + " & " + esc(display) + " & " + " & ".join(vals) + r" \\")
            if oi == 0: lines.append(r"\hline")
        lines.extend([r"\hline", r"\end{tabular}", rf"\label{{tab:aft_v8_{currency.lower()}_{contract.lower()}_univariate_{suffix}}}", r"\end{table}"])
        table = "\n".join(lines) + "\n"
        stem = f"{filename}_{currency.lower()}_{contract.lower()}"
        (OUT / f"{stem}.tex").write_text(table, encoding="utf-8"); combined.append(table)
    (OUT / f"{filename}.tex").write_text("\n".join(combined), encoding="utf-8")
    return summary


def coefficient_tables(coefficients: pd.DataFrame):
    rows = []
    # Give each univariate coefficient an explicit model prefix.
    for (market, currency, contract, origin, period, period_label, model_variable, model_label, param, covariate), g in coefficients.groupby(["market", "currency", "contract", "origin", "period", "period_label", "model_variable", "model_label", "param", "covariate"]):
        rows.append({"market": market, "currency": currency, "contract": contract, "origin": origin, "period": period, "period_label": period_label, "model_variable": model_variable, "model_label": model_label, "param": param, "covariate": covariate, "coef": g.coef.mean()})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "coefs_by_regime.csv", index=False)
    order = []
    for spec, (cov, _, display) in SPECS.items(): order += [(spec, f"alpha: {cov}"), (spec, "alpha: Intercept"), (spec, "beta: Intercept")]
    order += [("null", "alpha: Intercept"), ("null", "beta: Intercept")]
    labels = {"alpha_": "alpha", "beta_": "beta"}
    combined = []
    for currency, contract in [("BTC", "Linear"), ("BTC", "Inverse"), ("ETH", "Linear"), ("ETH", "Inverse")]:
        sub = summary[(summary.currency == currency) & (summary.contract == contract)]
        lines = [r"\begin{table}[H]", r"\centering", f"\\caption[{currency} {contract.lower()} univariate AFT coefficients by VECM period]", f"{{\\small Period-specific arithmetic means of raw coefficient estimates from the univariate log-logistic AFT models for {currency} {contract.lower()}.}}", r"\begin{tabular}{|l|l|rrrrrr|}", r"\hline", r"\textbf{Origin} & \textbf{Coefficient} & " + " & ".join(rf"\textbf{{{x}}}" for x in PERIOD_LABELS) + r" \\", r"\hline"]
        for oi, origin in enumerate(ORIGINS):
            for ri, (model, label) in enumerate(order):
                if model == "null":
                    param, cov = label.split(": ", 1); display = f"null model ({label})"
                    model_cov = "null"
                else:
                    param, cov = label.split(": ", 1); display = f"{SPECS[model][2]} ({label})"
                    model_cov = SPECS[model][0]
                row = sub[(sub.origin == origin) & (sub.model_variable == model_cov) & (sub.param == param.replace("alpha", "alpha_").replace("beta", "beta_")) & (sub.covariate == cov)].set_index("period")
                vals = [f"{row.loc[key, 'coef']:.4f}" if key in row.index else "" for key in PERIOD_KEYS]
                cell = rf"\multirow{{{len(order)}}}{{*}}{{{origin.title()}}}" if ri == 0 else ""
                lines.append(cell + " & " + esc(display) + " & " + " & ".join(vals) + r" \\")
            if oi == 0: lines.append(r"\hline")
        lines.extend([r"\hline", r"\end{tabular}", rf"\label{{tab:aft_v8_{currency.lower()}_{contract.lower()}_univariate_coefficients_vecm_periods}}", r"\end{table}"])
        table = "\n".join(lines) + "\n"
        (OUT / f"{currency.lower()}_{contract.lower()}_coefs.tex").write_text(table, encoding="utf-8")
        combined.append(table)
    (OUT / "coefs_by_regime.tex").write_text("\n".join(combined), encoding="utf-8")


def total_coefficient_table(coefficients: pd.DataFrame):
    group_cols = ["currency", "contract", "origin", "model_variable", "model_label", "param", "covariate"]
    summary = coefficients.groupby(group_cols, as_index=False).agg(monthly_models=("coef", "count"), mean_coefficient=("coef", "mean"))
    summary.to_csv(OUT / "mean_coefs.csv", index=False)
    order = []
    for spec, (cov, _, display) in SPECS.items():
        order += [(spec, f"alpha: {cov}"), (spec, "alpha: Intercept"), (spec, "beta: Intercept")]
    order += [("null", "alpha: Intercept"), ("null", "beta: Intercept")]
    columns = [
        ("BTC", "Linear", "spot"), ("BTC", "Linear", "perp"),
        ("BTC", "Inverse", "spot"), ("BTC", "Inverse", "perp"),
        ("ETH", "Linear", "spot"), ("ETH", "Linear", "perp"),
        ("ETH", "Inverse", "spot"), ("ETH", "Inverse", "perp"),
    ]
    lines = [
        r"\begin{table}[H]", r"\centering",
        r"\caption[Overall-average univariate AFT coefficients]",
        r"{\small Arithmetic means of monthly raw coefficient estimates from the four univariate log-logistic AFT models over all available months.}",
        r"\begin{tabular}{|l|cc|cc|cc|cc|}", r"\hline",
        r" & \multicolumn{4}{c|}{\textbf{BTC}} & \multicolumn{4}{c|}{\textbf{ETH}} \\",
        r"\cline{2-9}",
        r" & \multicolumn{2}{c|}{\textbf{Linear}} & \multicolumn{2}{c|}{\textbf{Inverse}} & \multicolumn{2}{c|}{\textbf{Linear}} & \multicolumn{2}{c|}{\textbf{Inverse}} \\",
        r"\hline",
        r"\textbf{Coefficient} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} \\",
        r"\hline",
    ]
    for model, label in order:
        param, cov = label.split(": ", 1)
        model_cov = "null" if model == "null" else SPECS[model][0]
        display = f"null model ({label})" if model == "null" else f"{SPECS[model][2]} ({label})"
        vals = []
        for currency, contract, origin in columns:
            row = summary[
                (summary.currency == currency) & (summary.contract == contract)
                & (summary.origin == origin) & (summary.model_variable == model_cov)
                & (summary.param == param.replace("alpha", "alpha_").replace("beta", "beta_"))
                & (summary.covariate == cov)
            ]
            number = row.mean_coefficient.iloc[0] if not row.empty else float("nan")
            vals.append(f"{number:.4f}" if pd.notna(number) else "")
        lines.append(esc(display) + " & " + " & ".join(vals) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\label{tab:aft_v8_univariate_coefficients_all_months}", r"\end{table}"])
    (OUT / "mean_coefs.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Read from this local run instead of the bundled coefficients.")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    metrics, coefficients = read_results(args.input)
    metrics.to_csv(OUT / "monthly_likelihood.csv", index=False)
    coefficient_tables(coefficients)
    total_coefficient_table(coefficients)
    summary_table(metrics, "log_likelihood_difference", "ll_change", "univariate log-likelihood differences", "Period-specific arithmetic means of the univariate log-likelihood improvement relative to the matching null model.", "ll_difference")
    summary_table(metrics, "two_delta_log_likelihood", "ll_ratio", "univariate likelihood-ratio statistics", "Period-specific arithmetic means of $2\\Delta\\ell=2(\\ell_{\\mathrm{univariate}}-\\ell_{\\mathrm{null}})$.", "2delta")
    summary_table(metrics, "percent_improvement", "ll_gain_pct", "univariate log-likelihood percentage improvements", "Period-specific arithmetic means of the monthly percentage improvement in log likelihood relative to the matching null model.", "ll_percent")
    all_month = metrics.groupby(["currency", "contract", "origin", "variable", "variable_label"], as_index=False).agg(monthly_models=("percent_improvement", "count"), mean_log_likelihood_difference=("log_likelihood_difference", "mean"), mean_two_delta_log_likelihood=("two_delta_log_likelihood", "mean"), mean_percent_improvement=("percent_improvement", "mean"))
    all_month.to_csv(OUT / "mean_likelihood.csv", index=False)
    write_all_month_table(all_month, "mean_log_likelihood_difference", "mean_change", "All-month univariate log-likelihood differences", "Mean log-likelihood difference relative to the matching null model over all available monthly fits.", "Log-likelihood difference", "univariate_ll_difference_all_months")
    write_all_month_table(all_month, "mean_two_delta_log_likelihood", "mean_ratio", "All-month univariate likelihood-ratio statistics", "Mean $2\\Delta\\ell=2(\\ell_{\\mathrm{univariate}}-\\ell_{\\mathrm{null}})$ over all available monthly fits.", "2 Delta log likelihood", "univariate_2delta_ll_all_months")
    write_all_month_table(all_month, "mean_percent_improvement", "mean_gain_pct", "All-month univariate log-likelihood percentage improvements", "The percentage improvement is calculated for each monthly fit before averaging over all available months.", "Improvement (\\%)", "univariate_ll_percent_all_months")
    print(f"Read {len(metrics)} monthly univariate results and {len(coefficients)} coefficient rows.")
    print(metrics.groupby(["variable", "market"]).size().to_string())


def write_all_month_table(data: pd.DataFrame, value: str, filename: str, title: str, caption: str, value_heading: str, label: str):
    order = [
        ("BTC", "Linear", "spot"), ("BTC", "Linear", "perp"),
        ("BTC", "Inverse", "spot"), ("BTC", "Inverse", "perp"),
        ("ETH", "Linear", "spot"), ("ETH", "Linear", "perp"),
        ("ETH", "Inverse", "spot"), ("ETH", "Inverse", "perp"),
    ]
    lines = [
        r"\begin{table}[H]", r"\centering",
        rf"\caption[{title}]",
        rf"{{\small {caption}}}",
        r"\begin{tabular}{|l|cc|cc|cc|cc|}", r"\hline",
        r" & \multicolumn{4}{c|}{\textbf{BTC}} & \multicolumn{4}{c|}{\textbf{ETH}} \\",
        r"\cline{2-9}",
        r" & \multicolumn{2}{c|}{\textbf{Linear}} & \multicolumn{2}{c|}{\textbf{Inverse}} & \multicolumn{2}{c|}{\textbf{Linear}} & \multicolumn{2}{c|}{\textbf{Inverse}} \\",
        r"\hline",
        rf"\textbf{{Univariate variable}} & \textbf{{Spot}} & \textbf{{Perp}} & \textbf{{Spot}} & \textbf{{Perp}} & \textbf{{Spot}} & \textbf{{Perp}} & \textbf{{Spot}} & \textbf{{Perp}} \\",
        r"\hline",
    ]
    for _, (covariate, _, display) in SPECS.items():
        vals = []
        for currency, contract, origin in order:
            row = data[(data.currency == currency) & (data.contract == contract) & (data.origin == origin) & (data.variable == covariate)]
            number = row[value].iloc[0] if not row.empty else float("nan")
            if pd.isna(number):
                vals.append("")
            elif value == "mean_percent_improvement":
                vals.append(f"{number:.4f}\\%")
            elif value == "mean_log_likelihood_difference":
                vals.append(f"{number:,.0f}")
            elif value == "mean_two_delta_log_likelihood":
                vals.append(f"{number:,.0f}")
            else:
                vals.append(f"{number:.4f}")
        lines.append(esc(display) + " & " + " & ".join(vals) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", rf"\label{{tab:aft_v8_{label}}}", r"\end{table}"])
    (OUT / f"{filename}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
