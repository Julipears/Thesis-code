from pathlib import Path
import pandas as pd

OUT = Path("sa_results/km_v8_final_01/analysis_outputs/aft_v8_univariate_results")


def make(side, side_label, filename, label):
    spread = pd.read_csv(OUT / "spread_midperiod_entire_adjacent_comparison.csv")
    shock = pd.read_csv(OUT / "shock_size_midperiod_entire_adjacent_comparison.csv")
    ratio = f"middle_as_percent_of_{side}"
    rows = []
    for data, coefficient in [(spread, "spread"), (shock, "shock size")]:
        for model in ["full", "univariate"]:
            x = data[data.model == model].groupby("currency", as_index=False)[ratio].mean()
            for _, row in x.iterrows():
                rows.append({"coefficient": coefficient, "model": model, "currency": row.currency, "percentage": row[ratio]})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / f"spread_shock_midperiod_percent_vs_entire_{side}_by_currency.csv", index=False)
    lines = [
        r"\begin{table}[H]", r"\centering",
        rf"\caption[Middle-period coefficients relative to the entire period {side}, averaged by currency]",
        rf"{{\small July 2022--March 2023 alpha coefficients expressed as percentages of the pooled monthly means over {side_label}, averaged across contract type and origin market within each currency.}}",
        r"\begin{tabular}{|l|cc|}", r"\hline",
        r"\textbf{Model and coefficient} & \textbf{BTC} & \textbf{ETH} \\", r"\hline",
    ]
    for coefficient in ["spread", "shock size"]:
        for model, model_label in [("full", "Full seven-variable"), ("univariate", "Univariate")]:
            x = summary[(summary.coefficient == coefficient) & (summary.model == model)].set_index("currency")
            lines.append(f"{model_label} ({coefficient}) & {x.loc['BTC', 'percentage']:.2f}\\% & {x.loc['ETH', 'percentage']:.2f}\\% \\\\")
    lines += [r"\hline", r"\end{tabular}", rf"\label{{tab:{label}}}", r"\end{table}"]
    (OUT / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


make("before", "2021-01--2022-06", "spread_shock_midperiod_percent_vs_entire_before_by_currency.tex", "spread_shock_midperiod_percent_vs_entire_before_by_currency")
make("after", "2023-04--2025-12", "spread_shock_midperiod_percent_vs_entire_after_by_currency.tex", "spread_shock_midperiod_percent_vs_entire_after_by_currency")
