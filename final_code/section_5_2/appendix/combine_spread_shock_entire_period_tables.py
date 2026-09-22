from pathlib import Path
import pandas as pd

OUT = Path("sa_results/km_v8_final_01/analysis_outputs/aft_v8_univariate_results")
ORDER = [("BTC", "Linear", "spot"), ("BTC", "Linear", "perp"), ("BTC", "Inverse", "spot"), ("BTC", "Inverse", "perp"), ("ETH", "Linear", "spot"), ("ETH", "Linear", "perp"), ("ETH", "Inverse", "spot"), ("ETH", "Inverse", "perp")]


def make(side, side_label, filename, label):
    spread = pd.read_csv(OUT / "spread_midperiod_entire_adjacent_comparison.csv")
    shock = pd.read_csv(OUT / "shock_size_midperiod_entire_adjacent_comparison.csv")
    ratio = f"middle_as_percent_of_{side}"
    lines = [
        r"\begin{table}[H]", r"\centering",
        rf"\caption[Middle-period coefficients relative to the entire period {side}]",
        rf"{{\small July 2022--March 2023 alpha coefficients expressed as percentages of the pooled monthly means over {side_label}.}}",
        r"\begin{tabular}{|l|cc|cc|cc|cc|}", r"\hline",
        r" & \multicolumn{4}{c|}{\textbf{BTC}} & \multicolumn{4}{c|}{\textbf{ETH}} \\",
        r"\cline{2-9}",
        r" & \multicolumn{2}{c|}{\textbf{Linear}} & \multicolumn{2}{c|}{\textbf{Inverse}} & \multicolumn{2}{c|}{\textbf{Linear}} & \multicolumn{2}{c|}{\textbf{Inverse}} \\",
        r"\hline",
        r"\textbf{Model and coefficient} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} \\",
        r"\hline",
    ]
    for data, coefficient, model_labels in [(spread, "spread", {"full": "Full seven-variable", "univariate": "Univariate"}), (shock, "shock size", {"full": "Full seven-variable", "univariate": "Univariate"})]:
        for model in ["full", "univariate"]:
            x = data[data.model == model].set_index(["currency", "contract", "origin"])
            vals = [f"{x.loc[k, ratio]:.2f}\\%" for k in ORDER]
            lines.append(model_labels[model] + " (" + coefficient + ") & " + " & ".join(vals) + r" \\")
    lines += [r"\hline", r"\end{tabular}", rf"\label{{tab:{label}}}", r"\end{table}"]
    (OUT / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


make("before", "2021-01--2022-06", "spread_shock_midperiod_percent_vs_entire_before.tex", "spread_shock_midperiod_percent_vs_entire_before")
make("after", "2023-04--2025-12", "spread_shock_midperiod_percent_vs_entire_after.tex", "spread_shock_midperiod_percent_vs_entire_after")
