from pathlib import Path
import pandas as pd

OUT = Path("sa_results/km_v8_final_01/analysis_outputs/aft_v8_univariate_results")
ORDER = [("BTC", "Linear", "spot"), ("BTC", "Linear", "perp"), ("BTC", "Inverse", "spot"), ("BTC", "Inverse", "perp"), ("ETH", "Linear", "spot"), ("ETH", "Linear", "perp"), ("ETH", "Inverse", "spot"), ("ETH", "Inverse", "perp")]


def make(stem, title, label):
    d = pd.read_csv(OUT / f"{stem}_each_side.csv")
    lines = [
        r"\begin{table}[H]", r"\centering",
        rf"\caption[{title} relative to each adjacent VECM period]",
        rf"{{\small The July 2022--March 2023 {label} alpha coefficient expressed separately as a percentage of each adjacent VECM-period coefficient.}}",
        r"\begin{tabular}{|l|cc|cc|cc|cc|}", r"\hline",
        r" & \multicolumn{4}{c|}{\textbf{BTC}} & \multicolumn{4}{c|}{\textbf{ETH}} \\",
        r"\cline{2-9}",
        r" & \multicolumn{2}{c|}{\textbf{Linear}} & \multicolumn{2}{c|}{\textbf{Inverse}} & \multicolumn{2}{c|}{\textbf{Linear}} & \multicolumn{2}{c|}{\textbf{Inverse}} \\",
        r"\hline",
        r"\textbf{Model and comparison period} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} \\",
        r"\hline",
    ]
    for model, model_label in [("full", "Full seven-variable"), ("univariate", f"Univariate {label}")]:
        x = d[d.model == model].set_index(["currency", "contract", "origin"])
        for side, side_label in [("prev", "vs Jan--Jun 2022"), ("next", "vs Apr--Dec 2023")]:
            vals = [f"{x.loc[k, f'middle_as_percent_of_{side}']:.2f}\\%" for k in ORDER]
            lines.append(model_label + " (" + side_label + ") & " + " & ".join(vals) + r" \\")
    lines += [r"\hline", r"\end{tabular}", rf"\label{{tab:{stem}_percent_vs_both_adjacent_periods}}", r"\end{table}"]
    (OUT / f"{stem}_percent_vs_both_adjacent_periods.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


make("spread_coefficient_midperiod_comparison", "Middle-period spread coefficient", "spread")
make("shock_size_coefficient_midperiod_comparison", "Middle-period shock-size coefficient", "shock size")
