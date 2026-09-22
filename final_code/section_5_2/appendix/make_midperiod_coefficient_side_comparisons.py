"""Create separate LaTeX tables comparing the middle VECM period with each adjacent period."""

from pathlib import Path
import pandas as pd


OUT = Path("sa_results/km_v8_final_01/analysis_outputs/aft_v8_univariate_results")
ORDER = [("BTC", "Linear", "spot"), ("BTC", "Linear", "perp"), ("BTC", "Inverse", "spot"), ("BTC", "Inverse", "perp"), ("ETH", "Linear", "spot"), ("ETH", "Linear", "perp"), ("ETH", "Inverse", "spot"), ("ETH", "Inverse", "perp")]


def make_table(data, coefficient, side, side_label, filename, label):
    ratio_col = f"middle_as_percent_of_{side}"
    lines = [
        r"\begin{table}[H]", r"\centering",
        rf"\caption[Middle-period {coefficient} coefficient relative to {side_label}]",
        rf"{{\small The July 2022--March 2023 {coefficient} alpha coefficient expressed as a percentage of the {side_label} coefficient.}}",
        r"\begin{tabular}{|l|cc|cc|cc|cc|}", r"\hline",
        r" & \multicolumn{4}{c|}{\textbf{BTC}} & \multicolumn{4}{c|}{\textbf{ETH}} \\",
        r"\cline{2-9}",
        r" & \multicolumn{2}{c|}{\textbf{Linear}} & \multicolumn{2}{c|}{\textbf{Inverse}} & \multicolumn{2}{c|}{\textbf{Linear}} & \multicolumn{2}{c|}{\textbf{Inverse}} \\",
        r"\hline",
        r"\textbf{Model} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} \\",
        r"\hline",
    ]
    for model, row_label in [("full", "Full seven-variable"), ("univariate", f"Univariate {coefficient}")]:
        x = data[data.model == model].set_index(["currency", "contract", "origin"])
        vals = [f"{x.loc[k, ratio_col]:.2f}\\%" for k in ORDER]
        lines.append(row_label + " & " + " & ".join(vals) + r" \\")
    lines += [r"\hline", r"\end{tabular}", rf"\label{{tab:{label}}}", r"\end{table}"]
    (OUT / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    outputs = []
    for stem, coefficient in [("spread_coefficient_midperiod_comparison", "spread"), ("shock_size_coefficient_midperiod_comparison", "shock size")]:
        d = pd.read_csv(OUT / f"{stem}.csv")
        d["middle_as_percent_of_prev"] = 100 * d.middle / d.prev
        d["middle_as_percent_of_next"] = 100 * d.middle / d.next
        d.to_csv(OUT / f"{stem}_each_side.csv", index=False)
        prefix = stem.replace("_comparison", "")
        make_table(d, coefficient, "prev", "January--June 2022", f"{prefix}_percent_vs_jan_jun_2022.tex", f"{prefix}_percent_vs_jan_jun_2022")
        make_table(d, coefficient, "next", "April--December 2023", f"{prefix}_percent_vs_apr_dec_2023.tex", f"{prefix}_percent_vs_apr_dec_2023")
        outputs.append(d)
    pd.concat(outputs, ignore_index=True).to_csv(OUT / "spread_shock_midperiod_each_side_comparisons.csv", index=False)


if __name__ == "__main__":
    main()
