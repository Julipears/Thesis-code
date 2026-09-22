"""Compute paired log-likelihood differences relative to the V5 null model."""

from pathlib import Path

import pandas as pd


INPUT = Path("open_interest_figures/native/aft_v5_univariate_baseline_results/log_likelihood_by_univariate_model.csv")
OUTPUT_DIR = Path("open_interest_figures/native/aft_v5_univariate_baseline_results")
SPECS = ("shock_size", "volatility", "spread")
ORDER = [
    ("BTC", "linear", "spot"), ("BTC", "linear", "perp"),
    ("BTC", "inverse", "spot"), ("BTC", "inverse", "perp"),
    ("ETH", "linear", "spot"), ("ETH", "linear", "perp"),
    ("ETH", "inverse", "spot"), ("ETH", "inverse", "perp"),
]


def main() -> None:
    data = pd.read_csv(INPUT, keep_default_na=False)
    key = ["market", "source_file", "currency", "contract_type", "origin"]
    ll = data.pivot_table(index=key, columns="specification", values="log_likelihood", aggfunc="first")
    for specification in SPECS:
        ll[f"difference_{specification}_minus_null"] = ll[specification] - ll["null_model"]

    rows = []
    for specification in SPECS:
        column = f"difference_{specification}_minus_null"
        grouped = ll.reset_index().groupby(["currency", "contract_type", "origin"])[column]
        summary = grouped.agg(
            paired_months="count",
            mean_difference="mean",
            median_difference="median",
            std_difference="std",
            min_difference="min",
            max_difference="max",
        ).reset_index()
        summary.insert(0, "specification", specification)
        rows.append(summary)
    summary = pd.concat(rows, ignore_index=True)
    summary.to_csv(OUTPUT_DIR / "log_likelihood_difference_from_null_summary.csv", index=False)

    differences = ll.reset_index()[key + [f"difference_{s}_minus_null" for s in SPECS]]
    differences.to_csv(OUTPUT_DIR / "log_likelihood_difference_from_null_by_model.csv", index=False)

    lookup = summary.set_index(["specification", "currency", "contract_type", "origin"])
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption[Average log-likelihood difference from the null AFT model]",
        r"{\small Average paired monthly difference $\ell_{\mathrm{univariate}}-\ell_{\mathrm{null}}$ for the V5 log-logistic AFT models. Positive values indicate an improved fit relative to the intercept-only specification.}",
        r"\begin{tabular}{|l|cc|cc|cc|cc|}",
        r"\hline",
        r"& \multicolumn{4}{c|}{\textbf{BTC}} & \multicolumn{4}{c|}{\textbf{ETH}} \\",
        r"\cline{2-9}",
        r"& \multicolumn{2}{c|}{\textbf{Linear}} & \multicolumn{2}{c|}{\textbf{Inverse}} & \multicolumn{2}{c|}{\textbf{Linear}} & \multicolumn{2}{c|}{\textbf{Inverse}} \\",
        r"\hline",
        r"\textbf{Specification} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} \\",
        r"\hline",
    ]
    labels = {
        "shock_size": r"shock\_size\_bps",
        "volatility": r"vol\_spot\_5min\_pct",
        "spread": r"spread\_spot\_5min",
    }
    for specification in SPECS:
        values = [
            f"{lookup.loc[(specification, *key), 'mean_difference']:.1f}"
            for key in ORDER
        ]
        lines.append(labels[specification] + "\n& " + " & ".join(values) + r" \\")
        lines.append("")
    lines.extend([r"\hline", r"\end{tabular}", r"\label{tab:aft_v5_ll_difference_null}", r"\end{table}"])
    (OUTPUT_DIR / "log_likelihood_difference_from_null_v5_table.tex").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
