"""Generate Appendix F Tables 18 and 19 from monthly AFT coefficients."""

import argparse
from pathlib import Path
import re

import pandas as pd

ROOT = Path("sa_results/km_v8_final_01")
MARKETS = {"btc_um": ("BTC", "Linear"), "btc_cm": ("BTC", "Inverse"), "eth_um": ("ETH", "Linear"), "eth_cm": ("ETH", "Inverse")}
ORDER = [("BTC", "Linear", "spot"), ("BTC", "Linear", "perp"), ("BTC", "Inverse", "spot"), ("BTC", "Inverse", "perp"), ("ETH", "Linear", "spot"), ("ETH", "Linear", "perp"), ("ETH", "Inverse", "spot"), ("ETH", "Inverse", "perp")]


def bucket(month):
    if month < "2022-07":
        return "before"
    if month < "2023-04":
        return "middle"
    return "after"


def read_monthly(run_dir, model, covariate):
    rows = []
    for market, (currency, contract) in MARKETS.items():
        if model == "full":
            root = run_dir / "aft_results_v5_without_spot_volatility" / market / "loglogistic" / "coefficients"
        else:
            result_dir = {"spread_spot_5min": "aft_results_v8_univariate_spread", "shock_size_bps": "aft_results_v8_univariate_shock_size"}[covariate]
            root = run_dir / result_dir / market / "loglogistic" / "coefficients"
        for path in sorted(root.glob("*_20??-??_coefficients.csv")):
            m = re.match(r"(spot|perp)_(\d{4}-\d{2})_coefficients\.csv$", path.name)
            if not m:
                continue
            origin, month = m.groups()
            d = pd.read_csv(path)
            x = d[(d.param == "alpha_") & (d.covariate == covariate)]
            if not x.empty:
                rows.append({"currency": currency, "contract": contract, "origin": origin, "month": month, "bucket": bucket(month), "coef": float(x.coef.iloc[0]), "model": model})
    return pd.DataFrame(rows)


def write_table(spread, shock, side, side_label, output_dir):
    label = f"spread_shock_midperiod_percent_vs_entire_{side}"
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
    (output_dir / f"{label}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(run_dir, covariate, label):
    detail = pd.concat(
        [read_monthly(run_dir, model, covariate) for model in ("full", "univariate")],
        ignore_index=True,
    )
    if detail.empty:
        raise ValueError(f"No monthly AFT coefficients found for {covariate} under {run_dir}")
    means = detail.groupby(["currency", "contract", "origin", "model", "bucket"])["coef"].mean()
    means = means.unstack("bucket")
    expected = pd.MultiIndex.from_tuples(
        [(*key, model) for key in ORDER for model in ("full", "univariate")],
        names=means.index.names,
    )
    means = means.reindex(index=expected, columns=["before", "middle", "after"])
    if means.isna().any().any():
        raise ValueError(f"Missing monthly AFT coefficient groups for {covariate} under {run_dir}")
    wide = means.reset_index()
    for side in ("before", "after"):
        wide[f"middle_as_percent_of_{side}"] = 100 * wide["middle"] / wide[side]
    wide["coefficient"] = label
    return wide


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output_dir = args.output or args.input / "analysis_outputs/aft_v8_univariate_results"
    spread = summarize(args.input, "spread_spot_5min", "spread")
    shock = summarize(args.input, "shock_size_bps", "shock size")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, summary in (("spread", spread), ("shock_size", shock)):
        summary.to_csv(output_dir / f"{name}_midperiod_entire_adjacent_comparison.csv", index=False)
    pd.concat([spread, shock], ignore_index=True).to_csv(
        output_dir / "midperiod_entire_adjacent_comparison.csv", index=False
    )
    write_table(spread, shock, "before", "2021-01--2022-06", output_dir)
    write_table(spread, shock, "after", "2023-04--2025-12", output_dir)


if __name__ == "__main__":
    main()
