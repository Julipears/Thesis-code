"""Compare the middle VECM-period coefficient with pooled periods before and after."""

from pathlib import Path
import re
import pandas as pd

ROOT = Path("sa_results/km_v8_final_01")
OUT = ROOT / "analysis_outputs/aft_v8_univariate_results"
MARKETS = {"btc_um": ("BTC", "Linear"), "btc_cm": ("BTC", "Inverse"), "eth_um": ("ETH", "Linear"), "eth_cm": ("ETH", "Inverse")}
ORDER = [("BTC", "Linear", "spot"), ("BTC", "Linear", "perp"), ("BTC", "Inverse", "spot"), ("BTC", "Inverse", "perp"), ("ETH", "Linear", "spot"), ("ETH", "Linear", "perp"), ("ETH", "Inverse", "spot"), ("ETH", "Inverse", "perp")]


def bucket(month):
    if month < "2022-07":
        return "before"
    if month < "2023-04":
        return "middle"
    return "after"


def read_monthly(model, covariate):
    rows = []
    for market, (currency, contract) in MARKETS.items():
        if model == "full":
            root = ROOT / "aft_results_v5_without_spot_volatility" / market / "loglogistic" / "coefficients"
        else:
            result_dir = {"spread_spot_5min": "aft_results_v8_univariate_spread", "shock_size_bps": "aft_results_v8_univariate_shock_size"}[covariate]
            root = ROOT / result_dir / market / "loglogistic" / "coefficients"
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


def write_table(summary, coefficient, side, side_label, filename, label):
    ratio = f"middle_as_percent_of_{side}"
    lines = [r"\begin{table}[H]", r"\centering", rf"\caption[Middle-period {coefficient} coefficient relative to the entire period {side}]", rf"{{\small The July 2022--March 2023 {coefficient} alpha coefficient expressed as a percentage of the pooled monthly mean over {side_label}.}}", r"\begin{tabular}{|l|cc|cc|cc|cc|}", r"\hline", r" & \multicolumn{4}{c|}{\textbf{BTC}} & \multicolumn{4}{c|}{\textbf{ETH}} \\", r"\cline{2-9}", r" & \multicolumn{2}{c|}{\textbf{Linear}} & \multicolumn{2}{c|}{\textbf{Inverse}} & \multicolumn{2}{c|}{\textbf{Linear}} & \multicolumn{2}{c|}{\textbf{Inverse}} \\", r"\hline", r"\textbf{Model} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} \\", r"\hline"]
    for model, label_text in [("full", "Full seven-variable"), ("univariate", f"Univariate {coefficient}")]:
        x = summary[summary.model == model].set_index(["currency", "contract", "origin"])
        vals = [f"{x.loc[k, ratio]:.2f}\\%" for k in ORDER]
        lines.append(label_text + " & " + " & ".join(vals) + r" \\")
    lines += [r"\hline", r"\end{tabular}", rf"\label{{tab:{label}}}", r"\end{table}"]
    (OUT / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    all_rows = []
    for covariate, coefficient in [("spread_spot_5min", "spread"), ("shock_size_bps", "shock size")]:
        frames = [read_monthly("full", covariate), read_monthly("univariate", covariate)]
        detail = pd.concat(frames, ignore_index=True)
        means = detail.groupby(["currency", "contract", "origin", "model", "bucket"], as_index=False).agg(mean_coefficient=("coef", "mean"), monthly_models=("coef", "count"))
        wide = means.pivot(index=["currency", "contract", "origin", "model"], columns="bucket", values="mean_coefficient").reset_index()
        wide["middle_as_percent_of_before"] = 100 * wide.middle / wide.before
        wide["middle_as_percent_of_after"] = 100 * wide.middle / wide.after
        wide["coefficient"] = coefficient
        all_rows.append(wide)
        wide.to_csv(OUT / f"{coefficient.replace(' ', '_')}_midperiod_entire_adjacent_comparison.csv", index=False)
        write_table(wide, coefficient, "before", "2021-01--2022-06", f"{coefficient.replace(' ', '_')}_midperiod_percent_vs_entire_before.tex", f"{coefficient.replace(' ', '_')}_midperiod_percent_vs_entire_before")
        write_table(wide, coefficient, "after", "2023-04--2025-12", f"{coefficient.replace(' ', '_')}_midperiod_percent_vs_entire_after.tex", f"{coefficient.replace(' ', '_')}_midperiod_percent_vs_entire_after")
    pd.concat(all_rows, ignore_index=True).to_csv(OUT / "midperiod_entire_adjacent_comparison.csv", index=False)


if __name__ == "__main__":
    main()
