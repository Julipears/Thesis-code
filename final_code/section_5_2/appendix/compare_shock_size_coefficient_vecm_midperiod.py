"""Compare the middle-period shock-size alpha coefficient with adjacent periods."""

from pathlib import Path
import pandas as pd

ROOT = Path("sa_results/km_v8_final_01")
FULL = ROOT / "analysis_outputs/aft_v5_without_spot_volatility_results/coefficient_tables"
UNIV = ROOT / "analysis_outputs/aft_v8_univariate_results/univariate_coefficients_vecm_periods.csv"
OUT = ROOT / "analysis_outputs/aft_v8_univariate_results"
MID = "2022-07_2023-04"
ORDER = [("BTC", "Linear", "spot"), ("BTC", "Linear", "perp"), ("BTC", "Inverse", "spot"), ("BTC", "Inverse", "perp"), ("ETH", "Linear", "spot"), ("ETH", "Linear", "perp"), ("ETH", "Inverse", "spot"), ("ETH", "Inverse", "perp")]


def full():
    rows = []
    for currency, contract in [("BTC", "Linear"), ("BTC", "Inverse"), ("ETH", "Linear"), ("ETH", "Inverse")]:
        d = pd.read_csv(FULL / f"coefficients_{currency.lower()}_{contract.lower()}_by_vecm_period.csv")
        d = d[d.coefficient == "alpha_|shock_size_bps"].copy()
        d["model"] = "full"
        rows.append(d.rename(columns={"2022-01_2022-07": "prev", MID: "middle", "2023-04_2024-01": "next"})[["currency", "contract", "origin", "model", "prev", "middle", "next"]])
    return pd.concat(rows, ignore_index=True)


def univariate():
    d = pd.read_csv(UNIV)
    d = d[(d.model_variable == "shock_size_bps") & (d.param == "alpha_") & (d.covariate == "shock_size_bps")]
    d = d.groupby(["currency", "contract", "origin", "period"], as_index=False).coef.mean()
    d = d.pivot(index=["currency", "contract", "origin"], columns="period", values="coef").reset_index()
    d["model"] = "univariate"
    return d.rename(columns={"2022-01_2022-07": "prev", MID: "middle", "2023-04_2024-01": "next"})[["currency", "contract", "origin", "model", "prev", "middle", "next"]]


def main():
    d = pd.concat([full(), univariate()], ignore_index=True)
    d["adjacent_average"] = (d.prev + d.next) / 2
    d["middle_as_percent_of_adjacent"] = 100 * d.middle / d.adjacent_average
    d["middle_minus_adjacent_percent"] = 100 * (d.middle - d.adjacent_average) / d.adjacent_average.abs()
    d.to_csv(OUT / "shock_size_coefficient_midperiod_comparison.csv", index=False)
    lines = [r"\begin{table}[H]", r"\centering", r"\caption[Middle-period shock-size coefficient relative to adjacent VECM periods]", r"{\small The July 2022--March 2023 shock-size alpha coefficient expressed as a percentage of the arithmetic mean of the January--June 2022 and April--December 2023 coefficients.}", r"\begin{tabular}{|l|cc|cc|cc|cc|}", r"\hline", r" & \multicolumn{4}{c|}{\textbf{BTC}} & \multicolumn{4}{c|}{\textbf{ETH}} \\", r"\cline{2-9}", r" & \multicolumn{2}{c|}{\textbf{Linear}} & \multicolumn{2}{c|}{\textbf{Inverse}} & \multicolumn{2}{c|}{\textbf{Linear}} & \multicolumn{2}{c|}{\textbf{Inverse}} \\", r"\hline", r"\textbf{Model} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} \\", r"\hline"]
    for model, label in [("full", "Full seven-variable"), ("univariate", "Univariate shock size")]:
        x = d[d.model == model].set_index(["currency", "contract", "origin"])
        vals = [f"{x.loc[k, 'middle_as_percent_of_adjacent']:.2f}\\%" for k in ORDER]
        lines.append(label + " & " + " & ".join(vals) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\label{tab:shock_size_coefficient_midperiod_percent_adjacent}", r"\end{table}"]
    (OUT / "shock_size_coefficient_midperiod_percent_of_adjacent.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(d.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
