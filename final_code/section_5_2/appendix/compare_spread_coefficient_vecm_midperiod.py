"""Compare the Jul-2022--Mar-2023 spread slope with adjacent VECM periods."""

from pathlib import Path
import re
import pandas as pd


ROOT = Path("sa_results/km_v8_final_01")
OUT = ROOT / "analysis_outputs/aft_v8_univariate_results"
PERIODS = [
    ("2021-01", "2022-01", "2021"),
    ("2022-01", "2022-07", "-June 2022"),
    ("2022-07", "2023-04", "Jul 2022-Mar 2023"),
    ("2023-04", "2024-01", "Apr- 2023"),
    ("2024-01", "2025-01", "2024"),
    ("2025-01", "2026-01", "2025"),
]
MID = "2022-07_2023-04"
MARKETS = {"btc_um": ("BTC", "Linear"), "btc_cm": ("BTC", "Inverse"), "eth_um": ("ETH", "Linear"), "eth_cm": ("ETH", "Inverse")}
ORDER = [(c, k, o) for c, k in [("BTC", "Linear"), ("BTC", "Inverse"), ("ETH", "Linear"), ("ETH", "Inverse")] for o in ["spot", "perp"]]


def period_for(month):
    for start, end, label in PERIODS:
        if start <= month < end:
            return f"{start}_{end}"
    return None


def read_full():
    rows = []
    base = ROOT / "analysis_outputs/aft_v5_without_spot_volatility_results/coefficient_tables"
    for currency, contract in [("BTC", "Linear"), ("BTC", "Inverse"), ("ETH", "Linear"), ("ETH", "Inverse")]:
        d = pd.read_csv(base / f"coefficients_{currency.lower()}_{contract.lower()}_by_vecm_period.csv")
        d = d[d.coefficient == "alpha_|spread_spot_5min"].copy()
        d["model"] = "full"
        rows.append(d.rename(columns={"coefficient": "row", "2022-01_2022-07": "prev", MID: "middle", "2023-04_2024-01": "next"})[["currency", "contract", "origin", "model", "prev", "middle", "next"]])
    return pd.concat(rows, ignore_index=True)


def read_univariate():
    d = pd.read_csv(OUT / "univariate_coefficients_vecm_periods.csv")
    d = d[(d.model_variable == "spread_spot_5min") & (d.param == "alpha_") & (d.covariate == "spread_spot_5min")].copy()
    d = d.groupby(["currency", "contract", "origin", "period"], as_index=False).coef.mean()
    wide = d.pivot(index=["currency", "contract", "origin"], columns="period", values="coef").reset_index()
    wide["model"] = "univariate"
    return wide.rename(columns={"2022-01_2022-07": "prev", MID: "middle", "2023-04_2024-01": "next"})[["currency", "contract", "origin", "model", "prev", "middle", "next"]]


def read_bivariate():
    rows = []
    base = ROOT / "aft_results_v8_bivariate_shock_spread"
    for market, (currency, contract) in MARKETS.items():
        directory = base / market / "loglogistic" / "coefficients"
        for path in sorted(directory.glob("*_20??-??_coefficients.csv")):
            m = re.match(r"(spot|perp)_(\d{4}-\d{2})_coefficients\.csv$", path.name)
            if not m:
                continue
            origin, month = m.groups(); period = period_for(month)
            if period is None:
                continue
            d = pd.read_csv(path)
            x = d[(d.param == "alpha_") & (d.covariate == "spread_spot_5min")]
            if not x.empty:
                rows.append({"currency": currency, "contract": contract, "origin": origin, "period": period, "coef": float(x.coef.iloc[0])})
    d = pd.DataFrame(rows).groupby(["currency", "contract", "origin", "period"], as_index=False).coef.mean()
    wide = d.pivot(index=["currency", "contract", "origin"], columns="period", values="coef").reset_index()
    wide["model"] = "bivariate"
    return wide.rename(columns={"2022-01_2022-07": "prev", MID: "middle", "2023-04_2024-01": "next"})[["currency", "contract", "origin", "model", "prev", "middle", "next"]]


def main():
    frames = [read_full(), read_univariate()]
    d = pd.concat(frames, ignore_index=True)
    d["adjacent_average"] = (d["prev"] + d["next"]) / 2
    d["middle_as_percent_of_adjacent"] = 100 * d["middle"] / d["adjacent_average"]
    d["middle_minus_adjacent_percent"] = 100 * (d["middle"] - d["adjacent_average"]) / d["adjacent_average"].abs()
    d.to_csv(OUT / "spread_coefficient_midperiod_comparison.csv", index=False)
    columns = ORDER
    lines = [
        r"\begin{table}[H]", r"\centering",
        r"\caption[Middle-period spread coefficient relative to adjacent VECM periods]",
        r"{\small The July 2022--March 2023 spread alpha coefficient expressed as a percentage of the arithmetic mean of the January--June 2022 and April--December 2023 coefficients.}",
        r"\begin{tabular}{|l|cc|cc|cc|cc|}", r"\hline",
        r" & \multicolumn{4}{c|}{\textbf{BTC}} & \multicolumn{4}{c|}{\textbf{ETH}} \\",
        r"\cline{2-9}",
        r" & \multicolumn{2}{c|}{\textbf{Linear}} & \multicolumn{2}{c|}{\textbf{Inverse}} & \multicolumn{2}{c|}{\textbf{Linear}} & \multicolumn{2}{c|}{\textbf{Inverse}} \\",
        r"\hline",
        r"\textbf{Model} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} \\",
        r"\hline",
    ]
    for model in ["full", "univariate"]:
        row = d[d.model == model].set_index(["currency", "contract", "origin"])
        vals = [f"{row.loc[key, 'middle_as_percent_of_adjacent']:.2f}\\%" for key in columns]
        label = "Full seven-variable" if model == "full" else "Univariate spread"
        lines.append(label + " & " + " & ".join(vals) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\label{tab:spread_coefficient_midperiod_percent_adjacent}", r"\end{table}"]
    (OUT / "spread_coefficient_midperiod_percent_of_adjacent.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(d.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
