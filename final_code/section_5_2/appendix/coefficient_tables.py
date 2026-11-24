"""Create yearly and overall coefficient tables for the seven-variable V8 AFT fit."""

import argparse
from pathlib import Path

import pandas as pd

from final_code import RESULTS
from final_code.section_5_2.saved_fits import read_monthly_coefficients

OUT = RESULTS / "aft" / "coefficients"

MARKETS = {
    "btc_um": ("BTC", "Linear"),
    "btc_cm": ("BTC", "Inverse"),
    "eth_um": ("ETH", "Linear"),
    "eth_cm": ("ETH", "Inverse"),
}
ORIGINS = ["spot", "perp"]
YEARS = ["2021", "2022", "2023", "2024", "2025"]
COEFFICIENTS = [
    ("alpha_", "basis_5min"),
    ("alpha_", "fundingRate_bps"),
    ("alpha_", "log_open_interest"),
    ("alpha_", "shock_size_bps"),
    ("alpha_", "spread_spot_5min"),
    ("alpha_", "log_taker_long_short"),
    ("alpha_", "log_volume"),
    ("alpha_", "Intercept"),
    ("beta_", "Intercept"),
]
COEF_KEYS = [f"{p}|{c}" for p, c in COEFFICIENTS]
LABELS = {
    f"{param}|{cov}": f"{'alpha' if param == 'alpha_' else 'beta'}: {cov.replace('_', r'\_')}"
    for param, cov in COEFFICIENTS
}

VECM_PERIODS = [
    ("2021-01", "2022-01", "2021"),
    ("2022-01", "2022-07", "-June 2022"),
    ("2022-07", "2023-04", "Jul 2022-Mar 2023"),
    ("2023-04", "2024-01", "Apr- 2023"),
    ("2024-01", "2025-01", "2024"),
    ("2025-01", "2026-01", "2025"),
]


def read_coefficients(run_dir: Path | None = None) -> pd.DataFrame:
    table = read_monthly_coefficients("full", "aft_results_v5_without_spot_volatility", run_dir)
    out = table[["market", "first", "period", "param", "covariate", "coef"]].rename(
        columns={"first": "origin", "period": "month"}
    )
    out["currency"] = out["market"].map({market: label[0] for market, label in MARKETS.items()})
    out["contract"] = out["market"].map({market: label[1] for market, label in MARKETS.items()})
    out["year"] = out["month"].str[:4]
    periods = out["month"].map(period_for)
    out["period"], out["period_label"] = periods.str[0], periods.str[1]
    out["key"] = out["param"].astype(str) + "|" + out["covariate"].astype(str)
    out["coef"] = pd.to_numeric(out["coef"], errors="coerce")
    out = out[out["key"].isin(COEF_KEYS)].copy()
    return out


def fmt(value):
    if pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def period_for(month: str):
    for start, end, label in VECM_PERIODS:
        if start <= month < end:
            return f"{start}_{end}", label
    return None


def write_yearly_tables(data: pd.DataFrame, output: Path = OUT):
    for currency, contract in [("BTC", "Linear"), ("BTC", "Inverse"), ("ETH", "Linear"), ("ETH", "Inverse")]:
        subset = data[(data.currency == currency) & (data.contract == contract)]
        rows = []
        for origin in ORIGINS:
            for key in COEF_KEYS:
                p, cov = key.split("|", 1)
                row = {"currency": currency, "contract": contract, "origin": origin, "coefficient": key}
                for year in YEARS:
                    vals = subset[(subset.origin == origin) & (subset.year == year) & (subset.key == key)]["coef"]
                    row[year] = vals.mean() if len(vals) else float("nan")
                rows.append(row)
        result = pd.DataFrame(rows)
        stem = f"{currency.lower()}_{contract.lower()}_yearly"
        result.to_csv(output / f"{stem}.csv", index=False)

        lines = [
            r"\begin{table}[H]",
            r"\centering",
            f"\\caption[Yearly-average {currency} {contract.lower()} seven-variable AFT coefficients]",
            f"{{\\small Arithmetic means of monthly log-logistic AFT coefficient estimates for {currency} {contract.lower()} contracts, separated by shock-origin market.}}",
            r"\begin{tabular}{|l|l|ccccc|}",
            r"\hline",
            r"\textbf{Origin} & \textbf{Coefficient} & \textbf{2021} & \textbf{2022} & \textbf{2023} & \textbf{2024} & \textbf{2025} \\",
            r"\hline",
        ]
        for origin_index, origin in enumerate(ORIGINS):
            for i, key in enumerate(COEF_KEYS):
                row = result[(result.origin == origin) & (result.coefficient == key)].iloc[0]
                origin_cell = rf"\multirow{{{len(COEF_KEYS)}}}{{*}}{{\textbf{{{origin.title()}}}}}" if i == 0 else ""
                vals = " & ".join(fmt(row[y]) for y in YEARS)
                lines.append(f"{origin_cell} & {LABELS[key]} & {vals} \\\\")
            if origin_index == 0:
                lines.append(r"\cline{1-7}")
        lines += [r"\hline", r"\end{tabular}", f"\\label{{tab:aft_v8_{currency.lower()}_{contract.lower()}_coefficients_by_year}}", r"\end{table}"]
        (output / f"{stem}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_total_table(data: pd.DataFrame, output: Path = OUT):
    columns = [
        ("BTC", "Linear", "spot", "BTC Linear Spot"),
        ("BTC", "Linear", "perp", "BTC Linear Perp"),
        ("BTC", "Inverse", "spot", "BTC Inverse Spot"),
        ("BTC", "Inverse", "perp", "BTC Inverse Perp"),
        ("ETH", "Linear", "spot", "ETH Linear Spot"),
        ("ETH", "Linear", "perp", "ETH Linear Perp"),
        ("ETH", "Inverse", "spot", "ETH Inverse Spot"),
        ("ETH", "Inverse", "perp", "ETH Inverse Perp"),
    ]
    rows = []
    for key in COEF_KEYS:
        row = {"coefficient": key}
        for currency, contract, origin, label in columns:
            vals = data[(data.currency == currency) & (data.contract == contract) & (data.origin == origin) & (data.key == key)]["coef"]
            row[label] = vals.mean() if len(vals) else float("nan")
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(output / "average.csv", index=False)

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption[Overall-average seven-variable AFT coefficients]",
        r"{\small Arithmetic means of all available monthly log-logistic AFT coefficient estimates in the seven-variable specification, reported by currency, contract type, and shock-origin market.}",
        r"\begin{tabular}{|l|cc|cc|cc|cc|}",
        r"\hline",
        r"& \multicolumn{4}{c|}{\textbf{BTC}} & \multicolumn{4}{c|}{\textbf{ETH}} \\",
        r"\cline{2-9}",
        r"& \multicolumn{2}{c|}{\textbf{Linear}} & \multicolumn{2}{c|}{\textbf{Inverse}} & \multicolumn{2}{c|}{\textbf{Linear}} & \multicolumn{2}{c|}{\textbf{Inverse}} \\",
        r"\hline",
        r"\textbf{Coefficient} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} \\",
        r"\hline",
    ]
    for _, row in result.iterrows():
        vals = " & ".join(fmt(row[label]) for _, _, _, label in columns)
        lines.append(f"{LABELS[row.coefficient]} & {vals} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\label{tab:aft_v8_seven_variable_coefficients_total_average}", r"\end{table}"]
    (output / "average.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_vecm_period_tables(data: pd.DataFrame, output: Path = OUT):
    period_keys = [f"{start}_{end}" for start, end, _ in VECM_PERIODS]
    period_labels = [label for _, _, label in VECM_PERIODS]
    for currency, contract in [("BTC", "Linear"), ("BTC", "Inverse"), ("ETH", "Linear"), ("ETH", "Inverse")]:
        subset = data[(data.currency == currency) & (data.contract == contract)]
        rows = []
        for origin in ORIGINS:
            for key in COEF_KEYS:
                row = {"currency": currency, "contract": contract, "origin": origin, "coefficient": key}
                for period_key in period_keys:
                    vals = subset[(subset.origin == origin) & (subset.period == period_key) & (subset.key == key)]["coef"]
                    row[period_key] = vals.mean() if len(vals) else float("nan")
                rows.append(row)
        result = pd.DataFrame(rows)
        stem = f"{currency.lower()}_{contract.lower()}_by_regime"
        result.to_csv(output / f"{stem}.csv", index=False)

        lines = [
            r"\begin{table}[H]",
            r"\centering",
            f"\\caption[{currency} {contract.lower()} AFT coefficients by VECM period]",
            f"{{\\small Period-specific arithmetic means of raw coefficient estimates from the seven-variable log-logistic AFT model for {currency} {contract.lower()}. Rows are separated by shock-origin market.}}",
            r"\begin{tabular}{|l|l|rrrrrr|}",
            r"\hline",
            r"\textbf{Origin} & \textbf{Coefficient} & "
            + " & ".join(rf"\textbf{{{label}}}" for label in period_labels)
            + r" \\",
            r"\hline",
        ]
        for origin_index, origin in enumerate(ORIGINS):
            for i, key in enumerate(COEF_KEYS):
                row = result[(result.origin == origin) & (result.coefficient == key)].iloc[0]
                origin_cell = rf"\multirow{{{len(COEF_KEYS)}}}{{*}}{{{origin.title()}}}" if i == 0 else ""
                vals = " & ".join(fmt(row[p]) for p in period_keys)
                lines.append(f"{origin_cell} & {LABELS[key]} & {vals} \\\\")
            if origin_index == 0:
                lines.append(r"\hline")
        lines += [
            r"\hline",
            r"\end{tabular}",
            f"\\label{{tab:aft_v8_{currency.lower()}_{contract.lower()}_coefficients_vecm_periods}}",
            r"\end{table}",
        ]
        (output / f"{stem}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Read from this local run instead of the bundled coefficients.")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = read_coefficients(args.input)
    write_yearly_tables(data, args.output)
    write_vecm_period_tables(data, args.output)
    write_total_table(data, args.output)
    print(f"Read {data[['market', 'origin', 'month']].drop_duplicates().shape[0]} monthly fits and {len(data)} coefficient rows.")
    print(f"Wrote coefficient tables to {args.output}")


if __name__ == "__main__":
    main()
