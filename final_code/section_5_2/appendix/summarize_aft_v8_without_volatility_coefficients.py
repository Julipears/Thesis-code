"""Create yearly and overall coefficient tables for the seven-variable V8 AFT fit."""

from pathlib import Path
import re

import pandas as pd


ROOT = Path("sa_results/km_v8_final_01/aft_results_v5_without_spot_volatility")
OUT = Path("sa_results/km_v8_final_01/analysis_outputs/aft_v5_without_spot_volatility_results/coefficient_tables")

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


def read_coefficients() -> pd.DataFrame:
    rows = []
    for market, (currency, contract) in MARKETS.items():
        directory = ROOT / market / "loglogistic" / "coefficients"
        for path in sorted(directory.glob("*_20??-??_coefficients.csv")):
            m = re.match(r"(spot|perp)_(\d{4})-(\d{2})_coefficients\.csv$", path.name)
            if not m:
                continue
            origin, year, month = m.groups()
            df = pd.read_csv(path)
            keep = df[["param", "covariate", "coef"]].copy()
            keep["market"] = market
            keep["currency"] = currency
            keep["contract"] = contract
            keep["origin"] = origin
            keep["year"] = year
            month_key = f"{year}-{month}"
            keep["month"] = month_key
            period = period_for(month_key)
            keep["period"] = period[0] if period else None
            keep["period_label"] = period[1] if period else None
            keep["key"] = keep["param"].astype(str) + "|" + keep["covariate"].astype(str)
            rows.append(keep)
    if not rows:
        raise FileNotFoundError(f"No coefficient files found below {ROOT}")
    out = pd.concat(rows, ignore_index=True)
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


def write_yearly_tables(data: pd.DataFrame):
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
        stem = f"coefficients_{currency.lower()}_{contract.lower()}_by_year"
        result.to_csv(OUT / f"{stem}.csv", index=False)

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
        (OUT / f"{stem}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_total_table(data: pd.DataFrame):
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
    result.to_csv(OUT / "coefficients_total_average.csv", index=False)

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
    (OUT / "coefficients_total_average.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_vecm_period_tables(data: pd.DataFrame):
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
        stem = f"coefficients_{currency.lower()}_{contract.lower()}_by_vecm_period"
        result.to_csv(OUT / f"{stem}.csv", index=False)

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
        (OUT / f"{stem}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = read_coefficients()
    write_yearly_tables(data)
    write_vecm_period_tables(data)
    write_total_table(data)
    print(f"Read {data[['market', 'origin', 'month']].drop_duplicates().shape[0]} monthly fits and {len(data)} coefficient rows.")
    print(f"Wrote coefficient tables to {OUT}")


if __name__ == "__main__":
    main()
