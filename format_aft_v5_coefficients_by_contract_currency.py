"""Format VECM-period AFT coefficients as one table per contract/currency type."""

from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("open_interest_figures/native/aft_v5_log_covariate_results_plots")
SUMMARY = OUTPUT_DIR / "complete_v5_average_coefficients_vecm_periods.csv"
COVARIATES = [
    "basis_5min",
    "fundingRate_bps",
    "log_open_interest",
    "shock_size_bps",
    "spread_spot_5min",
    "log_taker_long_short",
    "vol_spot_5min_pct",
    "log_volume",
]
ROW_ORDER = ["alpha: " + c for c in COVARIATES] + ["alpha: Intercept", "beta: Intercept"]
PERIODS = [
    ("2021-01_2022-01", "2021"),
    ("2022-01_2022-07", "Jan--Jun 2022"),
    ("2022-07_2023-04", "Jul 2022--Mar 2023"),
    ("2023-04_2024-01", "Apr--Dec 2023"),
    ("2024-01_2025-01", "2024"),
    ("2025-01_2026-01", "2025"),
]
TYPES = [
    ("BTC", "linear", "BTC Linear", "btc_linear"),
    ("BTC", "inverse", "BTC Inverse", "btc_inverse"),
    ("ETH", "linear", "ETH Linear", "eth_linear"),
    ("ETH", "inverse", "ETH Inverse", "eth_inverse"),
]


def latex_label(value: str) -> str:
    return value.replace("_", r"\_")


def make_table(summary: pd.DataFrame, currency: str, contract_type: str, title: str) -> str:
    sub = summary[
        summary.currency.eq(currency) & summary.contract_type.eq(contract_type)
    ].copy()
    sub["column_key"] = list(zip(sub.period, sub.origin))
    wide = sub.set_index(["row_label", "origin", "period"])["mean_coefficient"].unstack("period")
    wide = wide.reindex(index=pd.MultiIndex.from_product([ROW_ORDER, ["spot", "perp"]]))
    wide = wide.reindex(columns=[p[0] for p in PERIODS])

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption[{title} AFT coefficients by VECM period]",
        rf"{{\small Period-specific arithmetic means of raw coefficient estimates from the complete eight-covariate V5 log-logistic AFT model for {title}. Rows are separated by shock-origin market.}}",
        r"\begin{tabular}{|l|l|rrrrrr|}",
        r"\hline",
        r"\textbf{Origin market} & \textbf{Coefficient} & \textbf{2021} & \textbf{Jan--Jun 2022} & \textbf{Jul 2022--Mar 2023} & \textbf{Apr--Dec 2023} & \textbf{2024} & \textbf{2025} \\",
        r"\hline",
    ]
    for origin in ["spot", "perp"]:
        origin_label = "Spot" if origin == "spot" else "Perp"
        for i, row_label in enumerate(ROW_ORDER):
            vals = []
            for period, _label in PERIODS:
                value = wide.loc[(row_label, origin), period]
                vals.append(f"{value:.4f}" if pd.notna(value) else "")
            section = rf"\multirow{{{len(ROW_ORDER)}}}{{*}}{{{origin_label}}}" if i == 0 else ""
            lines.append(section + " & " + latex_label(row_label) + " & " + " & ".join(vals) + r" \\")
        lines.append(r"\hline")
    lines.extend([r"\end{tabular}", rf"\label{{tab:aft_v5_{currency.lower()}_{contract_type}_coefficients_vecm_periods}}", r"\end{table}"])
    return "\n".join(lines)


def main() -> None:
    summary = pd.read_csv(SUMMARY)
    combined = []
    for currency, contract_type, title, stem in TYPES:
        table = make_table(summary, currency, contract_type, title)
        path = OUTPUT_DIR / f"complete_v5_{stem}_coefficients_vecm_periods_table.tex"
        path.write_text(table, encoding="utf-8")
        combined.append(table)
    (OUTPUT_DIR / "complete_v5_coefficients_by_contract_currency_vecm_periods_tables.tex").write_text(
        "\n\n".join(combined), encoding="utf-8"
    )
    print("Saved four contract-currency tables and combined LaTeX file to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
