"""Average complete V5 AFT coefficients over the periods used in the VECM analysis."""

from pathlib import Path

import pandas as pd


MARKETS = ("btc_um", "btc_cm", "eth_um", "eth_cm")
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
RESULTS_DIR = "aft_results_v5_log_covariates"
OUTPUT_DIR = Path("open_interest_figures/native/aft_v5_log_covariate_results_plots")
ORDER = [
    ("BTC", "linear", "spot"), ("BTC", "linear", "perp"),
    ("BTC", "inverse", "spot"), ("BTC", "inverse", "perp"),
    ("ETH", "linear", "spot"), ("ETH", "linear", "perp"),
    ("ETH", "inverse", "spot"), ("ETH", "inverse", "perp"),
]

# Exact period boundaries used by the VECM period-mean summaries.
VECM_PERIODS = [
    ("2021-01", "2022-01", "2021"),
    ("2022-01", "2022-07", "Jan--Jun 2022"),
    ("2022-07", "2023-04", "Jul 2022--Mar 2023"),
    ("2023-04", "2024-01", "Apr--Dec 2023"),
    ("2024-01", "2025-01", "2024"),
    ("2025-01", "2026-01", "2025"),
]
ROW_ORDER = ["alpha: " + c for c in COVARIATES] + ["alpha: Intercept", "beta: Intercept"]


def period_for(month: str) -> tuple[str, str] | None:
    for start, end, label in VECM_PERIODS:
        if start <= month < end:
            return start + "_" + end, label
    return None


def main() -> None:
    rows = []
    for market in MARKETS:
        root = Path(f"sa_{market}") / RESULTS_DIR
        manifest = pd.read_csv(root / "aft_fit_manifest.csv")
        coefficient_dir = root / "loglogistic" / "coefficients"
        complete = manifest[manifest.status.isin(["complete", "skipped_existing"])]
        for source_file in complete.source_file:
            path = coefficient_dir / f"{Path(source_file).stem}_coefficients.csv"
            if not path.exists():
                continue
            coefficient = pd.read_csv(path)
            coefficient = coefficient[
                coefficient.covariate.isin([*COVARIATES, "Intercept"])
            ][["param", "covariate", "coef"]].copy()
            coefficient["coef"] = pd.to_numeric(coefficient["coef"], errors="coerce")
            stem = Path(source_file).stem
            first, month = stem.split("_", 1)
            period = period_for(month)
            if period is None:
                continue
            period_key, period_label = period
            coefficient["market"] = market
            coefficient["currency"] = market[:3].upper()
            coefficient["contract_type"] = "linear" if market.endswith("um") else "inverse"
            coefficient["origin"] = first
            coefficient["month"] = month
            coefficient["period"] = period_key
            coefficient["period_label"] = period_label
            rows.append(coefficient)

    coefficients = pd.concat(rows, ignore_index=True).dropna(subset=["coef"])
    coefficients.to_csv(OUTPUT_DIR / "complete_v5_coefficients_vecm_periods_long.csv", index=False)

    group_cols = ["period", "period_label", "param", "covariate", "currency", "contract_type", "origin"]
    summary = coefficients.groupby(group_cols, as_index=False).agg(
        monthly_models=("coef", "count"),
        mean_coefficient=("coef", "mean"),
        median_coefficient=("coef", "median"),
        std_coefficient=("coef", "std"),
    )
    summary["row_label"] = summary.apply(
        lambda row: f"{row['param'].replace('_', '')}: {row['covariate']}", axis=1
    )
    summary.to_csv(OUTPUT_DIR / "complete_v5_average_coefficients_vecm_periods.csv", index=False)

    summary["column_key"] = list(zip(summary.currency, summary.contract_type, summary.origin))
    wide = summary.set_index(["period", "period_label", "row_label", "column_key"])["mean_coefficient"].unstack("column_key")
    wide.columns = pd.MultiIndex.from_tuples(wide.columns)
    wide = wide.reindex(columns=ORDER, level=0).reindex(index=ROW_ORDER, level=2)
    wide.to_csv(OUTPUT_DIR / "complete_v5_average_coefficients_vecm_periods_wide.csv")

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption[VECM-period-average coefficients for the complete V5 AFT model]",
        r"{\small Arithmetic means of raw coefficient estimates from the complete eight-covariate V5 log-logistic AFT model, averaged within the six time periods used for the VECM summaries. Missing monthly fits are omitted from the corresponding period mean.}",
        r"\begin{tabular}{|l|l|cc|cc|cc|cc|}",
        r"\hline",
        r"\textbf{VECM period} & \textbf{Coefficient} & \multicolumn{2}{c|}{\textbf{BTC Linear}} & \multicolumn{2}{c|}{\textbf{BTC Inverse}} & \multicolumn{2}{c|}{\textbf{ETH Linear}} & \multicolumn{2}{c|}{\textbf{ETH Inverse}} \\",
        r"\cline{3-10}",
        r" &  & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} \\",
        r"\hline",
    ]
    for period_key, period_label in [(x[0] + "_" + x[1], x[2]) for x in VECM_PERIODS]:
        sub = summary[summary.period.eq(period_key)]
        subwide = sub.set_index(["row_label", "column_key"])["mean_coefficient"].unstack("column_key")
        subwide.columns = pd.MultiIndex.from_tuples(subwide.columns)
        subwide = subwide.reindex(index=ROW_ORDER, columns=ORDER)
        for i, row_label in enumerate(ROW_ORDER):
            values = [f"{subwide.loc[row_label, key]:.4f}" if pd.notna(subwide.loc[row_label, key]) else "" for key in ORDER]
            label = row_label.replace("_", r"\_")
            period_cell = rf"\multirow{{{len(ROW_ORDER)}}}{{*}}{{{period_label}}}" if i == 0 else ""
            lines.append(period_cell + " & " + label + " & " + " & ".join(values) + r" \\")
        lines.append(r"\hline")
    lines.extend([r"\end{tabular}", r"\label{tab:aft_v5_complete_coefficients_vecm_periods}", r"\end{table}"])
    (OUTPUT_DIR / "complete_v5_average_coefficients_vecm_periods_table.tex").write_text("\n".join(lines), encoding="utf-8")

    print("Saved VECM-period coefficient summaries and LaTeX table to", OUTPUT_DIR)
    print(summary.groupby(["period", "period_label"]).monthly_models.max().to_string())


if __name__ == "__main__":
    main()
