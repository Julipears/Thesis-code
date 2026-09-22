"""Average raw coefficients from the complete V5 AFT model by VECM month."""

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
            first, period = stem.split("_", 1)
            coefficient["market"] = market
            coefficient["currency"] = market[:3].upper()
            coefficient["contract_type"] = "linear" if market.endswith("um") else "inverse"
            coefficient["origin"] = first
            coefficient["period"] = period
            rows.append(coefficient)

    coefficients = pd.concat(rows, ignore_index=True)
    coefficients = coefficients.loc[coefficients.period.between("2021-01", "2025-12")]
    coefficients = coefficients.dropna(subset=["coef"])
    coefficients.to_csv(OUTPUT_DIR / "complete_v5_coefficients_monthly_long.csv", index=False)

    group_cols = ["param", "covariate", "currency", "contract_type", "origin"]
    summary = (
        coefficients.groupby(group_cols, as_index=False)
        .agg(
            monthly_models=("coef", "count"),
            mean_coefficient=("coef", "mean"),
            median_coefficient=("coef", "median"),
            std_coefficient=("coef", "std"),
        )
    )
    summary["row_label"] = summary.apply(
        lambda row: f"{row['param'].replace('_', '')}: {row['covariate']}", axis=1
    )
    summary.to_csv(OUTPUT_DIR / "complete_v5_average_coefficients_monthly.csv", index=False)

    row_order = ["alpha: " + covariate for covariate in COVARIATES]
    row_order += ["alpha: Intercept", "beta: Intercept"]
    summary["column_key"] = list(zip(summary.currency, summary.contract_type, summary.origin))
    wide = summary.set_index(["row_label", "column_key"])["mean_coefficient"].unstack("column_key")
    wide.columns = pd.MultiIndex.from_tuples(wide.columns)
    wide = wide.reindex(index=row_order, columns=ORDER)
    wide.to_csv(OUTPUT_DIR / "complete_v5_average_coefficients_monthly_wide.csv")

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption[Monthly-average coefficients for the complete V5 AFT model]",
        r"{\small Arithmetic means of raw coefficient estimates from the complete eight-covariate V5 log-logistic AFT model over the 2021-01--2025-12 period used for the monthly VECM summaries. Values are reported by currency, contract type, and shock-origin market.}",
        r"\begin{tabular}{|l|cc|cc|cc|cc|}",
        r"\hline",
        r"& \multicolumn{4}{c|}{\textbf{BTC}} & \multicolumn{4}{c|}{\textbf{ETH}} \\",
        r"\cline{2-9}",
        r"& \multicolumn{2}{c|}{\textbf{Linear}} & \multicolumn{2}{c|}{\textbf{Inverse}} & \multicolumn{2}{c|}{\textbf{Linear}} & \multicolumn{2}{c|}{\textbf{Inverse}} \\",
        r"\hline",
        r"\textbf{Coefficient} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} \\",
        r"\hline",
    ]
    for row_label in row_order:
        values = [f"{wide.loc[row_label, key]:.4f}" for key in ORDER]
        label = row_label.replace("_", r"\_")
        lines.append(label + "\n& " + " & ".join(values) + r" \\")
        lines.append("")
    lines.extend([r"\hline", r"\end{tabular}", r"\label{tab:aft_v5_complete_coefficients_monthly}", r"\end{table}"])
    (OUTPUT_DIR / "complete_v5_average_coefficients_monthly_table.tex").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
