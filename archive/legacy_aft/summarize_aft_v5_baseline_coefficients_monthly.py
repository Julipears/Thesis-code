"""Average V5 baseline AFT coefficients over the VECM monthly period."""

from pathlib import Path

import pandas as pd


MARKETS = ("btc_um", "btc_cm", "eth_um", "eth_cm")
SPECS = {
    "shock_size": ("aft_results_v5_univariate_shock_size", "shock_size_bps"),
    "volatility": ("aft_results_v5_univariate_volatility", "vol_spot_5min_pct"),
    "spread": ("aft_results_v5_univariate_spread", "spread_spot_5min"),
    "null_model": ("aft_results_v5_null", "intercept_only"),
}
START = "2021-01"
END = "2025-12"
OUTPUT_DIR = Path("open_interest_figures/native/aft_v5_univariate_baseline_results")
ORDER = [
    ("BTC", "linear", "spot"), ("BTC", "linear", "perp"),
    ("BTC", "inverse", "spot"), ("BTC", "inverse", "perp"),
    ("ETH", "linear", "spot"), ("ETH", "linear", "perp"),
    ("ETH", "inverse", "spot"), ("ETH", "inverse", "perp"),
]


def main() -> None:
    rows = []
    for specification, (results_dir, variable) in SPECS.items():
        for market in MARKETS:
            root = Path(f"sa_{market}") / results_dir
            manifest = pd.read_csv(root / "aft_fit_manifest.csv")
            coefficient_dir = root / "loglogistic" / "coefficients"
            complete = manifest[manifest.status.isin(["complete", "skipped_existing"])]
            for source_file in complete.source_file:
                path = coefficient_dir / f"{Path(source_file).stem}_coefficients.csv"
                if not path.exists():
                    continue
                coefficient = pd.read_csv(path)
                coefficient = coefficient[["param", "covariate", "coef"]].copy()
                coefficient["coef"] = pd.to_numeric(coefficient["coef"], errors="coerce")
                period = Path(source_file).stem.split("_", 1)[1]
                first = Path(source_file).stem.split("_", 1)[0]
                coefficient["specification"] = specification
                coefficient["variable"] = variable
                coefficient["market"] = market
                coefficient["currency"] = market[:3].upper()
                coefficient["contract_type"] = "linear" if market.endswith("um") else "inverse"
                coefficient["origin"] = first
                coefficient["period"] = period
                rows.append(coefficient)

    coefficients = pd.concat(rows, ignore_index=True)
    coefficients = coefficients.loc[
        coefficients.period.between(START, END) & coefficients.coef.notna()
    ].copy()
    coefficients.to_csv(OUTPUT_DIR / "monthly_aft_baseline_coefficients_long.csv", index=False)

    group_cols = ["specification", "variable", "param", "covariate", "currency", "contract_type", "origin"]
    summary = (
        coefficients.groupby(group_cols, as_index=False)
        .agg(
            monthly_models=("coef", "count"),
            mean_coefficient=("coef", "mean"),
            median_coefficient=("coef", "median"),
            std_coefficient=("coef", "std"),
        )
    )
    summary.to_csv(OUTPUT_DIR / "average_aft_baseline_coefficients_monthly.csv", index=False)

    summary["row_label"] = summary.apply(
        lambda row: (
            f"{row['variable']} ({row['param'].replace('_', '')}, {row['covariate']})"
            if row["specification"] != "null_model"
            else f"null model ({row['param'].replace('_', '')})"
        ),
        axis=1,
    )
    row_order = (
        summary[["specification", "param", "covariate", "row_label"]]
        .drop_duplicates()
        .sort_values(["specification", "param", "covariate"])
        .row_label.tolist()
    )
    summary["column_key"] = list(zip(summary.currency, summary.contract_type, summary.origin))
    wide = summary.set_index(["row_label", "column_key"])["mean_coefficient"].unstack("column_key")
    wide.columns = pd.MultiIndex.from_tuples(wide.columns)
    wide = wide.reindex(index=row_order, columns=ORDER)
    wide.to_csv(OUTPUT_DIR / "average_aft_baseline_coefficients_monthly_wide.csv")

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption[Monthly-average V5 baseline AFT coefficients]",
        r"{\small Arithmetic means of monthly log-logistic AFT coefficient estimates over the 2021-01--2025-12 period used for the monthly VECM summaries. Values are reported by currency, contract type, and shock-origin market.}",
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
        values = [f"{wide.loc[row_label, key]:.5g}" for key in ORDER]
        label = row_label.replace("_", r"\_")
        lines.append(label + "\n& " + " & ".join(values) + r" \\")
        lines.append("")
    lines.extend([r"\hline", r"\end{tabular}", r"\label{tab:aft_v5_baseline_coefficients_monthly}", r"\end{table}"])
    (OUTPUT_DIR / "average_aft_baseline_coefficients_monthly_table.tex").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
