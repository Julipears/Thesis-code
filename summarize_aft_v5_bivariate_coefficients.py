"""Summarize all-month coefficients from the V5 bivariate AFT model."""

from pathlib import Path

import pandas as pd


MARKETS = ("btc_um", "btc_cm", "eth_um", "eth_cm")
OUT = Path("open_interest_figures/native/aft_v5_log_covariate_results_plots")
RESULTS_DIR = "aft_results_v5_bivariate_shock_spread"
ORDER = [
    ("BTC", "linear", "spot"), ("BTC", "linear", "perp"),
    ("BTC", "inverse", "spot"), ("BTC", "inverse", "perp"),
    ("ETH", "linear", "spot"), ("ETH", "linear", "perp"),
    ("ETH", "inverse", "spot"), ("ETH", "inverse", "perp"),
]
ROW_ORDER = [
    ("alpha_", "Intercept", "Intercept (alpha)"),
    ("alpha_", "shock_size_bps", r"shock\_size\_bps (alpha)"),
    ("alpha_", "spread_spot_5min", r"spread\_spot\_5min (alpha)"),
    ("beta_", "Intercept", "Intercept (beta)"),
]


def main() -> None:
    rows = []
    for market in MARKETS:
        root = Path(f"sa_{market}") / RESULTS_DIR / "loglogistic" / "coefficients"
        currency = market[:3].upper()
        contract_type = "linear" if market.endswith("um") else "inverse"
        for path in sorted(root.glob("*_coefficients.csv")):
            frame = pd.read_csv(path, usecols=["param", "covariate", "coef"])
            allowed = {(p, c) for p, c, _ in ROW_ORDER}
            frame = frame[frame.apply(lambda row: (row["param"], row["covariate"]) in allowed, axis=1)].copy()
            frame["coef"] = pd.to_numeric(frame["coef"], errors="coerce")
            frame = frame.dropna(subset=["coef"])
            frame["currency"] = currency
            frame["contract_type"] = contract_type
            frame["origin"] = path.stem.split("_", 1)[0]
            frame["month"] = path.stem.split("_", 1)[1].replace("_coefficients", "")
            rows.append(frame)
    detail = pd.concat(rows, ignore_index=True)
    detail.to_csv(OUT / "aft_v5_bivariate_coefficients_monthly.csv", index=False)
    summary = detail.groupby(["param", "covariate", "currency", "contract_type", "origin"], as_index=False).agg(
        monthly_models=("coef", "count"),
        mean_coefficient=("coef", "mean"),
        median_coefficient=("coef", "median"),
        std_coefficient=("coef", "std"),
    )
    summary.to_csv(OUT / "aft_v5_bivariate_coefficients_all_months.csv", index=False)

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption[Monthly-average V5 bivariate AFT coefficients]",
        r"{\small Arithmetic means of monthly log-logistic AFT coefficient estimates from the bivariate specification containing shock size and spot spread, averaged over 2021-01--2025-12. Values are reported by currency, contract type, and shock-origin market.}",
        r"\begin{tabular}{|l|cc|cc|cc|cc|}",
        r"\hline",
        r"& \multicolumn{4}{c|}{\textbf{BTC}} & \multicolumn{4}{c|}{\textbf{ETH}} \\",
        r"\cline{2-9}",
        r"& \multicolumn{2}{c|}{\textbf{Linear}} & \multicolumn{2}{c|}{\textbf{Inverse}} & \multicolumn{2}{c|}{\textbf{Linear}} & \multicolumn{2}{c|}{\textbf{Inverse}} \\",
        r"\hline",
        r"\textbf{Coefficient} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} \\",
        r"\hline",
    ]
    for param, covariate, label in ROW_ORDER:
        vals = []
        for currency, contract_type, origin in ORDER:
            row = summary[
                summary.param.eq(param)
                & summary.covariate.eq(covariate)
                & summary.currency.eq(currency)
                & summary.contract_type.eq(contract_type)
                & summary.origin.eq(origin)
            ]
            value = row.mean_coefficient.iloc[0] if not row.empty else float("nan")
            vals.append(f"{value:.4f}" if pd.notna(value) else "")
        lines.append(label + "\n& " + " & ".join(vals) + r" \\")
        lines.append("")
    lines.extend([r"\hline", r"\end{tabular}", r"\label{tab:aft_v5_bivariate_coefficients_monthly}", r"\end{table}"])
    (OUT / "aft_v5_bivariate_coefficients_all_months_table.tex").write_text("\n".join(lines), encoding="utf-8")
    print("Saved bivariate coefficient summaries to", OUT)


if __name__ == "__main__":
    main()
