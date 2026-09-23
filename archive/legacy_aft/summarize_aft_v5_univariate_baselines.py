"""Summarize likelihoods from the V5 univariate and null AFT baselines."""

from pathlib import Path

import pandas as pd


MARKETS = ("btc_um", "btc_cm", "eth_um", "eth_cm")
SPECS = {
    "shock_size": ("aft_results_v5_univariate_shock_size", "shock_size_bps"),
    "volatility": ("aft_results_v5_univariate_volatility", "vol_spot_5min_pct"),
    "spread": ("aft_results_v5_univariate_spread", "spread_spot_5min"),
    "null_model": ("aft_results_v5_null", "intercept_only"),
}
OUTPUT_DIR = Path("open_interest_figures/native/aft_v5_univariate_baseline_results")


def main() -> None:
    rows = []
    for specification, (results_dir, variable) in SPECS.items():
        for market in MARKETS:
            root = Path(f"sa_{market}") / results_dir
            manifest = pd.read_csv(root / "aft_fit_manifest.csv")
            coefficient_dir = root / "loglogistic" / "coefficients"
            for record in manifest.to_dict("records"):
                if record.get("status") not in {"complete", "skipped_existing"}:
                    continue
                source_file = str(record["source_file"])
                coefficient_path = coefficient_dir / f"{Path(source_file).stem}_coefficients.csv"
                if not coefficient_path.exists():
                    continue
                coefficients = pd.read_csv(coefficient_path)
                if coefficients.empty or "log_likelihood" not in coefficients:
                    continue
                row = coefficients.iloc[0]
                rows.append(
                    {
                        "specification": specification,
                        "variable": variable,
                        "market": market,
                        "currency": market[:3].upper(),
                        "contract_type": "linear" if market.endswith("um") else "inverse",
                        "origin": source_file.split("_", 1)[0],
                        "source_file": source_file,
                        "fit_status": record.get("status"),
                        "observations": row.get("observations"),
                        "events": row.get("events"),
                        "censored": row.get("censored"),
                        "log_likelihood": pd.to_numeric(row.get("log_likelihood"), errors="coerce"),
                        "AIC": pd.to_numeric(row.get("AIC"), errors="coerce"),
                        "BIC": pd.to_numeric(row.get("BIC"), errors="coerce"),
                        "n_parameters": row.get("n_parameters"),
                    }
                )

    long = pd.DataFrame(rows).dropna(subset=["log_likelihood"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    long.to_csv(OUTPUT_DIR / "log_likelihood_by_univariate_model.csv", index=False)

    group_cols = ["specification", "currency", "contract_type", "origin"]
    summary = (
        long.groupby(group_cols, as_index=False)["log_likelihood"]
        .agg(
            fitted_models="count",
            mean_log_likelihood="mean",
            median_log_likelihood="median",
            std_log_likelihood="std",
            min_log_likelihood="min",
            max_log_likelihood="max",
        )
    )
    summary.to_csv(OUTPUT_DIR / "log_likelihood_univariate_summary.csv", index=False)

    order = [
        ("BTC", "linear", "spot"), ("BTC", "linear", "perp"),
        ("BTC", "inverse", "spot"), ("BTC", "inverse", "perp"),
        ("ETH", "linear", "spot"), ("ETH", "linear", "perp"),
        ("ETH", "inverse", "spot"), ("ETH", "inverse", "perp"),
    ]
    lookup = summary.set_index(group_cols)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption[Log-likelihood of univariate and null V5 AFT models]",
        r"{\small Mean log-likelihood across completed monthly V5 log-logistic AFT fits. The univariate specifications contain shock size, spot volatility, or spot spread; the null specification contains only intercept terms.}",
        r"\begin{tabular}{|l|cc|cc|cc|cc|}",
        r"\hline",
        r"& \multicolumn{4}{c|}{\textbf{BTC}} & \multicolumn{4}{c|}{\textbf{ETH}} \\",
        r"\cline{2-9}",
        r"& \multicolumn{2}{c|}{\textbf{Linear}} & \multicolumn{2}{c|}{\textbf{Inverse}} & \multicolumn{2}{c|}{\textbf{Linear}} & \multicolumn{2}{c|}{\textbf{Inverse}} \\",
        r"\hline",
        r"\textbf{Specification} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} \\",
        r"\hline",
    ]
    for specification, (_results_dir, variable) in SPECS.items():
        values = [
            f"{lookup.loc[(specification, *key), 'mean_log_likelihood']:.1f}"
            for key in order
        ]
        label = variable.replace("_", r"\_")
        lines.append(label + "\n& " + " & ".join(values) + r" \\")
        lines.append("")
    lines.extend([r"\hline", r"\end{tabular}", r"\label{tab:aft_v5_univariate_log_likelihood}", r"\end{table}"])
    (OUTPUT_DIR / "log_likelihood_univariate_v5_table.tex").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
