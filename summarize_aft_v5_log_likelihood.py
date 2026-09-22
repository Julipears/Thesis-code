"""Summarize log-likelihoods from the V5 log-logistic AFT fits."""

from pathlib import Path

import pandas as pd


MARKETS = ("btc_um", "btc_cm", "eth_um", "eth_cm")
RESULTS_DIR = "aft_results_v5_log_covariates"
OUTPUT_DIR = Path("open_interest_figures/native/aft_v5_log_covariate_results_plots")


def main() -> None:
    rows = []
    for market in MARKETS:
        root = Path(f"sa_{market}")
        coefficient_dir = root / RESULTS_DIR / "loglogistic" / "coefficients"
        manifest_path = root / RESULTS_DIR / "aft_fit_manifest.csv"
        manifest = pd.read_csv(manifest_path)
        status_by_source = dict(zip(manifest["source_file"], manifest["status"]))

        for path in sorted(coefficient_dir.glob("*_coefficients.csv")):
            coefficients = pd.read_csv(path)
            if coefficients.empty or "log_likelihood" not in coefficients:
                continue
            row = coefficients.iloc[0]
            source_file = str(row.get("source_file", path.stem.removesuffix("_coefficients") + ".parquet"))
            rows.append(
                {
                    "market": market,
                    "currency": market[:3].upper(),
                    "contract_type": "linear" if market.endswith("um") else "inverse",
                    "origin": path.name.split("_", 1)[0],
                    "source_file": source_file,
                    "status": status_by_source.get(source_file, "unknown"),
                    "observations": row.get("observations"),
                    "events": row.get("events"),
                    "censored": row.get("censored"),
                    "concordance": row.get("concordance"),
                    "log_likelihood": row.get("log_likelihood"),
                    "AIC": row.get("AIC"),
                    "BIC": row.get("BIC"),
                    "n_parameters": row.get("n_parameters"),
                }
            )

    long = pd.DataFrame(rows)
    long["log_likelihood"] = pd.to_numeric(long["log_likelihood"], errors="coerce")
    long = long.dropna(subset=["log_likelihood"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    long.to_csv(OUTPUT_DIR / "log_likelihood_by_model.csv", index=False)

    group_cols = ["currency", "contract_type", "origin"]
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
    summary.to_csv(OUTPUT_DIR / "log_likelihood_summary.csv", index=False)

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
        r"\caption[Log-likelihood of V5 log-logistic AFT models]",
        r"{\small Mean log-likelihood across completed monthly V5 log-logistic AFT fits, reported by currency, contract type, and shock-origin market.}",
        r"\begin{tabular}{|l|cc|cc|cc|cc|}",
        r"\hline",
        r"& \multicolumn{4}{c|}{\textbf{BTC}} & \multicolumn{4}{c|}{\textbf{ETH}} \\",
        r"\cline{2-9}",
        r"& \multicolumn{2}{c|}{\textbf{Linear}} & \multicolumn{2}{c|}{\textbf{Inverse}} & \multicolumn{2}{c|}{\textbf{Linear}} & \multicolumn{2}{c|}{\textbf{Inverse}} \\",
        r"\hline",
        r"\textbf{Statistic} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} & \textbf{Spot} & \textbf{Perp} \\",
        r"\hline",
        r"Mean log-likelihood",
    ]
    values = []
    for key in order:
        values.append(f"{lookup.loc[key, 'mean_log_likelihood']:.1f}")
    lines.append("& " + " & ".join(values) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\label{tab:aft_v5_log_likelihood}", r"\end{table}"])
    (OUTPUT_DIR / "log_likelihood_v5_table.tex").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
