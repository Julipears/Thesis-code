"""Compare original final AFT coverage and row counts with the V5 bivariate fit."""

from pathlib import Path

import pandas as pd


MARKETS = ("btc_um", "btc_cm", "eth_um", "eth_cm")
MONTHS = [f"{year}-{month:02d}" for year in range(2021, 2026) for month in range(1, 13)]
EXPECTED = [f"{origin}_{month}.parquet" for origin in ("spot", "perp") for month in MONTHS]
OUT = Path("open_interest_figures/native/aft_v5_log_covariate_results_plots")


def first_observations(path: Path):
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path, nrows=1)
        return int(frame["observations"].iloc[0])
    except Exception:
        return None


def main():
    missing_rows = []
    comparisons = []
    for market in MARKETS:
        old_root = Path(f"sa_{market}/aft_results_final/loglogistic/coefficients")
        new_root = Path(f"sa_{market}/aft_results_v5_bivariate_shock_spread/loglogistic/coefficients")
        manifest_path = Path(f"sa_{market}/aft_results_final/aft_fit_manifest.csv")
        manifest = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
        logit_manifest = manifest[manifest.model.eq("loglogistic")] if not manifest.empty else pd.DataFrame()
        status_by_source = dict(zip(logit_manifest.source_file, logit_manifest.status))
        for source in EXPECTED:
            stem = Path(source).stem
            old_file = old_root / f"{stem}_coefficients.csv"
            new_file = new_root / f"{stem}_coefficients.csv"
            old_obs = first_observations(old_file)
            new_obs = first_observations(new_file)
            if old_obs is None:
                missing_rows.append({
                    "market": market,
                    "source_file": source,
                    "origin": stem.split("_", 1)[0],
                    "month": stem.split("_", 1)[1],
                    "original_status": status_by_source.get(source, "not_in_manifest"),
                })
            if old_obs is not None and new_obs is not None:
                comparisons.append({
                    "market": market,
                    "source_file": source,
                    "origin": stem.split("_", 1)[0],
                    "month": stem.split("_", 1)[1],
                    "original_observations": old_obs,
                    "bivariate_observations": new_obs,
                    "difference_bivariate_minus_original": new_obs - old_obs,
                })
    missing = pd.DataFrame(missing_rows)
    comparison = pd.DataFrame(comparisons)
    missing.to_csv(OUT / "aft_original_missing_months_vs_bivariate.csv", index=False)
    comparison.to_csv(OUT / "aft_original_vs_bivariate_row_counts.csv", index=False)
    print("Missing original files:")
    print(missing.groupby(["market", "origin", "month"], as_index=False).size().to_string(index=False))
    print("\nRow-count comparison:")
    print(comparison.groupby("market")["difference_bivariate_minus_original"].agg(["count", "min", "max", "mean", "median"]).to_string())


if __name__ == "__main__":
    main()
