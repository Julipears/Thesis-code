"""Fit V5 univariate/null log-logistic AFT baselines and retain likelihoods.

These fits intentionally omit the expensive full-sample concordance-index
calculation.  The coefficient files retain log-likelihood, AIC, and BIC.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import LogLogisticAFTFitter

from survival_analysis_utils_final import (
    COVARIATE_TIMING_VERSION,
    LIQUIDITY_TIMING_VERSION,
    atomic_pandas_csv,
)


MARKETS = ("btc_um", "btc_cm", "eth_um", "eth_cm")
INPUT_DIR_NAME = "aft_data_liquidity_v5_log_covariates"
SPECS = {
    "shock_size": ("aft_results_v5_univariate_shock_size", ("shock_size_bps",)),
    "volatility": ("aft_results_v5_univariate_volatility", ("vol_spot_5min_pct",)),
    "spread": ("aft_results_v5_univariate_spread", ("spread_spot_5min",)),
    "null": ("aft_results_v5_null", ()),
}


def fit_one(path: Path, market: str, results_dir: str, covariates: tuple[str, ...]) -> dict:
    coefficient_dir = Path(f"sa_{market}") / results_dir / "loglogistic" / "coefficients"
    coefficient_path = coefficient_dir / f"{path.stem}_coefficients.csv"
    base = {
        "source_file": path.name,
        "market": market,
        "model": "loglogistic",
        "covariates": "|".join(covariates),
    }
    if coefficient_path.exists():
        return {**base, "status": "skipped_existing", "coefficient_file": str(coefficient_path)}

    frame = pd.read_parquet(path)
    required = ["Length", "Status", *covariates]
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        return {**base, "status": "failed", "error": f"missing columns: {missing}"}
    model_data = frame[required].copy()
    for column in required:
        model_data[column] = pd.to_numeric(model_data[column], errors="coerce")
    model_data = model_data.replace([np.inf, -np.inf], np.nan)
    model_data["Length"] = model_data["Length"].replace(0, 0.001)
    model_data = model_data[model_data["Length"] > 0]
    model_data["Status"] = model_data["Status"].astype(float)
    model_data = model_data[model_data["Status"].isin([0, 1])]
    eligible = len(model_data)
    model_data = model_data.dropna().reset_index(drop=True)
    base.update(
        {
            "source_rows": len(frame),
            "eligible_rows_before_dropna": eligible,
            "complete_rows": len(model_data),
            "dropped_rows": len(frame) - len(model_data),
            "events": int(model_data["Status"].sum()),
            "censored": int(len(model_data) - model_data["Status"].sum()),
        }
    )
    if len(model_data) < 100:
        return {**base, "status": "skipped_too_few_observations"}

    try:
        fitter = LogLogisticAFTFitter()
        fitter.fit(model_data, duration_col="Length", event_col="Status", ancillary=False)
        log_likelihood = float(fitter.log_likelihood_)
        n_parameters = int(len(fitter.params_))
        bic = n_parameters * np.log(len(model_data)) - 2 * log_likelihood
        coefficients = fitter.summary.reset_index()
        coefficients["source_file"] = path.name
        coefficients["model"] = "loglogistic"
        coefficients["observations"] = len(model_data)
        coefficients["events"] = int(model_data["Status"].sum())
        coefficients["censored"] = int(len(model_data) - model_data["Status"].sum())
        coefficients["concordance"] = np.nan
        coefficients["log_likelihood"] = log_likelihood
        coefficients["AIC"] = float(fitter.AIC_)
        coefficients["BIC"] = float(bic)
        coefficients["n_parameters"] = n_parameters
        coefficients["covariate_timing_version"] = COVARIATE_TIMING_VERSION
        coefficients["liquidity_timing_version"] = LIQUIDITY_TIMING_VERSION
        coefficient_dir.mkdir(parents=True, exist_ok=True)
        atomic_pandas_csv(coefficients, coefficient_path)
        return {
            **base,
            "status": "complete",
            "coefficient_file": str(coefficient_path),
            "concordance": np.nan,
            "log_likelihood": log_likelihood,
            "AIC": float(fitter.AIC_),
            "BIC": float(bic),
            "n_parameters": n_parameters,
        }
    except Exception as exc:
        return {**base, "status": "failed", "error": repr(exc)}
    finally:
        del model_data
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", choices=tuple(SPECS), required=True)
    parser.add_argument("--market", choices=MARKETS, default=None)
    args = parser.parse_args()
    results_dir, covariates = SPECS[args.spec]
    selected_markets = (args.market,) if args.market else MARKETS
    manifests = {}
    for market in selected_markets:
        manifest_path = Path(f"sa_{market}") / results_dir / "aft_fit_manifest.csv"
        manifests[market] = (
            pd.read_csv(manifest_path).to_dict("records") if manifest_path.exists() else []
        )

    tasks = []
    for market in selected_markets:
        input_dir = Path(f"sa_{market}") / INPUT_DIR_NAME
        tasks.extend((path, market) for path in sorted(input_dir.glob("*.parquet")))
    start = time.monotonic()
    for index, (path, market) in enumerate(tasks, 1):
        row = fit_one(path, market, results_dir, covariates)
        existing = [r for r in manifests[market] if r.get("source_file") != row.get("source_file")]
        existing.append(row)
        manifests[market] = existing
        manifest_path = Path(f"sa_{market}") / results_dir / "aft_fit_manifest.csv"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_pandas_csv(pd.DataFrame(existing).sort_values(["source_file", "model"]), manifest_path)
        if index % 10 == 0 or index == len(tasks):
            elapsed = (time.monotonic() - start) / 60
            print(f"[{index}/{len(tasks)}] {args.spec} {market} {row.get('source_file')} {row.get('status')} elapsed={elapsed:.1f}m", flush=True)

    for market in selected_markets:
        manifest = pd.read_csv(Path(f"sa_{market}") / results_dir / "aft_fit_manifest.csv")
        print(market)
        print(manifest.groupby("status").size().to_string())


if __name__ == "__main__":
    main()
