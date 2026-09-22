"""Fit V8-event univariate and null log-logistic AFT models.

Each univariate model's null comparison is refit on the same complete-case
rows as that univariate model. This makes the likelihood differences and
percentage improvements comparable within each monthly file.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import LogLogisticAFTFitter

from survival_analysis_utils_final import atomic_pandas_csv


ROOT = Path("sa_results/km_v8_final_01")
INPUT_DIR = ROOT / "aft_data_liquidity_v5_log_covariates"
MARKETS = ("btc_um", "btc_cm", "eth_um", "eth_cm")
SPECS = {
    "shock_size": ("shock_size_bps", "aft_results_v8_univariate_shock_size"),
    "spread": ("spread_spot_5min", "aft_results_v8_univariate_spread"),
    "log_volume": ("log_volume", "aft_results_v8_univariate_log_volume"),
    "basis": ("basis_5min", "aft_results_v8_univariate_basis"),
}
NULL_DIR = "aft_results_v8_null"


def clean(path: Path, covariate: str | None) -> pd.DataFrame:
    columns = ["Length", "Status"] + ([covariate] if covariate else [])
    data = pd.read_parquet(path, columns=columns).copy()
    for column in columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    data["Length"] = data["Length"].replace(0, 0.001)
    data = data[(data["Length"] > 0) & data["Status"].isin([0, 1])]
    return data.dropna().reset_index(drop=True)


def fit_model(data: pd.DataFrame, covariate: str | None):
    columns = ["Length", "Status"] + ([covariate] if covariate else [])
    fitter = LogLogisticAFTFitter()
    fitter.fit(data[columns], duration_col="Length", event_col="Status", ancillary=False)
    return fitter


def fit_file(path: Path, market: str, spec_name: str, covariate: str | None, result_dir_name: str) -> dict:
    data = clean(path, covariate)
    base = {"source_file": path.name, "market": market, "first": path.stem.split("_", 1)[0], "period": path.stem.split("_", 1)[1], "specification": spec_name, "covariate": covariate or "Intercept-only", "source_rows": len(pd.read_parquet(path, columns=["Length"])), "complete_rows": len(data), "events": int(data.Status.sum()), "censored": int(len(data) - data.Status.sum())}
    if len(data) < 100:
        return {**base, "status": "skipped_too_few_observations"}
    try:
        model = fit_model(data, covariate)
        null_model = fit_model(data, None)
        ll = float(model.log_likelihood_); null_ll = float(null_model.log_likelihood_)
        delta = ll - null_ll
        two_delta = 2.0 * delta
        pct = 100.0 * delta / abs(null_ll) if null_ll else np.nan
        result_dir = ROOT / result_dir_name / market / "loglogistic" / "coefficients"
        result_dir.mkdir(parents=True, exist_ok=True)
        coeff = model.summary.reset_index()
        coeff["source_file"] = path.name; coeff["market"] = market; coeff["specification"] = spec_name; coeff["model"] = "loglogistic"; coeff["observations"] = len(data); coeff["events"] = int(data.Status.sum()); coeff["censored"] = int(len(data) - data.Status.sum()); coeff["log_likelihood"] = ll; coeff["null_log_likelihood_matching_rows"] = null_ll; coeff["log_likelihood_difference"] = delta; coeff["two_delta_log_likelihood"] = two_delta; coeff["percent_improvement"] = pct; coeff["AIC"] = model.AIC_; coeff["BIC"] = len(model.params_) * np.log(len(data)) - 2 * ll; coeff["concordance"] = model.concordance_index_
        atomic_pandas_csv(coeff, result_dir / f"{path.stem}_coefficients.csv")
        return {**base, "status": "complete", "log_likelihood": ll, "null_log_likelihood": null_ll, "log_likelihood_difference": delta, "two_delta_log_likelihood": two_delta, "percent_improvement": pct, "AIC": model.AIC_, "BIC": coeff["BIC"].iloc[0], "concordance": model.concordance_index_}
    except Exception as exc:
        return {**base, "status": "failed", "error": repr(exc)}
    finally:
        del data
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--workers", type=int, default=1); parser.add_argument("--spec", nargs="+", choices=[*SPECS, "null"], default=[*SPECS, "null"]); args = parser.parse_args()
    started = time.monotonic()
    for spec_name in args.spec:
        if spec_name == "null":
            covariate, result_dir_name = None, NULL_DIR
        else:
            covariate, result_dir_name = SPECS[spec_name]
        for market in MARKETS:
            paths = sorted((INPUT_DIR / market).glob("*.parquet")); rows = []
            for i, path in enumerate(paths, 1):
                row = fit_file(path, market, spec_name, covariate, result_dir_name); rows.append(row)
                if i % 20 == 0 or i == len(paths): print(f"[{spec_name}] {market} {i}/{len(paths)} {row['status']} elapsed={(time.monotonic()-started)/60:.1f}m", flush=True)
            out = ROOT / result_dir_name / market / "aft_fit_manifest.csv"; out.parent.mkdir(parents=True, exist_ok=True); atomic_pandas_csv(pd.DataFrame(rows).sort_values("source_file"), out)
        print(f"[complete] {spec_name}", flush=True)


if __name__ == "__main__":
    main()
