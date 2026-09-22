"""Fit the V8-event bivariate log-logistic AFT model.

The model contains shock_size_bps and spread_spot_5min.  For each monthly
file, the matching null model is fit on the same complete-case observations
so likelihood comparisons remain valid.
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
RESULTS_DIR = ROOT / "aft_results_v8_bivariate_shock_spread"
MARKETS = ("btc_um", "btc_cm", "eth_um", "eth_cm")
COVARIATES = ("shock_size_bps", "spread_spot_5min")


def clean(path: Path) -> pd.DataFrame:
    columns = ["Length", "Status", *COVARIATES]
    data = pd.read_parquet(path, columns=columns).copy()
    for column in columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    data["Length"] = data["Length"].replace(0, 0.001)
    data = data[(data["Length"] > 0) & data["Status"].isin([0, 1])]
    return data.dropna().reset_index(drop=True)


def fit(data: pd.DataFrame, covariates: tuple[str, ...]) -> LogLogisticAFTFitter:
    columns = ["Length", "Status", *covariates]
    model = LogLogisticAFTFitter()
    model.fit(data[columns], duration_col="Length", event_col="Status", ancillary=False)
    return model


def fit_file(path: Path, market: str) -> dict:
    data = clean(path)
    source_rows = len(pd.read_parquet(path, columns=["Length"]))
    base = {
        "source_file": path.name,
        "market": market,
        "first": path.stem.split("_", 1)[0],
        "period": path.stem.split("_", 1)[1],
        "specification": "bivariate_shock_size_spread",
        "covariates": "|".join(COVARIATES),
        "source_rows": source_rows,
        "complete_rows": len(data),
        "events": int(data.Status.sum()),
        "censored": int(len(data) - data.Status.sum()),
    }
    if len(data) < 100:
        return {**base, "status": "skipped_too_few_observations"}
    try:
        model = fit(data, COVARIATES)
        null_model = fit(data, tuple())
        ll = float(model.log_likelihood_)
        null_ll = float(null_model.log_likelihood_)
        delta = ll - null_ll
        result_dir = RESULTS_DIR / market / "loglogistic" / "coefficients"
        result_dir.mkdir(parents=True, exist_ok=True)
        coefficients = model.summary.reset_index()
        coefficients["source_file"] = path.name
        coefficients["market"] = market
        coefficients["model"] = "loglogistic"
        coefficients["specification"] = "bivariate_shock_size_spread"
        coefficients["observations"] = len(data)
        coefficients["events"] = int(data.Status.sum())
        coefficients["censored"] = int(len(data) - data.Status.sum())
        coefficients["concordance"] = model.concordance_index_
        coefficients["log_likelihood"] = ll
        coefficients["null_log_likelihood_matching_rows"] = null_ll
        coefficients["log_likelihood_difference"] = delta
        coefficients["two_delta_log_likelihood"] = 2.0 * delta
        coefficients["percent_improvement"] = 100.0 * delta / abs(null_ll) if null_ll else np.nan
        coefficients["AIC"] = model.AIC_
        coefficients["BIC"] = len(model.params_) * np.log(len(data)) - 2.0 * ll
        atomic_pandas_csv(coefficients, result_dir / f"{path.stem}_coefficients.csv")
        return {
            **base,
            "status": "complete",
            "log_likelihood": ll,
            "null_log_likelihood": null_ll,
            "log_likelihood_difference": delta,
            "two_delta_log_likelihood": 2.0 * delta,
            "percent_improvement": coefficients["percent_improvement"].iloc[0],
            "AIC": model.AIC_,
            "BIC": coefficients["BIC"].iloc[0],
            "concordance": model.concordance_index_,
        }
    except Exception as exc:
        return {**base, "status": "failed", "error": repr(exc)}
    finally:
        del data
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", nargs="+", choices=MARKETS, default=list(MARKETS))
    args = parser.parse_args()
    started = time.monotonic()
    for market in args.markets:
        paths = sorted((INPUT_DIR / market).glob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No inputs under {INPUT_DIR / market}")
        rows = []
        for index, path in enumerate(paths, 1):
            row = fit_file(path, market)
            rows.append(row)
            if index % 20 == 0 or index == len(paths):
                print(f"[bivariate] {market} {index}/{len(paths)} {row['status']} elapsed={(time.monotonic()-started)/60:.1f}m", flush=True)
        out = RESULTS_DIR / market / "aft_fit_manifest.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        atomic_pandas_csv(pd.DataFrame(rows).sort_values("source_file"), out)
        print(f"[summary] {market} {pd.DataFrame(rows).groupby('status').size().to_dict()}", flush=True)
    print("[complete] bivariate_shock_size_spread", flush=True)


if __name__ == "__main__":
    main()
