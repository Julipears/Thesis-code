"""Fit a separate all-covariate log-logistic AFT specification.

The input rows are the unit-agreeing-OI AFT files.  The model includes all
eight covariates and writes to a separate v3
results directory so earlier fits remain unchanged.
"""

from __future__ import annotations

import argparse
import gc
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
INPUT_DIR_NAME = "aft_data_liquidity_unit_agreeing_oi"
RESULTS_DIR_NAME = "aft_results_unit_agreeing_oi_v3_all_covariates"
COVARIATES = (
    "basis_5min",
    "fundingRate_bps",
    "open_interest",
    "shock_size_bps",
    "spread_spot_5min",
    "taker_long_short",
    "vol_spot_5min_pct",
    "volume",
)


def fit_one(path_str: str, market: str, overwrite: bool = False) -> dict:
    input_path = Path(path_str)
    results_dir = Path(f"sa_{market}") / RESULTS_DIR_NAME
    coefficient_dir = results_dir / "loglogistic" / "coefficients"
    coefficient_path = coefficient_dir / f"{input_path.stem}_coefficients.csv"
    if coefficient_path.exists() and not overwrite:
        return {
            "source_file": input_path.name,
            "market": market,
            "model": "loglogistic",
            "status": "skipped_existing",
            "coefficient_file": str(coefficient_path),
        }

    # These saved AFT rows have already passed the timing audit.  Preserve the
    # historical timing labels and apply the same numeric model-row cleaning as
    # the completed unit-agreeing-OI refit, rather than rejecting legacy labels.
    frame = pd.read_parquet(input_path)
    required = ["Length", "Status", *COVARIATES]
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise KeyError(f"{input_path} missing model columns: {missing}")
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
    diagnostics = {
        "source_rows": len(frame),
        "eligible_rows_before_dropna": eligible,
        "complete_rows": len(model_data),
        "dropped_rows": len(frame) - len(model_data),
        "events": int(model_data["Status"].sum()),
        "censored": int(len(model_data) - model_data["Status"].sum()),
    }
    base = {
        "source_file": input_path.name,
        "market": market,
        "model": "loglogistic",
        "covariates": "|".join(COVARIATES),
        **diagnostics,
    }
    if len(model_data) < 100:
        return {**base, "status": "skipped_too_few_observations"}

    fitter = LogLogisticAFTFitter()
    try:
        fitter.fit(model_data, duration_col="Length", event_col="Status", ancillary=False)
        coefficients = fitter.summary.reset_index()
        n_parameters = int(len(fitter.params_))
        bic = n_parameters * np.log(len(model_data)) - 2 * fitter.log_likelihood_
        coefficients["source_file"] = input_path.name
        coefficients["model"] = "loglogistic"
        coefficients["observations"] = len(model_data)
        coefficients["events"] = int(model_data["Status"].sum())
        coefficients["censored"] = int(len(model_data) - model_data["Status"].sum())
        coefficients["concordance"] = fitter.concordance_index_
        coefficients["log_likelihood"] = fitter.log_likelihood_
        coefficients["AIC"] = fitter.AIC_
        coefficients["BIC"] = bic
        coefficients["n_parameters"] = n_parameters
        coefficients["covariate_timing_version"] = COVARIATE_TIMING_VERSION
        coefficients["liquidity_timing_version"] = LIQUIDITY_TIMING_VERSION
        coefficient_dir.mkdir(parents=True, exist_ok=True)
        atomic_pandas_csv(coefficients, coefficient_path)
        return {
            **base,
            "status": "complete",
            "coefficient_file": str(coefficient_path),
            "concordance": fitter.concordance_index_,
            "log_likelihood": fitter.log_likelihood_,
            "AIC": fitter.AIC_,
            "BIC": bic,
            "n_parameters": n_parameters,
        }
    except Exception as exc:
        return {**base, "status": "failed", "error": repr(exc)}
    finally:
        del model_data
        gc.collect()


def save_manifest_retry(frame: pd.DataFrame, path: Path) -> None:
    last_error = None
    for attempt in range(12):
        try:
            atomic_pandas_csv(frame, path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.5 * (attempt + 1))
    raise last_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--markets", nargs="+", choices=MARKETS, default=list(MARKETS))
    args = parser.parse_args()

    tasks = []
    selected_markets = tuple(args.markets)
    for market in selected_markets:
        input_dir = Path(f"sa_{market}") / INPUT_DIR_NAME
        files = sorted(input_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No input files under {input_dir}")
        tasks.extend((str(path), market) for path in files)

    manifest_by_market = {}
    for market in selected_markets:
        manifest_path = Path(f"sa_{market}") / RESULTS_DIR_NAME / "aft_fit_manifest.csv"
        manifest_by_market[market] = (
            pd.read_csv(manifest_path).to_dict("records") if manifest_path.exists() else []
        )

    start = time.monotonic()
    completed = 0
    if args.workers <= 1:
        completed_tasks = ((path, market, None) for path, market in tasks)
    else:
        pool = ThreadPoolExecutor(max_workers=args.workers)
        futures = {
            pool.submit(fit_one, path, market, args.overwrite): (path, market)
            for path, market in tasks
        }
        completed_tasks = (
            (futures[future][0], futures[future][1], future)
            for future in as_completed(futures)
        )

    try:
        for path, market, future in completed_tasks:
            try:
                row = fit_one(path, market, args.overwrite) if future is None else future.result()
            except Exception as exc:
                row = {
                    "source_file": Path(path).name,
                    "market": market,
                    "model": "loglogistic",
                    "status": "failed",
                    "error": repr(exc),
                }
            existing = [r for r in manifest_by_market[market] if r.get("source_file") != row.get("source_file")]
            existing.append(row)
            manifest_by_market[market] = existing
            manifest_path = Path(f"sa_{market}") / RESULTS_DIR_NAME / "aft_fit_manifest.csv"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            save_manifest_retry(
                pd.DataFrame(existing).sort_values(["source_file", "model"]),
                manifest_path,
            )
            completed += 1
            if completed % 10 == 0 or completed == len(tasks):
                elapsed = time.monotonic() - start
                print(
                    f"[{completed}/{len(tasks)}] {market} {row.get('source_file')} "
                    f"{row.get('status')} elapsed={elapsed/60:.1f}m",
                    flush=True,
                )
    finally:
        if args.workers > 1:
            pool.shutdown(wait=True)

    print("[complete]")
    for market in selected_markets:
        manifest_path = Path(f"sa_{market}") / RESULTS_DIR_NAME / "aft_fit_manifest.csv"
        manifest = pd.read_csv(manifest_path)
        print(market)
        print(manifest.groupby("status").size().to_string())


if __name__ == "__main__":
    main()
