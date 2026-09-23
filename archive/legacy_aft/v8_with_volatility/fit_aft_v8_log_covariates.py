"""Fit the V5 log-logistic specification on the V8 event set."""

from __future__ import annotations

import argparse
import gc
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import LogLogisticAFTFitter

from survival_analysis_utils_final import atomic_pandas_csv


MARKETS = ("btc_um", "btc_cm", "eth_um", "eth_cm")
ROOT = Path("sa_results/km_v8_final_01")
INPUT_DIR = ROOT / "aft_data_liquidity_v5_log_covariates"
RESULTS_DIR = ROOT / "aft_results_v5_log_covariates"
COVARIATES = (
    "basis_5min",
    "fundingRate_bps",
    "log_open_interest",
    "shock_size_bps",
    "spread_spot_5min",
    "log_taker_long_short",
    "vol_spot_5min_pct",
    "log_volume",
)


def fit_one(path: Path, market: str, overwrite: bool = False) -> dict:
    coefficient_dir = RESULTS_DIR / market / "loglogistic" / "coefficients"
    coefficient_path = coefficient_dir / f"{path.stem}_coefficients.csv"
    if coefficient_path.exists() and not overwrite:
        return {"source_file": path.name, "market": market, "status": "skipped_existing", "coefficient_file": str(coefficient_path)}

    frame = pd.read_parquet(path)
    required = ["Length", "Status", *COVARIATES]
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise KeyError(f"{path} missing model columns: {missing}")
    data = frame[required].copy()
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    data["Length"] = data["Length"].replace(0, 0.001)
    data = data[data["Length"] > 0]
    data["Status"] = data["Status"].astype(float)
    data = data[data["Status"].isin([0, 1])]
    eligible = len(data)
    data = data.dropna().reset_index(drop=True)
    base = {
        "source_file": path.name,
        "market": market,
        "first": path.stem.split("_", 1)[0],
        "period": path.stem.split("_", 1)[1],
        "source_rows": len(frame),
        "eligible_rows_before_dropna": eligible,
        "complete_rows": len(data),
        "dropped_rows": len(frame) - len(data),
        "events": int(data["Status"].sum()),
        "censored": int(len(data) - data["Status"].sum()),
        "covariates": "|".join(COVARIATES),
        "model": "loglogistic",
    }
    if len(data) < 100:
        return {**base, "status": "skipped_too_few_observations"}

    fitter = LogLogisticAFTFitter()
    try:
        fitter.fit(data, duration_col="Length", event_col="Status", ancillary=False)
        n_parameters = int(len(fitter.params_))
        bic = n_parameters * np.log(len(data)) - 2 * fitter.log_likelihood_
        coefficients = fitter.summary.reset_index()
        coefficients["source_file"] = path.name
        coefficients["market"] = market
        coefficients["model"] = "loglogistic"
        coefficients["observations"] = len(data)
        coefficients["events"] = int(data["Status"].sum())
        coefficients["censored"] = int(len(data) - data["Status"].sum())
        coefficients["concordance"] = fitter.concordance_index_
        coefficients["log_likelihood"] = fitter.log_likelihood_
        coefficients["AIC"] = fitter.AIC_
        coefficients["BIC"] = bic
        coefficients["n_parameters"] = n_parameters
        coefficient_dir.mkdir(parents=True, exist_ok=True)
        atomic_pandas_csv(coefficients, coefficient_path)
        return {**base, "status": "complete", "coefficient_file": str(coefficient_path), "concordance": fitter.concordance_index_, "log_likelihood": fitter.log_likelihood_, "AIC": fitter.AIC_, "BIC": bic, "n_parameters": n_parameters}
    except Exception as exc:
        return {**base, "status": "failed", "error": repr(exc)}
    finally:
        del data
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--markets", nargs="+", choices=MARKETS, default=list(MARKETS))
    args = parser.parse_args()

    tasks = []
    for market in args.markets:
        paths = sorted((INPUT_DIR / market).glob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No V8 inputs under {INPUT_DIR / market}")
        tasks.extend((path, market) for path in paths)

    manifests = {market: [] for market in args.markets}
    total = len(tasks)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(fit_one, path, market, args.overwrite): (path, market) for path, market in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            path, market = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {"source_file": path.name, "market": market, "status": "failed", "error": repr(exc)}
            manifests[market].append(row)
            if index % 10 == 0 or index == total:
                elapsed = (time.monotonic() - started) / 60
                print(f"[{index}/{total}] {market} {path.name} {row['status']} elapsed={elapsed:.1f}m", flush=True)

    for market, rows in manifests.items():
        out = RESULTS_DIR / market / "aft_fit_manifest.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        atomic_pandas_csv(pd.DataFrame(rows).sort_values(["source_file", "status"]), out)
        print(market, pd.DataFrame(rows).groupby("status").size().to_dict())


if __name__ == "__main__":
    main()
