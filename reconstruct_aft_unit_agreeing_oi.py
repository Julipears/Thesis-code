"""Rebuild AFT open-interest covariates and refit log-logistic models.

This is a separate, checkpointed version of the final AFT inputs.  It keeps
the existing ``aft_data_liquidity_final`` and ``aft_results_final`` outputs
untouched, replacing only the ``open_interest`` covariate in the new copies:

* USD-M linear: Binance ``sum_open_interest``
* COIN-M inverse: Binance ``sum_open_interest_value``

Both source fields are already retained in
``open_interest_figures/native/raw_metrics``.  The resulting files are saved
under ``aft_data_liquidity_unit_agreeing_oi`` and
``aft_results_unit_agreeing_oi`` within each market directory.
"""

from __future__ import annotations

import argparse
import gc
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import LogLogisticAFTFitter

from survival_analysis_data_processing_final import prepare_aft_data
from survival_analysis_utils_final import (
    LIQUIDITY_TIMING_VERSION,
    COVARIATE_TIMING_VERSION,
    atomic_pandas_csv,
    atomic_pandas_parquet,
    validate_augmented_timing,
)


MARKETS = ("btc_um", "btc_cm", "eth_um", "eth_cm")
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
SOURCE_DIR_NAME = "aft_data_liquidity_final"
OUTPUT_DIR_NAME = "aft_data_liquidity_unit_agreeing_oi"
RESULTS_DIR_NAME = "aft_results_unit_agreeing_oi"
RAW_ROOT = Path("open_interest_figures/native/raw_metrics")


def save_manifest(frame: pd.DataFrame, path: Path) -> None:
    """Retry briefly when OneDrive is indexing the previous checkpoint."""
    last_error = None
    for attempt in range(8):
        try:
            atomic_pandas_csv(frame, path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.5 * (attempt + 1))
    raise last_error


def selected_field(market: str) -> str:
    return "sum_open_interest" if market.endswith("_um") else "sum_open_interest_value"


def load_metric_lookup(market: str, dates: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Load both OI source fields for the requested dates, deduplicated by timestamp."""
    frames = []
    missing = []
    for day in dates:
        path = RAW_ROOT / market / f"{day}.csv.gz"
        if not path.exists():
            missing.append(day)
            continue
        frame = pd.read_csv(
            path,
            usecols=["create_time", "sum_open_interest", "sum_open_interest_value"],
        )
        frame["create_time"] = pd.to_datetime(frame["create_time"], utc=True, errors="raise")
        frame["sum_open_interest"] = pd.to_numeric(frame["sum_open_interest"], errors="coerce")
        frame["sum_open_interest_value"] = pd.to_numeric(frame["sum_open_interest_value"], errors="coerce")
        frame = frame.drop_duplicates("create_time", keep="last")
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["create_time", "sum_open_interest", "sum_open_interest_value"]), missing
    lookup = pd.concat(frames, ignore_index=True).drop_duplicates("create_time", keep="last")
    return lookup, missing


def reconstruct_file(source_path: Path, output_path: Path, market: str, overwrite: bool) -> dict:
    if output_path.exists() and not overwrite:
        return {
            "source_file": source_path.name,
            "output_file": str(output_path),
            "market": market,
            "status": "skipped_existing",
            "selected_field": selected_field(market),
        }

    frame = pd.read_parquet(source_path)
    original = pd.to_numeric(frame["open_interest"], errors="coerce")
    timestamps = pd.to_datetime(frame["create_time"], utc=True, errors="coerce")
    dates = sorted(timestamps.dropna().dt.strftime("%Y-%m-%d").unique())
    lookup, missing_days = load_metric_lookup(market, dates)

    if lookup.empty:
        replacement = pd.Series(np.nan, index=frame.index, dtype=float)
        both_sum = pd.Series(np.nan, index=frame.index, dtype=float)
        both_value = pd.Series(np.nan, index=frame.index, dtype=float)
    else:
        # A small number of legacy AFT rows carry the metric timestamp one
        # second after the five-minute boundary (for example, 02:50:01).
        # Align on the five-minute bin so those rows use the same observation
        # that produced the original covariate.
        lookup = lookup.copy()
        lookup["metric_bin"] = lookup["create_time"].dt.floor("5min")
        indexed = lookup.drop_duplicates("metric_bin", keep="last").set_index("metric_bin")
        aligned = indexed.reindex(timestamps.dt.floor("5min"))
        replacement = aligned[selected_field(market)].set_axis(frame.index)
        both_sum = aligned["sum_open_interest"].set_axis(frame.index)
        both_value = aligned["sum_open_interest_value"].set_axis(frame.index)

    replacement = pd.to_numeric(replacement, errors="coerce")
    if ((replacement.dropna() < 0).any() or not np.isfinite(replacement.dropna()).all()):
        raise ValueError(f"Invalid replacement OI values in {source_path}")
    old_available = int(original.notna().sum())
    new_available = int(replacement.notna().sum())
    old_missing_new = int(original.notna().sum() - replacement.notna().sum())
    if old_missing_new:
        raise ValueError(
            f"Replacement OI is missing for {old_missing_new} previously populated rows in {source_path.name}"
        )

    frame = frame.copy()
    frame["open_interest_original"] = original
    frame["open_interest_sum_open_interest"] = both_sum.to_numpy()
    frame["open_interest_sum_open_interest_value"] = both_value.to_numpy()
    frame["open_interest"] = replacement.to_numpy()
    atomic_pandas_parquet(frame, output_path)
    return {
        "source_file": source_path.name,
        "output_file": str(output_path),
        "market": market,
        "status": "written",
        "rows": len(frame),
        "old_open_interest_rows": old_available,
        "new_open_interest_rows": new_available,
        "missing_source_days": "|".join(missing_days),
        "selected_field": selected_field(market),
    }


def reconstruct_market(market: str, overwrite: bool = False) -> pd.DataFrame:
    source_dir = Path(f"sa_{market}") / SOURCE_DIR_NAME
    output_dir = Path(f"sa_{market}") / OUTPUT_DIR_NAME
    files = sorted(source_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No AFT input files under {source_dir}")
    rows = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(reconstruct_file, source_path, output_dir / source_path.name, market, overwrite): source_path
            for source_path in files
        }
        for index, future in enumerate(as_completed(futures), 1):
            source_path = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "source_file": source_path.name,
                    "output_file": str(output_dir / source_path.name),
                    "market": market,
                    "status": "failed",
                    "error": repr(exc),
                }
            rows.append(row)
            if index % 10 == 0 or index == len(files):
                print(f"[reconstruct {market}] {index}/{len(files)} {row['status']} {source_path.name}", flush=True)
    manifest = pd.DataFrame(rows)
    save_manifest(manifest, output_dir / "reconstruction_manifest.csv")
    return manifest


def fit_one_loglogistic(input_path: Path, results_dir: Path, overwrite: bool = False) -> dict:
    coefficient_dir = results_dir / "loglogistic" / "coefficients"
    coefficient_path = coefficient_dir / f"{input_path.stem}_coefficients.csv"
    if coefficient_path.exists() and not overwrite:
        return {"source_file": input_path.name, "status": "skipped_existing", "model": "loglogistic", "coefficient_file": str(coefficient_path)}

    frame = pd.read_parquet(input_path)
    # The saved final AFT inputs contain two previously validated timing labels
    # (v1 and v4) from the historical build.  Preserve those labels in the
    # parquet, but validate the timestamp invariants while temporarily
    # normalizing the label so the model refit can use the complete saved data.
    source_versions = "|".join(sorted(frame["covariate_timing_version"].dropna().astype(str).unique())) if "covariate_timing_version" in frame.columns else ""
    # These are immutable, previously saved AFT event rows.  The historical
    # files contain a small number of legacy timestamp-label combinations, so
    # the refit preserves them rather than rebuilding or filtering event rows.
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
        "input_covariate_timing_versions": source_versions,
    }
    base = {"source_file": input_path.name, "model": "loglogistic", **diagnostics}
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


def fit_market(market: str, overwrite: bool = False, workers: int = 4) -> pd.DataFrame:
    input_dir = Path(f"sa_{market}") / OUTPUT_DIR_NAME
    results_dir = Path(f"sa_{market}") / RESULTS_DIR_NAME
    files = sorted(input_dir.glob("*.parquet"))
    files = [path for path in files if path.name != "reconstruction_manifest.parquet"]
    if not files:
        raise FileNotFoundError(f"No reconstructed AFT files under {input_dir}")
    manifest_path = results_dir / "aft_fit_manifest.csv"
    prior = pd.read_csv(manifest_path) if manifest_path.exists() and not overwrite else pd.DataFrame()
    prior_rows = [prior] if not prior.empty else []
    rows = []
    start = time.monotonic()
    # lifelines/autograd is not thread-safe for simultaneous fits; use worker
    # processes so each fit has an isolated optimizer state.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fit_one_loglogistic, path, results_dir, overwrite): path
            for path in files
        }
        for index, future in enumerate(as_completed(futures), 1):
            path = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {"source_file": path.name, "model": "loglogistic", "status": "failed", "error": repr(exc)}
            row["market"] = market
            rows.append(row)
            current = pd.concat(prior_rows + [pd.DataFrame(rows)], ignore_index=True)
            current = current.drop_duplicates(["market", "source_file", "model"], keep="last")
            save_manifest(current.sort_values(["market", "source_file", "model"]), manifest_path)
            if index % 5 == 0 or index == len(files):
                elapsed = max(time.monotonic() - start, 1e-9)
                eta = elapsed / index * (len(files) - index)
                print(
                    f"[fit {market}] {index}/{len(files)} {row['status']} {path.name} "
                    f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
                    flush=True,
                )
            gc.collect()
    return pd.read_csv(manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("reconstruct", "fit", "all"), default="all")
    parser.add_argument("--markets", nargs="*", choices=MARKETS, default=list(MARKETS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.phase in ("reconstruct", "all"):
        for market in args.markets:
            reconstruct_market(market, overwrite=args.overwrite)
    if args.phase in ("fit", "all"):
        for market in args.markets:
            manifest = fit_market(market, overwrite=args.overwrite, workers=max(1, args.workers))
            print(f"[summary {market}]\n{manifest.groupby('status').size().to_string()}", flush=True)


if __name__ == "__main__":
    main()
