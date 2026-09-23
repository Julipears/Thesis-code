"""Stratified gap-filling audit for the three remaining contracts.

Samples the 5th, 15th, and 25th of every month from 2021--2025 and compares
the production-style observed-bin fit with the identical fit on a full grid.
The job is resumable: completed contract/date/latency/specification cells are
read from the output CSV and skipped.
"""

from __future__ import annotations

import gc
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

import audit_hasbrouck_one_hour as hasbrouck
from trade_data_pull import TradeData
from vecm_hasbrouck3 import generate_multiple_lags


CONTRACTS = {
    "btc_cm": ("BTCUSDT", "cm"),
    "eth_um": ("ETHUSDT", "um"),
    "eth_cm": ("ETHUSDT", "cm"),
}
LATENCIES = ["10ms", "100ms", "1s"]
SAMPLE_DAYS = (5, 15, 25)
HOUR = 0
OUT = Path("hasbrouck_spec_audit/stratified_other_contracts")
DAILY_FILE = OUT / "daily_results.csv"
GRID_FILE = OUT / "daily_grid_diagnostics.csv"
ERROR_FILE = OUT / "errors.csv"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def save_upsert(path: Path, new: pd.DataFrame, keys: list[str]) -> None:
    old = read_csv(path)
    combined = pd.concat([old, new], ignore_index=True, sort=False)
    if "day" in combined.columns:
        combined["day"] = pd.to_datetime(combined["day"], format="mixed").dt.strftime("%Y-%m-%d")
    combined = combined.drop_duplicates(keys, keep="last").sort_values(keys)
    combined.to_csv(path, index=False)


def sample_dates() -> pd.DatetimeIndex:
    months = pd.period_range("2021-01", "2025-12", freq="M")
    return pd.DatetimeIndex(
        [pd.Timestamp(year=m.year, month=m.month, day=d) for m in months for d in SAMPLE_DAYS]
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prior = read_csv(DAILY_FILE)
    completed = set()
    if not prior.empty:
        prior["day"] = pd.to_datetime(prior["day"])
        completed = set(prior[["market", "day", "latency", "specification"]].itertuples(index=False, name=None))

    dates = sample_dates()
    lag_structure = generate_multiple_lags(10, LATENCIES, max_length="10s")
    specifications = (
        "current_observed_intercept_float32",
        "full_grid_intercept_float32",
    )

    total = len(CONTRACTS) * len(dates)
    counter = 0
    for market, (symbol, margin) in CONTRACTS.items():
        for day in dates:
            counter += 1
            needed = [
                latency for latency in LATENCIES
                if any((market, day, latency, spec) not in completed for spec in specifications)
            ]
            if not needed:
                print(f"[skip {counter}/{total}] {market} {day.date()} complete", flush=True)
                continue

            start = day + pd.Timedelta(hours=HOUR)
            end = start + pd.Timedelta(hours=1)
            print(f"[run {counter}/{total}] {market} {day.date()} ({', '.join(needed)})", flush=True)
            try:
                data = TradeData(symbol, "Binance", margin)
                data.grab_trades_data(day.to_pydatetime(), days=1, n_jobs=2)
                if data.df_trades_spots.is_empty() or data.df_trades_perps.is_empty():
                    raise RuntimeError("spot or perpetual daily archive unavailable/empty")

                result_rows, grid_rows = [], []
                for latency in needed:
                    frames = {}
                    for fill_gaps in (False, True):
                        frame = data.agg_last_trade_to_intervals(
                            freq=latency, start=start, end=end, fill_gaps=fill_gaps,
                            rename_for_vecm=True, retain_initial_grid_row=True,
                        )
                        if frame.empty:
                            raise RuntimeError(f"no usable {latency} data")
                        frames[fill_gaps] = frame[["log_midpoint_spot", "log_midpoint_perp"]]

                    observed, full = frames[False], frames[True]
                    expected = int(pd.Timedelta(hours=1) / pd.Timedelta(latency))
                    grid_rows.append({
                        "market": market, "day": day, "hour_utc": HOUR, "latency": latency,
                        "expected_calendar_bins": expected, "observed_union_bins": len(observed),
                        "complete_grid_bins": len(full), "omitted_bins_current": expected - len(observed),
                        "omitted_pct_current": 100 * (expected - len(observed)) / expected,
                    })

                    # Hold the estimator, precision, intercept, EC term, lag
                    # structure, and inverse calculation fixed.  Only the
                    # construction of the time grid differs between the fits.
                    fits = (
                        (specifications[0], observed),
                        (specifications[1], full),
                    )
                    for label, frame in fits:
                        fit, n_obs, condition = hasbrouck.current_fit(frame, latency, lag_structure)
                        rows = hasbrouck.rows_for_result(latency, label, fit, n_obs, condition, len(frame))
                        for row in rows:
                            row.update(market=market, day=day, hour_start_utc=start)
                        result_rows.extend(rows)

                save_upsert(DAILY_FILE, pd.DataFrame(result_rows), ["market", "day", "latency", "specification", "series"])
                save_upsert(GRID_FILE, pd.DataFrame(grid_rows), ["market", "day", "latency"])
                completed.update((market, day, latency, spec) for latency in needed for spec in specifications)
                print(f"[saved] {market} {day.date()}", flush=True)
            except Exception as exc:
                save_upsert(ERROR_FILE, pd.DataFrame([{
                    "market": market, "day": day, "hour_utc": HOUR,
                    "error_type": type(exc).__name__, "error": str(exc),
                    "traceback": traceback.format_exc(limit=5).replace("\n", " | "),
                }]), ["market", "day"])
                print(f"[error] {market} {day.date()}: {type(exc).__name__}: {exc}", flush=True)
            finally:
                if "data" in locals():
                    del data
                gc.collect()

    results = read_csv(DAILY_FILE)
    perp = results[results["series"].str.contains("perp")].copy()
    perp["leader"] = np.where(perp["his_mid"] > 0.5, "perp", "spot")
    summary = perp.groupby(["market", "latency", "specification"], as_index=False).agg(
        days=("day", "nunique"), mean_perp_is=("his_mid", "mean"),
        median_perp_is=("his_mid", "median"),
        perp_leader_days=("leader", lambda x: int((x == "perp").sum())),
    )
    summary["perp_leader_pct"] = 100 * summary["perp_leader_days"] / summary["days"]
    summary.to_csv(OUT / "summary.csv", index=False)


if __name__ == "__main__":
    main()
