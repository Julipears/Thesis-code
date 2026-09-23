"""Resumable full-sample Hasbrouck audit using one UTC hour per day."""

from __future__ import annotations

import gc
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

from audit_hasbrouck_one_hour import (
    SYMBOL,
    MARKET,
    current_fit,
    rows_for_result,
    stable_fit,
)
from trade_data_pull import TradeData
from vecm_hasbrouck3 import generate_multiple_lags


START_DAY = pd.Timestamp("2021-01-01")
END_DAY = pd.Timestamp("2025-12-31")
HOUR = 0
LATENCIES = ["10ms", "100ms", "1s"]
OUT = Path("hasbrouck_spec_audit/full_sample_daily_hours")
DAILY_FILE = OUT / "daily_results.csv"
GRID_FILE = OUT / "daily_grid_diagnostics.csv"
ERROR_FILE = OUT / "errors.csv"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def save_upsert(path: Path, new: pd.DataFrame, keys: list[str]) -> None:
    old = read_csv(path)
    combined = pd.concat([old, new], ignore_index=True, sort=False)
    # CSV reloads represent dates as strings, while freshly produced rows use
    # Timestamps.  Normalize before sorting/upserting so an error on a later
    # day cannot crash the resumable audit with mixed-type comparisons.
    if "day" in combined.columns:
        combined["day"] = pd.to_datetime(
            combined["day"], format="mixed", errors="raise"
        ).dt.strftime("%Y-%m-%d")
    combined = combined.drop_duplicates(keys, keep="last").sort_values(keys)
    combined.to_csv(path, index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    completed_df = read_csv(DAILY_FILE)
    if not completed_df.empty:
        completed_df["day"] = pd.to_datetime(completed_df["day"])
        completed = {
            (row.day.normalize(), row.latency, row.specification)
            for row in completed_df[["day", "latency", "specification"]].drop_duplicates().itertuples(index=False)
        }
    else:
        completed = set()

    lag_structure = generate_multiple_lags(10, LATENCIES, max_length="10s")
    days = pd.date_range(START_DAY, END_DAY, freq="D")

    for day_number, day in enumerate(days, start=1):
        needed = [
            latency for latency in LATENCIES
            if (day, latency, "current_observed_intercept_float32") not in completed
            or (day, latency, "full_grid_demeaned_ec_no_intercept_float64") not in completed
        ]
        if not needed:
            print(f"[skip {day_number}/{len(days)}] {day.date()} complete", flush=True)
            continue

        hour_start = day + pd.Timedelta(hours=HOUR)
        hour_end = hour_start + pd.Timedelta(hours=1)
        print(f"[day {day_number}/{len(days)}] {day.date()} ({', '.join(needed)})", flush=True)

        try:
            data = TradeData(SYMBOL, "Binance", "um")
            data.grab_trades_data(day.to_pydatetime(), days=1, n_jobs=2)
            if data.df_trades_spots.is_empty() or data.df_trades_perps.is_empty():
                raise RuntimeError("spot or perpetual daily archive unavailable/empty")

            day_rows: list[dict] = []
            grid_rows: list[dict] = []
            for latency in needed:
                frames = {}
                for fill_gaps in (False, True):
                    frame = data.agg_last_trade_to_intervals(
                        freq=latency,
                        start=hour_start,
                        end=hour_end,
                        fill_gaps=fill_gaps,
                        rename_for_vecm=True,
                        retain_initial_grid_row=True,
                    )
                    if frame.empty:
                        raise RuntimeError(f"no usable {latency} data")
                    frames[fill_gaps] = frame[["log_midpoint_spot", "log_midpoint_perp"]]

                observed, full = frames[False], frames[True]
                expected = int(pd.Timedelta(hours=1) / pd.Timedelta(latency))
                grid_rows.append({
                    "market": MARKET,
                    "day": day,
                    "hour_utc": HOUR,
                    "latency": latency,
                    "expected_calendar_bins": expected,
                    "observed_union_bins": len(observed),
                    "complete_grid_bins": len(full),
                    "omitted_bins_current": expected - len(observed),
                    "omitted_pct_current": 100 * (expected - len(observed)) / expected,
                })

                fits = [
                    ("current_observed_intercept_float32", observed, False),
                    ("full_grid_demeaned_ec_no_intercept_float64", full, True),
                ]
                for label, frame, corrected in fits:
                    if corrected:
                        result, n_obs, condition = stable_fit(
                            frame, latency, lag_structure,
                            intercept=False, demean_ec=True,
                        )
                    else:
                        result, n_obs, condition = current_fit(frame, latency, lag_structure)
                    rows = rows_for_result(latency, label, result, n_obs, condition, len(frame))
                    for row in rows:
                        row["day"] = day
                        row["hour_start_utc"] = hour_start
                    day_rows.extend(rows)

            save_upsert(
                DAILY_FILE, pd.DataFrame(day_rows),
                ["day", "latency", "specification", "series"],
            )
            save_upsert(GRID_FILE, pd.DataFrame(grid_rows), ["day", "latency"])
            completed.update(
                (day, latency, specification)
                for latency in needed
                for specification in (
                    "current_observed_intercept_float32",
                    "full_grid_demeaned_ec_no_intercept_float64",
                )
            )
            print(f"[saved] {day.date()}", flush=True)

        except Exception as exc:
            error = pd.DataFrame([{
                "day": day,
                "hour_utc": HOUR,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=5).replace("\n", " | "),
            }])
            save_upsert(ERROR_FILE, error, ["day"])
            print(f"[error] {day.date()}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            if "data" in locals():
                del data
            gc.collect()

    results = read_csv(DAILY_FILE)
    if results.empty:
        raise RuntimeError("No daily audit results were produced.")
    perp = results[results["series"].str.contains("perp")].copy()
    perp["leader"] = np.where(perp["his_mid"] > 0.5, "perp", "spot")
    summary = (
        perp.groupby(["latency", "specification"], as_index=False)
        .agg(
            days=("day", "nunique"),
            mean_perp_is=("his_mid", "mean"),
            median_perp_is=("his_mid", "median"),
            perp_leader_days=("leader", lambda x: int((x == "perp").sum())),
        )
    )
    summary["perp_leader_pct"] = 100 * summary["perp_leader_days"] / summary["days"]
    summary.to_csv(OUT / "full_sample_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
