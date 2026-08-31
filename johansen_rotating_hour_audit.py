"""Resumable Johansen diagnostics using one rotating UTC hour per day.

Each sampled hour is tested independently, so gaps between sampled days never
become artificial price transitions.  The primary specification matches the
existing VECM diagnostics: observed-trade union bins, det_order=0, and one
lagged difference in statsmodels' Johansen trace test.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from trade_data_pull import TradeData


MARKETS = {
    "btc_um": ("BTCUSDT", "um"),
    "btc_cm": ("BTCUSDT", "cm"),
    "eth_um": ("ETHUSDT", "um"),
    "eth_cm": ("ETHUSDT", "cm"),
}
LATENCIES = ("1s", "500ms", "200ms", "100ms", "50ms", "10ms")
DET_ORDER = 0
K_AR_DIFF = 1
OUTPUT_ROOT = Path("hasbrouck_spec_audit/johansen_rotating_hour")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def save_upsert(path: Path, new: pd.DataFrame, keys: list[str]) -> None:
    old = read_csv(path)
    combined = pd.concat([old, new], ignore_index=True, sort=False)
    if "day" in combined.columns:
        combined["day"] = pd.to_datetime(
            combined["day"], format="mixed", errors="raise"
        ).dt.strftime("%Y-%m-%d")
    combined = combined.drop_duplicates(keys, keep="last").sort_values(keys)
    combined.to_csv(path, index=False)


def clear_error_day(path: Path, day: pd.Timestamp) -> None:
    """Remove a stale error once a resumed day completes successfully."""
    errors = read_csv(path)
    if errors.empty:
        return
    error_days = pd.to_datetime(errors["day"], format="mixed").dt.normalize()
    remaining = errors.loc[error_days.ne(day.normalize())].copy()
    remaining.to_csv(path, index=False)


def johansen_row(prices: pd.DataFrame) -> dict:
    values = prices[["log_midpoint_spot", "log_midpoint_perp"]].astype(np.float64)
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) <= K_AR_DIFF + 5:
        raise RuntimeError(f"too few price rows for Johansen test: {len(values)}")
    if (values.nunique() < 2).any():
        raise RuntimeError("one or more price series is constant in sampled hour")

    test = coint_johansen(values, det_order=DET_ORDER, k_ar_diff=K_AR_DIFF)
    trace = np.asarray(test.lr1, dtype=float)
    trace_cv = np.asarray(test.cvt, dtype=float)
    maxeig = np.asarray(test.lr2, dtype=float)
    maxeig_cv = np.asarray(test.cvm, dtype=float)

    reject_r0_5pct = bool(trace[0] > trace_cv[0, 1])
    reject_r1_5pct = bool(trace[1] > trace_cv[1, 1])
    rank_5pct = 0 if not reject_r0_5pct else (2 if reject_r1_5pct else 1)

    beta = np.asarray(test.evec[:, 0], dtype=float)
    beta = beta / beta[0] if beta[0] != 0 else beta
    return {
        "n_price_rows": len(values),
        "trace_r0": trace[0],
        "trace_r0_cv_90": trace_cv[0, 0],
        "trace_r0_cv_95": trace_cv[0, 1],
        "trace_r0_cv_99": trace_cv[0, 2],
        "trace_r1": trace[1],
        "trace_r1_cv_90": trace_cv[1, 0],
        "trace_r1_cv_95": trace_cv[1, 1],
        "trace_r1_cv_99": trace_cv[1, 2],
        "maxeig_r0": maxeig[0],
        "maxeig_r0_cv_95": maxeig_cv[0, 1],
        "maxeig_r1": maxeig[1],
        "maxeig_r1_cv_95": maxeig_cv[1, 1],
        "reject_r0_5pct": reject_r0_5pct,
        "reject_r1_5pct": reject_r1_5pct,
        "rank_5pct": rank_5pct,
        "beta_spot_normalized": beta[0],
        "beta_perp_normalized": beta[1],
    }


def write_period_summary(results_file: Path, summary_file: Path) -> None:
    results = read_csv(results_file)
    if results.empty:
        return
    results["day"] = pd.to_datetime(results["day"], format="mixed")
    results["period"] = results["day"].dt.year.astype(str)
    summary = (
        results.groupby(["market", "period", "latency"], as_index=False)
        .agg(
            sampled_days=("day", "nunique"),
            median_price_rows=("n_price_rows", "median"),
            trace_r0_median=("trace_r0", "median"),
            trace_r0_cv_95_median=("trace_r0_cv_95", "median"),
            rank_at_least_1_days=("reject_r0_5pct", "sum"),
            rank_2_days=("reject_r1_5pct", "sum"),
            median_rank_5pct=("rank_5pct", "median"),
            median_beta_perp_normalized=("beta_perp_normalized", "median"),
        )
    )
    summary["rank_at_least_1_pct"] = (
        100 * summary["rank_at_least_1_days"] / summary["sampled_days"]
    )
    summary["rank_2_pct"] = 100 * summary["rank_2_days"] / summary["sampled_days"]
    summary.to_csv(summary_file, index=False)


def run_market(market: str, start: pd.Timestamp, end: pd.Timestamp) -> None:
    symbol, contract = MARKETS[market]
    market_out = OUTPUT_ROOT / market
    market_out.mkdir(parents=True, exist_ok=True)
    results_file = market_out / "daily_results.csv"
    errors_file = market_out / "errors.csv"
    summary_file = market_out / "annual_summary.csv"

    completed_df = read_csv(results_file)
    if completed_df.empty:
        completed: set[tuple[pd.Timestamp, str]] = set()
    else:
        completed_df["day"] = pd.to_datetime(completed_df["day"], format="mixed").dt.normalize()
        completed = set(
            completed_df[["day", "latency"]].drop_duplicates().itertuples(index=False, name=None)
        )

    days = pd.date_range(start, end, freq="D")
    rotation_origin = pd.Timestamp("2021-01-01")
    for number, day in enumerate(days, start=1):
        missing = [latency for latency in LATENCIES if (day, latency) not in completed]
        if not missing:
            print(f"[skip {market} {number}/{len(days)}] {day.date()} complete", flush=True)
            continue

        hour = int((day - rotation_origin).days % 24)
        hour_start = day + pd.Timedelta(hours=hour)
        hour_end = hour_start + pd.Timedelta(hours=1)
        print(
            f"[day {market} {number}/{len(days)}] {day.date()} hour={hour:02d}:00 "
            f"({', '.join(missing)})",
            flush=True,
        )

        try:
            data = TradeData(symbol, "Binance", contract)
            data.grab_trades_data(day.to_pydatetime(), days=1, n_jobs=2)
            if data.df_trades_spots.is_empty() or data.df_trades_perps.is_empty():
                raise RuntimeError("spot or perpetual daily archive unavailable/empty")

            rows = []
            for latency in missing:
                frame = data.agg_last_trade_to_intervals(
                    freq=latency,
                    start=hour_start,
                    end=hour_end,
                    fill_gaps=False,
                    rename_for_vecm=True,
                    retain_initial_grid_row=True,
                )
                result = johansen_row(frame)
                result.update({
                    "market": market,
                    "symbol": symbol,
                    "contract": contract,
                    "day": day,
                    "period": str(day.year),
                    "sample_hour_utc": hour,
                    "hour_start_utc": hour_start,
                    "latency": latency,
                    "det_order": DET_ORDER,
                    "k_ar_diff": K_AR_DIFF,
                    "grid": "observed_trade_union",
                })
                rows.append(result)

            save_upsert(results_file, pd.DataFrame(rows), ["day", "latency"])
            clear_error_day(errors_file, day)
            completed.update((day, latency) for latency in missing)
            print(f"[saved] {market} {day.date()}", flush=True)

        except Exception as exc:
            error = pd.DataFrame([{
                "market": market,
                "day": day,
                "sample_hour_utc": hour,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=6).replace("\n", " | "),
            }])
            save_upsert(errors_file, error, ["day"])
            print(f"[error] {market} {day.date()}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            if "data" in locals():
                del data
            gc.collect()

        # The daily result CSV is authoritative.  OneDrive can briefly lock the
        # derived annual summary; that should not terminate a multi-day audit.
        try:
            write_period_summary(results_file, summary_file)
        except PermissionError as exc:
            print(f"[summary deferred] {market}: {exc}", flush=True)

    try:
        write_period_summary(results_file, summary_file)
    except PermissionError as exc:
        print(f"[summary deferred] {market}: {exc}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", nargs="+", choices=MARKETS, default=list(MARKETS))
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2025-12-31")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize()
    if end < start:
        raise ValueError("--end must be on or after --start")
    for market in args.markets:
        run_market(market, start, end)


if __name__ == "__main__":
    main()
