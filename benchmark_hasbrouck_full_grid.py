"""Time a small full-grid-only sample using the existing audit estimator."""
from pathlib import Path
from time import perf_counter
import gc
import json

import pandas as pd

from audit_hasbrouck_one_hour import current_fit, rows_for_result
from trade_data_pull import TradeData
from vecm_hasbrouck3 import generate_multiple_lags


def main():
    out = Path("hasbrouck_spec_audit/full_grid_timing_sample")
    out.mkdir(parents=True, exist_ok=True)
    contracts = {"btc_um": ("BTCUSDT", "um"), "btc_cm": ("BTCUSDT", "cm"),
                 "eth_um": ("ETHUSDT", "um"), "eth_cm": ("ETHUSDT", "cm")}
    latencies = ["1s", "100ms", "10ms"]
    lags = generate_multiple_lags(10, latencies, max_length="10s")
    timing, results, days = [], [], []
    for day_text in ["2021-07-01", "2025-07-01"]:
        day = pd.Timestamp(day_text)
        for market, (symbol, margin) in contracts.items():
            started = perf_counter()
            print(f"START {market} {day_text}", flush=True)
            data = TradeData(symbol, "Binance", margin)
            data.grab_trades_data(day.to_pydatetime(), days=1, n_jobs=2)
            load_seconds = perf_counter() - started
            if data.df_trades_spots.is_empty() or data.df_trades_perps.is_empty():
                raise RuntimeError(f"Missing data for {market} {day_text}")
            print(f"LOADED {market} {day_text}: {load_seconds:.2f}s", flush=True)
            for latency in latencies:
                begin = perf_counter()
                frame = data.agg_last_trade_to_intervals(
                    freq=latency, start=day, end=day + pd.Timedelta(hours=1),
                    fill_gaps=True, rename_for_vecm=True, retain_initial_grid_row=True,
                )[["log_midpoint_spot", "log_midpoint_perp"]]
                aggregation_seconds = perf_counter() - begin
                begin = perf_counter()
                fit, n_obs, condition = current_fit(frame, latency, lags)
                fit_seconds = perf_counter() - begin
                timing.append(dict(market=market, day=day_text, latency=latency,
                                   load_seconds=load_seconds, aggregation_seconds=aggregation_seconds,
                                   fit_seconds=fit_seconds, n_rows=len(frame), n_obs=n_obs))
                rows = rows_for_result(latency, "full_grid_intercept_float32", fit, n_obs, condition, len(frame))
                for row in rows:
                    row.update(market=market, day=day_text, hour_start_utc=day)
                results.extend(rows)
                pd.DataFrame(timing).to_csv(out / "timings.csv", index=False)
                pd.DataFrame(results).to_csv(out / "results.csv", index=False)
                print(f"FIT {market} {day_text} {latency}: aggregation={aggregation_seconds:.2f}s fit={fit_seconds:.2f}s rows={len(frame)}", flush=True)
                del frame, fit
            del data
            gc.collect()
            elapsed = perf_counter() - started
            days.append(dict(market=market, day=day_text, load_seconds=load_seconds, total_seconds=elapsed))
            pd.DataFrame(days).to_csv(out / "contract_day_timings.csv", index=False)
            print(f"DONE {market} {day_text}: {elapsed:.2f}s", flush=True)
    summary = dict(sample_contract_days=len(days), fits=len(timing),
                   mean_contract_day_seconds=pd.DataFrame(days).total_seconds.mean(),
                   projected_sequential_days=pd.DataFrame(days).total_seconds.mean() * 7304 / 86400)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
