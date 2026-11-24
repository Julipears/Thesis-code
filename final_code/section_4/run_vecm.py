"""Resumable Hasbrouck-3 production run for all paper markets."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import traceback

import pandas as pd

from final_code.survival_utils import as_timestamp
from vecm_hasbrouck3 import VECMHasbrouck2


START = pd.Timestamp("2021-01-01")
END = pd.Timestamp("2026-01-01")
AGGS = ["1s", "500ms", "200ms", "100ms", "50ms", "10ms"]
LAG_BASES = [10] # default lag, was found before that lag base didn't really change price leadership
PERIOD_DAYS = 10 # prevent memory blowup by splitting the run into 10-day chunks
INTERVAL = "1h"
CONFIGS = {
    "btc_um": {"symbol": "BTCUSDT", "margin": "um"},
    "btc_cm": {"symbol": "BTCUSDT", "margin": "cm"},
    "eth_um": {"symbol": "ETHUSDT", "margin": "um"},
    "eth_cm": {"symbol": "ETHUSDT", "margin": "cm"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=as_timestamp, default=START, help="First UTC date, inclusive.")
    parser.add_argument("--end", type=as_timestamp, default=END, help="Last UTC date, exclusive.")
    parser.add_argument("--markets", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--latencies", nargs="+", choices=AGGS, default=AGGS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("."), help="Parent directory for the market results.")
    args = parser.parse_args()
    if args.end <= args.start or args.workers < 1:
        parser.error("Use an end date after the start and at least one worker.")
    failures = []
    for market in args.markets:
        config = CONFIGS[market]
        asset, margin = market.split("_")
        folder = args.output / f"vecm_hasbrouck3_{market}"
        folder.mkdir(parents=True, exist_ok=True)
        print(f"[market start] {market}", flush=True)
        try:
            model = VECMHasbrouck2(config["symbol"], "Binance", margin)
            model.get_data_multiperiod(
                args.start,
                args.end,
                aggs=args.latencies,
                interval=INTERVAL,
                period=PERIOD_DAYS,
                folder_name=str(folder),
                prefix=f"hasbrouck3_{asset}",
                lag_is=LAG_BASES,
                fill_gaps=False, # if True, will take a long time to run... but audited
                max_fill_gap_ms=None,
                drop_both_stale=False,
                stale_after_ms=None,
                n_jobs=args.workers,
            )
            print(f"[market complete] {market}", flush=True)
        except Exception as exc:
            failures.append({
                "market": market,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
            pd.DataFrame(failures).to_csv(args.output / "hasbrouck3_run_errors.csv", index=False)
            print(f"[market error] {market}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            if "model" in locals():
                del model
            gc.collect()

    if failures:
        raise RuntimeError(f"Hasbrouck 3 run ended with {len(failures)} market-level error(s).")


if __name__ == "__main__":
    main()
