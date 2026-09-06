"""Full Hasbrouck 3 production run for BTC linear and inverse contracts.

Matches the ETH Hasbrouck 3 setup: last-trade prices, observed-bin alignment,
hourly models, six sampling frequencies, ten-second distributed lag horizon,
and ten-day resumable output chunks.
"""

from __future__ import annotations

import gc
from pathlib import Path
import traceback

import pandas as pd

from vecm_hasbrouck3 import VECMHasbrouck2


START = pd.Timestamp("2021-01-01")
END = pd.Timestamp("2026-01-01")
AGGS = ["1s", "500ms", "200ms", "100ms", "50ms", "10ms"]
LAG_BASES = [10]
PERIOD_DAYS = 10
INTERVAL = "1H"
CONFIGS = {
    "btc_um": {
        "symbol": "BTCUSDT",
        "margin": "um",
        "folder": Path("vecm_hasbrouck3_btc_um"),
    },
    "btc_cm": {
        "symbol": "BTCUSDT",
        "margin": "cm",
        "folder": Path("vecm_hasbrouck3_btc_cm"),
    },
}


def main() -> None:
    failures = []
    for market, config in CONFIGS.items():
        folder = config["folder"]
        folder.mkdir(parents=True, exist_ok=True)
        print(f"[market start] {market}", flush=True)
        try:
            model = VECMHasbrouck2(config["symbol"], "Binance", config["margin"])
            model.get_data_multiperiod(
                START,
                END,
                aggs=AGGS,
                interval=INTERVAL,
                period=PERIOD_DAYS,
                folder_name=str(folder),
                prefix="hasbrouck3_btc",
                lag_is=LAG_BASES,
                fill_gaps=False,
                max_fill_gap_ms=None,
                drop_both_stale=False,
                stale_after_ms=None,
                n_jobs=4,
            )
            print(f"[market complete] {market}", flush=True)
        except Exception as exc:
            failures.append({
                "market": market,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
            pd.DataFrame(failures).to_csv("hasbrouck3_btc_run_errors.csv", index=False)
            print(f"[market error] {market}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            if "model" in locals():
                del model
            gc.collect()

    if failures:
        raise RuntimeError(f"Hasbrouck 3 BTC run ended with {len(failures)} market-level error(s).")


if __name__ == "__main__":
    main()
