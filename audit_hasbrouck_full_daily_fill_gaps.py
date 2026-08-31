"""Run the gap-filling-only Hasbrouck audit for every day and contract.

This is a thin configuration wrapper around the resumable stratified runner.
It retains the same 00:00--01:00 UTC daily window and compares the existing
observed-bin implementation with the identical implementation using a full
regular grid (``fill_gaps=True``).
"""

from pathlib import Path

import pandas as pd

import audit_hasbrouck_stratified_other_contracts as audit


audit.CONTRACTS = {
    "btc_um": ("BTCUSDT", "um"),
    "btc_cm": ("BTCUSDT", "cm"),
    "eth_um": ("ETHUSDT", "um"),
    "eth_cm": ("ETHUSDT", "cm"),
}
audit.OUT = Path("hasbrouck_spec_audit/full_daily_fill_gaps")
audit.DAILY_FILE = audit.OUT / "daily_results.csv"
audit.GRID_FILE = audit.OUT / "daily_grid_diagnostics.csv"
audit.ERROR_FILE = audit.OUT / "errors.csv"
audit.sample_dates = lambda: pd.date_range("2021-01-01", "2025-12-31", freq="D")


if __name__ == "__main__":
    audit.main()
