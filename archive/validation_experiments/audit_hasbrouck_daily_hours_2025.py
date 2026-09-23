"""Independent November-December 2025 shard of the resumable audit."""

from pathlib import Path
import pandas as pd

import audit_hasbrouck_daily_hours as audit


audit.START_DAY = pd.Timestamp("2025-11-01")
audit.END_DAY = pd.Timestamp("2025-12-31")
audit.OUT = Path("hasbrouck_spec_audit/full_sample_daily_hours_2025_nov_dec")
audit.DAILY_FILE = audit.OUT / "daily_results.csv"
audit.GRID_FILE = audit.OUT / "daily_grid_diagnostics.csv"
audit.ERROR_FILE = audit.OUT / "errors.csv"


if __name__ == "__main__":
    audit.main()
