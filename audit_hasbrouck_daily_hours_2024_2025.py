"""Independent 2024-2025 shard of the resumable daily-hour audit."""

from pathlib import Path
import pandas as pd

import audit_hasbrouck_daily_hours as audit


audit.START_DAY = pd.Timestamp("2024-01-01")
audit.END_DAY = pd.Timestamp("2025-12-31")
audit.OUT = Path("hasbrouck_spec_audit/full_sample_daily_hours_2024_2025")
audit.DAILY_FILE = audit.OUT / "daily_results.csv"
audit.GRID_FILE = audit.OUT / "daily_grid_diagnostics.csv"
audit.ERROR_FILE = audit.OUT / "errors.csv"


if __name__ == "__main__":
    audit.main()
