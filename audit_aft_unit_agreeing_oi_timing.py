"""Audit strict pre-event timing in the reconstructed AFT inputs."""

from pathlib import Path
import pandas as pd


MARKETS = ("btc_um", "btc_cm", "eth_um", "eth_cm")
INPUT_DIR = "aft_data_liquidity_unit_agreeing_oi"


def audit_file(path: Path, market: str) -> dict:
    frame = pd.read_parquet(path, columns=["start_ts", "prev_ts", "covariate_5min_ts", "create_time"])
    start = pd.to_datetime(frame["start_ts"], utc=True, errors="coerce")
    prev = pd.to_datetime(frame["prev_ts"], utc=True, errors="coerce")
    five = pd.to_datetime(frame["covariate_5min_ts"], utc=True, errors="coerce")
    metric = pd.to_datetime(frame["create_time"], utc=True, errors="coerce")
    return {
        "market": market,
        "file": path.name,
        "rows": len(frame),
        "bad_metric_at_or_after_start": int((start.notna() & metric.notna() & (metric >= start)).sum()),
        "bad_5min_at_or_after_start": int((start.notna() & five.notna() & (five >= start)).sum()),
        "bad_5min_after_prev_trade": int((prev.notna() & five.notna() & (five > prev)).sum()),
        "nonmissing_metric": int(metric.notna().sum()),
        "nonmissing_5min": int(five.notna().sum()),
    }


def main() -> None:
    rows = []
    for market in MARKETS:
        directory = Path(f"sa_{market}") / INPUT_DIR
        for path in sorted(directory.glob("*.parquet")):
            rows.append(audit_file(path, market))
    result = pd.DataFrame(rows)
    output = Path("open_interest_figures/native/aft_unit_agreeing_oi_timing_audit.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.groupby("market")[["rows", "bad_metric_at_or_after_start", "bad_5min_at_or_after_start", "bad_5min_after_prev_trade"]].sum().to_string())
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
