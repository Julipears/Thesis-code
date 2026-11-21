"""Build the V5 log-covariate AFT inputs from the V8 event set.

The existing ``sa_*`` AFT inputs are based on the older saved-KM event set.
This script uses the V8 monthly event/covariate files, replaces open interest
with unit-agreeing Binance fields (USD-M ``sum_open_interest`` and COIN-M
``sum_open_interest_value``), and applies the V5 log1p transformations.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from survival_analysis_utils_final import atomic_pandas_csv, atomic_pandas_parquet


MARKETS = ("btc_um", "btc_cm", "eth_um", "eth_cm")
ROOT = Path("sa_results/km_v8_final_01")
SOURCE_DIR = ROOT / "aft_data_liquidity_monthly"
OUTPUT_DIR = ROOT / "aft_data_liquidity_v5_log_covariates"
RAW_METRICS = Path("open_interest_figures/native/raw_metrics")


def _load_oi_metrics(market: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Load raw OI observations covering a monthly event file."""
    days = pd.date_range(start.floor("D") - pd.Timedelta(days=1), end.ceil("D"), freq="D")
    frames = []
    for day in days:
        path = RAW_METRICS / market / f"{day:%Y-%m-%d}.csv.gz"
        if not path.exists():
            continue
        frame = pd.read_csv(path, usecols=["create_time", "sum_open_interest", "sum_open_interest_value"])
        frame["create_time"] = pd.to_datetime(frame["create_time"], utc=True, errors="coerce")
        for column in ("sum_open_interest", "sum_open_interest_value"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frames.append(frame.dropna(subset=["create_time"]))
    if not frames:
        return pd.DataFrame(columns=["create_time", "open_interest"])
    metrics = pd.concat(frames, ignore_index=True).drop_duplicates("create_time", keep="last")
    source_column = "sum_open_interest" if market.endswith("_um") else "sum_open_interest_value" # this ensures the unit will always be units of the cryptocurrency
    metrics = metrics.rename(columns={source_column: "open_interest"})[["create_time", "open_interest"]]
    return metrics.sort_values("create_time")


def reconstruct_file(source_path: Path, output_path: Path, market: str) -> dict:
    frame = pd.read_parquet(source_path).copy()
    frame["start_ts"] = pd.to_datetime(frame["start_ts"], utc=True, errors="coerce")
    if frame["start_ts"].notna().any():
        metrics = _load_oi_metrics(market, frame["start_ts"].min(), frame["start_ts"].max())
    else:
        metrics = pd.DataFrame(columns=["create_time", "open_interest"])

    valid = frame.sort_values("start_ts").copy()
    valid = valid.drop(columns=["create_time", "open_interest"], errors="ignore")
    if metrics.empty:
        valid["create_time"] = pd.NaT
        valid["open_interest"] = np.nan
    else:
        valid = pd.merge_asof(
            valid,
            metrics,
            left_on="start_ts",
            right_on="create_time",
            direction="backward",
            allow_exact_matches=False,
        )
        check = valid["create_time"].notna() & ~(valid["create_time"] < valid["start_ts"])
        if check.any():
            raise AssertionError(f"lookahead in {source_path}: {int(check.sum())} rows")

    for source_column, output_column in (
        ("open_interest", "log_open_interest"),
        ("volume", "log_volume"),
        ("taker_long_short", "log_taker_long_short"),
    ):
        values = pd.to_numeric(valid[source_column], errors="coerce")
        if values.dropna().lt(0).any():
            raise ValueError(f"negative values in {source_column}: {source_path.name}")
        valid[output_column] = np.log1p(values.to_numpy(dtype=float))

    valid = valid.sort_index() if len(valid) == len(frame) else valid
    atomic_pandas_parquet(valid, output_path)
    return {
        "source_file": source_path.name,
        "output_file": str(output_path),
        "market": market,
        "rows": len(valid),
        "open_interest_nonnull": int(valid["open_interest"].notna().sum()),
        "status": "written",
    }


def main() -> None:
    manifests = []
    for market in MARKETS:
        source_dir = SOURCE_DIR / market
        output_dir = OUTPUT_DIR / market
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = sorted(source_dir.glob("*.parquet"))
        rows = []
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {
                pool.submit(reconstruct_file, path, output_dir / path.name, market): path
                for path in paths
            }
            for index, future in enumerate(as_completed(futures), 1):
                path = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {"source_file": path.name, "market": market, "status": "failed", "error": repr(exc)}
                rows.append(row)
                if index % 20 == 0 or index == len(paths):
                    print(f"[{market}] {index}/{len(paths)} {row['status']}", flush=True)
        manifest = pd.DataFrame(rows).sort_values("source_file")
        atomic_pandas_csv(manifest, output_dir / "reconstruction_manifest.csv")
        manifests.append(manifest)
    summary = pd.concat(manifests, ignore_index=True)
    summary.to_csv(ROOT / "v5_log_covariate_reconstruction_manifest.csv", index=False)
    print(summary.groupby(["market", "status"]).size().to_string())


if __name__ == "__main__":
    main()
