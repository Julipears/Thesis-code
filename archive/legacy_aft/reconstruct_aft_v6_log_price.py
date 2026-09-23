"""Add the unstandardized prior-day log spot price to the V5 AFT inputs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from survival_analysis_utils_final import atomic_pandas_csv, atomic_pandas_parquet


MARKETS = ("btc_um", "btc_cm", "eth_um", "eth_cm")
SOURCE_DIR_NAME = "aft_data_liquidity_v5_log_covariates"
PRICE_DIR_NAME = "aft_data_liquidity_v4_daily_price"
OUTPUT_DIR_NAME = "aft_data_liquidity_v6_log_price"


def reconstruct_file(source_path: Path, price_path: Path, output_path: Path) -> dict:
    existed = output_path.exists()
    frame = pd.read_parquet(source_path).copy()
    price_frame = pd.read_parquet(price_path, columns=["event_tick", "daily_log_price"])
    if len(frame) != len(price_frame) or not frame["event_tick"].reset_index(drop=True).equals(
        price_frame["event_tick"].reset_index(drop=True)
    ):
        raise ValueError(f"V5 and price rows do not align: {source_path.name}")
    prices = pd.to_numeric(price_frame["daily_log_price"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(prices).all():
        raise ValueError(f"Missing or non-finite daily log prices: {source_path.name}")
    frame["daily_log_price"] = prices
    atomic_pandas_parquet(frame, output_path)
    return {
        "source_file": source_path.name,
        "status": "updated_existing" if existed else "written",
        "output_file": str(output_path),
        "rows": len(frame),
    }


def main() -> None:
    manifests = []
    for market in MARKETS:
        source_dir = Path(f"sa_{market}") / SOURCE_DIR_NAME
        price_dir = Path(f"sa_{market}") / PRICE_DIR_NAME
        output_dir = Path(f"sa_{market}") / OUTPUT_DIR_NAME
        output_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(source_dir.glob("*.parquet"))
        rows = []
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {
                pool.submit(reconstruct_file, path, price_dir / path.name, output_dir / path.name): path
                for path in files
            }
            for index, future in enumerate(as_completed(futures), 1):
                path = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {"source_file": path.name, "status": "failed", "error": repr(exc)}
                row["market"] = market
                rows.append(row)
                if index % 20 == 0 or index == len(files):
                    print(f"[reconstruct {market}] {index}/{len(files)} {row['status']}", flush=True)
        manifest = pd.DataFrame(rows).sort_values("source_file")
        atomic_pandas_csv(manifest, output_dir / "reconstruction_manifest.csv")
        manifests.append(manifest)
    summary = pd.concat(manifests, ignore_index=True)
    summary.to_csv(
        Path("open_interest_figures/native/aft_v6_log_price_reconstruction_summary.csv"),
        index=False,
    )
    print(summary.groupby(["market", "status"]).size().to_string())


if __name__ == "__main__":
    main()
