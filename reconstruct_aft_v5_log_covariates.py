"""Create V5 AFT inputs with log1p-transformed liquidity covariates."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from survival_analysis_utils_final import atomic_pandas_csv, atomic_pandas_parquet


MARKETS = ("btc_um", "btc_cm", "eth_um", "eth_cm")
SOURCE_DIR_NAME = "aft_data_liquidity_unit_agreeing_oi"
OUTPUT_DIR_NAME = "aft_data_liquidity_v5_log_covariates"
TRANSFORMS = {
    "open_interest": "log_open_interest",
    "volume": "log_volume",
    "taker_long_short": "log_taker_long_short",
}


def reconstruct_file(source_path: Path, output_path: Path) -> dict:
    existed = output_path.exists()
    frame = pd.read_parquet(source_path).copy()
    for source_column, output_column in TRANSFORMS.items():
        values = pd.to_numeric(frame[source_column], errors="coerce")
        # Preserve missing source observations as missing after transformation;
        # only negative values are invalid for the requested log1p transform.
        if values.dropna().lt(0).any():
            raise ValueError(f"Negative values in {source_column}: {source_path.name}")
        frame[output_column] = np.log1p(values.to_numpy(dtype=float))
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
        output_dir = Path(f"sa_{market}") / OUTPUT_DIR_NAME
        output_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(source_dir.glob("*.parquet"))
        rows = []
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {
                pool.submit(reconstruct_file, path, output_dir / path.name): path
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
        Path("open_interest_figures/native/aft_v5_log_covariates_reconstruction_summary.csv"),
        index=False,
    )
    print(summary.groupby(["market", "status"]).size().to_string())


if __name__ == "__main__":
    main()
