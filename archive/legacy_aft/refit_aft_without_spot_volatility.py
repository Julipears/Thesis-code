"""Checkpointed monthly AFT refit after removing spot volatility for VIF."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import pandas as pd

from survival_analysis_pipeline_final import fit_one_aft_file
from survival_analysis_utils_final import atomic_pandas_csv


COVARIATES = [
    "basis_5min",
    "fundingRate_bps",
    "open_interest",
    "shock_size_bps",
    "spread_spot_5min",
    "taker_long_short",
    "volume",
]
MODELS = ("weibull", "lognormal", "loglogistic")
MARKETS = ("btc_um", "btc_cm", "eth_um", "eth_cm")


def fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def is_complete(results_dir: Path, path: Path) -> bool:
    return all(
        (results_dir / model / "coefficients" / f"{path.stem}_coefficients.csv").exists()
        for model in MODELS
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="sa_results/km_v8_final_01")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    input_root = root / "aft_data_liquidity_monthly"
    results_root = root / "aft_results_no_spot_volatility_by_market"
    manifest_path = results_root / "aft_fit_manifest.csv"
    results_root.mkdir(parents=True, exist_ok=True)

    files = [
        path
        for market in MARKETS
        for first in ("spot", "perp")
        for path in sorted((input_root / market).glob(f"{first}_*.parquet"))
    ]
    if not files:
        raise FileNotFoundError(f"No monthly AFT inputs under {input_root}")

    prior = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
    manifest_frames = [prior] if not prior.empty else []
    start = time.monotonic()
    completed_at_start = sum(
        is_complete(results_root / path.parent.name, path) for path in files
    )
    processed_this_run = 0

    print(
        f"[start] files={len(files)} already_complete={completed_at_start} "
        f"covariates={'|'.join(COVARIATES)}",
        flush=True,
    )

    for index, path in enumerate(files, start=1):
        market = path.parent.name
        first, period = path.stem.split("_", 1)
        market_results_dir = results_root / market
        if is_complete(market_results_dir, path) and not args.overwrite:
            print(
                f"[skip {index}/{len(files)}] market={market} contract={first} "
                f"month={period} already_complete",
                flush=True,
            )
            continue

        file_start = time.monotonic()
        rows = fit_one_aft_file(
            path,
            market_results_dir,
            covariates=COVARIATES,
            overwrite=args.overwrite,
            min_complete_observations=100,
        )
        rows.insert(0, "market", market)
        rows.insert(1, "first", first)
        rows.insert(2, "period", period)
        rows["covariates"] = "|".join(COVARIATES)
        manifest_frames.append(rows)
        manifest = pd.concat(manifest_frames, ignore_index=True)
        manifest = manifest.drop_duplicates(
            ["market", "first", "period", "model"], keep="last"
        )
        atomic_pandas_csv(
            manifest.sort_values(["market", "first", "period", "model"]),
            manifest_path,
        )
        manifest_frames = [manifest]

        processed_this_run += 1
        elapsed = time.monotonic() - start
        average = elapsed / processed_this_run
        remaining = len(files) - index
        statuses = ",".join(
            f"{row.model}:{row.status}" for row in rows.itertuples(index=False)
        )
        print(
            f"[done {index}/{len(files)}] market={market} contract={first} "
            f"month={period} file_time={fmt_seconds(time.monotonic() - file_start)} "
            f"run_elapsed={fmt_seconds(elapsed)} eta={fmt_seconds(average * remaining)} "
            f"{statuses}",
            flush=True,
        )
        del rows, manifest
        gc.collect()

    final = pd.read_csv(manifest_path)
    print("[complete]", flush=True)
    print(final.groupby(["market", "first", "status"]).size().to_string(), flush=True)


if __name__ == "__main__":
    main()
