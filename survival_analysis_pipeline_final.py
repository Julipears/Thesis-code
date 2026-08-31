"""Orchestration and model fitting for the final survival-analysis workflow."""

from __future__ import annotations

import gc
import os
import shutil
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import polars as pl
from lifelines import KaplanMeierFitter, LogLogisticAFTFitter, LogNormalAFTFitter, WeibullAFTFitter

from survival_analysis_data_processing_final import (
    BasisSurvivalAnalysis,
    augment_liquidity_metrics,
    detect_basis_events_from_event_time,
    build_aft_covariates,
    reconstruct_saved_event_ticks,
    assert_saved_km_outcomes_unchanged,
    prepare_aft_data,
)
from survival_analysis_data_pull_final import OUTPUT_METRIC_COLUMNS, pull_or_load_market_metrics
from survival_analysis_utils_final import (
    COVARIATE_TIMING_VERSION,
    LIQUIDITY_TIMING_VERSION,
    PIPELINE_VERSION,
    as_timestamp,
    atomic_pandas_csv,
    atomic_pandas_parquet,
    atomic_polars_parquet,
    build_periods,
    iter_chunks,
    parquet_has_timing_version,
    validate_augmented_timing,
    validate_base_covariate_timing,
    write_manifest_row,
)


# Current specification used in the liquidity-augmented AFT notebook.
DEFAULT_AFT_COVARIATES = [
    "basis_5min",
    "fundingRate_bps",
    "open_interest",
    "shock_size_bps",
    "spread_spot_5min",
    "taker_long_short",
    # "trader_long_short",  # available in the data; leave excluded unless intentionally added.
    "vol_spot_5min_pct",
    "volume",
]

AFT_MODELS = {
    "weibull": WeibullAFTFitter,
    "lognormal": LogNormalAFTFitter,
    "loglogistic": LogLogisticAFTFitter,
}

DEFAULT_MARKETS = {
    # km_checkpoint_dir points at the EXISTING v6 root containing event_store/.
    # run_dir is also where the new corrected AFT-only outputs are written.
    "btc_um": {"run_dir": Path("./sa_btc_um"), "km_checkpoint_dir": Path("./sa_btc_um"), "symbol": "BTCUSDT", "cm_um": "um"},
    "btc_cm": {"run_dir": Path("./sa_btc_cm"), "km_checkpoint_dir": Path("./sa_btc_cm"), "symbol": "BTCUSDT", "cm_um": "cm"},
    "eth_um": {"run_dir": Path("./sa_eth_um"), "km_checkpoint_dir": Path("./sa_eth_um"), "symbol": "ETHUSDT", "cm_um": "um"},
    "eth_cm": {"run_dir": Path("./sa_eth_cm"), "km_checkpoint_dir": Path("./sa_eth_cm"), "symbol": "ETHUSDT", "cm_um": "cm"},
}


def _read_parquets(paths) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in paths if Path(path).exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _file_is_current_base(path: Path) -> bool:
    return parquet_has_timing_version(path, augmented=False)


def _file_is_current_augmented(path: Path) -> bool:
    return parquet_has_timing_version(path, augmented=True)


FINAL_BASE_AFT_DIR = "aft_data_final"
FINAL_LIQUIDITY_AFT_DIR = "aft_data_liquidity_final"
FINAL_AFT_RESULTS_DIR = "aft_results_final"
KM_REUSE_VERSION = "v5_monthly_km_events_read_only_final_v2"


def _discover_saved_km_event_files(km_checkpoint_dir, first: str) -> list[Path]:
    """Find the event rows underlying the saved v5 monthly KM curves.

    V5 fits ``km/<first>_YYYY-MM.csv`` from the pooled episode rows saved as
    ``events/<first>_YYYY-MM.parquet``.  Those monthly event files are therefore
    authoritative.  Chunked ``event_store`` files remain a compatibility
    fallback only.  ``aft_data`` is deliberately excluded because its legacy
    builder could redetect events independently.
    """
    root = Path(km_checkpoint_dir)
    v5_events = root / "events"
    if v5_events.exists():
        files = sorted(v5_events.glob(f"{first}_*.parquet"))
        if files:
            return files

    event_store = root / "event_store" / first
    if event_store.exists():
        files = sorted(event_store.glob("*.parquet"))
        if files:
            return files

    raise FileNotFoundError(
        f"No saved event-level KM checkpoints found for {first!r} under {root}. "
        f"Expected {v5_events}/{first}_*.parquet (v5) or "
        f"{event_store}/*.parquet (fallback). "
        "KM curve CSVs by themselves are not sufficient to rebuild AFT covariates."
    )


def _split_saved_km_file_into_source_chunks(
    path: Path,
    primary_resolution_pct: int,
    start=None,
    end=None,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]]:
    """Return immutable saved KM episode groups with their original source bounds."""
    frame = pd.read_parquet(path)
    if frame.empty:
        return []
    required = {"start_ts", "prev_ts", "Status", "Length"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"{path.name} is missing saved KM event columns: {sorted(missing)}")

    frame = frame.copy()
    frame["start_ts"] = pd.to_datetime(frame["start_ts"], errors="coerce")
    frame["prev_ts"] = pd.to_datetime(frame["prev_ts"], errors="coerce")
    frame = frame.dropna(subset=["start_ts", "prev_ts"])
    if "resolution_pct" in frame.columns:
        frame = frame[frame["resolution_pct"].astype(int).eq(int(primary_resolution_pct))].copy()
    else:
        frame["resolution_pct"] = int(primary_resolution_pct)

    if start is not None:
        frame = frame[frame["start_ts"] >= as_timestamp(start)]
    if end is not None:
        frame = frame[frame["start_ts"] < as_timestamp(end)]
    if frame.empty:
        return []

    if {"source_chunk_start", "source_chunk_end"}.issubset(frame.columns):
        bound_cols = ["source_chunk_start", "source_chunk_end"]
    elif {"period_start", "period_end"}.issubset(frame.columns):
        bound_cols = ["period_start", "period_end"]
    else:
        # Last-resort compatibility for a saved monthly event file that lacks
        # source-bound metadata.  This remains safe because exact start_ts/prev_ts
        # reconstruction below must succeed before covariates are written.
        left = frame["start_ts"].min().floor("D")
        right = frame["start_ts"].max().ceil("D")
        if right <= left:
            right = left + pd.Timedelta(days=1)
        return [(left, right, frame)]

    groups = []
    for bounds, subset in frame.groupby(bound_cols, dropna=False, sort=True):
        if not isinstance(bounds, tuple):
            bounds = (bounds,)
        left = pd.to_datetime(bounds[0])
        right = pd.to_datetime(bounds[1])
        if pd.isna(left) or pd.isna(right) or right <= left:
            raise ValueError(f"{path.name} contains invalid source chunk bounds: {bounds}")
        groups.append((pd.Timestamp(left), pd.Timestamp(right), subset.copy()))
    return groups


def build_aft_from_saved_km_all_directions(
    symbol: str,
    cm_um: str,
    km_checkpoint_dir,
    run_dir=None,
    start: str = "20210101",
    end: str = "20260101",
    firsts=("spot", "perp"),
    primary_resolution_pct: int = 90,
    load_lookback_days: int = 1,
    load_followup_days: int = 1,
    resume: bool = True,
    parquet_compression: str = "zstd",
) -> dict[str, pd.DataFrame]:
    """Build corrected AFT covariates from existing KM episode checkpoints.

    No shock detection, basis-resolution calculation, KM fitting, or KM plotting is
    performed.  ``Length`` and ``Status`` are immutable inputs from the saved KM
    event store.  Raw trades are reloaded only so the original event-time rows can
    be reconstructed and leakage-safe covariates attached.
    """
    firsts = tuple(firsts)
    if any(first not in {"spot", "perp"} for first in firsts):
        raise ValueError("firsts may only contain 'spot' and/or 'perp'")

    km_root = Path(km_checkpoint_dir)
    run_root = Path(run_dir) if run_dir is not None else km_root
    aft_dir = run_root / FINAL_BASE_AFT_DIR
    chunk_root = run_root / "aft_reuse_km_chunks_final"
    aft_dir.mkdir(parents=True, exist_ok=True)
    chunk_root.mkdir(parents=True, exist_ok=True)

    # Build an index of original source chunks across both shock directions.
    chunk_groups: dict[tuple[pd.Timestamp, pd.Timestamp], dict[str, list[tuple[Path, pd.DataFrame]]]] = {}
    source_files = {}
    for first in firsts:
        files = _discover_saved_km_event_files(km_root, first)
        source_files[first] = files
        for path in files:
            for left, right, subset in _split_saved_km_file_into_source_chunks(
                path,
                primary_resolution_pct=primary_resolution_pct,
                start=start,
                end=end,
            ):
                chunk_groups.setdefault((left, right), {}).setdefault(first, []).append((path, subset))

    if not chunk_groups:
        raise ValueError("Saved KM checkpoints contained no episodes in the requested date range.")

    analysis = BasisSurvivalAnalysis(symbol, source="Binance", cm_um=cm_um)
    manifest_rows = []

    for (chunk_start, chunk_end), by_first in sorted(chunk_groups.items()):
        chunk_label = f"{chunk_start:%Y-%m-%d}_to_{chunk_end:%Y-%m-%d}"
        needed = []
        for first in firsts:
            if first not in by_first:
                continue
            output_path = chunk_root / first / f"{chunk_label}.parquet"
            if resume and _file_is_current_base(output_path):
                try:
                    validate_base_covariate_timing(pd.read_parquet(output_path), output_path.name)
                    print(f"[skip reused-KM chunk] {symbol} {cm_um} {first} {chunk_label}")
                    continue
                except Exception as exc:
                    print(f"[rebuild reused-KM chunk] {output_path.name}: {exc}")
            needed.append(first)

        if not needed:
            continue

        print(f"[raw pull for saved KM] {symbol} {cm_um} {chunk_label}; directions={needed}")
        data = event_time = None
        try:
            load_start = chunk_start - pd.Timedelta(days=load_lookback_days)
            load_end = chunk_end + pd.Timedelta(days=load_followup_days)
            data, event_time = analysis._get_event_time_data(
                load_start,
                load_end,
                retain_raw_data=True,
            )

            for first in needed:
                entries = by_first[first]
                saved = pd.concat([subset for _, subset in entries], ignore_index=True)
                # Defensive deduplication only; conflicting duplicate outcomes fail below.
                dedup_keys = ["start_ts", "resolution_pct"]
                if saved.duplicated(dedup_keys).any():
                    duplicate_block = saved[saved.duplicated(dedup_keys, keep=False)]
                    outcome_counts = duplicate_block.groupby(dedup_keys)[["Status", "Length"]].nunique(dropna=False)
                    if (outcome_counts > 1).any().any():
                        raise AssertionError(
                            f"Conflicting duplicate KM outcomes in source chunk {chunk_label} ({first})."
                        )
                    saved = saved.drop_duplicates(dedup_keys, keep="first")

                reconstructed = reconstruct_saved_event_ticks(event_time, saved)
                enriched = build_aft_covariates(
                    start=chunk_start,
                    end=chunk_end,
                    data=data,
                    event_time_prices=event_time,
                    events=reconstructed,
                    primary_resolution_pct=primary_resolution_pct,
                )
                assert_saved_km_outcomes_unchanged(
                    reconstructed,
                    enriched,
                    source_name=f"{first} {chunk_label}",
                )

                enriched["period"] = pd.to_datetime(enriched["start_ts"]).dt.to_period("M").astype(str)
                enriched["chunk"] = chunk_label
                enriched["first"] = first
                enriched["resolution_pct"] = int(primary_resolution_pct)
                if "shock_significance" not in enriched.columns:
                    enriched["shock_significance"] = np.nan
                enriched["pipeline_version"] = PIPELINE_VERSION
                enriched["km_reuse_version"] = KM_REUSE_VERSION
                enriched["km_source_mode"] = "saved_event_checkpoint"
                enriched["km_source_files"] = ";".join(sorted({path.name for path, _ in entries}))

                output_path = chunk_root / first / f"{chunk_label}.parquet"
                atomic_pandas_parquet(enriched, output_path, compression=parquet_compression)
                manifest_rows.append({
                    "first": first,
                    "chunk": chunk_label,
                    "status": "complete",
                    "saved_km_rows": len(saved),
                    "aft_rows": len(enriched),
                    "km_source_files": enriched["km_source_files"].iloc[0] if len(enriched) else "",
                    "covariate_timing_version": COVARIATE_TIMING_VERSION,
                    "km_reuse_version": KM_REUSE_VERSION,
                })
                print(f"  [saved corrected covariates] {first} {chunk_label}: {len(enriched):,} rows")
        finally:
            del data, event_time
            gc.collect()

    # Pool corrected chunk outputs into the same monthly granularity used for AFT fits.
    start_ts = as_timestamp(start)
    end_ts = as_timestamp(end)
    for first in firsts:
        chunk_files = sorted((chunk_root / first).glob("*.parquet"))
        if not chunk_files:
            continue
        pooled = _read_parquets(chunk_files)
        if pooled.empty:
            continue
        pooled["start_ts"] = pd.to_datetime(pooled["start_ts"])
        pooled = pooled[(pooled["start_ts"] >= start_ts) & (pooled["start_ts"] < end_ts)].copy()
        pooled = pooled.sort_values("start_ts")
        pooled = pooled.drop_duplicates(["start_ts", "resolution_pct"], keep="first")

        for month_start, month_end, period in build_periods(start_ts, end_ts, "MS"):
            monthly = pooled[(pooled["start_ts"] >= month_start) & (pooled["start_ts"] < month_end)].copy()
            output_path = aft_dir / f"{first}_{period}.parquet"
            if monthly.empty:
                if output_path.exists():
                    output_path.unlink()
                continue
            validate_base_covariate_timing(monthly, f"{first}_{period}")
            atomic_pandas_parquet(monthly, output_path, compression=parquet_compression)

        del pooled
        gc.collect()

    manifest = pd.DataFrame(manifest_rows)
    atomic_pandas_csv(manifest, run_root / "aft_reuse_km_manifest_final.csv")
    return {
        first: manifest[manifest["first"].eq(first)].reset_index(drop=True)
        if not manifest.empty else pd.DataFrame()
        for first in firsts
    }


def build_market_event_and_aft_data(
    symbol: str,
    cm_um: str,
    run_dir,
    start: str = "20210101",
    end: str = "20260101",
    first: str = "spot",
    shock_significance: float = 0.001,
    method: str = "lee_mykland",
    resolution_pcts=(90,),
    primary_resolution_pct: int = 90,
    max_duration_seconds: float = 120.0,
    refractory_seconds: float = 5.0,
    min_hold_seconds: float = 0.25,
    basis_tolerance_bps: float = 0.5,
    min_basis_shock_bps: float = 0.0,
    load_followup_days: int = 1,
    load_lookback_days: int = 1,
    chunk_days: int = 17,
    resume: bool = True,
    keep_chunk_files: bool = False,
    parquet_compression: str = "zstd",
) -> pd.DataFrame:
    """Pull once per chunk, then save both event and covariate-enriched outputs.

    The exact event-time frame used for shock detection is handed directly to
    build_aft_covariates, so event_tick cannot be reinterpreted by a second
    reconstruction.
    """
    if primary_resolution_pct not in {int(x) for x in resolution_pcts}:
        raise ValueError("primary_resolution_pct must be included in resolution_pcts")

    run_dir = Path(run_dir)
    events_dir = run_dir / "events"
    aft_dir = run_dir / "aft_data"
    chunk_root = run_dir / "final_chunks" / first
    manifest_path = run_dir / f"final_build_manifest_{first}.csv"
    for directory in (events_dir, aft_dir, chunk_root):
        directory.mkdir(parents=True, exist_ok=True)

    analysis = BasisSurvivalAnalysis(symbol, source="Binance", cm_um=cm_um)

    for month_start, month_end, period in build_periods(start, end, "MS"):
        event_output = events_dir / f"{first}_{period}.parquet"
        aft_output = aft_dir / f"{first}_{period}.parquet"

        if resume and event_output.exists() and _file_is_current_base(aft_output):
            try:
                validate_base_covariate_timing(pd.read_parquet(aft_output), aft_output.name)
                print(f"[skip current] {symbol} {cm_um} {first} {period}")
                continue
            except Exception as exc:
                print(f"[rebuild unverified] {aft_output.name}: {exc}")

        month_chunk_dir = chunk_root / period
        month_chunk_dir.mkdir(parents=True, exist_ok=True)
        event_chunk_paths = []
        aft_chunk_paths = []

        try:
            for chunk_start, chunk_end, chunk_label in iter_chunks(month_start, month_end, chunk_days):
                event_chunk_path = month_chunk_dir / f"events_{chunk_label}.parquet"
                aft_chunk_path = month_chunk_dir / f"aft_{chunk_label}.parquet"
                event_chunk_paths.append(event_chunk_path)
                aft_chunk_paths.append(aft_chunk_path)

                if resume and event_chunk_path.exists() and _file_is_current_base(aft_chunk_path):
                    try:
                        validate_base_covariate_timing(pd.read_parquet(aft_chunk_path), aft_chunk_path.name)
                        print(f"  [skip chunk] {chunk_label}")
                        continue
                    except Exception as exc:
                        print(f"  [rebuild chunk] {chunk_label}: {exc}")

                print(f"  [build chunk] {symbol} {cm_um} {first}: {chunk_label}")
                data = event_time = events = aft = None
                try:
                    data, event_time, events = analysis.fit_basis_events(
                        start=chunk_start,
                        end=chunk_end,
                        shock_significance=shock_significance,
                        method=method,
                        resolution_pcts=resolution_pcts,
                        max_duration_seconds=max_duration_seconds,
                        refractory_seconds=refractory_seconds,
                        min_hold_seconds=min_hold_seconds,
                        basis_tolerance_bps=basis_tolerance_bps,
                        min_basis_shock_bps=min_basis_shock_bps,
                        first=first,
                        load_followup_days=load_followup_days,
                        load_lookback_days=load_lookback_days,
                        retain_raw_data=True,
                    )

                    if events.height:
                        events = events.with_columns(
                            pl.lit(period).alias("period"),
                            pl.lit(chunk_label).alias("chunk"),
                            pl.lit(PIPELINE_VERSION).alias("pipeline_version"),
                        )
                        aft = build_aft_covariates(
                            start=chunk_start,
                            end=chunk_end,
                            data=data,
                            event_time_prices=event_time,
                            events=events,
                            primary_resolution_pct=primary_resolution_pct,
                        )
                        aft["period"] = period
                        aft["chunk"] = chunk_label
                        aft["first"] = first
                        aft["resolution_pct"] = primary_resolution_pct
                        aft["shock_significance"] = shock_significance
                        aft["pipeline_version"] = PIPELINE_VERSION
                    else:
                        aft = pd.DataFrame()

                    atomic_polars_parquet(events, event_chunk_path, compression=parquet_compression)
                    atomic_pandas_parquet(aft, aft_chunk_path, compression=parquet_compression)
                finally:
                    del data, event_time, events, aft
                    gc.collect()

            monthly_events = _read_parquets(event_chunk_paths)
            monthly_aft = _read_parquets(aft_chunk_paths)
            if not monthly_aft.empty:
                validate_base_covariate_timing(monthly_aft, f"{first}_{period}")
            atomic_pandas_parquet(monthly_events, event_output, compression=parquet_compression)
            atomic_pandas_parquet(monthly_aft, aft_output, compression=parquet_compression)

            write_manifest_row(
                manifest_path,
                {
                    "period": period,
                    "first": first,
                    "status": "complete",
                    "symbol": symbol,
                    "cm_um": cm_um,
                    "event_rows": len(monthly_events),
                    "aft_rows": len(monthly_aft),
                    "chunk_days": chunk_days,
                    "covariate_timing_version": COVARIATE_TIMING_VERSION,
                    "pipeline_version": PIPELINE_VERSION,
                    "events_file": str(event_output),
                    "aft_file": str(aft_output),
                },
            )
            print(f"[saved] {symbol} {cm_um} {first} {period}: {len(monthly_aft):,} AFT rows")

            if not keep_chunk_files:
                shutil.rmtree(month_chunk_dir, ignore_errors=True)
        except Exception as exc:
            write_manifest_row(
                manifest_path,
                {
                    "period": period,
                    "first": first,
                    "status": "failed",
                    "symbol": symbol,
                    "cm_um": cm_um,
                    "error": repr(exc),
                    "covariate_timing_version": COVARIATE_TIMING_VERSION,
                    "pipeline_version": PIPELINE_VERSION,
                },
            )
            raise

    return pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()



def build_market_all_directions(
    symbol: str,
    cm_um: str,
    run_dir,
    start: str = "20210101",
    end: str = "20260101",
    firsts=("spot", "perp"),
    shock_significance: float = 0.001,
    method: str = "lee_mykland",
    resolution_pcts=(90,),
    primary_resolution_pct: int = 90,
    max_duration_seconds: float = 120.0,
    refractory_seconds: float = 5.0,
    min_hold_seconds: float = 0.25,
    basis_tolerance_bps: float = 0.5,
    min_basis_shock_bps: float = 0.0,
    load_followup_days: int = 1,
    load_lookback_days: int = 1,
    chunk_days: int = 17,
    resume: bool = True,
    keep_chunk_files: bool = False,
    parquet_compression: str = "zstd",
) -> dict[str, pd.DataFrame]:
    """Build spot-leading and perp-leading outputs from ONE raw pull per chunk."""
    firsts = tuple(firsts)
    if primary_resolution_pct not in {int(x) for x in resolution_pcts}:
        raise ValueError("primary_resolution_pct must be included in resolution_pcts")
    if any(first not in {"spot", "perp"} for first in firsts):
        raise ValueError("firsts may only contain 'spot' and/or 'perp'")

    run_dir = Path(run_dir)
    events_dir = run_dir / "events"
    aft_dir = run_dir / "aft_data"
    chunk_root = run_dir / "final_chunks"
    for directory in (events_dir, aft_dir, chunk_root):
        directory.mkdir(parents=True, exist_ok=True)

    analysis = BasisSurvivalAnalysis(symbol, source="Binance", cm_um=cm_um)
    manifest_paths = {first: run_dir / f"final_build_manifest_{first}.csv" for first in firsts}

    for month_start, month_end, period in build_periods(start, end, "MS"):
        month_current = True
        for first in firsts:
            event_output = events_dir / f"{first}_{period}.parquet"
            aft_output = aft_dir / f"{first}_{period}.parquet"
            if not (event_output.exists() and _file_is_current_base(aft_output)):
                month_current = False
                break
            try:
                validate_base_covariate_timing(pd.read_parquet(aft_output), aft_output.name)
            except Exception:
                month_current = False
                break
        if resume and month_current:
            print(f"[skip current month] {symbol} {cm_um} {period} (spot + perp)")
            continue

        month_chunk_dir = chunk_root / period
        month_chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_paths = {first: {"events": [], "aft": []} for first in firsts}

        for chunk_start, chunk_end, chunk_label in iter_chunks(month_start, month_end, chunk_days):
            needed = []
            for first in firsts:
                event_chunk = month_chunk_dir / f"events_{first}_{chunk_label}.parquet"
                aft_chunk = month_chunk_dir / f"aft_{first}_{chunk_label}.parquet"
                chunk_paths[first]["events"].append(event_chunk)
                chunk_paths[first]["aft"].append(aft_chunk)
                current = event_chunk.exists() and _file_is_current_base(aft_chunk)
                if current:
                    try:
                        validate_base_covariate_timing(pd.read_parquet(aft_chunk), aft_chunk.name)
                    except Exception:
                        current = False
                if not (resume and current):
                    needed.append(first)

            if not needed:
                print(f"  [skip chunk] {chunk_label} (spot + perp verified)")
                continue

            print(f"  [raw pull once] {symbol} {cm_um}: {chunk_label}; building {needed}")
            data = event_time = None
            try:
                load_start = chunk_start - pd.Timedelta(days=load_lookback_days)
                load_end = chunk_end + pd.Timedelta(days=load_followup_days)
                data, event_time = analysis._get_event_time_data(
                    load_start,
                    load_end,
                    retain_raw_data=True,
                )

                for first in needed:
                    event_chunk = month_chunk_dir / f"events_{first}_{chunk_label}.parquet"
                    aft_chunk = month_chunk_dir / f"aft_{first}_{chunk_label}.parquet"
                    events = detect_basis_events_from_event_time(
                        event_time=event_time,
                        start=chunk_start,
                        end=chunk_end,
                        shock_significance=shock_significance,
                        method=method,
                        resolution_pcts=resolution_pcts,
                        max_duration_seconds=max_duration_seconds,
                        refractory_seconds=refractory_seconds,
                        min_hold_seconds=min_hold_seconds,
                        basis_tolerance_bps=basis_tolerance_bps,
                        min_basis_shock_bps=min_basis_shock_bps,
                        first=first,
                    )
                    if events.height:
                        events = events.with_columns(
                            pl.lit(period).alias("period"),
                            pl.lit(chunk_label).alias("chunk"),
                            pl.lit(PIPELINE_VERSION).alias("pipeline_version"),
                        )
                        aft = build_aft_covariates(
                            start=chunk_start,
                            end=chunk_end,
                            data=data,
                            event_time_prices=event_time,
                            events=events,
                            primary_resolution_pct=primary_resolution_pct,
                        )
                        aft["period"] = period
                        aft["chunk"] = chunk_label
                        aft["first"] = first
                        aft["resolution_pct"] = primary_resolution_pct
                        aft["shock_significance"] = shock_significance
                        aft["pipeline_version"] = PIPELINE_VERSION
                    else:
                        aft = pd.DataFrame(columns=[
                            "start_ts", "prev_ts", "covariate_5min_ts",
                            "daily_covariate_date", "covariate_timing_version",
                            "period", "chunk", "first", "resolution_pct",
                            "shock_significance", "pipeline_version",
                        ])

                    atomic_polars_parquet(events, event_chunk, compression=parquet_compression)
                    atomic_pandas_parquet(aft, aft_chunk, compression=parquet_compression)
                    del events, aft
                    gc.collect()
            finally:
                del data, event_time
                gc.collect()

        # Pool the small checkpoint rows into monthly outputs for each direction.
        for first in firsts:
            monthly_events = _read_parquets(chunk_paths[first]["events"])
            monthly_aft = _read_parquets(chunk_paths[first]["aft"])
            if not monthly_aft.empty:
                validate_base_covariate_timing(monthly_aft, f"{first}_{period}")

            event_output = events_dir / f"{first}_{period}.parquet"
            aft_output = aft_dir / f"{first}_{period}.parquet"
            atomic_pandas_parquet(monthly_events, event_output, compression=parquet_compression)
            atomic_pandas_parquet(monthly_aft, aft_output, compression=parquet_compression)
            write_manifest_row(
                manifest_paths[first],
                {
                    "period": period,
                    "first": first,
                    "status": "complete",
                    "symbol": symbol,
                    "cm_um": cm_um,
                    "event_rows": len(monthly_events),
                    "aft_rows": len(monthly_aft),
                    "chunk_days": chunk_days,
                    "covariate_timing_version": COVARIATE_TIMING_VERSION,
                    "pipeline_version": PIPELINE_VERSION,
                    "events_file": str(event_output),
                    "aft_file": str(aft_output),
                },
            )
            print(f"[saved] {symbol} {cm_um} {first} {period}: {len(monthly_aft):,} AFT rows")
            del monthly_events, monthly_aft

        if not keep_chunk_files:
            shutil.rmtree(month_chunk_dir, ignore_errors=True)
        gc.collect()

    return {
        first: (pd.read_csv(path) if path.exists() else pd.DataFrame())
        for first, path in manifest_paths.items()
    }


def augment_market_liquidity(
    run_dir,
    symbol: str,
    cm_um: str,
    overwrite: bool = False,
    redownload_metrics: bool = False,
    source_dir_name: str = "aft_data",
    output_dir_name: str = "aft_data_liquidity",
) -> pd.DataFrame:
    """Attach strictly-pre-shock Binance metrics to every base AFT parquet."""
    run_dir = Path(run_dir)
    source_dir = run_dir / source_dir_name
    output_dir = run_dir / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    event_files = sorted(source_dir.glob("*.parquet"))
    if not event_files:
        raise FileNotFoundError(f"No base AFT files in {source_dir}")

    for path in event_files:
        validate_base_covariate_timing(pd.read_parquet(path), path.name)

    metrics = pull_or_load_market_metrics(
        run_dir=run_dir,
        symbol=symbol,
        cm_um=cm_um,
        event_files=event_files,
        redownload=redownload_metrics,
    )

    rows = []
    for index, input_path in enumerate(event_files, start=1):
        output_path = output_dir / input_path.name
        if output_path.exists() and not overwrite and _file_is_current_augmented(output_path):
            try:
                validate_augmented_timing(pd.read_parquet(output_path), output_path.name)
                rows.append({
                    "input_file": str(input_path),
                    "output_file": str(output_path),
                    "status": "skipped_existing_verified",
                })
                print(f"[{index}/{len(event_files)}] {input_path.name}: skipped verified")
                continue
            except Exception as exc:
                print(f"Rebuilding unverified augmented file {output_path.name}: {exc}")

        base = pd.read_parquet(input_path)
        augmented = augment_liquidity_metrics(base, metrics)
        atomic_pandas_parquet(augmented, output_path)
        rows.append({
            "input_file": str(input_path),
            "output_file": str(output_path),
            "status": "written_verified",
            "rows": len(augmented),
            "open_interest_missing_share": float(augmented["open_interest"].isna().mean()),
            "trader_long_short_missing_share": float(augmented["trader_long_short"].isna().mean()),
            "taker_long_short_missing_share": float(augmented["taker_long_short"].isna().mean()),
            "liquidity_timing_version": LIQUIDITY_TIMING_VERSION,
        })
        print(f"[{index}/{len(event_files)}] {input_path.name}: written verified")

    manifest = pd.DataFrame(rows)
    atomic_pandas_csv(manifest, output_dir / "liquidity_augmentation_manifest.csv")
    return manifest


def verify_market_timing(
    run_dir,
    require_liquidity: bool = True,
    base_dir_name: str = "aft_data",
    liquidity_dir_name: str = "aft_data_liquidity",
) -> pd.DataFrame:
    """Read every saved parquet and fail on the first timing violation."""
    run_dir = Path(run_dir)
    directory = run_dir / (liquidity_dir_name if require_liquidity else base_dir_name)
    paths = sorted(directory.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files in {directory}")

    rows = []
    for path in paths:
        frame = pd.read_parquet(path)
        if require_liquidity:
            validate_augmented_timing(frame, path.name)
        else:
            validate_base_covariate_timing(frame, path.name)
        rows.append({"file": path.name, "rows": len(frame), "status": "verified"})
    return pd.DataFrame(rows)



def fit_km_grid(
    events: pd.DataFrame,
    timeline_step_seconds: float = 1.0,
    max_duration_seconds: float = 120.0,
) -> pd.DataFrame:
    """Fit Kaplan-Meier curves on a common time grid from saved event rows."""
    if events.empty:
        return pd.DataFrame()
    group_cols = [
        column for column in ("period", "first", "shock_significance", "resolution_pct")
        if column in events.columns
    ]
    grouped = events.groupby(group_cols, dropna=False, sort=True) if group_cols else [((), events)]
    rows = []
    for group_key, subset in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        subset = subset.dropna(subset=["Length", "Status"])
        if subset.empty:
            continue
        timeline = np.arange(
            0.0,
            float(max_duration_seconds) + timeline_step_seconds * 0.5,
            timeline_step_seconds,
        )
        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=subset["Length"].astype(float),
            event_observed=subset["Status"].astype(int),
            timeline=timeline,
            label="survival",
        )
        curve = pd.DataFrame({
            "time_s": timeline,
            "survival": kmf.survival_function_.iloc[:, 0].to_numpy(),
            "ci_lower": kmf.confidence_interval_.iloc[:, 0].to_numpy(),
            "ci_upper": kmf.confidence_interval_.iloc[:, 1].to_numpy(),
        })
        for column, value in zip(group_cols, group_key):
            curve[column] = value
        curve["n_episodes"] = len(subset)
        curve["n_resolved"] = int(subset["Status"].sum())
        curve["censoring_share"] = 1.0 - subset["Status"].mean()
        rows.append(curve)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def fit_market_km_results(
    run_dir,
    timeline_step_seconds: float = 1.0,
    max_duration_seconds: float = 120.0,
) -> pd.DataFrame:
    """Fit and save monthly/directional KM grids from event parquets; no raw repull."""
    run_dir = Path(run_dir)
    source_dir = run_dir / "events"
    output_dir = run_dir / "km"
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(source_dir.glob("*.parquet"))
    rows = []
    for path in files:
        events = pd.read_parquet(path)
        grid = fit_km_grid(events, timeline_step_seconds, max_duration_seconds)
        output = output_dir / f"{path.stem}.csv"
        atomic_pandas_csv(grid, output)
        rows.append({
            "source_file": path.name,
            "km_file": str(output),
            "event_rows": len(events),
            "grid_rows": len(grid),
            "status": "complete",
        })
    manifest = pd.DataFrame(rows)
    atomic_pandas_csv(manifest, output_dir / "km_manifest.csv")
    return manifest


def fit_one_aft_file(
    input_path,
    results_dir,
    covariates: Sequence[str] = DEFAULT_AFT_COVARIATES,
    overwrite: bool = False,
    min_complete_observations: int = 100,
) -> pd.DataFrame:
    """Fit Weibull, Log-Normal, and Log-Logistic AFT models to one file."""
    input_path = Path(input_path)
    results_dir = Path(results_dir)
    model_data, diagnostics = prepare_aft_data(input_path, covariates)

    if len(model_data) < min_complete_observations:
        return pd.DataFrame([{
            "source_file": input_path.name,
            "model": None,
            "status": "skipped_too_few_observations",
            **diagnostics,
        }])

    rows = []
    for model_name, model_class in AFT_MODELS.items():
        coefficient_dir = results_dir / model_name / "coefficients"
        coefficient_dir.mkdir(parents=True, exist_ok=True)
        coefficient_path = coefficient_dir / f"{input_path.stem}_coefficients.csv"
        if coefficient_path.exists() and not overwrite:
            rows.append({
                "source_file": input_path.name,
                "model": model_name,
                "status": "skipped_existing",
                **diagnostics,
            })
            continue

        fitter = model_class()
        try:
            fitter.fit(model_data, duration_col="Length", event_col="Status", ancillary=False)
            coefficients = fitter.summary.reset_index()
            try:
                n_parameters = int(len(fitter.params_))
            except Exception:
                n_parameters = int(len(coefficients))
            bic = n_parameters * np.log(len(model_data)) - 2 * fitter.log_likelihood_

            coefficients["source_file"] = input_path.name
            coefficients["model"] = model_name
            coefficients["observations"] = len(model_data)
            coefficients["events"] = int(model_data["Status"].sum())
            coefficients["censored"] = int(len(model_data) - model_data["Status"].sum())
            coefficients["concordance"] = fitter.concordance_index_
            coefficients["log_likelihood"] = fitter.log_likelihood_
            coefficients["AIC"] = fitter.AIC_
            coefficients["BIC"] = bic
            coefficients["n_parameters"] = n_parameters
            coefficients["covariate_timing_version"] = COVARIATE_TIMING_VERSION
            coefficients["liquidity_timing_version"] = LIQUIDITY_TIMING_VERSION
            atomic_pandas_csv(coefficients, coefficient_path)

            rows.append({
                "source_file": input_path.name,
                "model": model_name,
                "status": "complete",
                "coefficient_file": str(coefficient_path),
                "concordance": fitter.concordance_index_,
                "log_likelihood": fitter.log_likelihood_,
                "AIC": fitter.AIC_,
                "BIC": bic,
                "n_parameters": n_parameters,
                **diagnostics,
            })
        except Exception as exc:
            rows.append({
                "source_file": input_path.name,
                "model": model_name,
                "status": "failed",
                "error": repr(exc),
                **diagnostics,
            })
    return pd.DataFrame(rows)


def fit_market_aft_models(
    run_dir,
    covariates: Sequence[str] = DEFAULT_AFT_COVARIATES,
    overwrite: bool = False,
    min_complete_observations: int = 100,
    input_dir_name: str = "aft_data_liquidity",
    results_dir_name: str = "aft_results_liquidity",
) -> pd.DataFrame:
    """Fit all three AFT distributions to every augmented monthly/directional file."""
    run_dir = Path(run_dir)
    input_dir = run_dir / input_dir_name
    results_dir = run_dir / results_dir_name
    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No augmented AFT files in {input_dir}")

    result_frames = []
    for index, path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] fitting {path.name}")
        validate_augmented_timing(pd.read_parquet(path), path.name)
        result_frames.append(
            fit_one_aft_file(
                path,
                results_dir,
                covariates=covariates,
                overwrite=overwrite,
                min_complete_observations=min_complete_observations,
            )
        )

    manifest = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    results_dir.mkdir(parents=True, exist_ok=True)
    atomic_pandas_csv(manifest, results_dir / "aft_fit_manifest.csv")
    return manifest


def run_market_pipeline_from_saved_km(
    market: Mapping,
    start: str = "20210101",
    end: str = "20260101",
    primary_resolution_pct: int = 90,
    load_lookback_days: int = 1,
    load_followup_days: int = 1,
    covariates: Sequence[str] = DEFAULT_AFT_COVARIATES,
    resume: bool = True,
    overwrite_augmented: bool = False,
    redownload_metrics: bool = False,
    overwrite_results: bool = True,
    min_complete_observations: int = 100,
):
    """Reuse saved KM episode outcomes and rerun only covariates + AFT models.

    ``market['km_checkpoint_dir']`` may point at the old v6 checkpoint root. If
    omitted, ``market['run_dir']`` is used. Existing KM event/curve files are read
    only; they are never overwritten or refit by this function.
    """
    run_dir = Path(market["run_dir"])
    km_checkpoint_dir = Path(market.get("km_checkpoint_dir", run_dir))
    manifests = {}

    manifests.update({
        f"km_reuse_{first}": value
        for first, value in build_aft_from_saved_km_all_directions(
            symbol=market["symbol"],
            cm_um=market["cm_um"],
            km_checkpoint_dir=km_checkpoint_dir,
            run_dir=run_dir,
            start=start,
            end=end,
            firsts=("spot", "perp"),
            primary_resolution_pct=primary_resolution_pct,
            load_lookback_days=load_lookback_days,
            load_followup_days=load_followup_days,
            resume=resume,
        ).items()
    })

    # Deliberately NO fit_market_km_results() call here.
    manifests["base_verification"] = verify_market_timing(
        run_dir,
        require_liquidity=False,
        base_dir_name=FINAL_BASE_AFT_DIR,
        liquidity_dir_name=FINAL_LIQUIDITY_AFT_DIR,
    )
    manifests["liquidity"] = augment_market_liquidity(
        run_dir=run_dir,
        symbol=market["symbol"],
        cm_um=market["cm_um"],
        overwrite=overwrite_augmented,
        redownload_metrics=redownload_metrics,
        source_dir_name=FINAL_BASE_AFT_DIR,
        output_dir_name=FINAL_LIQUIDITY_AFT_DIR,
    )
    manifests["liquidity_verification"] = verify_market_timing(
        run_dir,
        require_liquidity=True,
        base_dir_name=FINAL_BASE_AFT_DIR,
        liquidity_dir_name=FINAL_LIQUIDITY_AFT_DIR,
    )
    manifests["aft_models"] = fit_market_aft_models(
        run_dir,
        covariates=covariates,
        overwrite=overwrite_results,
        min_complete_observations=min_complete_observations,
        input_dir_name=FINAL_LIQUIDITY_AFT_DIR,
        results_dir_name=FINAL_AFT_RESULTS_DIR,
    )
    manifests["km_status"] = pd.DataFrame([{
        "status": "reused_existing_event_checkpoints",
        "km_checkpoint_dir": str(km_checkpoint_dir),
        "km_refit_performed": False,
        "km_reuse_version": KM_REUSE_VERSION,
    }])
    return manifests


def run_market_pipeline(
    market: Mapping,
    start: str = "20210101",
    end: str = "20260101",
    build_settings: Optional[Mapping] = None,
    covariates: Sequence[str] = DEFAULT_AFT_COVARIATES,
    resume: bool = True,
    overwrite_augmented: bool = False,
    redownload_metrics: bool = False,
    overwrite_results: bool = True,
    min_complete_observations: int = 100,
):
    """Final public runner: reuse saved KM episodes and rebuild AFT inputs only.

    This function intentionally does NOT redetect shocks, recompute survival
    outcomes, refit Kaplan-Meier models, or overwrite KM outputs. It is an alias
    for :func:`run_market_pipeline_from_saved_km` so the final workflow has only
    one supported execution path.
    """
    if build_settings:
        raise ValueError(
            "build_settings is not used in the final reuse-KM workflow. "
            "Existing KM episode checkpoints are immutable inputs."
        )
    return run_market_pipeline_from_saved_km(
        market=market,
        start=start,
        end=end,
        covariates=covariates,
        resume=resume,
        overwrite_augmented=overwrite_augmented,
        redownload_metrics=redownload_metrics,
        overwrite_results=overwrite_results,
        min_complete_observations=min_complete_observations,
    )


def run_all_markets(
    markets: Mapping = DEFAULT_MARKETS,
    start: str = "20210101",
    end: str = "20260101",
    build_settings: Optional[Mapping] = None,
    covariates: Sequence[str] = DEFAULT_AFT_COVARIATES,
    **kwargs,
):
    outputs = {}
    for name, market in markets.items():
        print("\n" + "=" * 80)
        print(f"FINAL REUSE-KM PIPELINE: {name}")
        print("=" * 80)
        outputs[name] = run_market_pipeline(
            market=market,
            start=start,
            end=end,
            build_settings=build_settings,
            covariates=covariates,
            **kwargs,
        )
    return outputs
