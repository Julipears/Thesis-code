"""Shared utilities and timing-contract validation for the final survival pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import pandas as pd

PathLike = Union[str, Path]

# -----------------------------------------------------------------------------
# Frozen timing contract.
# -----------------------------------------------------------------------------
COVARIATE_TIMING_VERSION = "prior_day_completed_ffill_1s_event_tick_minus_1_5min_final_v4"
LIQUIDITY_TIMING_VERSION = "strict_pre_shock_metric_final_v1"
PIPELINE_VERSION = "survival_analysis_final_v6_event_tick_minus_1_covariates"


def as_timestamp(value) -> pd.Timestamp:
    """Parse YYYYMMDD strings and ordinary pandas-compatible date values."""
    if isinstance(value, str) and len(value) == 8 and value.isdigit():
        return pd.to_datetime(value, format="%Y%m%d")
    return pd.Timestamp(value)


def format_yyyymmdd(value) -> str:
    return as_timestamp(value).strftime("%Y%m%d")


def naive_utc_ns(values):
    """Return datetime64[ns] values normalized to naive UTC."""
    values = pd.to_datetime(values, errors="coerce", utc=True)
    if isinstance(values, pd.Series):
        return values.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]")
    return values.tz_convert("UTC").tz_localize(None).astype("datetime64[ns]")


def build_periods(start, end, freq: str = "MS") -> List[Tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Build non-overlapping [start, end) calendar periods."""
    start_ts = as_timestamp(start)
    end_ts = as_timestamp(end)
    if end_ts <= start_ts:
        raise ValueError("end must be later than start")

    boundaries = list(pd.date_range(start=start_ts, end=end_ts, freq=freq))
    if not boundaries or boundaries[0] != start_ts:
        boundaries.insert(0, start_ts)
    if boundaries[-1] != end_ts:
        boundaries.append(end_ts)

    periods = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        if right <= left:
            continue
        upper = freq.upper()
        if upper.startswith("M"):
            label = left.strftime("%Y-%m")
        elif upper.startswith("Q"):
            label = f"{left.year}-Q{left.quarter}"
        elif upper.startswith(("Y", "A")):
            label = str(left.year)
        else:
            label = f"{left:%Y-%m-%d} to {right:%Y-%m-%d}"
        periods.append((left, right, label))
    return periods


def iter_chunks(start, end, chunk_days: int):
    """Yield non-overlapping [start, end) chunks."""
    if chunk_days < 1:
        raise ValueError("chunk_days must be >= 1")
    left = as_timestamp(start)
    end_ts = as_timestamp(end)
    while left < end_ts:
        right = min(left + pd.Timedelta(days=chunk_days), end_ts)
        yield left, right, f"{left:%Y-%m-%d}_to_{right:%Y-%m-%d}"
        left = right


def safe_label(label: str) -> str:
    return str(label).replace("/", "-").replace(" ", "_")


def atomic_pandas_csv(frame: pd.DataFrame, path: PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def atomic_pandas_parquet(frame: pd.DataFrame, path: PathLike, compression: str = "zstd") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temp, index=False, compression=compression)
    os.replace(temp, path)


def atomic_polars_parquet(frame, path: PathLike, compression: str = "zstd") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.write_parquet(temp, compression=compression, statistics=False)
    os.replace(temp, path)


def write_manifest_row(manifest_path: PathLike, row: dict, key_cols: Sequence[str] = ("period", "first")) -> None:
    """Upsert a manifest row and save atomically."""
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        if all(col in manifest.columns for col in key_cols) and all(col in row for col in key_cols):
            mask = pd.Series(True, index=manifest.index)
            for col in key_cols:
                mask &= manifest[col].astype(str).eq(str(row[col]))
            manifest = manifest.loc[~mask]
    else:
        manifest = pd.DataFrame()

    manifest = pd.concat([manifest, pd.DataFrame([row])], ignore_index=True)
    present_sort = [col for col in key_cols if col in manifest.columns]
    if present_sort:
        manifest = manifest.sort_values(present_sort).reset_index(drop=True)
    atomic_pandas_csv(manifest, manifest_path)


def timing_values_ok(frame: pd.DataFrame, column: str, expected: str) -> bool:
    if column not in frame.columns:
        return False
    values = frame[column].dropna().astype(str).unique()
    return len(values) == 0 or (len(values) == 1 and values[0] == expected)


def validate_base_covariate_timing(frame: pd.DataFrame, source_name: str = "frame") -> bool:
    """Validate the frozen base-covariate timing contract."""
    required = {
        "start_ts",
        "prev_ts",
        "covariate_5min_ts",
        "daily_covariate_date",
        "covariate_timing_version",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"{source_name} is missing timing-audit columns: {missing}")

    if not timing_values_ok(frame, "covariate_timing_version", COVARIATE_TIMING_VERSION):
        raise RuntimeError(
            f"{source_name} has an old/unknown covariate timing version. "
            f"Expected {COVARIATE_TIMING_VERSION}."
        )

    start_ts = pd.to_datetime(frame["start_ts"], utc=True, errors="coerce")
    prev_ts = pd.to_datetime(frame["prev_ts"], utc=True, errors="coerce")
    cov5_ts = pd.to_datetime(frame["covariate_5min_ts"], utc=True, errors="coerce")

    bad_prev = start_ts.notna() & prev_ts.notna() & (prev_ts >= start_ts)
    if bad_prev.any():
        raise AssertionError(f"{source_name}: {int(bad_prev.sum())} rows have prev_ts >= start_ts")

    bad_5min = start_ts.notna() & cov5_ts.notna() & (cov5_ts >= start_ts)
    if bad_5min.any():
        raise AssertionError(
            f"{source_name}: {int(bad_5min.sum())} 5-minute covariate timestamps are not strictly before start_ts"
        )

    bad_5min_prev = prev_ts.notna() & cov5_ts.notna() & (cov5_ts > prev_ts)
    if bad_5min_prev.any():
        raise AssertionError(
            f"{source_name}: {int(bad_5min_prev.sum())} 5-minute covariate timestamps are after prev_ts"
        )

    daily_used = pd.to_datetime(frame["daily_covariate_date"], errors="coerce").dt.date
    required_daily = (start_ts.dt.normalize() - pd.Timedelta(days=1)).dt.date
    bad_daily = daily_used.notna() & required_daily.notna() & (daily_used != required_daily)
    if bad_daily.any():
        raise AssertionError(
            f"{source_name}: {int(bad_daily.sum())} daily covariate dates are not the previous calendar day"
        )

    return True


def validate_augmented_timing(frame: pd.DataFrame, source_name: str = "frame") -> bool:
    """Validate base timing plus strictly-pre-shock Binance metrics."""
    validate_base_covariate_timing(frame, source_name)

    if not timing_values_ok(frame, "liquidity_timing_version", LIQUIDITY_TIMING_VERSION):
        raise RuntimeError(
            f"{source_name} has an old/unknown liquidity timing version. "
            f"Expected {LIQUIDITY_TIMING_VERSION}."
        )
    if "create_time" not in frame.columns:
        raise KeyError(f"{source_name} is missing create_time")

    start_ts = pd.to_datetime(frame["start_ts"], utc=True, errors="coerce")
    metric_ts = pd.to_datetime(frame["create_time"], utc=True, errors="coerce")
    bad_metric = start_ts.notna() & metric_ts.notna() & (metric_ts >= start_ts)
    if bad_metric.any():
        raise AssertionError(
            f"{source_name}: {int(bad_metric.sum())} liquidity timestamps are not strictly before start_ts"
        )
    return True


def parquet_has_timing_version(path: PathLike, augmented: bool = False) -> bool:
    """Cheap resume check; False means rebuild rather than trust an old file."""
    path = Path(path)
    if not path.exists():
        return False
    columns = ["covariate_timing_version"]
    if augmented:
        columns.append("liquidity_timing_version")
    try:
        frame = pd.read_parquet(path, columns=columns)
    except Exception:
        return False
    if not timing_values_ok(frame, "covariate_timing_version", COVARIATE_TIMING_VERSION):
        return False
    if augmented and not timing_values_ok(frame, "liquidity_timing_version", LIQUIDITY_TIMING_VERSION):
        return False
    return True
