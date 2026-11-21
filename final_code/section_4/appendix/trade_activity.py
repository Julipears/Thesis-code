"""Trade/update-frequency diagnostics for the existing TradeData pipeline.

This module is designed to work with the user's current ``TradeData`` class
from ``trade_data_pull.py``.  It does not fit a VECM.  It measures the
transaction/update activity that the calendar-time VECM actually sees.

Important limitation
--------------------
The current downloader reduces the exchange files to at most one retained
observation per millisecond and trade side and discards quantity.  Therefore,
unless the loaded Polars frames contain a ``trade_count`` column, this module
cannot recover the exact raw exchange trade count.  It reports
``*_retained_observation_count`` instead.  The occupied-bin and price-change
metrics remain directly relevant to the last-trade VECM.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import polars as pl

from trade_data_pull import TradeData


ACTIVITY_DIAGNOSTIC_BASIS = "last_trade_update_activity_v1"


def _as_python_datetime(value):
    if value is None:
        return None
    return pd.Timestamp(value).to_pydatetime()


def _filter_trade_frame(
    frame: pl.DataFrame,
    start=None,
    end=None,
) -> pl.DataFrame:
    """Return ``frame`` restricted to the half-open interval [start, end)."""
    out = frame
    start_dt = _as_python_datetime(start)
    end_dt = _as_python_datetime(end)

    if start_dt is not None:
        out = out.filter(pl.col("timestamp") >= pl.lit(start_dt))
    if end_dt is not None:
        out = out.filter(pl.col("timestamp") < pl.lit(end_dt))
    return out.sort("timestamp")


def _market_bin_summary(
    data: TradeData,
    frame: pl.DataFrame,
    *,
    market: str,
    freq: str,
    window: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Create observed-bin prices and per-window market-only diagnostics."""
    if market not in {"spot", "perp"}:
        raise ValueError("market must be 'spot' or 'perp'.")

    price_col = f"last_trade_{market}"
    observed_count_col = f"{market}_retained_observation_count"
    exact_count_col = f"{market}_raw_trade_count"
    change_count_col = f"{market}_price_change_bin_count"
    median_gap_col = f"{market}_median_interupdate_ms"

    observed = (
        data.to_intervals_last_trade(frame, freq=freq, fill_gaps=False)
        .rename({"last_trade_price": price_col})
        .with_columns(
            pl.col("timestamp_bin")
            .cast(pl.Datetime("ms"))
            .alias("timestamp_bin")
        )
        .sort("timestamp_bin")
    )

    if observed.height == 0:
        empty_obs = pl.DataFrame({
            "timestamp_bin": pl.Series([], dtype=pl.Datetime("ms")),
            price_col: pl.Series([], dtype=pl.Float64),
        })
        empty_summary = pl.DataFrame({
            "window_start": pl.Series([], dtype=pl.Datetime("ms")),
            observed_count_col: pl.Series([], dtype=pl.Int64),
            exact_count_col: pl.Series([], dtype=pl.Float64),
            change_count_col: pl.Series([], dtype=pl.Int64),
            median_gap_col: pl.Series([], dtype=pl.Float64),
        })
        return empty_obs, empty_summary

    # A price-change bin is an occupied bin whose last-trade price differs from
    # the preceding occupied bin for that same market.  The first bin is not
    # counted as a change.
    observed = observed.with_columns([
        pl.col("timestamp_bin").dt.truncate(window).alias("window_start"),
        (
            pl.col(price_col).shift(1).is_not_null()
            & (pl.col(price_col) != pl.col(price_col).shift(1))
        ).alias(f"_{market}_price_changed"),
        (
            (
                pl.col("timestamp_bin")
                .cast(pl.Datetime("ns"))
                .cast(pl.Int64)
                .diff()
            )
            .cast(pl.Float64)
            / 1_000_000.0
        ).alias(f"_{market}_interupdate_ms"),
    ])

    binned_summary = (
        observed.group_by("window_start", maintain_order=True)
        .agg([
            pl.col(f"_{market}_price_changed")
            .cast(pl.Int64)
            .sum()
            .alias(change_count_col),
            pl.col(f"_{market}_interupdate_ms")
            .median()
            .alias(median_gap_col),
        ])
    )

    # The current loader normally stores one retained row per millisecond and
    # trade side.  This is not the exact exchange trade count, so it is named
    # explicitly as a retained-observation count.
    raw = frame.with_columns(
        pl.col("timestamp")
        .cast(pl.Datetime("ms"))
        .dt.truncate(window)
        .alias("window_start")
    )

    raw_aggregations = [
        pl.len().alias(observed_count_col),
    ]

    # A future/modified loader may preserve exact raw counts in each retained
    # row.  Use them when available; otherwise return NaN rather than silently
    # mislabelling retained observations as exact trades.
    if "trade_count" in raw.columns:
        raw_aggregations.append(
            pl.col("trade_count").sum().cast(pl.Float64).alias(exact_count_col)
        )
    else:
        raw_aggregations.append(
            pl.lit(None, dtype=pl.Float64).alias(exact_count_col)
        )

    raw_summary = (
        raw.group_by("window_start", maintain_order=True)
        .agg(raw_aggregations)
    )

    summary = (
        raw_summary.join(binned_summary, on="window_start", how="outer")
        .with_columns(
            pl.coalesce(["window_start", "window_start_right"])
            .alias("window_start")
        )
        .drop("window_start_right")
        .sort("window_start")
    )

    # Keep only the two columns needed for the aligned-market calculations.
    return observed.select(["timestamp_bin", price_col]), summary


def summarize_loaded_trade_activity(
    data: TradeData,
    *,
    freq: str,
    start=None,
    end=None,
    window: str = "1h",
    contract_type: Optional[str] = None,
) -> pd.DataFrame:
    """Summarize spot/perp update activity for already-loaded trade data.

    Parameters
    ----------
    data
        A ``TradeData`` instance after ``grab_trades_data`` has been called.
    freq
        Calendar-time aggregation used by the VECM, for example ``'10ms'``,
        ``'50ms'``, ``'100ms'``, ``'200ms'``, ``'500ms'``, or ``'1s'``.
    start, end
        Optional half-open sample range ``[start, end)``.
    window
        Diagnostic grouping window.  Use ``'1h'`` to match hourly VECMs.
    contract_type
        Optional label such as ``'linear'`` or ``'inverse'``.

    Returns
    -------
    pandas.DataFrame
        One row per diagnostic window.  Important columns include:

        - ``spot_observed_bin_count`` / ``perp_observed_bin_count``
        - ``spot_price_change_bin_count`` / ``perp_price_change_bin_count``
        - ``spot_median_interupdate_ms`` / ``perp_median_interupdate_ms``
        - ``spot_forward_fill_share`` / ``perp_forward_fill_share``
        - perp-to-spot ratios and log ratios

    Notes
    -----
    ``observed_bin_count`` is the main measure for the VECM mechanism: it
    counts how many calendar-time bins contain a fresh last-trade observation.
    ``forward_fill_share`` is measured on the union of occupied spot/perp bins
    after both markets have an initial observation, matching the last-trade
    alignment logic used by ``agg_last_trade_to_intervals(fill_gaps=False)``.
    """
    if not hasattr(data, "df_trades_spots") or not hasattr(data, "df_trades_perps"):
        raise AttributeError(
            "Trade data are not loaded. Call data.grab_trades_data(...) first."
        )

    spots = _filter_trade_frame(data.df_trades_spots, start=start, end=end)
    perps = _filter_trade_frame(data.df_trades_perps, start=start, end=end)

    if spots.height == 0 or perps.height == 0:
        return pd.DataFrame()

    spot_bins, spot_summary = _market_bin_summary(
        data,
        spots,
        market="spot",
        freq=freq,
        window=window,
    )
    perp_bins, perp_summary = _market_bin_summary(
        data,
        perps,
        market="perp",
        freq=freq,
        window=window,
    )

    joined = (
        spot_bins.join(
            perp_bins,
            on="timestamp_bin",
            how="outer",
            suffix="_perp_key",
        )
        .with_columns(
            pl.coalesce(["timestamp_bin", "timestamp_bin_perp_key"])
            .cast(pl.Datetime("ms"))
            .alias("timestamp_bin")
        )
        .drop("timestamp_bin_perp_key")
        .sort("timestamp_bin")
        .with_columns([
            pl.col("last_trade_spot").is_not_null().alias("_spot_observed"),
            pl.col("last_trade_perp").is_not_null().alias("_perp_observed"),
        ])
        .with_columns([
            pl.col("last_trade_spot").forward_fill().alias("_spot_filled"),
            pl.col("last_trade_perp").forward_fill().alias("_perp_filled"),
        ])
        .filter(
            pl.col("_spot_filled").is_not_null()
            & pl.col("_perp_filled").is_not_null()
        )
        .with_columns(
            pl.col("timestamp_bin").dt.truncate(window).alias("window_start")
        )
    )

    if joined.height == 0:
        return pd.DataFrame()

    aligned_summary = (
        joined.group_by("window_start", maintain_order=True)
        .agg([
            pl.len().alias("aligned_union_bin_count"),
            pl.col("_spot_observed")
            .cast(pl.Int64)
            .sum()
            .alias("spot_observed_bin_count"),
            pl.col("_perp_observed")
            .cast(pl.Int64)
            .sum()
            .alias("perp_observed_bin_count"),
            (~pl.col("_spot_observed"))
            .cast(pl.Float64)
            .mean()
            .alias("spot_forward_fill_share"),
            (~pl.col("_perp_observed"))
            .cast(pl.Float64)
            .mean()
            .alias("perp_forward_fill_share"),
            (pl.col("_spot_observed") & pl.col("_perp_observed"))
            .cast(pl.Float64)
            .mean()
            .alias("both_observed_share"),
            (pl.col("_spot_observed") & ~pl.col("_perp_observed"))
            .cast(pl.Float64)
            .mean()
            .alias("spot_only_observed_share"),
            (~pl.col("_spot_observed") & pl.col("_perp_observed"))
            .cast(pl.Float64)
            .mean()
            .alias("perp_only_observed_share"),
        ])
    )

    summary = (
        aligned_summary
        .join(spot_summary, on="window_start", how="left")
        .join(perp_summary, on="window_start", how="left")
        .sort("window_start")
        .with_columns([
            pl.lit(freq).alias("latency"),
            pl.lit(ACTIVITY_DIAGNOSTIC_BASIS).alias("diagnostic_basis"),
        ])
    )

    if contract_type is not None:
        summary = summary.with_columns(
            pl.lit(contract_type).alias("contract_type")
        )

    out = summary.to_pandas()
    out["window_start"] = pd.to_datetime(out["window_start"])
    out["date"] = out["window_start"].dt.normalize()
    out["year"] = out["window_start"].dt.year

    def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        return (numerator.astype(float) + 1.0) / (denominator.astype(float) + 1.0)

    out["perp_to_spot_observed_bin_ratio"] = safe_ratio(
        out["perp_observed_bin_count"],
        out["spot_observed_bin_count"],
    )
    out["log_perp_to_spot_observed_bin_ratio"] = np.log(
        out["perp_to_spot_observed_bin_ratio"]
    )

    out["perp_to_spot_price_change_ratio"] = safe_ratio(
        out["perp_price_change_bin_count"],
        out["spot_price_change_bin_count"],
    )
    out["log_perp_to_spot_price_change_ratio"] = np.log(
        out["perp_to_spot_price_change_ratio"]
    )

    out["perp_to_spot_retained_observation_ratio"] = safe_ratio(
        out["perp_retained_observation_count"],
        out["spot_retained_observation_count"],
    )
    out["log_perp_to_spot_retained_observation_ratio"] = np.log(
        out["perp_to_spot_retained_observation_ratio"]
    )

    out["spot_minus_perp_forward_fill_share"] = (
        out["spot_forward_fill_share"]
        - out["perp_forward_fill_share"]
    )

    # Exact trade ratio is available only when a modified loader preserved a
    # trade_count column.  Leave it NaN otherwise.
    if (
        "spot_raw_trade_count" in out.columns
        and "perp_raw_trade_count" in out.columns
    ):
        exact_mask = (
            out["spot_raw_trade_count"].notna()
            & out["perp_raw_trade_count"].notna()
        )
        out["perp_to_spot_raw_trade_ratio"] = np.nan
        out.loc[exact_mask, "perp_to_spot_raw_trade_ratio"] = safe_ratio(
            out.loc[exact_mask, "perp_raw_trade_count"],
            out.loc[exact_mask, "spot_raw_trade_count"],
        )

    return out


def run_sampled_trade_activity(
    *,
    symbol: str,
    source: str,
    cm_um: str,
    contract_type: str,
    start,
    end,
    freqs: Sequence[str] = (
        "10ms",
        "50ms",
        "100ms",
        "200ms",
        "500ms",
        "1s",
    ),
    sample_every_days: int = 30,
    sample_days: int = 1,
    window: str = "1h",
    n_jobs: int = 10,
    output_csv: Optional[str] = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Pull sampled days and calculate update-frequency diagnostics.

    This runner intentionally avoids VECM estimation.  A typical five-year
    robustness sample is one day every 30 days for each contract type.

    ``end`` is exclusive.  To include all of calendar year 2025, use
    ``end='2026-01-01'``.
    """
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()

    if end_ts <= start_ts:
        raise ValueError("end must be later than start.")
    if sample_every_days < 1 or sample_days < 1:
        raise ValueError("sample_every_days and sample_days must be positive.")

    existing = pd.DataFrame()
    output_path = Path(output_csv) if output_csv is not None else None

    if resume and output_path is not None and output_path.exists():
        existing = pd.read_csv(output_path, parse_dates=["sample_start", "sample_end", "window_start", "date"])

    all_results = [existing] if not existing.empty else []

    sampled_starts = pd.date_range(
        start=start_ts,
        end=end_ts - pd.Timedelta(days=1),
        freq=f"{sample_every_days}D",
    )

    for sample_start in sampled_starts:
        sample_end = min(
            sample_start + pd.Timedelta(days=sample_days),
            end_ts,
        )

        missing_freqs = list(freqs)
        if not existing.empty:
            completed = existing.loc[
                (existing["contract_type"] == contract_type)
                & (existing["sample_start"] == sample_start),
                "latency",
            ].astype(str)
            missing_freqs = [freq for freq in freqs if freq not in set(completed)]

        if not missing_freqs:
            print(f"Skipping completed sample {sample_start.date()} ({contract_type})")
            continue

        print(
            f"Loading {contract_type} data for {sample_start.date()} "
            f"through {(sample_end - pd.Timedelta(microseconds=1)).date()}"
        )

        data = TradeData(symbol, source, cm_um)

        # The current downloader enumerates dates backward from end_date and
        # includes end_date itself.
        download_end = (sample_end - pd.Timedelta(days=1)).normalize()
        days_to_pull = max(1, int((sample_end - sample_start).days))

        data.grab_trades_data(
            end_date=download_end.to_pydatetime(),
            days=days_to_pull,
            n_jobs=n_jobs,
        )

        for freq in missing_freqs:
            print(f"  Calculating {freq} activity diagnostics")
            result = summarize_loaded_trade_activity(
                data,
                freq=freq,
                start=sample_start,
                end=sample_end,
                window=window,
                contract_type=contract_type,
            )

            if result.empty:
                print(f"  No aligned data for {sample_start.date()} at {freq}")
                continue

            result["sample_start"] = sample_start
            result["sample_end"] = sample_end
            all_results.append(result)

            if output_path is not None:
                combined = pd.concat(all_results, ignore_index=True)
                dedupe_keys = [
                    "contract_type",
                    "sample_start",
                    "latency",
                    "window_start",
                ]
                combined = (
                    combined.sort_values(dedupe_keys)
                    .drop_duplicates(dedupe_keys, keep="last")
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                combined.to_csv(output_path, index=False)
                existing = combined
                all_results = [combined]

        # Release the potentially large Polars frames before the next sample.
        del data

    if not all_results:
        return pd.DataFrame()

    final = pd.concat(all_results, ignore_index=True)
    dedupe_keys = [
        "contract_type",
        "sample_start",
        "latency",
        "window_start",
    ]
    final = (
        final.sort_values(dedupe_keys)
        .drop_duplicates(dedupe_keys, keep="last")
        .reset_index(drop=True)
    )
    return final


def annual_activity_summary(activity: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly sampled diagnostics into annual contract/latency cells."""
    required = {
        "contract_type",
        "year",
        "latency",
        "spot_observed_bin_count",
        "perp_observed_bin_count",
        "spot_price_change_bin_count",
        "perp_price_change_bin_count",
        "spot_forward_fill_share",
        "perp_forward_fill_share",
        "spot_median_interupdate_ms",
        "perp_median_interupdate_ms",
    }
    missing = required.difference(activity.columns)
    if missing:
        raise KeyError(f"Activity data are missing columns: {sorted(missing)}")

    grouped = (
        activity.groupby(
            ["contract_type", "year", "latency"],
            observed=True,
        )
        .agg(
            n_windows=("window_start", "size"),
            spot_observed_bin_count=("spot_observed_bin_count", "sum"),
            perp_observed_bin_count=("perp_observed_bin_count", "sum"),
            spot_price_change_bin_count=("spot_price_change_bin_count", "sum"),
            perp_price_change_bin_count=("perp_price_change_bin_count", "sum"),
            spot_forward_fill_share=("spot_forward_fill_share", "mean"),
            perp_forward_fill_share=("perp_forward_fill_share", "mean"),
            spot_median_interupdate_ms=("spot_median_interupdate_ms", "median"),
            perp_median_interupdate_ms=("perp_median_interupdate_ms", "median"),
        )
        .reset_index()
    )

    grouped["perp_to_spot_observed_bin_ratio"] = (
        grouped["perp_observed_bin_count"] + 1.0
    ) / (
        grouped["spot_observed_bin_count"] + 1.0
    )
    grouped["perp_to_spot_price_change_ratio"] = (
        grouped["perp_price_change_bin_count"] + 1.0
    ) / (
        grouped["spot_price_change_bin_count"] + 1.0
    )
    grouped["spot_minus_perp_forward_fill_share"] = (
        grouped["spot_forward_fill_share"]
        - grouped["perp_forward_fill_share"]
    )

    return grouped
