"""All market-data processing for the final survival-analysis pipeline.

Frozen timing rules
-------------------
* High-frequency volatility and instantaneous basis: value at event_tick - 1.
* Rolling spread/basis: latest rolling observation strictly before start_ts.
* Daily funding, spot volume, Fear & Greed: previous calendar day exactly.
* Binance open-interest / long-short metrics: latest create_time strictly before start_ts.
* Price alignment uses last observed trade with forward fill only; no interpolation.

The original survival_analysis.py is not modified or imported by this final pipeline.
"""

from __future__ import annotations

import datetime as dt
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import polars as pl

from survival_analysis_data_pull_final import (
    OUTPUT_METRIC_COLUMNS,
    TradeData,
    get_fear_greed_history,
)
from survival_analysis_utils_final import (
    COVARIATE_TIMING_VERSION,
    LIQUIDITY_TIMING_VERSION,
    as_timestamp,
    format_yyyymmdd,
    naive_utc_ns,
    validate_augmented_timing,
    validate_base_covariate_timing,
)


# -----------------------------------------------------------------------------
# Raw-trade transformations
# -----------------------------------------------------------------------------

def to_intervals_bidask(df: pl.DataFrame, freq: str = "1s", fill_gaps: bool = False) -> pl.DataFrame:
    """Legacy bid/ask interval construction, moved out of the data-pull module."""
    df = df.with_columns(pl.col("timestamp").cast(pl.Datetime))
    # Group trades into the requested frequency (one second by default).
    df = df.with_columns(pl.col("timestamp").dt.truncate(freq).alias("timestamp_bin"))
    # Retain the last observed buy- and sell-side price in each interval.
    df_agg = (
        df.group_by(["timestamp_bin", "is_bid"], maintain_order=True)
        .agg(pl.col("price").last().alias("price"))
        .pivot(values="price", index="timestamp_bin", columns="is_bid")
        .rename({"true": "bid_price", "false": "ask_price"})
        .sort("timestamp_bin")
    )
    if not fill_gaps:
        return df_agg

    full_range = pl.DataFrame({
        "timestamp_bin": pl.datetime_range(
            start=df_agg["timestamp_bin"].min(),
            end=df_agg["timestamp_bin"].max(),
            interval=freq,
            eager=True,
        )
    })
    # Forward-filled prices are available for gap-filling callers but are stale
    # until a new trade updates the corresponding side.
    return full_range.join(df_agg, on="timestamp_bin", how="left").fill_null(strategy="forward")


def build_event_time_prices(df_trades_spots: pl.DataFrame, df_trades_perps: pl.DataFrame) -> pl.DataFrame:
    """Build the irregular forward-filled last-trade stream used for shocks/events."""
    required = {"timestamp", "price"}
    for name, frame in (("spot", df_trades_spots), ("perp", df_trades_perps)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} trade data missing columns: {sorted(missing)}")

    def _market_prices(frame: pl.DataFrame, price_name: str, id_name: str) -> pl.DataFrame:
        expressions = [
            pl.col("timestamp").cast(pl.Datetime("ns")),
            pl.col("price").cast(pl.Float64).alias(price_name),
        ]
        order = ["timestamp"]
        if "trade_id" in frame.columns:
            expressions.append(pl.col("trade_id").alias(id_name))
            order.append(id_name)
        else:
            order.append(price_name)
        # At irregular timestamps, retain the deterministically last trade when
        # multiple trades share the same exchange timestamp.
        return (
            frame.select(expressions)
            .sort(order, maintain_order=True)
            .group_by("timestamp", maintain_order=True)
            .agg(pl.col(price_name).last())
        )

    spot = (
        _market_prices(df_trades_spots, "p_spot", "_spot_trade_id")
        .group_by("timestamp", maintain_order=True)
        .agg(pl.col("p_spot").last())
        .with_row_index("id_spot")
    )
    perp = (
        _market_prices(df_trades_perps, "p_perp", "_perp_trade_id")
        .group_by("timestamp", maintain_order=True)
        .agg(pl.col("p_perp").last())
        .with_row_index("id_perp")
    )

    try:
        joined = spot.join(perp, on="timestamp", how="full", coalesce=True)
    except TypeError:
        joined = (
            spot.join(
                perp.rename({"timestamp": "timestamp_perp"}),
                left_on="timestamp",
                right_on="timestamp_perp",
                how="full",
            )
            .with_columns(pl.coalesce(["timestamp", "timestamp_perp"]).alias("timestamp"))
            .drop("timestamp_perp")
        )

    return (
        joined.sort("timestamp")
        # The inactive venue is carried forward at each trade timestamp, so many
        # rows intentionally contain a stale price for one side of the basis.
        .with_columns(
            pl.col("p_spot").forward_fill(),
            pl.col("p_perp").forward_fill(),
        )
        .drop_nulls(["p_spot", "p_perp"])
        # Returns are changes between adjacent rows of this event-time stream.
        .with_columns(
            (pl.col("p_spot").log() - pl.col("p_spot").log().shift(1))
            .fill_null(0.0)
            .alias("spot_ret"),
            (pl.col("p_perp").log() - pl.col("p_perp").log().shift(1))
            .fill_null(0.0)
            .alias("perp_ret"),
        )
    )


# -----------------------------------------------------------------------------
# Shock detection: copied from the established implementation, not reinterpreted.
# -----------------------------------------------------------------------------

def find_shocks2(
    df_ret,
    method: str = "lee_mykland",
    volatility_threshold: float = 3.0,
    quantile_threshold: float = 0.99,
    lm_k=None,
    lm_sampling_minutes: float = 5,
    lm_signif: float = 0.01,
    groupday: bool = True,
    first: str = "spot",
    follow_ticks: int = 5,
    follow_frac: float = 0.5,
):
    """Established shock-identification implementation from survival_analysis.py."""
    print(method, volatility_threshold, quantile_threshold, lm_k, lm_signif, groupday, first)
    df_ret2 = df_ret.sort("timestamp").with_row_index("tick")
    col = f"{first}_ret"
    id_col = f"id_{first}"
    thresh = df_ret2.filter(~pl.col(id_col).is_null())

    def next_cum_expr(n):
        if n <= 0:
            return None
        shifts = [pl.col(col).shift(-k) for k in range(1, n + 1)]
        return sum(shifts).alias("next_cum")

    if method == "lee_mykland":
        returns = thresh.select(col).to_numpy().flatten()
        if lm_k is None:
            if lm_sampling_minutes is None:
                median_spacing_sec = thresh.select(
                    pl.col("timestamp").diff().dt.total_milliseconds().median()
                ).item() / 1000
                lm_sampling_minutes = median_spacing_sec / 60
            lm_k = int(np.ceil(np.sqrt(252 * 24 * 60 / lm_sampling_minutes)))

        # Shift once to pair each return with its predecessor for bipower
        # variation, then shift the product so the volatility input is lagged.
        returns_shifted = np.concatenate([[np.nan], returns[:-1]])
        bpv = np.abs(returns) * np.abs(returns_shifted)
        bpv = np.concatenate([[np.nan], bpv[:-1]])
        sig = pd.Series(bpv).rolling(window=lm_k, min_periods=1).mean().values
        sig = np.sqrt(sig)
        sig[:lm_k - 1] = np.nan
        L = returns / sig

        n = len(returns)
        c = np.sqrt(2 / np.pi)
        Sn = c * np.sqrt(2 * np.log(n))
        Cn = (
            np.sqrt(2 * np.log(n)) / c
            - np.log(np.pi * np.log(n)) / (2 * c * np.sqrt(2 * np.log(n)))
        )
        beta_star = -np.log(-np.log(1 - lm_signif))
        T = (np.abs(L) - Cn) * Sn
        J = (T > beta_star).astype(float) * np.sign(returns)
        J[:lm_k] = np.nan

        df_day = thresh.with_columns([
            pl.Series("J", J),
            pl.Series("T", T),
            pl.Series("sig", sig),
        ])
        if follow_ticks > 0:
            df_day = df_day.with_columns(next_cum_expr(follow_ticks))
        cands = df_day.filter((pl.col("J") != 0) & (~pl.col("J").is_null()))
        if follow_ticks > 0:
            cands = cands.filter(
                pl.col("next_cum").is_null()
                | (pl.col("next_cum").sign() == pl.col(col).sign())
                | (pl.col("next_cum").abs() < follow_frac * pl.col(col).abs())
            )

    elif method == "volatility":
        df_day = thresh.with_columns(pl.col(col).abs().std().over("1h").alias("1h_std"))
        if follow_ticks > 0:
            df_day = df_day.with_columns(next_cum_expr(follow_ticks))
        cands = df_day.filter(pl.col(col).abs() >= volatility_threshold * pl.col("1h_std"))
        if follow_ticks > 0:
            cands = cands.filter(
                pl.col("next_cum").is_null()
                | (pl.col("next_cum").sign() == pl.col(col).sign())
                | (pl.col("next_cum").abs() < follow_frac * pl.col(col).abs())
            )

    else:
        if groupday:
            df_day = thresh.with_columns(pl.col("timestamp").dt.date().alias("day"))
            if follow_ticks > 0:
                df_day = df_day.with_columns(next_cum_expr(follow_ticks))
            cands = (
                df_day.with_columns(
                    pl.col(col).abs().quantile(quantile_threshold).over("day").alias("daily_threshold")
                )
                .filter(pl.col(col).abs() >= pl.col("daily_threshold"))
            )
            if follow_ticks > 0:
                cands = cands.filter(
                    pl.col("next_cum").is_null()
                    | (pl.col("next_cum").sign() == pl.col(col).sign())
                    | (pl.col("next_cum").abs() < follow_frac * pl.col(col).abs())
                )
        else:
            threshold = thresh.select(pl.col(col).abs().quantile(quantile_threshold)).item()
            df_thr = thresh
            if follow_ticks > 0:
                df_thr = df_thr.with_columns(next_cum_expr(follow_ticks))
            cands = df_thr.filter(pl.col(col).abs() >= threshold)
            if follow_ticks > 0:
                cands = cands.filter(
                    pl.col("next_cum").is_null()
                    | (pl.col("next_cum").sign() == pl.col(col).sign())
                    | (pl.col("next_cum").abs() < follow_frac * pl.col(col).abs())
                )

    cands = cands.select(["tick", col]).to_numpy()
    median_dt = df_ret2.select(
        (pl.col("timestamp").diff().dt.total_milliseconds() / 1e3).median().alias("median_dt")
    ).item()
    return None, cands, median_dt


def greedy_refractory_seconds(candidate_ticks: Sequence[int], times: np.ndarray, refractory_seconds: float) -> List[int]:
    ticks = np.sort(np.asarray(candidate_ticks, dtype=np.int64))
    if ticks.size == 0:
        return []
    if refractory_seconds <= 0:
        return ticks.tolist()
    selected = [int(ticks[0])]
    last_time = times[ticks[0]]
    for tick in ticks[1:]:
        elapsed = (times[tick] - last_time) / np.timedelta64(1, "s")
        if elapsed >= refractory_seconds:
            selected.append(int(tick))
            last_time = times[tick]
    return selected


def _first_sustained_entry_index(condition: np.ndarray, times: np.ndarray, min_hold_seconds: float) -> Optional[int]:
    condition = np.asarray(condition, dtype=bool)
    if not np.any(condition):
        return None
    if min_hold_seconds <= 0:
        return int(np.flatnonzero(condition)[0])

    run_start = None
    for idx, is_inside in enumerate(condition):
        if not is_inside:
            run_start = None
            continue
        if run_start is None:
            run_start = idx
        elapsed = (times[idx] - times[run_start]) / np.timedelta64(1, "s")
        if elapsed >= min_hold_seconds:
            return int(run_start)
    return None


def calculate_basis_resolution_events(
    df_prices: pl.DataFrame,
    shock_events: Sequence[int],
    resolution_pcts: Iterable[int] = (90,),
    max_duration_seconds: float = 120.0,
    min_hold_seconds: float = 0.0,
    basis_tolerance_bps: float = 0.0,
    min_basis_shock_bps: float = 0.0,
    require_full_horizon: bool = True,
) -> pl.DataFrame:
    """First-passage basis-resolution outcomes from the established event-store definition."""
    percentages = sorted({int(value) for value in resolution_pcts})
    invalid = [value for value in percentages if value <= 0 or value >= 100]
    if invalid:
        raise ValueError(f"resolution_pcts must be strictly between 0 and 100: {invalid}")

    frame = df_prices.select(["timestamp", "p_spot", "p_perp"]).drop_nulls().sort("timestamp")
    times = frame["timestamp"].to_numpy().astype("datetime64[ns]")
    spot = frame["p_spot"].to_numpy().astype(float)
    perp = frame["p_perp"].to_numpy().astype(float)
    basis = np.log(perp) - np.log(spot)

    tolerance_abs = basis_tolerance_bps / 10_000.0
    minimum_shock_abs = min_basis_shock_bps / 10_000.0
    horizon_ns = int(round(max_duration_seconds * 1_000_000_000))
    results = []

    for shock_tick_raw in shock_events:
        shock_tick = int(shock_tick_raw)
        if shock_tick < 1 or shock_tick >= len(basis) - 1:
            continue
        start_time = times[shock_tick]
        censor_time = start_time + np.timedelta64(horizon_ns, "ns")
        full_horizon_available = censor_time <= times[-1]
        if require_full_horizon and not full_horizon_available:
            continue

        window_end = min(int(np.searchsorted(times, censor_time, side="right")), len(times))
        if window_end <= shock_tick + 1:
            continue

        pre_basis = float(basis[shock_tick - 1])
        post_basis = float(basis[shock_tick])
        initial_deviation = post_basis - pre_basis
        initial_size = abs(initial_deviation)
        if not np.isfinite(initial_size) or initial_size <= max(1e-12, minimum_shock_abs):
            continue

        future_basis = basis[shock_tick + 1:window_end]
        future_times = times[shock_tick + 1:window_end]
        remaining_deviation = np.abs(future_basis - pre_basis)
        available_duration = float((future_times[-1] - start_time) / np.timedelta64(1, "s"))
        censor_duration = max_duration_seconds if full_horizon_available else available_duration

        for pct in percentages:
            target_remaining = max((1.0 - pct / 100.0) * initial_size, tolerance_abs)
            inside = remaining_deviation <= target_remaining
            event_idx = _first_sustained_entry_index(inside, future_times, min_hold_seconds)
            if event_idx is None:
                status = 0
                length = censor_duration
                resolution_time = None
                resolution_tick = None
            else:
                status = 1
                resolution_time = future_times[event_idx]
                length = float((resolution_time - start_time) / np.timedelta64(1, "s"))
                resolution_tick = shock_tick + 1 + event_idx

            results.append({
                "event_tick": shock_tick,
                "start_ts": start_time,
                "prev_ts": times[shock_tick - 1],
                "Status": status,
                "Length": max(float(length), 0.0),
                "resolution_pct": pct,
                "shock_size": initial_size,
                "shock_size_bps": initial_size * 10_000.0,
                "pre_shock_basis": pre_basis,
                "pre_shock_basis_bps": pre_basis * 10_000.0,
                "post_shock_basis": post_basis,
                "post_shock_basis_bps": post_basis * 10_000.0,
                "target_remaining_bps": target_remaining * 10_000.0,
                "resolution_tick": resolution_tick,
                "resolution_ts": resolution_time,
                "censor_duration_seconds": censor_duration,
            })

    if not results:
        return pl.DataFrame(schema={
            "event_tick": pl.Int64,
            "start_ts": pl.Datetime("ns"),
            "prev_ts": pl.Datetime("ns"),
            "Status": pl.Int64,
            "Length": pl.Float64,
            "resolution_pct": pl.Int64,
        })
    return pl.DataFrame(results)



def detect_basis_events_from_event_time(
    event_time: pl.DataFrame,
    start,
    end,
    shock_significance: float = 0.001,
    method: str = "lee_mykland",
    resolution_pcts: Iterable[int] = (90,),
    max_duration_seconds: float = 120.0,
    refractory_seconds: float = 5.0,
    min_hold_seconds: float = 0.0,
    basis_tolerance_bps: float = 0.0,
    min_basis_shock_bps: float = 0.0,
    first: str = "spot",
    groupday: bool = True,
    require_full_horizon: bool = True,
) -> pl.DataFrame:
    """Detect and resolve shocks on an already-loaded event-time frame."""
    analysis_start = as_timestamp(start)
    analysis_end = as_timestamp(end)
    if method == "lee_mykland":
        _, candidates, _ = find_shocks2(
            event_time,
            method=method,
            lm_signif=shock_significance,
            groupday=groupday,
            first=first,
            follow_ticks=0,
        )
    else:
        _, candidates, _ = find_shocks2(
            event_time,
            method=method,
            quantile_threshold=1.0 - shock_significance,
            groupday=groupday,
            first=first,
            follow_ticks=0,
        )

    times = event_time["timestamp"].to_numpy().astype("datetime64[ns]")
    candidate_ticks = candidates[:, 0].astype(np.int64) if len(candidates) else np.array([], dtype=np.int64)
    start_np = np.datetime64(analysis_start.to_datetime64(), "ns")
    end_np = np.datetime64(analysis_end.to_datetime64(), "ns")
    candidate_ticks = candidate_ticks[
        (times[candidate_ticks] >= start_np) & (times[candidate_ticks] < end_np)
    ]
    selected = greedy_refractory_seconds(candidate_ticks, times, refractory_seconds)
    event_table = calculate_basis_resolution_events(
        event_time,
        selected,
        resolution_pcts=resolution_pcts,
        max_duration_seconds=max_duration_seconds,
        min_hold_seconds=min_hold_seconds,
        basis_tolerance_bps=basis_tolerance_bps,
        min_basis_shock_bps=min_basis_shock_bps,
        require_full_horizon=require_full_horizon,
    )
    if event_table.height:
        event_table = event_table.with_columns(
            pl.lit(first).alias("first"),
            pl.lit(float(shock_significance)).alias("shock_significance"),
            pl.lit(analysis_start.strftime("%Y-%m-%d")).alias("period_start"),
            pl.lit(analysis_end.strftime("%Y-%m-%d")).alias("period_end"),
        )
    return event_table


class BasisSurvivalAnalysis:
    """Final basis-resolution analysis object; survival_analysis.py is not modified."""

    def __init__(self, ticker: str, source: str = "Binance", cm_um: str = "um"):
        self.ticker = ticker
        self.source = source
        self.cm_um = cm_um

    def _get_event_time_data(self, start, end, retain_raw_data: bool = False):
        start_str = format_yyyymmdd(start)
        end_str = format_yyyymmdd(end)
        start_t = dt.datetime.strptime(start_str, "%Y%m%d")
        end_t = dt.datetime.strptime(end_str, "%Y%m%d")

        days = (end_t - start_t).days
        if days < 1:
            raise ValueError("Raw trade interval must contain at least one calendar day")
        # TradeData includes ``end_date`` itself. Convert the requested
        # half-open [start_t, end_t) interval to its inclusive final archive
        # date so the lookback day is not dropped and a future day added.
        archive_end = end_t - dt.timedelta(days=1)
        data = TradeData(self.ticker, source=self.source, cm_um=self.cm_um)
        data.grab_trades_data(archive_end, days)
        for name, trades in (
            ("spot", data.df_trades_spots),
            ("perp", data.df_trades_perps),
        ):
            if trades.height == 0:
                raise ValueError(f"No {name} trades loaded for [{start_t}, {end_t})")
            observed_first = pd.Timestamp(trades["timestamp"].min()).normalize()
            observed_last = pd.Timestamp(trades["timestamp"].max()).normalize()
            if observed_first != pd.Timestamp(start_t) or observed_last != pd.Timestamp(archive_end):
                raise AssertionError(
                    f"{name} archive range mismatch: requested [{start_t:%Y-%m-%d}, "
                    f"{end_t:%Y-%m-%d}), observed {observed_first:%Y-%m-%d} through "
                    f"{observed_last:%Y-%m-%d}"
                )
        event_time = build_event_time_prices(data.df_trades_spots, data.df_trades_perps).sort("timestamp")

        if retain_raw_data:
            return data, event_time
        for attr in ("df_trades_spots", "df_trades_perps"):
            if hasattr(data, attr):
                delattr(data, attr)
        return None, event_time

    def fit_basis_events(
        self,
        start,
        end,
        shock_significance: float = 0.001,
        method: str = "lee_mykland",
        resolution_pcts: Iterable[int] = (90,),
        max_duration_seconds: float = 120.0,
        refractory_seconds: float = 5.0,
        min_hold_seconds: float = 0.0,
        basis_tolerance_bps: float = 0.0,
        min_basis_shock_bps: float = 0.0,
        first: str = "spot",
        groupday: bool = True,
        require_full_horizon: bool = True,
        load_followup_days: int = 1,
        load_lookback_days: int = 1,
        retain_raw_data: bool = False,
    ):
        """Return (TradeData or None, exact event-time frame, basis event table)."""
        analysis_start = as_timestamp(start)
        analysis_end = as_timestamp(end)
        load_start = analysis_start - pd.Timedelta(days=load_lookback_days)
        load_end = analysis_end + pd.Timedelta(days=load_followup_days)

        data, event_time = self._get_event_time_data(load_start, load_end, retain_raw_data=retain_raw_data)
        event_table = detect_basis_events_from_event_time(
            event_time=event_time,
            start=analysis_start,
            end=analysis_end,
            shock_significance=shock_significance,
            method=method,
            resolution_pcts=resolution_pcts,
            max_duration_seconds=max_duration_seconds,
            refractory_seconds=refractory_seconds,
            min_hold_seconds=min_hold_seconds,
            basis_tolerance_bps=basis_tolerance_bps,
            min_basis_shock_bps=min_basis_shock_bps,
            first=first,
            groupday=groupday,
            require_full_horizon=require_full_horizon,
        )
        return data, event_time, event_table


# -----------------------------------------------------------------------------
# Reuse saved KM episodes without redetecting shocks or recomputing outcomes
# -----------------------------------------------------------------------------

def reconstruct_saved_event_ticks(
    event_time_prices: pl.DataFrame,
    saved_events: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct ``event_tick`` from saved KM ``start_ts`` values.

    The v6 event-store checkpoints intentionally stored portable timestamps rather
    than chunk-local row numbers.  This function rebuilds those row numbers against
    the exact same irregular forward-filled event-time construction and fails if
    either ``start_ts`` or ``prev_ts`` does not match exactly.  It does *not*
    redetect shocks or recalculate Length/Status.
    """
    if saved_events.empty:
        return saved_events.copy()

    required = {"start_ts", "prev_ts", "Status", "Length"}
    missing = required.difference(saved_events.columns)
    if missing:
        raise KeyError(f"Saved KM events missing required columns: {sorted(missing)}")

    out = saved_events.copy()

    # V5's pandas/parquet round trip can represent an original millisecond
    # timestamp a few hundred nanoseconds to either side of the millisecond
    # boundary (for example .513999872 instead of .514000000).  Normalize only
    # that bounded serialization noise.  An unrestricted nearest match would be
    # unsafe because it could attach an episode to a different trade update.
    for column in ("start_ts", "prev_ts"):
        original = naive_utc_ns(out[column])
        normalized = original.dt.round("ms")
        drift = (original - normalized).abs()
        excessive = drift > pd.Timedelta("500us")
        if excessive.any():
            examples = out.loc[excessive, column].head(5).tolist()
            raise AssertionError(
                f"{int(excessive.sum())} saved v5 KM {column} values are more than "
                f"0.5 ms from the timestamp grid. Examples: {examples}"
            )
        out[column] = normalized

    times = event_time_prices.sort("timestamp")["timestamp"].to_numpy().astype("datetime64[ns]")
    starts = out["start_ts"].to_numpy(dtype="datetime64[ns]")
    ticks = np.searchsorted(times, starts, side="left")

    in_bounds = ticks < len(times)
    exact_start = np.zeros(len(ticks), dtype=bool)
    exact_start[in_bounds] = times[ticks[in_bounds]] == starts[in_bounds]
    if not exact_start.all():
        examples = out.loc[~exact_start, "start_ts"].head(5).tolist()
        raise AssertionError(
            f"{int((~exact_start).sum())} saved KM start_ts values were not found exactly "
            f"in the reconstructed event-time stream. Examples: {examples}"
        )

    if np.any(ticks <= 0):
        raise AssertionError("A saved KM event reconstructed to tick 0 and has no prior tick.")

    expected_prev = times[ticks - 1]
    saved_prev = out["prev_ts"].to_numpy(dtype="datetime64[ns]")
    exact_prev = expected_prev == saved_prev
    if not exact_prev.all():
        examples = out.loc[~exact_prev, ["start_ts", "prev_ts"]].head(5).to_dict("records")
        raise AssertionError(
            f"{int((~exact_prev).sum())} saved KM prev_ts values do not equal tick-1 in the "
            f"reconstructed event-time stream. Examples: {examples}"
        )

    out["event_tick"] = ticks.astype(np.int64)
    if "shock_size" not in out.columns:
        if "shock_size_bps" not in out.columns:
            raise KeyError("Saved KM events need shock_size or shock_size_bps for AFT covariates.")
        out["shock_size"] = pd.to_numeric(out["shock_size_bps"], errors="coerce") / 10_000.0
    return out


def assert_saved_km_outcomes_unchanged(
    saved_events: pd.DataFrame,
    enriched_events: pd.DataFrame,
    source_name: str = "saved KM events",
) -> bool:
    """Assert that covariate enrichment did not change KM outcomes.

    ``start_ts``, ``prev_ts``, ``Status`` and ``Length`` are treated as immutable
    inputs.  ``shock_size`` is also checked when present in both frames.
    """
    if saved_events.empty and enriched_events.empty:
        return True

    keys = ["start_ts"]
    compare_cols = ["prev_ts", "Status", "Length"]
    if "resolution_pct" in saved_events.columns and "resolution_pct" in enriched_events.columns:
        keys.append("resolution_pct")
    if "shock_size" in saved_events.columns and "shock_size" in enriched_events.columns:
        compare_cols.append("shock_size")

    left = saved_events[keys + compare_cols].copy()
    right = enriched_events[keys + compare_cols].copy()
    for frame in (left, right):
        frame["start_ts"] = naive_utc_ns(frame["start_ts"])
        if "prev_ts" in frame:
            frame["prev_ts"] = naive_utc_ns(frame["prev_ts"])

    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise AssertionError(f"{source_name}: duplicate event keys prevent a one-to-one KM reuse check.")

    merged = left.merge(right, on=keys, how="outer", suffixes=("_saved", "_aft"), indicator=True)
    if not merged["_merge"].eq("both").all():
        raise AssertionError(
            f"{source_name}: AFT enrichment changed the set of saved KM episodes "
            f"({merged['_merge'].value_counts().to_dict()})."
        )

    for column in compare_cols:
        a = merged[f"{column}_saved"]
        b = merged[f"{column}_aft"]
        if column in {"Length", "shock_size"}:
            equal = np.isclose(
                pd.to_numeric(a, errors="coerce"),
                pd.to_numeric(b, errors="coerce"),
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            )
        else:
            equal = (a == b) | (a.isna() & b.isna())
        if not np.all(equal):
            raise AssertionError(
                f"{source_name}: {int((~np.asarray(equal)).sum())} rows changed immutable KM field {column}."
            )
    return True


# -----------------------------------------------------------------------------
# Leakage-safe AFT covariates
# -----------------------------------------------------------------------------

def build_aft_covariates(
    start,
    end,
    data: TradeData,
    event_time_prices: pl.DataFrame,
    events,
    primary_resolution_pct: int = 90,
    rolling_window: str = "5min",
    min_rolling_periods: int = 100,
    legacy_min_event_tick: int = 6000,
) -> pd.DataFrame:
    """Attach the frozen AFT covariates to basis-resolution episodes.

    event_time_prices MUST be the exact dataframe used for shock detection.
    This removes the possibility of event_tick drift between detection and
    covariate assignment.
    """
    if isinstance(events, pl.DataFrame):
        event_pd = events.to_pandas()
    else:
        event_pd = events.copy()
    if event_pd.empty:
        return pd.DataFrame()
    if "resolution_pct" in event_pd.columns:
        event_pd = event_pd[event_pd["resolution_pct"].astype(int).eq(primary_resolution_pct)].copy()
    if event_pd.empty:
        return pd.DataFrame()

    required_events = {"event_tick", "start_ts", "prev_ts", "Status", "Length", "shock_size"}
    missing_events = required_events.difference(event_pd.columns)
    if missing_events:
        raise KeyError(f"Event table missing columns: {sorted(missing_events)}")

    # Raw Binance trade timestamps are millisecond-valued.  Normalize bounded
    # Polars/pandas conversion noise before enforcing exact event_tick identity.
    for column in ("start_ts", "prev_ts"):
        original = naive_utc_ns(event_pd[column])
        normalized = original.dt.round("ms")
        drift = (original - normalized).abs()
        if (drift > pd.Timedelta("500us")).any():
            raise AssertionError(f"{column} is not on the Binance millisecond timestamp grid")
        event_pd[column] = normalized

    # Exact same stream used by detection; no reconstruction and no interpolation.
    frame = event_time_prices.sort("timestamp")
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"].to_pandas())).tz_localize(None)
    spot_returns = pd.Series(frame["spot_ret"].to_numpy(), index=timestamps)
    perp_returns = pd.Series(frame["perp_ret"].to_numpy(), index=timestamps)
    spot_prices = pd.Series(frame["p_spot"].to_numpy(), index=timestamps)
    perp_prices = pd.Series(frame["p_perp"].to_numpy(), index=timestamps)

    # basis constructed from the raw prices
    basis = np.log(perp_prices) - np.log(spot_prices)

    # volatility represented by rolling standard deviation of returns
    vol_spot = spot_returns.rolling(window=rolling_window, min_periods=min_rolling_periods).std().to_numpy()
    vol_perp = perp_returns.rolling(window=rolling_window, min_periods=min_rolling_periods).std().to_numpy()

    hf_rows = []
    for _, shock in event_pd.iterrows():
        tick = int(shock["event_tick"])
        if tick <= 0 or tick >= len(timestamps):
            raise AssertionError(f"event_tick {tick} is outside the event-time frame")

        tick_ts = pd.Timestamp(timestamps[tick])
        prev_tick_ts = pd.Timestamp(timestamps[tick - 1])
        if tick_ts != pd.Timestamp(shock["start_ts"]):
            raise AssertionError(
                f"event_tick alignment failure: tick {tick} -> {tick_ts}, saved start_ts={shock['start_ts']}"
            )
        if prev_tick_ts != pd.Timestamp(shock["prev_ts"]):
            raise AssertionError(
                f"tick-1 alignment failure: tick-1 -> {prev_tick_ts}, saved prev_ts={shock['prev_ts']}"
            )

        # Preserve the original InProgress eligibility rule exactly.
        if tick < legacy_min_event_tick:
            hf_rows.append({
                "event_tick": tick,
                "vol_spot_5min": np.nan,
                "vol_perp_5min": np.nan,
                "basis_level": np.nan,
            })
        else:
            hf_rows.append({
                "event_tick": tick,
                "vol_spot_5min": vol_spot[tick - 1],
                "vol_perp_5min": vol_perp[tick - 1],
                "basis_level": basis.iloc[tick - 1],
            })

    result = event_pd.merge(pd.DataFrame(hf_rows), on="event_tick", how="left", validate="one_to_one")

    # 1-second bid/ask observations -> rolling 5-minute spreads and midpoint basis.
    spots = to_intervals_bidask(data.df_trades_spots, "1s", fill_gaps=True).to_pandas()
    spots["timestamp_bin"] = naive_utc_ns(spots["timestamp_bin"])
    spots = spots.set_index("timestamp_bin").sort_index()
    spots["spread_spot"] = spots["ask_price"] - spots["bid_price"]
    spots["midpoint_spot"] = (spots["ask_price"] + spots["bid_price"]) / 2
    spots["spread_spot_bps"] = spots["spread_spot"] / spots["midpoint_spot"] * 10_000

    perps = to_intervals_bidask(data.df_trades_perps, "1s", fill_gaps=True).to_pandas()
    perps["timestamp_bin"] = naive_utc_ns(perps["timestamp_bin"])
    perps = perps.set_index("timestamp_bin").sort_index()
    perps["spread_perp"] = perps["ask_price"] - perps["bid_price"]
    perps["midpoint_perp"] = (perps["ask_price"] + perps["bid_price"]) / 2
    perps["spread_perp_bps"] = perps["spread_perp"] / perps["midpoint_perp"] * 10_000

    # rolling 5-minute averages of spreads and basis, with a minimum of 30 seconds of data.
    spots_roll = (
        spots[["spread_spot_bps"]]
        .rolling(window=rolling_window, min_periods=30)
        .mean()
        .rename(columns={"spread_spot_bps": "spread_spot_5min"})
    )
    perps_roll = (
        perps[["spread_perp_bps"]]
        .rolling(window=rolling_window, min_periods=30)
        .mean()
        .rename(columns={"spread_perp_bps": "spread_perp_5min"})
    )
    midpoint_spot = spots["midpoint_spot"]
    midpoint_perp = perps["midpoint_perp"]
    basis_point = ((midpoint_perp.reindex(midpoint_spot.index) - midpoint_spot) / midpoint_spot) * 10_000
    basis_roll = basis_point.rolling(window=rolling_window, min_periods=30).mean().rename("basis_5min")

    bidask_roll = pd.concat([spots_roll, perps_roll, basis_roll], axis=1).dropna().reset_index()
    # ``timestamp_bin`` labels the START of [t, t+1s), while its spread and
    # midpoint use all trades in that second.  The observation is therefore
    # available only at the bin END.  Timestamp it at t+1s before performing
    # the strict backward merge, otherwise an event inside the same second can
    # accidentally receive trades that occur after the event.
    bidask_roll["timestamp_bin"] = (
        pd.to_datetime(bidask_roll["timestamp_bin"]) + pd.Timedelta(seconds=1)
    )
    bidask_roll = bidask_roll.rename(columns={"timestamp_bin": "covariate_5min_ts"})
    # Match to the latest fully completed 1-second observation available at
    # the actual event-time predecessor, not merely somewhere before start_ts.
    # A bucket labelled T contains [T-1s, T), so an exact match to prev_ts is
    # already known at prev_ts and cannot contain the shock trade.
    result = result.sort_values("prev_ts")
    bidask_roll = bidask_roll.sort_values("covariate_5min_ts")
    result = pd.merge_asof(
        result,
        bidask_roll,
        left_on="prev_ts",
        right_on="covariate_5min_ts",
        direction="backward",
        allow_exact_matches=True,
    )
    result = result.sort_values("start_ts").reset_index(drop=True)

    # Exact previous calendar day for all daily variables.
    start_day = as_timestamp(start)
    end_day = as_timestamp(end)
    daily_start = (start_day - pd.Timedelta(days=1)).strftime("%Y%m%d")
    daily_end = end_day.strftime("%Y%m%d")

    funding = data.get_funding_data(daily_start, daily_end).copy()
    if funding.empty:
        funding_daily = pd.DataFrame(columns=["covariate_date", "fundingRate"])
    else:
        funding["covariate_date"] = pd.to_datetime(funding["fundingTime"]).dt.date
        funding_daily = funding.groupby("covariate_date", as_index=False)["fundingRate"].mean()

    volume = data.get_klines(
        daily_start,
        daily_end,
        "spot",
        "1d",
        columns=["Open time", "Close time", "Open", "Close", "log_return", "Volume"],
    ).copy()
    volume["covariate_date"] = pd.to_datetime(volume["Open time"]).dt.date
    volume_daily = (
        volume[["covariate_date", "Volume"]]
        .rename(columns={"Volume": "volume"})
        .drop_duplicates("covariate_date", keep="last")
    )
    fear_greed = get_fear_greed_history()
    daily = (
        funding_daily.merge(volume_daily, on="covariate_date", how="outer")
        .merge(fear_greed, on="covariate_date", how="outer")
    )

    result["date"] = pd.to_datetime(result["start_ts"]).dt.date
    result["_required_daily_date"] = (
        pd.to_datetime(result["start_ts"]).dt.normalize() - pd.Timedelta(days=1)
    ).dt.date
    result = result.merge(
        daily,
        left_on="_required_daily_date",
        right_on="covariate_date",
        how="left",
        validate="many_to_one",
    )
    result = result.rename(columns={"covariate_date": "daily_covariate_date"})

    result["fundingRate_bps"] = result["fundingRate"] * 1e4
    result["shock_size_bps"] = result["shock_size"] * 1e4
    result["basis_level_bps"] = result["basis_level"] * 1e4
    result["vol_spot_5min_pct"] = result["vol_spot_5min"] * 100
    result["vol_perp_5min_pct"] = result["vol_perp_5min"] * 100
    result["Length"] = pd.to_numeric(result["Length"], errors="coerce").replace(0, 0.001)
    result["covariate_timing_version"] = COVARIATE_TIMING_VERSION

    # Preserve audit timestamps. Drop only redundant raw-scale columns.
    result = result.drop(
        columns=[
            "fundingRate",
            "basis_level",
            "vol_spot_5min",
            "vol_perp_5min",
            "_required_daily_date",
        ],
        errors="ignore",
    )
    result = result.sort_values("start_ts").reset_index(drop=True)
    validate_base_covariate_timing(result, "new AFT covariates")
    return result


def augment_liquidity_metrics(events: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Attach latest Binance metric strictly BEFORE each shock timestamp."""
    # Confirm the base-event timing contract before adding another covariate set.
    validate_base_covariate_timing(events, "base AFT events")
    frame = events.drop(
        columns=["create_time", "liquidity_timing_version", *OUTPUT_METRIC_COLUMNS],
        errors="ignore",
    ).copy()
    frame["start_ts"] = pd.to_datetime(frame["start_ts"], utc=True, errors="coerce")
    # merge_asof requires timestamp sorting. Preserve the incoming event order,
    # including the order of multiple trades sharing one timestamp.
    frame["_original_order"] = np.arange(len(frame), dtype=np.int64)

    metric_frame = metrics.copy()
    metric_frame["create_time"] = pd.to_datetime(metric_frame["create_time"], utc=True, errors="coerce")
    metric_frame = metric_frame.dropna(subset=["create_time"]).sort_values("create_time")

    valid = frame.loc[frame["start_ts"].notna()].sort_values("start_ts").copy()
    invalid = frame.loc[frame["start_ts"].isna()].copy()
    augmented = pd.merge_asof(
        valid,
        metric_frame,
        left_on="start_ts",
        right_on="create_time",
        direction="backward",
        # Metrics are matched strictly before the shock because create_time is
        # treated as the far end of the metric observation interval.
        allow_exact_matches=False,
    )
    if not invalid.empty:
        invalid["create_time"] = pd.NaT
        for column in OUTPUT_METRIC_COLUMNS:
            invalid[column] = np.nan
        augmented = pd.concat([augmented, invalid], ignore_index=True, sort=False)

    for column in OUTPUT_METRIC_COLUMNS:
        augmented[column] = pd.to_numeric(augmented[column], errors="coerce").astype("float64")
    # Record the timing contract for downstream audits and output provenance.
    augmented["liquidity_timing_version"] = LIQUIDITY_TIMING_VERSION
    # Restore the exact incoming event order after the timestamp-sorted join.
    augmented = (
        augmented.sort_values("_original_order")
        .drop(columns="_original_order")
        .reset_index(drop=True)
    )
    validate_augmented_timing(augmented, "augmented AFT events")
    return augmented


def prepare_aft_data(frame_or_path, covariates: Sequence[str]):
    """Select complete model rows after timing validation."""
    if isinstance(frame_or_path, (str, bytes)) or hasattr(frame_or_path, "suffix"):
        frame = pd.read_parquet(frame_or_path)
        source_name = str(frame_or_path)
    else:
        frame = frame_or_path.copy()
        source_name = "frame"

    validate_augmented_timing(frame, source_name)
    required = ["Length", "Status", *covariates]
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise KeyError(f"{source_name} missing model columns: {missing}")

    model = frame[required].copy()
    for column in required:
        model[column] = pd.to_numeric(model[column], errors="coerce")
    model = model.replace([np.inf, -np.inf], np.nan)
    model["Length"] = model["Length"].replace(0, 0.001)
    model = model[model["Length"] > 0]
    model["Status"] = model["Status"].astype(float)
    model = model[model["Status"].isin([0, 1])]
    eligible = len(model)
    model = model.dropna().reset_index(drop=True)

    diagnostics = {
        "source_rows": len(frame),
        "eligible_rows_before_dropna": eligible,
        "complete_rows": len(model),
        "dropped_rows": len(frame) - len(model),
        "events": int(model["Status"].sum()),
        "censored": int(len(model) - model["Status"].sum()),
    }
    return model, diagnostics
