"""
Basis-resolution survival analysis extensions.

This module leaves the original survival_analysis.py unchanged and adds:
1. Basis-resolution events rather than cross-market return absorption.
2. Clock-time censoring and optional clock-time refractory periods.
3. Period-by-period Kaplan-Meier estimation on a fixed time grid.
4. Period-by-period Log-Logistic AFT estimation with reproducible curve grids.

The primary event definition is:
    basis_t = log(perp_price_t) - log(spot_price_t)

For an event at t0, let b_pre be the basis immediately before the shock and
D0 = |basis_t0 - b_pre|. A q-percent resolution event occurs at the first time
when:
    |basis_t - b_pre| <= (1 - q/100) * D0
subject to an optional minimum absolute basis band and hold duration.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from lifelines import KaplanMeierFitter, LogLogisticAFTFitter

from survival_analysis import (
    SurvivalAnalysis,
    TradeData,
    all_formatting,
    find_shocks2,
    find_covariates as legacy_find_covariates,
)


DateLike = Union[str, dt.date, dt.datetime, pd.Timestamp]


def _as_timestamp(value: DateLike) -> pd.Timestamp:
    """Parse YYYYMMDD strings and ordinary pandas-compatible dates."""
    if isinstance(value, str) and len(value) == 8 and value.isdigit():
        return pd.to_datetime(value, format="%Y%m%d")
    return pd.Timestamp(value)


def _format_yyyymmdd(value: DateLike) -> str:
    return _as_timestamp(value).strftime("%Y%m%d")


def _normalise_percentages(values: Iterable[int]) -> List[int]:
    percentages = sorted({int(x) for x in values})
    invalid = [x for x in percentages if x <= 0 or x >= 100]
    if invalid:
        raise ValueError(
            "resolution_pcts must lie strictly between 0 and 100; "
            f"received {invalid}."
        )
    return percentages


def build_periods(
    start: DateLike,
    end: DateLike,
    freq: str = "MS",
) -> List[Tuple[pd.Timestamp, pd.Timestamp, str]]:
    """
    Construct non-overlapping [start, end) periods.

    Examples
    --------
    freq="MS" : calendar months
    freq="QS" : calendar quarters
    freq="YS" : calendar years
    freq="30D": fixed 30-day periods
    """
    start_ts = _as_timestamp(start)
    end_ts = _as_timestamp(end)
    if end_ts <= start_ts:
        raise ValueError("end must be later than start")

    boundaries = list(pd.date_range(start=start_ts, end=end_ts, freq=freq))
    if not boundaries or boundaries[0] != start_ts:
        boundaries.insert(0, start_ts)
    if boundaries[-1] != end_ts:
        boundaries.append(end_ts)

    periods: List[Tuple[pd.Timestamp, pd.Timestamp, str]] = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        if right <= left:
            continue
        if freq.upper().startswith("M"):
            label = left.strftime("%Y-%m")
        elif freq.upper().startswith("Q"):
            label = f"{left.year}-Q{left.quarter}"
        elif freq.upper().startswith(("Y", "A")):
            label = str(left.year)
        else:
            label = f"{left:%Y-%m-%d} to {right:%Y-%m-%d}"
        periods.append((left, right, label))
    return periods


def greedy_refractory_seconds(
    candidate_ticks: Sequence[int],
    times: np.ndarray,
    refractory_seconds: float,
) -> List[int]:
    """Keep shocks separated by at least refractory_seconds in clock time."""
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


def _first_sustained_entry_index(
    condition: np.ndarray,
    times: np.ndarray,
    min_hold_seconds: float,
) -> Optional[int]:
    """
    Return the first index entering a True run that remains True for the
    requested clock-time duration.

    The reported event time is the entry time, not the later confirmation time.
    """
    condition = np.asarray(condition, dtype=bool)
    if not np.any(condition):
        return None

    if min_hold_seconds <= 0:
        return int(np.flatnonzero(condition)[0])

    run_start: Optional[int] = None
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
    resolution_pcts: Iterable[int] = (50, 90, 95),
    max_duration_seconds: float = 120.0,
    min_hold_seconds: float = 0.0,
    basis_tolerance_bps: float = 0.0,
    min_basis_shock_bps: float = 0.0,
    require_full_horizon: bool = True,
) -> pl.DataFrame:
    """
    Calculate first-passage basis-resolution times for detected shocks.

    Parameters
    ----------
    df_prices:
        Polars frame containing timestamp, p_spot, and p_perp. Prices should
        reflect information available at each timestamp; forward-filled last
        trades are preferred to interpolation.
    shock_events:
        Row indices of detected spot or perp shocks.
    resolution_pcts:
        Percentages of the initial basis displacement that must be removed.
    max_duration_seconds:
        Fixed clock-time censoring horizon.
    min_hold_seconds:
        Optional duration for which the basis must remain in the target band.
    basis_tolerance_bps:
        Minimum absolute target band in basis points. This avoids making high
        resolution thresholds impossible because of price discreteness.
    min_basis_shock_bps:
        Skip detected return shocks that create a smaller basis displacement.
    require_full_horizon:
        If True, shocks without the full follow-up horizon are excluded rather
        than censored early at the end of the loaded data.
    """
    percentages = _normalise_percentages(resolution_pcts)
    if max_duration_seconds <= 0:
        raise ValueError("max_duration_seconds must be positive")

    required = {"timestamp", "p_spot", "p_perp"}
    missing = required.difference(df_prices.columns)
    if missing:
        raise ValueError(f"df_prices is missing required columns: {sorted(missing)}")

    frame = (
        df_prices
        .select(["timestamp", "p_spot", "p_perp"])
        .drop_nulls()
        .sort("timestamp")
    )
    times = frame["timestamp"].to_numpy().astype("datetime64[ns]")
    spot = frame["p_spot"].to_numpy().astype(float)
    perp = frame["p_perp"].to_numpy().astype(float)
    basis = np.log(perp) - np.log(spot)

    tolerance_abs = basis_tolerance_bps / 10_000.0
    minimum_shock_abs = min_basis_shock_bps / 10_000.0
    horizon_ns = int(round(max_duration_seconds * 1_000_000_000))

    results: List[dict] = []
    for shock_tick_raw in shock_events:
        shock_tick = int(shock_tick_raw)
        if shock_tick < 1 or shock_tick >= len(basis) - 1:
            continue

        start_time = times[shock_tick]
        censor_time = start_time + np.timedelta64(horizon_ns, "ns")
        full_horizon_available = censor_time <= times[-1]
        if require_full_horizon and not full_horizon_available:
            continue

        window_end = int(np.searchsorted(times, censor_time, side="right"))
        window_end = min(window_end, len(times))
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

        available_duration = float(
            (future_times[-1] - start_time) / np.timedelta64(1, "s")
        )
        censor_duration = (
            max_duration_seconds if full_horizon_available else available_duration
        )

        for pct in percentages:
            target_remaining = max(
                (1.0 - pct / 100.0) * initial_size,
                tolerance_abs,
            )
            inside = remaining_deviation <= target_remaining
            event_idx = _first_sustained_entry_index(
                inside,
                future_times,
                min_hold_seconds=min_hold_seconds,
            )

            if event_idx is None:
                status = 0
                length = censor_duration
                resolution_time = None
                resolution_tick = None
            else:
                status = 1
                resolution_time = future_times[event_idx]
                length = float(
                    (resolution_time - start_time) / np.timedelta64(1, "s")
                )
                resolution_tick = shock_tick + 1 + event_idx

            results.append(
                {
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
                }
            )

    if not results:
        return pl.DataFrame(
            schema={
                "event_tick": pl.Int64,
                "start_ts": pl.Datetime("ns"),
                "prev_ts": pl.Datetime("ns"),
                "Status": pl.Int64,
                "Length": pl.Float64,
                "resolution_pct": pl.Int64,
            }
        )
    return pl.DataFrame(results)


def fit_km_grid(
    events: Union[pd.DataFrame, pl.DataFrame],
    timeline_step_seconds: float = 1.0,
    max_duration_seconds: Optional[float] = None,
    group_cols: Sequence[str] = (
        "period",
        "first",
        "shock_significance",
        "resolution_pct",
    ),
    confidence_level: float = 0.95,
) -> Tuple[pd.DataFrame, Dict[Tuple, KaplanMeierFitter]]:
    """
    Fit Kaplan-Meier curves and save them on a common regular time grid.

    A one-second grid is recommended for reproducible plots. The underlying KM
    estimator remains a step function; evaluating it every second does not add
    artificial events.
    """
    if timeline_step_seconds <= 0:
        raise ValueError("timeline_step_seconds must be positive")

    df = events.to_pandas() if isinstance(events, pl.DataFrame) else events.copy()
    if df.empty:
        return pd.DataFrame(), {}

    actual_groups = [col for col in group_cols if col in df.columns]
    if not actual_groups:
        grouped = [((), df)]
    else:
        grouped = df.groupby(actual_groups, dropna=False, sort=True)

    rows: List[pd.DataFrame] = []
    models: Dict[Tuple, KaplanMeierFitter] = {}

    for group_key, subset in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        subset = subset.dropna(subset=["Length", "Status"])
        if subset.empty:
            continue

        group_horizon = float(subset["Length"].max())
        horizon = (
            float(max_duration_seconds)
            if max_duration_seconds is not None
            else group_horizon
        )
        timeline = np.arange(
            0.0,
            horizon + timeline_step_seconds * 0.5,
            timeline_step_seconds,
        )

        kmf = KaplanMeierFitter(alpha=1.0 - confidence_level)
        kmf.fit(
            durations=subset["Length"].astype(float),
            event_observed=subset["Status"].astype(int),
            timeline=timeline,
            label="survival",
        )
        models[group_key] = kmf

        curve = pd.DataFrame(
            {
                "time_s": timeline,
                "survival": kmf.survival_function_.iloc[:, 0].to_numpy(),
                "ci_lower": kmf.confidence_interval_.iloc[:, 0].to_numpy(),
                "ci_upper": kmf.confidence_interval_.iloc[:, 1].to_numpy(),
            }
        )
        for col, value in zip(actual_groups, group_key):
            curve[col] = value
        curve["n_episodes"] = len(subset)
        curve["n_resolved"] = int(subset["Status"].sum())
        curve["censoring_share"] = 1.0 - subset["Status"].mean()
        rows.append(curve)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(), models


def plot_periodic_km(
    km_grid: pd.DataFrame,
    resolution_pct: int = 90,
    first: Optional[str] = None,
    shock_significance: Optional[float] = None,
    marker_every_seconds: Optional[float] = 10.0,
    figsize: Tuple[float, float] = (8, 5),
):
    """Plot one KM line per period from the saved regular grid."""
    df = km_grid.copy()
    if "resolution_pct" in df.columns:
        df = df[df["resolution_pct"] == resolution_pct]
    if first is not None and "first" in df.columns:
        df = df[df["first"] == first]
    if shock_significance is not None and "shock_significance" in df.columns:
        df = df[np.isclose(df["shock_significance"], shock_significance)]
    if df.empty:
        raise ValueError("No KM rows match the requested filters")

    fig, ax = plt.subplots(figsize=figsize)
    period_col = "period" if "period" in df.columns else None
    groups = df.groupby(period_col) if period_col else [("KM", df)]

    for label, subset in groups:
        subset = subset.sort_values("time_s")
        step = float(np.diff(subset["time_s"].unique()).min()) if len(subset) > 1 else 1.0
        if marker_every_seconds is None:
            markevery = None
            marker = None
        else:
            markevery = max(1, int(round(marker_every_seconds / step)))
            marker = "o"
        line, = ax.step(
            subset["time_s"],
            subset["survival"],
            where="post",
            label=str(label),
            marker=marker,
            markevery=markevery,
            markersize=3,
        )
        ax.fill_between(
            subset["time_s"],
            subset["ci_lower"],
            subset["ci_upper"],
            step="post",
            alpha=0.15,
            color=line.get_color(),
        )

    ax.set_xlabel("Time since shock (seconds)")
    ax.set_ylabel("Probability dislocation remains unresolved")
    ax.set_title(f"Kaplan–Meier basis resolution: {resolution_pct}%")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Period")
    fig.tight_layout()
    return fig, ax


def fit_loglogistic_aft_by_period(
    model_data: pd.DataFrame,
    covariates: Sequence[str],
    period_col: str = "period",
    duration_col: str = "Length",
    event_col: str = "Status",
    resolution_pct: Optional[int] = 90,
    max_duration_seconds: Optional[float] = None,
    timeline_step_seconds: float = 1.0,
    percentile_step: int = 1,
    min_episodes: int = 100,
    ancillary: bool = False,
) -> Tuple[
    Dict[str, LogLogisticAFTFitter],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Fit one Log-Logistic AFT model per period.

    Returns
    -------
    models:
        Fitted lifelines models keyed by period.
    coefficient_table:
        Model coefficients and inference statistics.
    curve_grid:
        Predicted survival and hazard for the median covariate profile, evaluated
        every timeline_step_seconds.
    percentile_grid:
        Approximate first-passage quantiles of the model CDF at every requested
        percentage (default 1%, 2%, ..., 99%).

    Notes
    -----
    Fit the AFT model at one primary resolution definition, such as 90%. Fitting
    99 separate AFT models per period is usually unnecessary. The one-percent
    grid is more useful for summarising the fitted distribution.
    """
    df = model_data.copy()
    if resolution_pct is not None and "resolution_pct" in df.columns:
        df = df[df["resolution_pct"] == resolution_pct]
    if df.empty:
        return {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    missing = {period_col, duration_col, event_col, *covariates}.difference(df.columns)
    if missing:
        raise ValueError(f"model_data is missing columns: {sorted(missing)}")

    models: Dict[str, LogLogisticAFTFitter] = {}
    summaries: List[pd.DataFrame] = []
    curves: List[pd.DataFrame] = []
    percentiles: List[pd.DataFrame] = []

    for period, subset in df.groupby(period_col, sort=True):
        fit_df = (
            subset[[duration_col, event_col, *covariates]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .copy()
        )
        fit_df[duration_col] = fit_df[duration_col].clip(lower=0.001)
        if len(fit_df) < min_episodes or fit_df[event_col].sum() < 5:
            continue

        aft = LogLogisticAFTFitter()
        aft.fit(
            fit_df,
            duration_col=duration_col,
            event_col=event_col,
            ancillary=ancillary,
        )
        period_key = str(period)
        models[period_key] = aft

        summary = aft.summary.reset_index()
        summary[period_col] = period
        summary["n_episodes"] = len(fit_df)
        summary["n_resolved"] = int(fit_df[event_col].sum())
        summary["concordance_index"] = aft.concordance_index_
        summary["AIC"] = aft.AIC_
        if resolution_pct is not None:
            summary["resolution_pct"] = resolution_pct
        summaries.append(summary)

        reference = pd.DataFrame(
            [{col: float(fit_df[col].median()) for col in covariates}]
        )
        horizon = (
            float(max_duration_seconds)
            if max_duration_seconds is not None
            else float(fit_df[duration_col].max())
        )
        positive_times = np.arange(
            max(timeline_step_seconds, 0.001),
            horizon + timeline_step_seconds * 0.5,
            timeline_step_seconds,
        )
        survival = aft.predict_survival_function(reference, times=positive_times).iloc[:, 0]
        hazard = aft.predict_hazard(reference, times=positive_times).iloc[:, 0]
        curve = pd.DataFrame(
            {
                "time_s": positive_times,
                "survival": survival.to_numpy(),
                "hazard": hazard.to_numpy(),
                period_col: period,
            }
        )
        if resolution_pct is not None:
            curve["resolution_pct"] = resolution_pct
        curves.append(curve)

        cdf = 1.0 - curve["survival"].to_numpy()
        percentile_values = range(percentile_step, 100, percentile_step)
        pct_rows = []
        for pct in percentile_values:
            target = pct / 100.0
            reached = np.flatnonzero(cdf >= target)
            time_value = (
                float(curve["time_s"].iloc[reached[0]])
                if reached.size
                else np.nan
            )
            pct_rows.append(
                {
                    period_col: period,
                    "cdf_percentile": pct,
                    "time_s": time_value,
                    "within_timeline": bool(reached.size),
                }
            )
        pct_frame = pd.DataFrame(pct_rows)
        if resolution_pct is not None:
            pct_frame["resolution_pct"] = resolution_pct
        percentiles.append(pct_frame)

    return (
        models,
        pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(),
        pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(),
        pd.concat(percentiles, ignore_index=True) if percentiles else pd.DataFrame(),
    )


class BasisSurvivalAnalysis(SurvivalAnalysis):
    """SurvivalAnalysis subclass using basis resolution and clock-time censoring."""

    def _get_forward_filled_data(
        self,
        start: DateLike,
        end: DateLike,
    ) -> Tuple[TradeData, pl.DataFrame]:
        start_str = _format_yyyymmdd(start)
        end_str = _format_yyyymmdd(end)
        start_t = dt.datetime.strptime(start_str, "%Y%m%d")
        end_t = dt.datetime.strptime(end_str, "%Y%m%d")

        data = TradeData(self.ticker, source=self.source, cm_um=self.cm_um)
        data.grab_trades_data(end_t, (end_t - start_t).days)
        df_ret_ff, _ = all_formatting(
            data.df_trades_spots,
            data.df_trades_perps,
        )
        return data, df_ret_ff.sort("timestamp")

    def fit_basis_events(
        self,
        start: DateLike,
        end: DateLike,
        shock_significance: float = 0.001,
        method: str = "lee_mykland",
        resolution_pcts: Iterable[int] = (50, 90, 95),
        max_duration_seconds: float = 120.0,
        refractory_seconds: float = 5.0,
        min_hold_seconds: float = 0.0,
        basis_tolerance_bps: float = 0.0,
        min_basis_shock_bps: float = 0.0,
        first: str = "spot",
        groupday: bool = True,
        require_full_horizon: bool = True,
        load_followup_days: int = 1,
    ) -> Tuple[TradeData, List[int], pl.DataFrame]:
        """
        Detect shocks and calculate basis-resolution events for one analysis period.

        The data pull is extended beyond the period end so that shocks near the
        boundary can still receive their full clock-time follow-up. Only shocks
        whose start timestamps lie within [start, end) are retained.
        """
        analysis_start = _as_timestamp(start)
        analysis_end = _as_timestamp(end)
        load_end = analysis_end + pd.Timedelta(days=load_followup_days)

        data, df = self._get_forward_filled_data(analysis_start, load_end)

        if method == "lee_mykland":
            _, candidates, _ = find_shocks2(
                df,
                method=method,
                lm_signif=shock_significance,
                groupday=groupday,
                first=first,
                follow_ticks=0,
            )
        else:
            _, candidates, _ = find_shocks2(
                df,
                method=method,
                quantile_threshold=1.0 - shock_significance,
                groupday=groupday,
                first=first,
                follow_ticks=0,
            )

        times = df["timestamp"].to_numpy().astype("datetime64[ns]")
        candidate_ticks = (
            candidates[:, 0].astype(np.int64)
            if len(candidates)
            else np.array([], dtype=np.int64)
        )
        start_np = np.datetime64(analysis_start.to_datetime64(), "ns")
        end_np = np.datetime64(analysis_end.to_datetime64(), "ns")
        candidate_ticks = candidate_ticks[
            (times[candidate_ticks] >= start_np)
            & (times[candidate_ticks] < end_np)
        ]
        events = greedy_refractory_seconds(
            candidate_ticks,
            times,
            refractory_seconds=refractory_seconds,
        )

        event_table = calculate_basis_resolution_events(
            df_prices=df,
            shock_events=events,
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
        return data, events, event_table

    def fit_basis_over_periods(
        self,
        start: DateLike,
        end: DateLike,
        period_freq: str = "MS",
        shock_significance: float = 0.001,
        method: str = "lee_mykland",
        resolution_pcts: Iterable[int] = (50, 90, 95),
        max_duration_seconds: float = 120.0,
        refractory_seconds: float = 5.0,
        min_hold_seconds: float = 0.0,
        basis_tolerance_bps: float = 0.0,
        min_basis_shock_bps: float = 0.0,
        first: str = "spot",
        timeline_step_seconds: float = 1.0,
        output_events_csv: Optional[Union[str, Path]] = None,
        output_km_csv: Optional[Union[str, Path]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run basis-resolution survival analysis over calendar periods."""
        event_frames: List[pd.DataFrame] = []

        for period_start, period_end, label in build_periods(start, end, period_freq):
            _, _, events = self.fit_basis_events(
                start=period_start,
                end=period_end,
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
                frame = events.to_pandas()
                frame["period"] = label
                event_frames.append(frame)

        all_events = (
            pd.concat(event_frames, ignore_index=True)
            if event_frames
            else pd.DataFrame()
        )
        if output_events_csv is not None:
            Path(output_events_csv).parent.mkdir(parents=True, exist_ok=True)
            all_events.to_csv(output_events_csv, index=False)

        km_grid, _ = fit_km_grid(
            all_events,
            timeline_step_seconds=timeline_step_seconds,
            max_duration_seconds=max_duration_seconds,
        )
        if output_km_csv is not None:
            Path(output_km_csv).parent.mkdir(parents=True, exist_ok=True)
            km_grid.to_csv(output_km_csv, index=False)

        return all_events, km_grid

    def build_aft_data_over_periods(
        self,
        start: DateLike,
        end: DateLike,
        period_freq: str = "QS",
        shock_significance: float = 0.001,
        method: str = "lee_mykland",
        primary_resolution_pct: int = 90,
        max_duration_seconds: float = 120.0,
        refractory_seconds: float = 5.0,
        min_hold_seconds: float = 0.0,
        basis_tolerance_bps: float = 0.0,
        min_basis_shock_bps: float = 0.0,
        first: str = "spot",
        output_csv: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """
        Build a covariate-enriched event table for period-specific AFT models.

        This reuses the existing covariate pipeline from survival_analysis.py,
        but passes it the new basis-resolution event outcomes. Quarterly or
        annual periods are recommended because monthly AFT coefficients may be
        unstable even when monthly KM summaries are informative.
        """
        model_frames: List[pd.DataFrame] = []

        for period_start, period_end, label in build_periods(start, end, period_freq):
            data, _, events = self.fit_basis_events(
                start=period_start,
                end=period_end,
                shock_significance=shock_significance,
                method=method,
                resolution_pcts=[primary_resolution_pct],
                max_duration_seconds=max_duration_seconds,
                refractory_seconds=refractory_seconds,
                min_hold_seconds=min_hold_seconds,
                basis_tolerance_bps=basis_tolerance_bps,
                min_basis_shock_bps=min_basis_shock_bps,
                first=first,
            )
            if not events.height:
                continue

            # The legacy covariate function expects the old nested structure and
            # hard-codes the 0.001 shock key and 90% label. Restrict the frame to
            # the columns it needs, then provide a compatibility wrapper.
            event_pd = events.select([
                "event_tick",
                "start_ts",
                "prev_ts",
                "Status",
                "Length",
                "shock_size",
            ]).to_pandas()
            event_pd["Shock"] = f"{primary_resolution_pct}%"
            wrapped = {
                0.001: {
                    "90%": pl.from_pandas(event_pd)
                }
            }

            period_model = legacy_find_covariates(
                _format_yyyymmdd(period_start),
                _format_yyyymmdd(period_end),
                data,
                wrapped,
            )
            period_model["period"] = label
            period_model["first"] = first
            period_model["resolution_pct"] = primary_resolution_pct
            period_model["shock_significance"] = shock_significance
            model_frames.append(period_model)

        result = (
            pd.concat(model_frames, ignore_index=True)
            if model_frames
            else pd.DataFrame()
        )
        if output_csv is not None:
            Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(output_csv, index=False)
        return result
