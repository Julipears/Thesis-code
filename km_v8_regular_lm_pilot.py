"""Pilot comparison of saved v5 KM events with regular-grid Lee--Mykland events.

This script never writes into the v5 ``sa_*`` folders.  It detects shocks on
regular last-trade grids, maps each detected bin to the last actual trade of the
origin market in that bin, and measures basis resolution on the original
irregular event-time stream.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

from survival_analysis_data_processing_final import (
    build_aft_covariates,
    build_event_time_prices,
    calculate_basis_resolution_events,
    greedy_refractory_seconds,
)
from trade_data_pull import TradeData


MARKETS = {
    "btc_um": ("BTCUSDT", "um", Path("sa_btc_um")),
    "btc_cm": ("BTCUSDT", "cm", Path("sa_btc_cm")),
    "eth_um": ("ETHUSDT", "um", Path("sa_eth_um")),
    "eth_cm": ("ETHUSDT", "cm", Path("sa_eth_cm")),
}
DEFAULT_DATES = ["2021-07-01", "2023-07-01", "2025-07-01"]
DEFAULT_GRIDS = ["1s", "5s", "10s"]
METHODOLOGY_VERSION = "v8_regular_lm_daily_v4"


def grid_seconds(freq: str) -> float:
    return pd.Timedelta(freq).total_seconds()


def regular_lee_mykland(
    log_prices: pd.Series,
    target_start: pd.Timestamp,
    target_end: pd.Timestamp,
    freq: str,
    significance: float = 0.001,
    require_complete_grid: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Lee--Mykland test on genuinely regular, equally spaced log prices."""
    # Sampling frequency and LM local-volatility window:
    #   n_year = number of equally spaced observations in one year
    #   k       = ceil(sqrt(n_year))
    # The annual observation count is used only to choose the local window;
    # the extreme-value normalization below uses the target day's actual n.
    step_seconds = grid_seconds(freq)
    annual_observations = 365.0 * 24.0 * 60.0 * 60.0 / step_seconds
    k = int(np.ceil(np.sqrt(annual_observations)))

    log_prices = pd.to_numeric(log_prices, errors="coerce").sort_index()

    # Log return in bin i:
    #   r_i = log(P_i) - log(P_{i-1}).
    returns = log_prices.diff()

    # Local bipower-variation estimator (LM Equation 8):
    #   sigma_i^2 = (1 / (k - 2))
    #               * sum_{j=i-k+2}^{i-1} |r_j| |r_{j-1}|.
    #
    # First form |r_j||r_{j-1}|. The final shift(1) ensures that the
    # volatility estimate for r_i contains only information available before
    # bin i, so the tested return cannot inflate its own denominator.
    bpv = (returns.abs() * returns.shift(1).abs()).shift(1)

    # MATLAB ``movmean(bpv, [k-3 0])`` contains exactly k-2 lagged bipower
    # terms. rolling(...).mean() therefore supplies both the sum and the
    # 1/(k-2) denominator in Equation 8.
    bpv_terms = k - 2
    sigma2 = bpv.rolling(window=bpv_terms, min_periods=bpv_terms).mean()
    sigma = np.sqrt(sigma2)

    # Lee--Mykland standardized return statistic:
    #   L_i = r_i / sigma_i.
    statistic_l = returns / sigma

    target_index = returns.index[(returns.index >= target_start) & (returns.index < target_end)]
    expected_index = pd.date_range(
        target_start,
        target_end,
        freq=freq,
        inclusive="left",
    )
    grid_matches = (
        len(target_index) == len(expected_index)
        and np.array_equal(
            target_index.to_numpy(dtype="datetime64[ns]"),
            expected_index.to_numpy(dtype="datetime64[ns]"),
        )
    )
    if require_complete_grid and not grid_matches:
        missing = expected_index.difference(target_index)
        unexpected = target_index.difference(expected_index)
        raise AssertionError(
            f"LM target day is not a complete regular {freq} grid: "
            f"expected={len(expected_index)}, actual={len(target_index)}, "
            f"missing={len(missing)} {missing[:5].tolist()}, "
            f"unexpected={len(unexpected)} {unexpected[:5].tolist()}"
        )
    target = returns.index.isin(target_index)
    n = len(expected_index)
    if n < 2:
        raise ValueError(f"Only {n} regular observations in target day")

    # Extreme-value normalization constants for n tests:
    #   c   = sqrt(2/pi)
    #   S_n = c * sqrt(2 log n)
    #   C_n = sqrt(2 log n)/c
    #         - log(pi log n)/(2 c sqrt(2 log n)).
    # These transform max |L_i| to its asymptotic Gumbel scale.
    c = np.sqrt(2.0 / np.pi)
    sn = c * np.sqrt(2.0 * np.log(n))
    cn = (
        np.sqrt(2.0 * np.log(n)) / c
        - np.log(np.pi * np.log(n))
        / (2.0 * c * np.sqrt(2.0 * np.log(n)))
    )

    # Critical value at the requested family-wise significance level alpha:
    #   beta* = -log[-log(1 - alpha)].
    beta_star = -np.log(-np.log(1.0 - significance))

    # Per-bin transformed LM statistic:
    #   T_i = S_n (|L_i| - C_n).
    statistic_t = (statistic_l.abs() - cn) * sn

    # Jump decision rule for target-day bins:
    #   declare a jump at i when T_i > beta*.
    detected = target & statistic_t.gt(beta_star) & statistic_t.notna()
    candidates = pd.DataFrame({
        "timestamp_bin": returns.index[detected],
        "return": returns.loc[detected].to_numpy(),
        "sigma": sigma.loc[detected].to_numpy(),
        "L": statistic_l.loc[detected].to_numpy(),
        "T": statistic_t.loc[detected].to_numpy(),
    }).reset_index(drop=True)
    metadata = {
        "grid": freq,
        "grid_seconds": step_seconds,
        "k": k,
        "bpv_terms": k - 2,
        "lookback_seconds": (k - 1) * step_seconds,
        "n_target": n,
        "significance": significance,
        "beta_star": beta_star,
    }
    return candidates, metadata


def map_bins_to_origin_trade_ticks(
    candidates: pd.DataFrame,
    origin_trades: pl.DataFrame,
    event_time: pl.DataFrame,
    freq: str,
    refractory_seconds: float,
) -> tuple[list[int], pd.DataFrame]:
    """Map a detected regular bin to its last actual origin-market trade."""
    if candidates.empty:
        return [], candidates.assign(start_ts=pd.NaT, event_tick=pd.Series(dtype="int64"))

    # mapping potential jumps to the actual trades
    wanted = pl.from_pandas(candidates[["timestamp_bin"]]).with_columns(
        pl.col("timestamp_bin").cast(pl.Datetime("ns"))
    )
    # for each shock within a bin, find the last trade in that bin and use its timestamp as the shock timestamp
    # just keep in mind this is a limitation. Although trades that revert itself within the bucket will be neglected
    # (which is kind of the desired behavior anyways), and shock sizes may be understated
    # (which should have low impact considering we are only taking very large shocks anyways)
    mapped = (
        origin_trades
        .select(pl.col("timestamp").cast(pl.Datetime("ns")))
        .with_columns(pl.col("timestamp").dt.truncate(freq).alias("timestamp_bin"))
        .join(wanted, on="timestamp_bin", how="inner")
        .group_by("timestamp_bin")
        .agg(pl.col("timestamp").max().alias("start_ts"))
        .sort("start_ts")
        .to_pandas()
    )
    mapped["timestamp_bin"] = pd.to_datetime(mapped["timestamp_bin"])
    mapped["start_ts"] = pd.to_datetime(mapped["start_ts"])
    out = candidates.merge(mapped, on="timestamp_bin", how="inner", validate="one_to_one")

    times = event_time.sort("timestamp")["timestamp"].to_numpy().astype("datetime64[ns]")
    starts = out["start_ts"].to_numpy(dtype="datetime64[ns]")
    ticks = np.searchsorted(times, starts, side="left")
    exact = (ticks < len(times)) & (times[np.minimum(ticks, len(times) - 1)] == starts)
    out = out.loc[exact].copy()
    out["event_tick"] = ticks[exact].astype(np.int64)

    # calculating the refractory seconds to avoid double counting events that are too close to each other
    # this is set to 5s
    selected = greedy_refractory_seconds(
        out["event_tick"].to_numpy(dtype=np.int64),
        times,
        refractory_seconds,
    )
    out = out[out["event_tick"].isin(selected)].sort_values("start_ts").reset_index(drop=True)
    return out["event_tick"].astype(int).tolist(), out


def load_v5_events(root: Path, first: str, target_start: pd.Timestamp) -> pd.DataFrame:
    path = root / "events" / f"{first}_{target_start:%Y-%m}.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    frame["start_ts"] = pd.to_datetime(frame["start_ts"], errors="coerce").dt.round("ms")
    frame = frame[
        (frame["start_ts"] >= target_start)
        & (frame["start_ts"] < target_start + pd.Timedelta(days=1))
    ].copy()
    if "resolution_pct" in frame:
        frame = frame[frame["resolution_pct"].astype(int).eq(90)].copy()
    return frame.drop_duplicates(["start_ts"], keep="first").reset_index(drop=True)


def km_resolved_by(frame: pd.DataFrame, horizon: float = 10.0) -> float:
    if frame.empty:
        return np.nan
    clean = frame.dropna(subset=["Length", "Status"])
    if clean.empty:
        return np.nan
    kmf = KaplanMeierFitter().fit(clean["Length"].astype(float), clean["Status"].astype(int))
    return float(1.0 - kmf.predict(horizon))


def overlap_metrics(v8: pd.DataFrame, v5: pd.DataFrame, tolerance_seconds: float) -> dict:
    if v8.empty or v5.empty:
        return {"v8_matched_share": np.nan, "v5_matched_share": np.nan}
    a = np.sort(v8["start_ts"].to_numpy(dtype="datetime64[ns]"))
    b = np.sort(v5["start_ts"].to_numpy(dtype="datetime64[ns]"))
    tolerance_ns = int(tolerance_seconds * 1e9)

    def matched_share(source, target):
        positions = np.searchsorted(target, source)
        distances = np.full(len(source), np.iinfo(np.int64).max, dtype=np.int64)
        right = positions < len(target)
        distances[right] = np.minimum(
            distances[right],
            np.abs((target[positions[right]] - source[right]).astype("timedelta64[ns]").astype(np.int64)),
        )
        left = positions > 0
        distances[left] = np.minimum(
            distances[left],
            np.abs((target[positions[left] - 1] - source[left]).astype("timedelta64[ns]").astype(np.int64)),
        )
        return float(np.mean(distances <= tolerance_ns))

    return {"v8_matched_share": matched_share(a, b), "v5_matched_share": matched_share(b, a)}


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def plot_pooled_km_comparison(summary: pd.DataFrame, output_root: Path) -> None:
    """Plot pooled v5 and regular-grid KM curves for every completed market/direction."""
    if summary.empty:
        return
    grids = summary["grid"].dropna().astype(str).unique()
    if len(grids) != 1:
        return
    grid = grids[0]
    completed_markets = [market for market in MARKETS if market in set(summary["market"])]
    if not completed_markets:
        return
    fig, axes = plt.subplots(
        len(completed_markets), 2,
        figsize=(13, 4.25 * len(completed_markets)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    curve_rows = []
    for row, market in enumerate(completed_markets):
        _, _, v5_root = MARKETS[market]
        market_summary = summary[(summary["market"] == market) & (summary["grid"] == grid)]
        dates = sorted(pd.to_datetime(market_summary["date"].dropna().unique()))
        for col, first in enumerate(("spot", "perp")):
            ax = axes[row, col]
            frames = {"V5 irregular": [], f"V8 regular {grid}": []}
            for date in dates:
                target = pd.Timestamp(date)
                v5 = load_v5_events(v5_root, first, target)
                if not v5.empty:
                    frames["V5 irregular"].append(v5)
                event_path = output_root / "events" / market / f"{first}_{target:%Y-%m-%d}_{grid}.parquet"
                if event_path.exists():
                    frames[f"V8 regular {grid}"].append(pd.read_parquet(event_path))
            colors = {"V5 irregular": "#e6862a", f"V8 regular {grid}": "#158f63"}
            for label, pieces in frames.items():
                if not pieces:
                    continue
                pooled = pd.concat(pieces, ignore_index=True).dropna(subset=["Length", "Status"])
                if pooled.empty:
                    continue
                plot_label = f"{label} (n={len(pooled):,})"
                kmf = KaplanMeierFitter().fit(
                    pooled["Length"].astype(float), pooled["Status"].astype(int), label=plot_label
                )
                kmf.plot_survival_function(
                    ax=ax, ci_show=True, ci_alpha=0.12, color=colors[label]
                )
                curve = kmf.survival_function_.reset_index()
                curve.columns = ["timeline", "probability_unresolved"]
                curve["market"] = market
                curve["first"] = first
                curve["specification"] = label
                curve["n_events"] = len(pooled)
                curve_rows.append(curve)
            ax.set_title(f"{market.upper()} — {first}-first")
            ax.set_xlim(0, 120)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Seconds since shock")
            ax.set_ylabel("Probability unresolved")
            ax.grid(alpha=0.2)
    dates_all = pd.to_datetime(summary["date"].dropna())
    date_label = (
        f"{dates_all.min():%Y-%m-%d} to {dates_all.max():%Y-%m-%d}"
        if not dates_all.empty else "completed dates"
    )
    fig.suptitle(
        f"Pooled-event KM curves ({date_label}): V5 vs regular-grid {grid}", y=0.995
    )
    fig.tight_layout()
    plot_dir = output_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / f"km_curves_{grid}_v5_vs_v8.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    if curve_rows:
        atomic_csv(pd.concat(curve_rows, ignore_index=True), output_root / f"km_curves_{grid}_v5_vs_v8.csv")


def _weighted_km_inputs(
    paths: list[Path], v5: bool = False, horizon: float = 120.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse events into exact KM weights through an administrative horizon."""
    counts: dict[tuple[float, int], int] = {}
    columns = ["Length", "Status", "start_ts", "resolution_pct"] if v5 else ["Length", "Status"]
    for path in paths:
        frame = pd.read_parquet(path, columns=columns)
        if v5:
            frame = frame[frame["resolution_pct"].astype("Int64").eq(90)]
            frame = frame.drop_duplicates("start_ts", keep="first")
        frame["Length"] = pd.to_numeric(frame["Length"], errors="coerce")
        frame["Status"] = pd.to_numeric(frame["Status"], errors="coerce")
        frame = frame.dropna(subset=["Length", "Status"])
        beyond_horizon = frame["Length"].gt(horizon)
        frame.loc[beyond_horizon, "Status"] = 0
        frame.loc[beyond_horizon, "Length"] = horizon
        grouped = frame.groupby(["Length", "Status"], sort=False).size()
        for (length, status), weight in grouped.items():
            key = (float(length), int(status))
            counts[key] = counts.get(key, 0) + int(weight)
        del frame, grouped
    keys = list(counts)
    return (
        np.fromiter((key[0] for key in keys), dtype=float),
        np.fromiter((key[1] for key in keys), dtype=np.int8),
        np.fromiter((counts[key] for key in keys), dtype=np.int64),
    )


def plot_yearly_btc_km(summary: pd.DataFrame, output_root: Path) -> None:
    """Write exact pooled yearly V8 BTC KM plots using bounded-memory parquet reads."""
    output_root = Path(output_root)
    summary = summary[summary["market"].isin(["btc_um", "btc_cm"])].copy()
    if summary.empty:
        return
    grid = str(summary["grid"].dropna().iloc[0])
    years = sorted(pd.to_datetime(summary["date"], errors="coerce").dropna().dt.year.unique())
    plot_dir = output_root / "plots" / "yearly_btc"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for year in years:
        year_summary = summary[pd.to_datetime(summary["date"]).dt.year.eq(year)]
        markets = [market for market in ("btc_um", "btc_cm") if market in set(year_summary["market"])]
        fig, axes = plt.subplots(len(markets), 2, figsize=(13, 4.25 * len(markets)),
                                 sharex=True, sharey=True, squeeze=False)
        curve_rows = []
        for row, market in enumerate(markets):
            for col, first in enumerate(("spot", "perp")):
                ax = axes[row, col]
                sources = ((
                    f"KM {grid}",
                    sorted((output_root / "events" / market).glob(f"{first}_{year}-*_{grid}.parquet")),
                    False,
                    "#158f63",
                ),)
                for label, paths, is_v5, color in sources:
                    durations, observed, weights = _weighted_km_inputs(paths, v5=is_v5)
                    if not len(durations):
                        continue
                    n_events = int(weights.sum())
                    kmf = KaplanMeierFitter().fit(durations, observed, weights=weights,
                                                  label=f"{label} (n={n_events:,})")
                    kmf.plot_survival_function(ax=ax, ci_show=True, ci_alpha=0.12, color=color)
                    curve = kmf.survival_function_.reset_index()
                    curve.columns = ["timeline", "probability_unresolved"]
                    curve.insert(0, "year", year)
                    curve["market"] = market
                    curve["first"] = first
                    curve["specification"] = label
                    curve["n_events"] = n_events
                    curve_rows.append(curve)
                    del durations, observed, weights, kmf, curve
                ax.set_title(f"{market.upper()} — {first}-first")
                ax.set_xlim(0, 120)
                ax.set_ylim(0, 1)
                ax.set_xlabel("Seconds since shock")
                ax.set_ylabel("Probability unresolved")
                ax.grid(alpha=0.2)
        fig.suptitle(f"BTC Kaplan–Meier curves, {year} ({grid} event grid)", y=0.995)
        fig.tight_layout()
        fig.savefig(plot_dir / f"btc_km_curves_{year}_{grid}.png",
                    dpi=180, bbox_inches="tight")
        plt.close(fig)
        if curve_rows:
            atomic_csv(pd.concat(curve_rows, ignore_index=True),
                       plot_dir / f"btc_km_curves_{year}_{grid}.csv")
        gc.collect()


def plot_asset_km_month_comparison(
    output_root: Path,
    asset: str = "btc",
    months: tuple[str, ...] = ("2021-12", "2022-12", "2023-12", "2025-12"),
    grid: str = "1s",
    horizon: float = 120.0,
    require_complete: bool = True,
    plot_step_seconds: float = 0.25,
) -> Path:
    """Create the thesis KM panel for one asset from regular-grid event files."""
    output_root = Path(output_root)
    asset = asset.lower()
    if asset not in {"btc", "eth"}:
        raise ValueError("asset must be 'btc' or 'eth'")
    if plot_step_seconds <= 0:
        raise ValueError("plot_step_seconds must be positive")
    styles = {
        (f"{asset}_um", "spot"): ("Linear: spot", "tab:blue", "-"),
        (f"{asset}_um", "perp"): ("Linear: perp", "tab:blue", "--"),
        (f"{asset}_cm", "spot"): ("Inverse: spot", "tab:orange", "-"),
        (f"{asset}_cm", "perp"): ("Inverse: perp", "tab:orange", "--"),
    }
    missing_series = [
        (month, market, first)
        for month in months
        for market, first in styles
        if not list(
            (output_root / "events" / market)
            .glob(f"{first}_{month}-*_{grid}.parquet")
        )
    ]
    if missing_series and require_complete:
        raise ValueError(
            f"Cannot build the requested four-column {asset.upper()} KM panel; missing "
            f"month/market/direction series: {missing_series}"
        )
    complete_months = list(months)
    fig, axes = plt.subplots(
        1, len(complete_months), figsize=(4.5 * len(complete_months), 3.5),
        sharex=True, sharey=True, squeeze=False,
    )
    axes = axes.ravel()
    curve_rows = []
    # Shock detection remains on ``grid`` (normally 1s). Resolution durations
    # are measured on the millisecond event-time stream, so evaluate the fitted
    # KM curve more finely to display that retained subsecond information.
    timeline = np.arange(
        0.0, horizon + plot_step_seconds / 2.0, plot_step_seconds
    )
    for ax, month in zip(axes, complete_months):
        for (market, first), (label, color, linestyle) in styles.items():
            paths = sorted((output_root / "events" / market).glob(f"{first}_{month}-*_{grid}.parquet"))
            durations, observed, weights = _weighted_km_inputs(paths, horizon=horizon)
            if not len(durations):
                continue
            n_events = int(weights.sum())
            kmf = KaplanMeierFitter().fit(durations, observed, weights=weights, label=label)
            curve = pd.DataFrame({
                "timeline": timeline,
                "survival": kmf.survival_function_at_times(timeline).to_numpy(),
            })
            ax.step(curve["timeline"], curve["survival"], where="post", color=color,
                    linestyle=linestyle, linewidth=1.8, label=label)
            curve.insert(0, "period", month)
            curve["market"] = market
            curve["first"] = first
            curve["resolution_pct"] = 90
            curve["n_events"] = n_events
            curve_rows.append(curve)
        ax.set_title(month)
        ax.set_xlim(0, horizon)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Seconds since basis displacement")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Probability unresolved")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
                   ncol=4, frameon=False)
    fig.suptitle("Kaplan–Meier curves for 90% basis resolution", y=1.07)
    fig.tight_layout()
    plot_dir = output_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    horizon_suffix = "" if horizon == 120 else f"_{horizon:g}s"
    output_path = plot_dir / f"survival_all_graphs_{asset}_v8{horizon_suffix}.png"
    # Match the thesis notebook's default Matplotlib rendering.  In particular,
    # do not upscale to 180 DPI: that makes the 1.8-point step lines and text
    # appear materially heavier than the original figure.
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    if curve_rows:
        atomic_csv(pd.concat(curve_rows, ignore_index=True),
                   plot_dir / f"survival_all_graphs_{asset}_v8{horizon_suffix}.csv")
    return output_path


def plot_btc_km_month_comparison(
    output_root: Path,
    months: tuple[str, ...] = ("2021-12", "2022-12", "2023-12", "2025-12"),
    grid: str = "1s",
    horizon: float = 120.0,
    require_complete: bool = True,
    plot_step_seconds: float = 0.25,
) -> Path:
    """Backward-compatible wrapper for the BTC thesis panel."""
    return plot_asset_km_month_comparison(
        output_root=output_root,
        asset="btc",
        months=months,
        grid=grid,
        horizon=horizon,
        require_complete=require_complete,
        plot_step_seconds=plot_step_seconds,
    )


def run_one(
    market: str,
    date: str,
    grids: list[str],
    output_root: Path,
    significance: float,
    build_covariates: bool = False,
    data: TradeData | None = None,
    event_time=None,
) -> list[dict]:
    symbol, cm_um, v5_root = MARKETS[market]
    target_start = pd.Timestamp(date)
    target_end = target_start + pd.Timedelta(days=1)
    load_start = target_start - pd.Timedelta(days=1)
    load_end = target_end + pd.Timedelta(days=1)

    owns_data = data is None
    if owns_data:
        print(f"[load] {market} {date}", flush=True)
        data = TradeData(symbol, source="Binance", cm_um=cm_um)
        # ``grab_trades_data(end_date, days)`` includes end_date itself. Passing
        # target_end yields exactly: preceding day, target day, following day.
        data.grab_trades_data(target_end.to_pydatetime(), 3, n_jobs=4)
    if event_time is None:
        event_time = build_event_time_prices(
            data.df_trades_spots, data.df_trades_perps
        ).sort("timestamp")
    rows = []

    for freq in grids:
        print(f"  [grid] {freq}", flush=True)
        regular, regular_prices = data.agg_last_trade_to_intervals(
            freq=freq,
            start=load_start,
            end=load_end,
            fill_gaps=True,
            # Previous-tick interpolation is required to preserve a genuinely
            # regular sampling grid. Staleness creates zero returns; dropping
            # stale seconds would silently turn the series irregular again.
            max_fill_gap_ms=None,
            return_fill_diagnostics=True,
            rename_for_vecm=True,
            # The transformed model frame drops its first row because the
            # first log difference is undefined. LM consumes price levels, so
            # retain that row to avoid losing midnight when it is the first
            # valid bin in a sliced/archive block.
            retain_initial_grid_row=True,
        )
        if regular_prices.empty:
            raise ValueError(f"No regular {freq} prices for {market} {date}")
        # The general aggregation helper returns a float32 model frame for
        # memory efficiency. LM is sensitive to small high-frequency returns,
        # so calculate its log prices from the retained float64 transaction
        # prices instead of using quantized log columns.
        regular = pd.DataFrame(index=regular_prices.index)
        regular["log_last_trade_spot"] = np.log(
            regular_prices["last_trade_spot"].astype(np.float64)
        )
        regular["log_last_trade_perp"] = np.log(
            regular_prices["last_trade_perp"].astype(np.float64)
        )

        for first in ("spot", "perp"):
            log_column = f"log_last_trade_{first}"
            candidates, lm_meta = regular_lee_mykland(
                regular[log_column], target_start, target_end, freq, significance
            )
            # LM thresholds remain day-specific, but V5's five-second clock-time
            # refractory rule is continuous. Include the preceding day's
            # candidates when filtering so midnight does not reset that rule.
            previous_candidates, _ = regular_lee_mykland(
                regular[log_column], target_start - pd.Timedelta(days=1), target_start,
                freq, significance, require_complete_grid=False,
            )
            refractory_candidates = pd.concat(
                [previous_candidates, candidates], ignore_index=True
            ).sort_values("timestamp_bin").reset_index(drop=True)
            origin = data.df_trades_spots if first == "spot" else data.df_trades_perps
            _, mapped = map_bins_to_origin_trade_ticks(
                refractory_candidates, origin, event_time, freq, refractory_seconds=5.0
            )
            mapped = mapped[
                (mapped["start_ts"] >= target_start)
                & (mapped["start_ts"] < target_end)
            ].copy()
            ticks = mapped["event_tick"].astype(int).tolist()
            events = calculate_basis_resolution_events(
                event_time,
                ticks,
                resolution_pcts=[90],
                max_duration_seconds=120.0,
                min_hold_seconds=0.25,
                basis_tolerance_bps=0.5,
                min_basis_shock_bps=0.0,
                require_full_horizon=True,
            ).to_pandas()
            if not events.empty:
                # Binance trade timestamps are millisecond-valued.  A Polars ->
                # pandas conversion can otherwise preserve tiny float-derived
                # nanosecond tails (for example .967000064), which breaks the
                # exact event_tick contract despite referring to the same trade.
                for timestamp_column in ("start_ts", "prev_ts", "end_ts"):
                    if timestamp_column in events:
                        events[timestamp_column] = pd.to_datetime(
                            events[timestamp_column]
                        ).dt.round("ms")
                events["first"] = first
                events["market"] = market
                events["grid"] = freq
                events = events.merge(
                    mapped[["event_tick", "timestamp_bin", "return", "sigma", "L", "T"]],
                    on="event_tick",
                    how="left",
                    validate="one_to_one",
                )
            event_path = output_root / "events" / market / f"{first}_{target_start:%Y-%m-%d}_{freq}.parquet"
            event_path.parent.mkdir(parents=True, exist_ok=True)
            events.to_parquet(event_path, index=False)

            aft_path = output_root / "aft_data" / market / f"{first}_{target_start:%Y-%m-%d}_{freq}.parquet"
            aft_rows = np.nan
            if build_covariates:
                aft_path.parent.mkdir(parents=True, exist_ok=True)
                if events.empty:
                    # Preserve the timing-version schema even when a valid day
                    # has no detected events, so resume checks can distinguish
                    # it from an incomplete/legacy checkpoint.
                    aft = pd.DataFrame({
                        "covariate_timing_version": pd.Series(dtype="string")
                    })
                else:
                    aft = build_aft_covariates(
                        start=target_start,
                        end=target_end,
                        data=data,
                        event_time_prices=event_time,
                        events=events,
                        primary_resolution_pct=90,
                    )
                    aft["period"] = target_start.strftime("%Y-%m")
                    aft["first"] = first
                    aft["methodology_version"] = METHODOLOGY_VERSION
                aft.to_parquet(aft_path, index=False)
                aft_rows = len(aft)
                del aft

            v5 = load_v5_events(v5_root, first, target_start)
            overlap = overlap_metrics(events, v5, tolerance_seconds=grid_seconds(freq))
            rows.append({
                "market": market,
                "date": target_start.strftime("%Y-%m-%d"),
                "first": first,
                "methodology_version": METHODOLOGY_VERSION,
                **lm_meta,
                "v8_candidates_before_refractory": len(candidates),
                "v8_events": len(events),
                "v5_events": len(v5),
                "v8_resolved_by_10s": km_resolved_by(events, 10.0),
                "v5_resolved_by_10s": km_resolved_by(v5, 10.0),
                **overlap,
                "event_file": str(event_path),
                "aft_file": str(aft_path) if build_covariates else "",
                "aft_rows": aft_rows,
            })
        del regular
        gc.collect()

    if owns_data:
        del data, event_time
    gc.collect()
    return rows


def run_date_block(
    market: str,
    dates: list[str],
    grids: list[str],
    output_root: Path,
    significance: float,
    build_covariates: bool = False,
) -> list[dict]:
    """Pull one contiguous archive block, then process each UTC day separately.

    The pull includes one calendar-day margin on both sides. LM estimation,
    daily thresholds, refractory filtering, event resolution, and checkpoint
    files remain day-specific; only the raw download/parsing is shared.
    """
    if not dates:
        return []
    ordered = sorted(pd.Timestamp(d).normalize() for d in set(dates))
    expected = pd.date_range(ordered[0], ordered[-1], freq="D").tolist()
    if ordered != expected:
        raise ValueError("run_date_block requires contiguous UTC dates")

    symbol, cm_um, _ = MARKETS[market]
    archive_end = ordered[-1] + pd.Timedelta(days=1)
    archive_days = len(ordered) + 2
    print(
        f"[load block] {market} {ordered[0]:%Y-%m-%d}..{ordered[-1]:%Y-%m-%d} "
        f"({archive_days} archive days)",
        flush=True,
    )
    data = TradeData(symbol, source="Binance", cm_um=cm_um)
    data.grab_trades_data(archive_end.to_pydatetime(), archive_days, n_jobs=4)
    event_time = build_event_time_prices(
        data.df_trades_spots, data.df_trades_perps
    ).sort("timestamp")

    rows = []
    try:
        for day in ordered:
            target_end = day + pd.Timedelta(days=1)
            target_spot_rows = data.df_trades_spots.filter(
                (pl.col("timestamp") >= pl.lit(day.to_pydatetime()))
                & (pl.col("timestamp") < pl.lit(target_end.to_pydatetime()))
            ).height
            target_perp_rows = data.df_trades_perps.filter(
                (pl.col("timestamp") >= pl.lit(day.to_pydatetime()))
                & (pl.col("timestamp") < pl.lit(target_end.to_pydatetime()))
            ).height
            if target_spot_rows == 0 or target_perp_rows == 0:
                invalid_path = output_root / "invalid_days.csv"
                invalid = (
                    pd.read_csv(invalid_path)
                    if invalid_path.exists()
                    else pd.DataFrame()
                )
                invalid_row = pd.DataFrame([{
                    "market": market,
                    "date": day.strftime("%Y-%m-%d"),
                    "grid": ",".join(grids),
                    "significance": significance,
                    "methodology_version": METHODOLOGY_VERSION,
                    "spot_trade_rows": target_spot_rows,
                    "perp_trade_rows": target_perp_rows,
                    "reason": "missing target-day raw trades",
                }])
                invalid = pd.concat([invalid, invalid_row], ignore_index=True)
                invalid = invalid.drop_duplicates(
                    ["market", "date", "grid", "significance", "methodology_version"],
                    keep="last",
                )
                atomic_csv(invalid.sort_values(["market", "date"]), invalid_path)
                print(
                    f"  [skip invalid day] {market} {day:%Y-%m-%d}: "
                    f"spot_rows={target_spot_rows}, perp_rows={target_perp_rows}",
                    flush=True,
                )
                continue
            slice_start = day - pd.Timedelta(days=1)
            slice_end = day + pd.Timedelta(days=2)
            # Keep only the same three-day window that the original one-day
            # implementation used.  This avoids recomputing rolling covariates
            # over the entire 19-day archive block for every target day.
            day_data = TradeData(symbol, source="Binance", cm_um=cm_um)
            day_data.df_trades_spots = data.df_trades_spots.filter(
                (pl.col("timestamp") >= pl.lit(slice_start.to_pydatetime()))
                & (pl.col("timestamp") < pl.lit(slice_end.to_pydatetime()))
            )
            day_data.df_trades_perps = data.df_trades_perps.filter(
                (pl.col("timestamp") >= pl.lit(slice_start.to_pydatetime()))
                & (pl.col("timestamp") < pl.lit(slice_end.to_pydatetime()))
            )
            day_event_time = event_time.filter(
                (pl.col("timestamp") >= pl.lit(slice_start.to_pydatetime()))
                & (pl.col("timestamp") < pl.lit(slice_end.to_pydatetime()))
            )
            try:
                day_rows = run_one(
                    market=market,
                    date=day.strftime("%Y-%m-%d"),
                    grids=grids,
                    output_root=output_root,
                    significance=significance,
                    build_covariates=build_covariates,
                    data=day_data,
                    event_time=day_event_time,
                )
            except AssertionError as exc:
                if "LM target day is not a complete regular" not in str(exc):
                    raise
                invalid_path = output_root / "invalid_days.csv"
                invalid = (
                    pd.read_csv(invalid_path)
                    if invalid_path.exists()
                    else pd.DataFrame()
                )
                invalid_row = pd.DataFrame([{
                    "market": market,
                    "date": day.strftime("%Y-%m-%d"),
                    "grid": ",".join(grids),
                    "significance": significance,
                    "methodology_version": METHODOLOGY_VERSION,
                    "spot_trade_rows": target_spot_rows,
                    "perp_trade_rows": target_perp_rows,
                    "reason": str(exc),
                }])
                invalid = pd.concat([invalid, invalid_row], ignore_index=True)
                invalid = invalid.drop_duplicates(
                    ["market", "date", "grid", "significance", "methodology_version"],
                    keep="last",
                )
                atomic_csv(invalid.sort_values(["market", "date"]), invalid_path)
                print(
                    f"  [skip invalid grid] {market} {day:%Y-%m-%d}: {exc}",
                    flush=True,
                )
                del day_event_time, day_data
                gc.collect()
                continue
            except BaseException as exc:
                # Kernel/process failures are recovery conditions, not evidence
                # that the underlying market-day is invalid.
                if isinstance(exc, (KeyboardInterrupt, SystemExit, MemoryError)):
                    raise
                invalid_path = output_root / "invalid_days.csv"
                invalid = (
                    pd.read_csv(invalid_path)
                    if invalid_path.exists()
                    else pd.DataFrame()
                )
                invalid_row = pd.DataFrame([{
                    "market": market,
                    "date": day.strftime("%Y-%m-%d"),
                    "grid": ",".join(grids),
                    "significance": significance,
                    "methodology_version": METHODOLOGY_VERSION,
                    "spot_trade_rows": target_spot_rows,
                    "perp_trade_rows": target_perp_rows,
                    "reason": f"{type(exc).__name__}: {exc}",
                }])
                invalid = pd.concat([invalid, invalid_row], ignore_index=True)
                invalid = invalid.drop_duplicates(
                    ["market", "date", "grid", "significance", "methodology_version"],
                    keep="last",
                )
                atomic_csv(invalid.sort_values(["market", "date"]), invalid_path)
                print(
                    f"  [skip day error] {market} {day:%Y-%m-%d}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                del day_event_time, day_data
                gc.collect()
                continue
            rows.extend(day_rows)
            del day_event_time, day_data
            gc.collect()
    finally:
        del event_time, data
        gc.collect()
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", nargs="+", choices=sorted(MARKETS), default=sorted(MARKETS))
    parser.add_argument("--dates", nargs="+", default=DEFAULT_DATES)
    parser.add_argument("--grids", nargs="+", default=DEFAULT_GRIDS)
    parser.add_argument("--output", type=Path, default=Path("sa_results/km_v8_regular_lm_pilot"))
    parser.add_argument("--significance", type=float, default=0.001)
    parser.add_argument(
        "--build-covariates", action="store_true",
        help="Build base AFT covariates from the same raw pull and event-time stream.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "comparison_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
    else:
        summary = pd.DataFrame()

    completed = set()
    if not summary.empty and "methodology_version" in summary:
        current = summary[summary["methodology_version"].eq(METHODOLOGY_VERSION)]
        if args.build_covariates:
            if "aft_file" not in current:
                current = current.iloc[0:0]
            else:
                current = current[
                    current["aft_file"].fillna("").map(lambda p: bool(p) and Path(p).exists())
                ]
        complete_groups = current.groupby(["market", "date", "grid"])["first"].nunique()
        completed = set(complete_groups[complete_groups.eq(2)].index)

    for market in args.markets:
        for date in args.dates:
            missing_grids = [freq for freq in args.grids if (market, date, freq) not in completed]
            if not missing_grids:
                print(f"[skip] {market} {date}", flush=True)
                continue
            try:
                new_rows = run_one(
                    market, date, missing_grids, args.output, args.significance,
                    build_covariates=args.build_covariates,
                )
                summary = pd.concat([summary, pd.DataFrame(new_rows)], ignore_index=True)
                summary = summary.drop_duplicates(["market", "date", "first", "grid"], keep="last")
                atomic_csv(summary.sort_values(["market", "date", "grid", "first"]), summary_path)
            except Exception as exc:
                print(f"[failed] {market} {date}: {exc!r}", flush=True)
                error_path = args.output / "errors.jsonl"
                with error_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"market": market, "date": date, "error": repr(exc)}) + "\n")
            gc.collect()

    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        plot_pooled_km_comparison(summary, args.output)
        print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
