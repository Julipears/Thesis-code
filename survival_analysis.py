"""
This file contains the SurvivalAnalysis class, which pulls data for trades, shocks, and covariates.
This data can then be used to fit non-parametric (Kaplan-Meier) or parametric (AFT, Cox) survival models to estimate 
convergence times after shocks.

Shock identification can be performed through various methods, including volatility thresholds, quantile thresholds, or the Lee-Mykland test.

GenAI was used to implement the Lee-Mykland method in a fully vectorized manner for improved performance, 
and to optimize the overall shock detection and convergence time calculation process.
"""

import datetime
import requests
import zipfile
import io
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from joblib import Parallel, delayed
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import aiohttp
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin, quote
import time

from pytrends.request import TrendReq
from matplotlib import cm
from lifelines import KaplanMeierFitter, WeibullAFTFitter, CoxPHFitter

from trade_data_pull import *


def wiki_pageviews(page_title, start, end):
    page = quote(page_title, safe="")
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{page}/daily/{start}/{end}"

    headers = {
        "User-Agent": "JuliaHuang-Thesis/1.0 (contact: ydj.huang@mail.utoronto.ca)"
    }

    resp = requests.get(url, headers=headers, timeout=30)
    print("status:", resp.status_code, "content-type:", resp.headers.get("content-type"))

    resp.raise_for_status()
    r = resp.json()

    df = pd.DataFrame({
        "date": [pd.to_datetime(it["timestamp"][:8]) for it in r["items"]],
        "views": [it["views"] for it in r["items"]],
    })
    return df


def apply_returns(df_trades):
    # log returns
    df_ret = df_trades.with_columns([
        (pl.col("p_spot").log() - pl.col("p_spot").log().shift(1)).alias("spot_ret"),
        (pl.col("p_perp").log() - pl.col("p_perp").log().shift(1)).alias("perp_ret"),
    ]).drop_nulls(subset=["spot_ret","perp_ret"])

    return df_ret

def all_formatting(df_trades_spots, df_trades_perps):
    # give every trade a sequence number so we can tell real
    # successive trades from forward‑filled rows later
    df_spots = df_trades_spots.with_row_count('id_spot')
    df_perps = df_trades_perps.with_row_count('id_perp')

    df_trades = pl.sql("""
       select d1.price    as p_spot,
              d1.id_spot,
              d1.is_bid  as spot_bid,
              d1.timestamp as timestamp,
              d2.price    as p_perp,
              d2.id_perp,
              d2.is_bid  as perp_bid,
              d2.timestamp as timestamp2
       from df_spots as d1 full outer join df_perps as d2 
       using (timestamp)""").collect()
    
    df_trades = df_trades.with_columns(pl.col('timestamp').fill_null(pl.col('timestamp2'))).sort('timestamp')

    # forward‑fill prices **and ids** so we know which rows were imputed
    df_trades_ff = (df_trades
                    .with_columns(pl.col('p_spot').forward_fill())
                    .with_columns(pl.col('p_perp').forward_fill()))
    df_trades_int = (df_trades
                     .with_columns(pl.col('p_spot').interpolate_by(pl.col('timestamp')))   # id still ff
                     .with_columns(pl.col('p_perp').interpolate_by(pl.col('timestamp'))))

    df_ret_ff = apply_returns(df_trades_ff)
    df_ret_int = apply_returns(df_trades_int)

    return df_ret_ff, df_ret_int

def find_shocks2(
        df_ret,
        method="lee_mykland",
        volatility_threshold=3.0,
        quantile_threshold=0.99,
        lm_k=None,
        lm_sampling_minutes=5,
        lm_signif=0.01,
        groupday=True,
        first='spot',
        follow_ticks: int = 5,
        follow_frac: float = 0.5):
    """
    Identify candidate shock ticks.

    Parameters:
    -----------
    method : str
        "quantile", "volatility", or "lee_mykland"
    lm_k : int, optional
        Window size for Lee-Mykland (if method="lee_mykland")
    lm_sampling_minutes : float, optional
        Sampling frequency for Lee-Mykland k calculation
    lm_signif : float
        Significance level for Lee-Mykland test (default 0.01)
    follow_ticks : int
        number of future ticks to sum (cumulative return) for the follow-up check.
    follow_frac : float
        fraction of the shock size which the cumulative opposite-signed return
        must exceed in order to reject the candidate.
    """
    print(method, volatility_threshold, quantile_threshold, lm_k, lm_signif, groupday, first)
    df_ret2 = df_ret.sort("timestamp").with_row_index("tick")

    col = f"{first}_ret"
    id = f"id_{first}"

    # frame used to compute thresholds: drop only zeros produced by forward-fill
    thresh = df_ret2.filter(~(pl.col(id).is_null()))

    # helper: build forward-looking cumulative-return expression if requested
    def next_cum_expr(n):
        if n <= 0:
            return None
        shifts = [pl.col(col).shift(-k) for k in range(1, n + 1)]
        return sum(shifts).alias("next_cum")

    if method == "lee_mykland":
        # Extract prices and returns
        returns = thresh.select(col).to_numpy().flatten()
        
        # Calculate k if not provided
        if lm_k is None:
            if lm_sampling_minutes is None:
                # Use median spacing as default
                median_spacing_sec = thresh.select(
                    pl.col("timestamp").diff().dt.total_milliseconds().median()
                ).item() / 1000
                lm_sampling_minutes = median_spacing_sec / 60
            lm_k = int(np.ceil(np.sqrt(252 * 24 * 60 / lm_sampling_minutes)))
        
        # Bipower variation (vectorized)
        returns_shifted = np.concatenate([[np.nan], returns[:-1]])
        bpv = np.abs(returns) * np.abs(returns_shifted)
        bpv = np.concatenate([[np.nan], bpv[:-1]])
        
        # FAST rolling volatility using pandas
        sig = pd.Series(bpv).rolling(window=lm_k, min_periods=1).mean().values
        sig = np.sqrt(sig)
        sig[:lm_k-1] = np.nan
        
        # L statistic (vectorized)
        L = returns / sig
        
        # Test parameters (all vectorized)
        n = len(returns)
        c = np.sqrt(2 / np.pi)
        Sn = c * np.sqrt(2 * np.log(n))
        Cn = np.sqrt(2 * np.log(n)) / c - np.log(np.pi * np.log(n)) / (2 * c * np.sqrt(2 * np.log(n)))
        
        # Test statistic
        beta_star = -np.log(-np.log(1 - lm_signif))
        T = (np.abs(L) - Cn) * Sn
        
        # Jump indicator
        J = (T > beta_star).astype(float) * np.sign(returns)
        J[:lm_k] = np.nan
        
        # Add to dataframe (single operation)
        df_day = thresh.with_columns([
            pl.Series("J", J),
            pl.Series("T", T),
            pl.Series("sig", sig)
        ])
        
        if follow_ticks > 0:
            nc = next_cum_expr(follow_ticks)
            df_day = df_day.with_columns(nc)
        
        # Filter for jumps
        cands = df_day.filter(
            (pl.col("J") != 0) & (~pl.col("J").is_null())
        )
        
        # Apply follow-up filter
        if follow_ticks > 0:
            cands = cands.filter(
                (pl.col("next_cum").is_null()) |
                ((pl.col("next_cum").sign() == pl.col(col).sign()) |
                (pl.col("next_cum").abs() < follow_frac * pl.col(col).abs()))
            )

    elif method == "volatility":
        df_day = thresh.with_columns(
            pl.col(col).abs().std().over("1h").alias("1h_std")
        )

        if follow_ticks > 0:
            nc = next_cum_expr(follow_ticks)
            df_day = df_day.with_columns(nc)

        cands = (
            df_day
            .filter(pl.col(col).abs() >= volatility_threshold * pl.col("1h_std"))
        )

        if follow_ticks > 0:
            cands = cands.filter(
                (pl.col("next_cum").is_null()) |
                ((pl.col("next_cum").sign() == pl.col(col).sign()) |
                 (pl.col("next_cum").abs() < follow_frac * pl.col(col).abs()))
            )
        
    else:  # quantile method
        if groupday:
            df_day = thresh.with_columns(pl.col("timestamp").dt.date().alias("day"))
            if follow_ticks > 0:
                nc = next_cum_expr(follow_ticks)
                df_day = df_day.with_columns(nc)

            cands = (
                df_day
                .with_columns(
                    pl.col(col).abs().quantile(quantile_threshold).over("day").alias("daily_threshold")
                )
                .filter(pl.col(col).abs() >= pl.col("daily_threshold"))
            )

            if follow_ticks > 0:
                cands = cands.filter(
                    (pl.col("next_cum").is_null()) |
                    ((pl.col("next_cum").sign() == pl.col(col).sign()) |
                     (pl.col("next_cum").abs() < follow_frac * pl.col(col).abs()))
                )
        else:
            thr = thresh.select(pl.col(col).abs().quantile(quantile_threshold)).item()

            df_thr = thresh
            if follow_ticks > 0:
                nc = next_cum_expr(follow_ticks)
                df_thr = df_thr.with_columns(nc)

            cands = df_thr.filter(pl.col(col).abs() >= thr)

            if follow_ticks > 0:
                cands = cands.filter(
                    (pl.col("next_cum").is_null()) |
                    ((pl.col("next_cum").sign() == pl.col(col).sign()) |
                     (pl.col("next_cum").abs() < follow_frac * pl.col(col).abs()))
                )

    cands = cands.select(["tick", col]).to_numpy()

    median_dt = (
        df_ret2.select(
            (pl.col("timestamp").diff().dt.total_milliseconds() / 1e3).median().alias("median_dt")
        ).item()
    )

    return None, cands, median_dt

def find_convergence_rigorous(df_ret, shock_events, first='spot', 
                              tolerance=0.001, min_duration=5, max_window=500):
    """
    Find convergence time using tolerance band around pre-shock basis.
    
    Parameters:
    -----------
    df_ret : polars DataFrame
        Returns data with spot_ret and perp_ret
    shock_events : array
        Array of shock tick indices
    first : str
        Which series had the shock ('spot' or 'perp')
    tolerance : float
        Fractional tolerance (e.g., 0.001 = 0.1%)
    min_duration : int
        Number of ticks basis must stay within tolerance
    max_window : int
        Maximum ticks to search for convergence
        
    Returns:
    --------
    polars DataFrame with shock info and convergence times
    """
    
    # Calculate basis at each tick
    spot_price_cum = np.cumsum(df_ret['spot_ret'].to_numpy())
    perp_price_cum = np.cumsum(df_ret['perp_ret'].to_numpy())
    basis = perp_price_cum - spot_price_cum  # log price difference
    
    times = df_ret['timestamp'].to_numpy()
    
    results = []
    
    for shock_tick in shock_events:
        # Get pre-shock basis (tick before shock)
        if shock_tick < 1 or shock_tick >= len(basis) - max_window:
            continue
            
        pre_shock_basis = basis[shock_tick - 1]
        shock_size = abs(basis[shock_tick] - pre_shock_basis)
        
        if shock_size < 1e-12:  # Skip negligible shocks
            continue
        
        # Define tolerance band
        tol_abs = abs(pre_shock_basis) * tolerance
        upper_bound = pre_shock_basis + tol_abs
        lower_bound = pre_shock_basis - tol_abs
        
        # Search for convergence
        converged = False
        converge_tick = None
        
        window_end = min(shock_tick + max_window, len(basis))
        
        for t in range(shock_tick + 1, window_end - min_duration + 1):
            # Check if basis stays within tolerance for min_duration ticks
            window_basis = basis[t:t+min_duration]
            
            if np.all((window_basis >= lower_bound) & (window_basis <= upper_bound)):
                converged = True
                converge_tick = t
                break
        
        # Calculate duration
        if converged:
            duration_sec = (times[converge_tick] - times[shock_tick]) / np.timedelta64(1, 's')
            status = 1
        else:
            duration_sec = (times[window_end-1] - times[shock_tick]) / np.timedelta64(1, 's')
            status = 0  # censored
        
        results.append({
            'event_tick': shock_tick,
            'start_ts': times[shock_tick],
            'prev_ts': times[shock_tick - 1],
            'Status': status,
            'Length': duration_sec,
            'shock_size': shock_size,
            'pre_shock_basis': pre_shock_basis,
            'converge_tick': converge_tick if converged else None
        })
    
    return pl.DataFrame(results)

def single_iteration_v3_polars_optimized(events, ret1, ret2, times, delay_tick, delay_time, 
                                        window, all_vals, i, median_dt=None, 
                                        ret1_cum_global=None, ret2_cum_global=None):
    """Optimized version using pure NumPy operations, compatible with Polars"""
    
    p_close = i * 0.01  # Avoid repeated division
    
    # Use precomputed cumulative sums if available
    if ret1_cum_global is None:
        ret1_cum_global = np.cumsum(ret1)
    if ret2_cum_global is None:
        ret2_cum_global = np.cumsum(ret2)
    
    if not events:
        return [], delay_tick, delay_time
    
    # Convert to numpy array once for faster indexing
    events_array = np.array(events, dtype=np.int32)
    
    # Vectorized filtering of valid events
    valid_events_mask = (events_array + window) < len(ret1)
    if not np.any(valid_events_mask):
        return [], delay_tick, delay_time
    
    valid_events = events_array[valid_events_mask]
    n_valid = len(valid_events)
    
    # Pre-allocate result arrays
    delays = np.full(n_valid, window, dtype=np.int32)
    resolved = np.zeros(n_valid, dtype=bool)
    actually_valid = np.zeros(n_valid, dtype=bool)
    shock_sizes = np.zeros(n_valid, dtype=np.float64)  # <-- ADD THIS
    
    # Vectorized processing using advanced indexing
    for idx in range(n_valid):
        e = valid_events[idx]
        
        # Fast cumulative using prefix sums
        ret1_cum = ret1_cum_global[e+1:e+1+window] - ret1_cum_global[e]
        ret2_cum = ret2_cum_global[e+1:e+1+window] - ret2_cum_global[e]
        
        init_gap = ret1_cum[0] - ret2_cum[0]
        if abs(init_gap) < 1e-12:
            continue
            
        actually_valid[idx] = True
        shock_sizes[idx] = abs(init_gap)  # <-- ADD THIS: Store shock magnitude
        sign_shock = np.sign(ret1_cum[0])
        
        # Vectorized condition with early termination
        diff = np.abs(ret1_cum - ret2_cum)
        cond = (np.sign(ret2_cum) == sign_shock) & (diff <= (1 - p_close) * abs(init_gap))
        
        if np.any(cond):
            delays[idx] = np.argmax(cond)
            resolved[idx] = True
    
    # Filter to actually valid events
    final_mask = actually_valid
    if not np.any(final_mask):
        return [], delay_tick, delay_time
    
    final_events = valid_events[final_mask]
    final_delays = delays[final_mask]
    final_resolved = resolved[final_mask]
    final_shock_sizes = shock_sizes[final_mask]  # <-- ADD THIS
    
    # Vectorized time calculation using advanced indexing
    event_indices = final_events
    end_indices = np.clip(event_indices + final_delays, 0, len(times) - 1)

    # calculate previous‑tick timestamp (tick numbers are row indices)
    prev_indices = np.clip(event_indices - 1, 0, len(times) - 1)
    prev_ts = times[prev_indices]

    # Calculate delays_time vectorized
    delays_time = (times[end_indices] - times[event_indices]) / np.timedelta64(1, "s")

    # … existing storage logic …
    key = f"{i}%"

    if not all_vals:
        # Use numpy percentile for better performance
        if median_dt is not None:
            delays_sec = final_delays * median_dt
            delay_tick[key] = np.array([
                np.median(delays_sec), 
                np.percentile(delays_sec, 75), 
                np.percentile(delays_sec, 95)
            ])
        delay_time[key] = np.array([
            np.median(delays_time), 
            np.percentile(delays_time, 75), 
            np.percentile(delays_time, 95)
        ])
    else:
        n_results = len(final_delays)
        start_ts = times[final_events]

        if median_dt is not None:
            delay_tick[key] = pl.DataFrame({
                "Shock": [key] * n_results,
                "event_tick": final_events,
                "start_ts": start_ts,
                "prev_ts": prev_ts,                # <– new column
                "Status": final_resolved,
                "Length": final_delays * median_dt,
                "shock_size": final_shock_sizes,
            })

        delay_time[key] = pl.DataFrame({
            "Shock": [key] * n_results,
            "event_tick": final_events,
            "start_ts": start_ts,
            "prev_ts": prev_ts,                # <– new column
            "Status": final_resolved,
            "Length": delays_time,
            "shock_size": final_shock_sizes,
        })

    return final_events.tolist(), delay_tick, delay_time

def full_iteration_rigorous(df_ret, method, quantile_threshold, max_window=500, 
                           refractory=100, min_duration=5,
                           groupday=True, first='spot', all_vals=True):
    """
    Optimized version with reduced percentages and vectorization.
    """
    
    # Find shocks
    if method == 'lee_mykland':
        thr, cands, median_dt = find_shocks2(df_ret, method=method, 
                                            lm_signif=1-quantile_threshold, 
                                            groupday=groupday, first=first,
                                            follow_ticks=0)
    else:
        thr, cands, median_dt = find_shocks2(df_ret, method=method, 
                                            quantile_threshold=quantile_threshold, 
                                            groupday=groupday, first=first,
                                            follow_ticks=0)
    
    # Apply refractory period
    events = []
    if len(cands) > 0:
        ticks = np.array([tick[0] for tick in cands], dtype=np.int32)
        events = greedy_refractory_numpy(ticks, refractory)
    
    if not events:
        return [], {}, {}
    
    # Calculate basis once
    spot_price_cum = np.cumsum(df_ret['spot_ret'].to_numpy())
    perp_price_cum = np.cumsum(df_ret['perp_ret'].to_numpy())
    basis = perp_price_cum - spot_price_cum
    times = df_ret['timestamp'].to_numpy()
    
    # Reduced percentages for speed
    pcts_to_compute = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
    reversion_pcts = np.array(pcts_to_compute) / 100.0  # Vectorize division
    
    # Pre-allocate storage
    all_results = {f'{pct}%': [] for pct in pcts_to_compute}
    
    # Process each shock
    for shock_tick in events:
        if shock_tick < 1 or shock_tick >= len(basis) - max_window:
            continue
            
        pre_shock_basis = basis[shock_tick - 1]
        shock_basis = basis[shock_tick]
        shock_deviation = shock_basis - pre_shock_basis
        shock_size = abs(shock_deviation)
        
        if shock_size < 1e-12:
            continue
        
        # Get future basis values
        window_end = min(shock_tick + max_window, len(basis))
        future_basis = basis[shock_tick + 1:window_end]
        future_times = times[shock_tick + 1:window_end]
        
        if len(future_basis) < min_duration:
            continue
        
        tolerance = shock_size * 0.05
        
        # Vectorized: compute all target basis levels at once
        target_bases = pre_shock_basis + shock_deviation * (1 - reversion_pcts)  # Shape: (n_pcts,)
        
        # Vectorized: check all percentages at once for each time point
        # future_basis shape: (n_times,)
        # target_bases shape: (n_pcts,)
        # Result shape: (n_pcts, n_times)
        future_basis_2d = future_basis[np.newaxis, :]  # Shape: (1, n_times)
        target_bases_2d = target_bases[:, np.newaxis]  # Shape: (n_pcts, 1)
        
        if shock_deviation > 0:
            within_target = (future_basis_2d <= target_bases_2d + tolerance) & \
                          (future_basis_2d >= pre_shock_basis - tolerance)
        else:
            within_target = (future_basis_2d >= target_bases_2d - tolerance) & \
                          (future_basis_2d <= pre_shock_basis + tolerance)
        
        # For each percentage, find first sustained period
        for pct_idx, pct in enumerate(pcts_to_compute):
            converged = False
            converge_idx = None
            
            # Check for sustained min_duration period
            within_for_pct = within_target[pct_idx, :]  # Boolean array for this pct
            
            # Vectorized rolling window check (faster than loop for large arrays)
            if len(within_for_pct) >= min_duration:
                # Create rolling window view
                for i in range(len(within_for_pct) - min_duration + 1):
                    if np.all(within_for_pct[i:i+min_duration]):
                        converged = True
                        converge_idx = i
                        break
            
            if converged:
                duration_sec = (future_times[converge_idx] - times[shock_tick]) / np.timedelta64(1, 's')
                status = 1
            else:
                duration_sec = (future_times[-1] - times[shock_tick]) / np.timedelta64(1, 's')
                status = 0
            
            all_results[f'{pct}%'].append({
                'event_tick': shock_tick,
                'start_ts': times[shock_tick],
                'prev_ts': times[shock_tick - 1],
                'Status': status,
                'Length': duration_sec,
                'shock_size': shock_size,
            })
    
    # Convert to output format
    delay_tick = {}
    delay_time = {}
    
    for key, results in all_results.items():
        if len(results) == 0:
            continue
            
        if all_vals:
            delay_time[key] = pl.DataFrame(results).with_columns(
                pl.lit(key).alias("Shock")
            )
        else:
            resolved_lengths = [r['Length'] for r in results if r['Status'] == 1]
            if len(resolved_lengths) > 0:
                delay_time[key] = np.array([
                    np.median(resolved_lengths),
                    np.percentile(resolved_lengths, 75),
                    np.percentile(resolved_lengths, 95)
                ])
    
    return events, delay_tick, delay_time

def full_iteration_polars_optimized(df_ret, method, quantile_threshold, window, refractory, use_ticks=False, all_vals=False, groupday=True, first='spot'):
    """Optimized main function with precomputation and efficient refractory filtering"""
    
    if method == 'lee_mykland':
        thr, cands, median_dt = find_shocks2(df_ret, method=method, lm_signif=1-quantile_threshold, groupday=groupday, first=first)
    else:
        thr, cands, median_dt = find_shocks2(df_ret, method=method, quantile_threshold=quantile_threshold, groupday=groupday, first=first)

    # Optimized refractory period enforcement using numpy
    events = []
    if len(cands) > 0:
        ticks = np.array([tick[0] for tick in cands], dtype=np.int32)
        # Vectorized refractory filtering
        events = greedy_refractory_numpy(ticks, refractory)

    if not events:
        return [], {}, {}

    # Extract numpy arrays once (this is already optimized in your original code)
    if first=='spot':
        ret1 = df_ret["spot_ret"].to_numpy()
        ret2 = df_ret["perp_ret"].to_numpy()

    elif first=='perp':
        ret1 = df_ret["perp_ret"].to_numpy()
        ret2 = df_ret["spot_ret"].to_numpy()
        
    ret1_cum_global = np.cumsum(ret1)
    ret2_cum_global = np.cumsum(ret2)
    times = df_ret['timestamp'].to_numpy()

    delay_tick = {}
    delay_time = {}

    if not use_ticks:
        median_dt = None

    # Process iterations with precomputed cumulative sums
    for i in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]:
        valid_events, delay_tick, delay_time = single_iteration_v3_polars_optimized(
            events, ret1, ret2, times, delay_tick, delay_time, 
            window, all_vals, i, median_dt, ret1_cum_global, ret2_cum_global
        )

    return valid_events, delay_tick, delay_time

def greedy_refractory_numpy(ticks, refractory):
    """Optimized refractory filtering using numpy operations"""
    if len(ticks) == 0:
        return []
    
    # Sort ticks if they're not already sorted
    ticks = np.sort(ticks)
    
    # Vectorized approach for small to medium arrays
    if len(ticks) < 10000:  # For larger arrays, use the loop approach
        selected = [ticks[0]]  # Always include first tick
        last_selected = ticks[0]
        
        for tick in ticks[1:]:
            if tick - last_selected >= refractory:
                selected.append(tick)
                last_selected = tick
        return selected
    
    # For very large arrays, use a more memory-efficient approach
    else:
        selected_mask = np.zeros(len(ticks), dtype=bool)
        selected_mask[0] = True
        last_idx = 0
        
        for i in range(1, len(ticks)):
            if ticks[i] - ticks[last_idx] >= refractory:
                selected_mask[i] = True
                last_idx = i
                
        return ticks[selected_mask].tolist()
    
def plot_dataframe_list(df_list, titles=None, ncols=2, figsize=(12, 6), sharex=True, sharey=False, lines=None, first=None):
    """
    Plot multiple pandas DataFrames in a grid of subplots.

    Parameters
    ----------
    df_list : list of pd.DataFrame
        Each DataFrame will be plotted in its own subplot.
    titles : list of str, optional
        Titles for each subplot (defaults to no title).
    ncols : int
        Number of subplot columns.
    figsize : tuple
        Figure size for the whole grid.
    sharex, sharey : bool
        Whether to share x/y axes across subplots.

    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    n = len(df_list)
    nrows = int(np.ceil(n / ncols))

    blues = cm.get_cmap("Blues", 10)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey)
    axes = np.array(axes).reshape(-1)  # flatten in case it's 2D

    kmfs = {}

    for idx, (ax, df) in enumerate(zip(axes, df_list)):
        if lines is None:
            for i in range(1, 100, 10):
                kmf = KaplanMeierFitter(label=f"{i}% absorbed")  # moved inside
                kmf.fit(
                    df[f'{i}%']['Length'].to_numpy(),
                    event_observed=df[f'{i}%']['Status'].to_numpy()
                )
                kmf.plot_survival_function(ax=ax, label=f'{i}% absorbed', color=blues(i//10))
                kmfs[(float(titles[idx].split('-')[0]), i)] = kmf
        else:
            for x, i in enumerate(lines):
                kmf = KaplanMeierFitter(label=f"{i}% absorbed")  # moved inside
                kmf.fit(
                    df[f'{i}%']['Length'].to_numpy(),
                    event_observed=df[f'{i}%']['Status'].to_numpy()
                )
                kmf.plot_survival_function(ax=ax, label=f'{i}% absorbed', color=blues(2*x+3))
                kmfs[(float(titles[idx].split('-')[0]), i)] = kmf

        if titles is not None and idx < len(titles):
            ax.set_title(titles[idx])
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Lifetime (s)')
        ax.set_ylabel('Fraction of survival')
        ax.legend()

    # Hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    if first is None:
        plot_title = 'Survival Lifetime'
    elif first == 'spot':
        plot_title = 'Spot to Perp Survival Lifetime'
    elif first == 'perp':
        plot_title = 'Perp to Spot Survival Lifetime'

    plt.tight_layout()
    plt.suptitle(plot_title, y=1.02, fontsize=14)
    return fig, axes, kmfs

class SurvivalAnalysis:
    def __init__(self, ticker, source='Binance', cm_um='um'):
        self.ticker = ticker
        self.source = source
        self.cm_um = cm_um

    def _get_and_parse_data(self, start, end, calc_ff=False):
        def combine_funding_rates():
            fr_binance = (
                pl.from_pandas(ff)
                .with_columns(
                    pl.col("fundingTime")
                    .cast(pl.Datetime("ms"))
                    .alias("timestamp"),
                    pl.col("fundingTime")
                    .cast(pl.Datetime("ms"))
                )
            )

            df_joined = (
                pl.concat([df_ret_int, fr_binance], how="diagonal")
                .with_columns(
                    pl.col("timestamp").cast(pl.Datetime("ms")),
                    pl.col("fundingTime").cast(pl.Datetime("ms")),
                )
                .sort("timestamp")
                .with_columns(
                    pl.col("fundingTime").fill_null(strategy="backward"),
                )
                .with_columns(
                    (pl.col("fundingTime") - pl.col("timestamp"))
                    .dt.total_milliseconds()
                    .alias("time_to_fr")
                )
                .with_columns(
                    pl.col('fundingRate').fill_nan(0).fill_null(0)
                )
            )

            return df_joined

        data = TradeData(self.ticker, source=self.source, cm_um=self.cm_um)
        start_t = datetime.datetime.strptime(start, '%Y%m%d')
        end_t = datetime.datetime.strptime(end, '%Y%m%d')
        data.grab_trades_data(end_t, (end_t-start_t).days)
        df_trades_spots, df_trades_perps = data.df_trades_spots, data.df_trades_perps
        df_ret_ff, df_ret_int = all_formatting(df_trades_spots, df_trades_perps)
        
        if calc_ff:
            ff = data.get_funding_data(start, end)
            df_ret_joined = combine_funding_rates()
        else:
            df_ret_joined = df_ret_int

        return data, df_ret_joined

    def fit_KM(self, start, end, tols, method='lee_mykland', max_trades=500, blockout_period=100, all_vals=True, first='spot'):
        # tol is in decimal
        data, df_ret_int = self._get_and_parse_data(start, end, calc_ff=False)

        events = {}
        shocks = {}
        for tol in tols:
            valid_events, _, shock = full_iteration_polars_optimized(df_ret_int, method, 1-tol, max_trades, blockout_period, all_vals=all_vals, first=first)
            events[tol] = valid_events
            # times = df_ret_int.loc[valid_events]
            # shock['time'] = times
            shocks[tol] = shock
        return data, events, shocks

    
    def graph_results(self, shocks, ptiles, first, ncols=3, figsize=(20,5), lines=[50,90,95]):
        a,b,kmfs = plot_dataframe_list([shocks[i] for i in ptiles], [f'{i*100}-%tile shocks' for i in ptiles], ncols=ncols, figsize=figsize, lines=lines, first=first)
        return kmfs
    
    def find_covariates(self, start, end, data, shocks_sp):
        ################
        # finding spreads and basis
        ################

        spots_bidask = data.to_intervals_bidask(data.df_trades_spots, '1s').to_pandas().set_index('timestamp_bin')
        spots_bidask['spread_spot'] = spots_bidask['ask_price'] - spots_bidask['bid_price']

        perps_bidask = data.to_intervals_bidask(data.df_trades_perps, '1s').to_pandas().set_index('timestamp_bin')
        perps_bidask['spread_perp'] = perps_bidask['ask_price'] - perps_bidask['bid_price']

        bidask = spots_bidask.merge(perps_bidask, how='inner', left_index=True, right_index=True, suffixes=['_spot', '_perp']).dropna()
        bidask['midpoint_spot'] = (bidask['ask_price_spot'] + bidask['bid_price_spot'])/2
        bidask['midpoint_perp'] = (bidask['ask_price_perp'] + bidask['bid_price_perp'])/2
        bidask['basis'] = bidask['midpoint_perp'] - bidask['midpoint_spot']
        bidask['spread_spot'] = bidask['spread_spot']/bidask['midpoint_spot']*10000
        bidask['spread_perp'] = bidask['spread_perp']/bidask['midpoint_perp']*10000
        bidask['basis'] = bidask['basis']/bidask['midpoint_spot']*10000
        bidask = bidask.reset_index()

        # aggregated by day
        bidask["date"] = pd.to_datetime(bidask['timestamp_bin']).dt.date

        bidask =  bidask.merge(
            bidask
            .groupby("date")[["spread_spot", 'spread_perp', 'basis']]
            .mean().reset_index().rename(columns={'spread_spot':'spread_spot_avg','spread_perp':'spread_perp_avg', 'basis':'basis_avg'}), how='outer', on='date'
        )

        spread_basis_daily = bidask[['date', 'spread_spot_avg', 'spread_perp_avg', 'basis_avg']].drop_duplicates().set_index('date')

        ##########
        # trade volume
        ##########
        spot_volume = data.get_klines(start, end, 'spot', '1d', columns=["Open time", "Close time", 'Open', "Close", "log_return", 'Volume'])
        perp_volume = data.get_klines(start, end, 'perp', '1d', columns=["Open time", "Close time", 'Open', "Close", "log_return", 'Volume'])
        spot_volume_daily = spot_volume.rename(columns={'Open time':'date','Volume':'volume'})[['date','volume']].set_index('date')

        ##########
        # funding fees
        ##########
        funding = data.get_funding_data(start, end)
        funding["date"] = pd.to_datetime(funding["fundingTime"]).dt.date

        daily_avg = (
            funding
            .groupby("date")["fundingRate"]
            .mean()
        )

        ##########
        # Google trends
        ##########
        pytrends = TrendReq(hl='en-US', tz=0)

        pytrends.build_payload(
            kw_list=["Bitcoin"],
            timeframe=f"{datetime.datetime.strfptime(start, '%Y%m%d').strftime('%Y-%m-%d')} {datetime.datetime.strfptime(end, '%Y%m%d').strftime('%Y-%m-%d')}",
            geo=""
        )

        data = pytrends.interest_over_time()
        google_sentiment = data[["Bitcoin"]]

        ##########
        # Wikipedia views
        ##########
        wikipedia_views = wiki_pageviews("Bitcoin", start, end).set_index('date')

        ##########
        # Fear and Greed
        ##########
        url = "https://api.alternative.me/fng/?limit=0"  # 0 = all available history
        resp = requests.get(url)
        resp_data = resp.json()["data"]
        market_status = pd.DataFrame(resp_data)

        # convert types
        market_status["date"] = pd.to_datetime(market_status["timestamp"].astype(int), unit="s")
        market_status["fear_greed"] = market_status["value"].astype(int)

        # keep just what you need
        market_status = market_status[["date", "fear_greed", "value_classification"]]

        market_status.sort_values("date", inplace=True)
        market_status.reset_index(drop=True, inplace=True)
        market_status = market_status.set_index('date')

        market_status.head()

        ##########
        # combining into one dataframe
        ##########
        covariates = pd.concat([spread_basis_daily, daily_avg, market_status], axis=1).dropna().drop('value_classification', axis=1)
        
        shocks_formatted = shocks_sp[0.001]['90%'].to_pandas().drop(['Shock', 'event_tick'], axis=1)
        shocks_formatted['date'] = pd.to_datetime(shocks_formatted["start_ts"]).dt.date
        shocks_formatted = shocks_formatted.merge(covariates.shift(1).reset_index().dropna(), how='inner', on='date')

        shocks_formatted[['fundingRate_bps', 'shock_size']] = shocks_formatted[['fundingRate', 'shock_size']] *1e4
        shocks_formatted["shock_logt"] = shocks_formatted["shock_size"] * np.log(shocks_formatted["Length"] + 1)

        df = shocks_formatted.drop(['start_ts', 'fundingRate', 'spread_perp_avg', 'fear_greed'], axis=1)
        df['Length'] = df['Length'].replace(0,0.001)

        return df

    def fit_cox_ph(self, start, end, data, shocks_sp, df=None):
        if df is None:
            df = self.find_covariates(start, end, data, shocks_sp)
        cph = CoxPHFitter()
        cph.fit(df, duration_col='Length', event_col='Status', cluster_col="date")

        cph.print_summary()

        return cph, df

    def fit_parametric(self, start, end, data, shocks_sp, df=None):
        if df is None:
            df = self.find_covariates(start, end, data, shocks_sp)
        aft = WeibullAFTFitter()
        aft.fit(df.drop('date', axis=1), duration_col='Length', event_col='Status')

        aft.print_summary(3)  

        return aft, df
    
def find_covariates(start, end, data, shocks_sp):
    """
    Covariates at 5-minute scale (consistent with Lee-Mykland window).
    Calculates returns internally from raw trade data.
    """
    
    ################
    # Calculate returns from raw trades
    ################
    
    # Get the raw trade data
    df_spots = data.df_trades_spots
    df_perps = data.df_trades_perps
    
    # Merge spot and perp on timestamp (same logic as your all_formatting)
    df_spots_indexed = df_spots.with_row_count('id_spot')
    df_perps_indexed = df_perps.with_row_count('id_perp')
    
    df_trades = pl.sql("""
       select d1.price    as p_spot,
              d1.id_spot,
              d1.timestamp as timestamp,
              d2.price    as p_perp,
              d2.id_perp,
              d2.timestamp as timestamp2
       from df_spots_indexed as d1 full outer join df_perps_indexed as d2 
       using (timestamp)""").collect()
    
    df_trades = (df_trades
                 .with_columns(pl.col('timestamp').fill_null(pl.col('timestamp2')))
                 .sort('timestamp')
                 .with_columns(pl.col('p_spot').interpolate_by(pl.col('timestamp')))
                 .with_columns(pl.col('p_perp').interpolate_by(pl.col('timestamp'))))
    
    # Calculate log returns
    df_ret = df_trades.with_columns([
        (pl.col("p_spot").log() - pl.col("p_spot").log().shift(1)).alias("spot_ret"),
        (pl.col("p_perp").log() - pl.col("p_perp").log().shift(1)).alias("perp_ret"),
    ]).drop_nulls(subset=["spot_ret","perp_ret"])
    
    ################
    # Time-based rolling covariates at shock time (5-minute rolling windows)
    ################

    # Build pandas Series indexed by timestamps to allow time-based rolling
    # Ensure timestamps are datetime64
    timestamps_pd = pd.to_datetime(df_ret['timestamp'].to_pandas() if hasattr(df_ret['timestamp'], 'to_pandas') else df_ret['timestamp'])
    returns_spot_pd = pd.Series(df_ret['spot_ret'].to_numpy(), index=timestamps_pd)
    returns_perp_pd = pd.Series(df_ret['perp_ret'].to_numpy(), index=timestamps_pd)
    spot_prices_pd = pd.Series(df_ret['p_spot'].to_numpy(), index=timestamps_pd)
    perp_prices_pd = pd.Series(df_ret['p_perp'].to_numpy(), index=timestamps_pd)

    # Calculate basis as a time-indexed series
    basis_pd = np.log(perp_prices_pd) - np.log(spot_prices_pd)

    # Use a time-window string for rolling operations
    window_str = '5min'
    window_5min = 6000

    # Time-based rolling volatilities (5-minute window). Keep a modest min_periods.
    vol_spot_5min = returns_spot_pd.rolling(window=window_str, min_periods=100).std().reindex(timestamps_pd).to_numpy()
    vol_perp_5min = returns_perp_pd.rolling(window=window_str, min_periods=100).std().reindex(timestamps_pd).to_numpy()

    # Basis volatility (5-minute) computed from basis changes
    # basis_changes_pd = basis_pd.diff().fillna(0)
    # basis_vol_5min = basis_changes_pd.rolling(window=window_str, min_periods=100).std().reindex(timestamps_pd).to_numpy()

    # Extract covariates at each shock tick
    shocks_formatted = shocks_sp[0.001]['90%'].to_pandas()
    shocks_formatted['start_ts'] = pd.to_datetime(shocks_formatted['start_ts'])
    
    hf_covariates = []
    for _, shock in shocks_formatted.iterrows():
        tick = shock['event_tick']
        
        # Need sufficient history (5 min)
        if tick < window_5min or tick >= len(returns_spot_pd):
            hf_covariates.append({
                'event_tick': tick,
                'vol_spot_5min': np.nan,
                'vol_perp_5min': np.nan,
                'basis_level': np.nan,
                # 'basis_vol_5min': np.nan,
            })
            continue
        
        hf_covariates.append({
            'event_tick': tick,
            'vol_spot_5min': vol_spot_5min[tick],
            'vol_perp_5min': vol_perp_5min[tick],
            'basis_level': basis_pd[tick-1],  # Pre-shock level
            # 'basis_vol_5min': basis_vol_5min[tick],
        })
    
    hf_df = pd.DataFrame(hf_covariates)
    shocks_formatted = shocks_formatted.merge(hf_df, on='event_tick', how='left')
    
    ################
    # Spreads resampled to 5-minute intervals
    ################
    
    # Build 1s-level bid/ask series, then compute time-based 5-minute rolling means
    spots_bidask = data.to_intervals_bidask(data.df_trades_spots, '1s').to_pandas()
    spots_bidask['timestamp_bin'] = pd.to_datetime(spots_bidask['timestamp_bin'])
    spots_bidask = spots_bidask.set_index('timestamp_bin').sort_index()
    spots_bidask['spread_spot'] = spots_bidask['ask_price'] - spots_bidask['bid_price']
    spots_bidask['midpoint_spot'] = (spots_bidask['ask_price'] + spots_bidask['bid_price']) / 2
    spots_bidask['spread_spot_bps'] = spots_bidask['spread_spot'] / spots_bidask['midpoint_spot'] * 10000

    perps_bidask = data.to_intervals_bidask(data.df_trades_perps, '1s').to_pandas()
    perps_bidask['timestamp_bin'] = pd.to_datetime(perps_bidask['timestamp_bin'])
    perps_bidask = perps_bidask.set_index('timestamp_bin').sort_index()
    perps_bidask['spread_perp'] = perps_bidask['ask_price'] - perps_bidask['bid_price']
    perps_bidask['midpoint_perp'] = (perps_bidask['ask_price'] + perps_bidask['bid_price']) / 2
    perps_bidask['spread_perp_bps'] = perps_bidask['spread_perp'] / perps_bidask['midpoint_perp'] * 10000

    # Compute rolling (time-based) averages over 5 minutes
    spots_roll = spots_bidask[['spread_spot_bps']].rolling(window=window_str, min_periods=30).mean().rename(columns={'spread_spot_bps':'spread_spot_rolling'})
    perps_roll = perps_bidask[['spread_perp_bps']].rolling(window=window_str, min_periods=30).mean().rename(columns={'spread_perp_bps':'spread_perp_rolling'})

    # Rolling basis (in bps) using midpoints, then 5-min rolling mean
    midpoint_spot = spots_bidask['midpoint_spot']
    midpoint_perp = perps_bidask['midpoint_perp']
    # align indexes; compute pointwise basis then rolling mean
    basis_point = ((midpoint_perp.reindex(midpoint_spot.index) - midpoint_spot) / midpoint_spot) * 10000
    basis_roll = basis_point.rolling(window=window_str, min_periods=30).mean().rename('basis_rolling')

    # Consolidate rolling bidask values into a single DataFrame keyed by timestamp
    bidask_roll = pd.concat([spots_roll, perps_roll, basis_roll], axis=1).dropna()
    bidask_roll = bidask_roll.reset_index().rename(columns={'timestamp_bin':'timestamp_5min'})

    # Keep legacy column names so downstream code remains unchanged
    bidask_roll = bidask_roll.rename(columns={
        'spread_spot_rolling':'spread_spot_5min',
        'spread_perp_rolling':'spread_perp_5min',
        'basis_rolling':'basis_5min'
    })

    # Merge rolling values to shocks by nearest prior timestamp using merge_asof
    shocks_formatted = shocks_formatted.sort_values('start_ts')
    bidask_roll = bidask_roll.sort_values('timestamp_5min')
    # Ensure both sides use the same datetime resolution and no tz info
    shocks_formatted['start_ts'] = pd.to_datetime(shocks_formatted['start_ts']).dt.tz_localize(None).astype('datetime64[ns]')
    bidask_roll['timestamp_5min'] = pd.to_datetime(bidask_roll['timestamp_5min']).dt.tz_localize(None).astype('datetime64[ns]')
    shocks_formatted = pd.merge_asof(shocks_formatted, bidask_roll, left_on='start_ts', right_on='timestamp_5min', direction='backward')
    
    ################
    # Daily/regime covariates (funding, sentiment)
    ################
    
    shocks_formatted['date'] = pd.to_datetime(shocks_formatted["start_ts"]).dt.date
    
    # Funding rate (8-hour updates, averaged by day)
    funding = data.get_funding_data(start, end)
    funding["date"] = pd.to_datetime(funding["fundingTime"]).dt.date
    funding_daily = funding.groupby("date")["fundingRate"].mean().reset_index()
    
    # Trade volume (daily) - FIX: Convert date column to date type
    spot_volume = data.get_klines(start, end, 'spot', '1d', 
                                  columns=["Open time", "Close time", 'Open', "Close", "log_return", 'Volume'])
    spot_volume['date'] = pd.to_datetime(spot_volume['Open time']).dt.date  # Convert to date
    volume_daily = spot_volume[['date','Volume']].rename(columns={'Volume':'volume'})
    
    # Fear and Greed (daily)
    url = "https://api.alternative.me/fng/?limit=0"
    resp = requests.get(url)
    resp_data = resp.json()["data"]
    market_status = pd.DataFrame(resp_data)
    market_status["date"] = pd.to_datetime(market_status["timestamp"].astype(int), unit="s").dt.date
    market_status["fear_greed"] = market_status["value"].astype(int)
    market_status = market_status[["date", "fear_greed"]].drop_duplicates()
    
    # Merge daily covariates (all dates are now date objects)
    daily_covariates = (
        funding_daily
        .merge(volume_daily, on='date', how='outer')
        .merge(market_status, on='date', how='outer')
    )
    
    shocks_formatted = shocks_formatted.merge(
        daily_covariates,
        on='date',
        how='left'
    )
    
    ################
    # Final processing
    ################
    
    # Scale to basis points where appropriate
    shocks_formatted['fundingRate_bps'] = shocks_formatted['fundingRate'] * 1e4
    shocks_formatted['shock_size_bps'] = shocks_formatted['shock_size'] * 1e4
    shocks_formatted['basis_level_bps'] = shocks_formatted['basis_level'] * 1e4
    # shocks_formatted['basis_vol_5min_bps'] = shocks_formatted['basis_vol_5min'] * 1e4
    
    # Volatility in percentage terms for readability
    shocks_formatted['vol_spot_5min_pct'] = shocks_formatted['vol_spot_5min'] * 100
    shocks_formatted['vol_perp_5min_pct'] = shocks_formatted['vol_perp_5min'] * 100
    
    # Drop redundant columns
    df = shocks_formatted.drop([
        'start_ts', 'prev_ts', 'Shock', 'timestamp_5min',
        'fundingRate', 'shock_size', 'basis_level', 'vol_spot_5min', 'vol_perp_5min'
    ], axis=1, errors='ignore')
    
    # Handle zero durations
    df['Length'] = df['Length'].replace(0, 0.001)
    
    # Drop rows with missing critical data
    df = df.dropna(subset=['vol_spot_5min_pct', 'spread_spot_5min'])
    
    return df

# ...existing code...
def plot_shock_prices(shock_row, data_sp, window_s=5, resample='100ms', show_trades=True):
    """
    shock_row: pandas Series or dict with 'start_ts' (str/datetime) or 'event_tick'
    data_sp: TradeData instance (data_sp)
    window_s: seconds before/after to show
    resample: frequency to build midprice series from trades ('100ms','1s',...)
    """
    ts = shock_row.get('start_ts', shock_row.get('timestamp', None))
    ts0 = shock_row.get('prev_ts', shock_row.get('timestamp', None))
    if ts is None:
        raise ValueError("shock_row must contain 'start_ts' or 'timestamp'")
    start_ts = pd.to_datetime(ts)
    start_ts0 = pd.to_datetime(ts0)

    spots = getattr(data_sp, 'df_trades_spots')
    perps = getattr(data_sp, 'df_trades_perps')
    if hasattr(spots, 'to_pandas'):
        spots = spots.to_pandas()
    if hasattr(perps, 'to_pandas'):
        perps = perps.to_pandas()

    for df in (spots, perps):
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

    start, end = start_ts - pd.Timedelta(seconds=window_s), start_ts + pd.Timedelta(seconds=window_s)
    sp_win = spots[(spots['timestamp'] >= start) & (spots['timestamp'] <= end)].copy()
    pp_win = perps[(perps['timestamp'] >= start) & (perps['timestamp'] <= end)].copy()

    def midseries(trades):
        if trades.empty:
            return pd.Series(dtype=float)
        trades = trades.set_index('timestamp').sort_index()
        price_col = 'price' if 'price' in trades.columns else 'p_spot' if 'p_spot' in trades.columns else trades.columns[0]
        if 'qty' in trades.columns or 'size' in trades.columns:
            w = trades.get('qty', trades.get('size'))
            s = (trades[price_col] * w).resample(resample).sum() / w.resample(resample).sum()
        else:
            s = trades[price_col].resample(resample).median()
        return s.interpolate()

    sp_mid = midseries(spots[(spots['timestamp'] >= start) & (spots['timestamp'] <= end)])
    pp_mid = midseries(perps[(perps['timestamp'] >= start) & (perps['timestamp'] <= end)])

    fig, ax = plt.subplots(2,1, figsize=(10,6), sharex=True)
    if show_trades:
        if not sp_win.empty:
            ax[0].scatter(sp_win['timestamp'], sp_win['price'], s=10, c='C0', alpha=0.6, label='spot trades')
        if not pp_win.empty:
            ax[0].scatter(pp_win['timestamp'], pp_win['price'], s=10, c='C1', alpha=0.6, label='perp trades')
    if not sp_mid.empty:
        ax[0].plot(sp_mid.index, sp_mid.values, c='C0', lw=2, label=f'spot mid (resampled {resample})')
    if not pp_mid.empty:
        ax[0].plot(pp_mid.index, pp_mid.values, c='C1', lw=2, label=f'perp mid (resampled {resample})')

    ax[0].axvline(start_ts, color='k', linestyle='--', label='shock time')

    # add resolved line if shock has Length and is marked resolved (Status==1)
    resolved_ts = None
    if 'Length' in shock_row and pd.notna(shock_row['Length']) and shock_row.get('Status', None) == 1:
        try:
            resolved_ts = start_ts + pd.Timedelta(seconds=float(shock_row['Length']))
        except Exception:
            resolved_ts = None

    if resolved_ts is not None:
        ax[0].axvline(resolved_ts, color='g', linestyle='--', label='resolved')
        ax[0].annotate('resolved', xy=(resolved_ts, ax[0].get_ylim()[1]), xytext=(5,-10), textcoords='offset points', color='g', rotation=90, va='top')

    ax[0].legend(); ax[0].set_ylabel('Price'); ax[0].set_title('Prices around shock')

    sp_mid = sp_mid.reset_index()
    sp_mid['timestamp'] = pd.to_datetime(sp_mid['timestamp'])
    sp_mid = sp_mid.set_index('timestamp')

    pp_mid = pp_mid.reset_index()
    pp_mid['timestamp'] = pd.to_datetime(pp_mid['timestamp'])
    pp_mid = pp_mid.set_index('timestamp')

    combined = pd.concat([sp_mid,pp_mid], axis=1).dropna()
    combined.columns = ['spot', 'perp']

    if not combined.empty:
        gap = combined['perp'] - combined['spot']
        ax[1].plot(gap.index, gap.values, color='purple')
        ax[1].axhline(0, color='k', lw=0.7)
        # ax[1].axhline(gap.loc[start_ts:].iloc[0], color='purple', linestyle=':')
        # ax[1].axvline(start_ts, color='k', linestyle='--')

    # Basis BEFORE the shock (at prev_ts/start_ts0)
    pre_shock_gap = gap.loc[:start_ts0].iloc[-1] if len(gap.loc[:start_ts0]) > 0 else None
    if pre_shock_gap is not None:
        ax[1].axhline(pre_shock_gap, color='purple', linestyle=':', label='pre-shock basis')

    ax[1].axvline(start_ts, color='k', linestyle='--')  # Changed from start_ts0

    if resolved_ts is not None:
        # Basis AT resolution time
        resolved_gap = gap.loc[:resolved_ts].iloc[-1] if len(gap.loc[:resolved_ts]) > 0 else None
        if resolved_gap is not None:
            ax[1].axhline(resolved_gap, color='green', linestyle=':', label='resolved basis')
        ax[1].axvline(resolved_ts, color='g', linestyle='--')
        # if resolved_ts is not None:
        #     ax[1].axhline(gap.loc[:resolved_ts].iloc[-1], color='green', linestyle=':')
        #     ax[1].axvline(resolved_ts, color='g', linestyle='--')
        ax[1].set_ylabel('Perp - Spot'); ax[1].set_xlabel('Time')
    else:
        ax[1].text(0.1,0.5, 'No midprice overlap to compute gap', transform=ax[1].transAxes)

    plt.tight_layout()
    plt.show()

# Example usage:
# shock_df = shocks_sp[0.001]['90%'].to_pandas()
# row = shock_df.iloc[1]
# plot_shock_prices(row, data_sp, window_s=10, resample='50ms')
# ...existing code...