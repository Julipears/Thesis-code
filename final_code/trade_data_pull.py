"""
This file pulls trade data from Binance, Kucoin, and OKX, processes it into bid/ask intervals, 
and provides functions to aggregate and analyze the data. It also includes functionality to pull 
funding rate data and implied volatility data from various sources. 

There are two versions of the data retrieval functions: the original version (TradeData) which is compatible with 
a single ticker, and the new version (TradeDataMulti) which can handle multiple tickers and sources.

GenAI was used to optimize the data retrieval and processing functions for better performance and memory efficiency.
"""
import datetime
import requests
import zipfile
import io
import polars as pl
import pandas as pd
import numpy as np
import re
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

import os
import time
from pull_binance_data import *
    
from typing import Dict, Tuple, Optional, List


STALENESS_DIAGNOSTIC_BASIS = "actual_observation_fill_age_v4"
LAST_TRADE_DIAGNOSTIC_BASIS = "actual_last_trade_observation_fill_age_v1"


def check_price_staleness(
    fill_details: pd.DataFrame,
    stale_after_ms: Optional[float] = None,
    *,
    return_details: bool = False,
    print_summary: bool = True,
):
    """Summarize age created by forward-filling actual observations.

    Unlike the earlier diagnostic, this function does *not* infer staleness
    from whether a numerical price changed. It expects observation-age columns
    produced by ``TradeData.agg_to_intervals(...,
    return_fill_diagnostics=True)``. A same-price trade therefore resets the
    relevant side's age to zero.

    Parameters
    ----------
    fill_details : pd.DataFrame
        Row-level output from ``agg_to_intervals`` containing
        ``spot_fill_age_ms`` and ``perp_fill_age_ms``. Each midpoint age is the
        older (maximum) age of the bid-side and ask-side inputs used to build
        that midpoint.
    stale_after_ms : float, optional
        An observation is stale when its fill age is greater than this value.
        If omitted, any positive fill age is treated as forward-filled/stale.
    return_details : bool, default False
        Return ``(summary, row_details)``.
    print_summary : bool, default True
        Print the one-row summary table.
    """
    if not isinstance(fill_details, pd.DataFrame):
        raise TypeError("fill_details must be a pandas DataFrame.")
    if not isinstance(fill_details.index, pd.DatetimeIndex):
        raise TypeError("fill_details must have a DatetimeIndex.")
    if stale_after_ms is not None and stale_after_ms < 0:
        raise ValueError("stale_after_ms must be non-negative.")

    required = {"spot_fill_age_ms", "perp_fill_age_ms"}
    missing = required.difference(fill_details.columns)
    if missing:
        raise KeyError(
            "Fill diagnostics are missing required observation-age columns: "
            f"{sorted(missing)}. Obtain the frame with "
            "TradeData.agg_to_intervals(..., return_fill_diagnostics=True)."
        )

    frame = fill_details.copy().sort_index()
    if frame.index.has_duplicates:
        raise ValueError("fill_details.index contains duplicate timestamps.")

    # Rows without both midpoint ages cannot support a two-price comparison.
    frame = frame.dropna(subset=["spot_fill_age_ms", "perp_fill_age_ms"])
    if frame.empty:
        summary = pd.DataFrame([{
            "diagnostic_basis": STALENESS_DIAGNOSTIC_BASIS,
            "n_rows": 0,
            "stale_after_ms": (
                np.nan if stale_after_ms is None else float(stale_after_ms)
            ),
        }])
        if print_summary:
            print(summary.to_string(index=False))
        return (summary, frame) if return_details else summary

    spot_age = pd.to_numeric(frame["spot_fill_age_ms"], errors="coerce")
    perp_age = pd.to_numeric(frame["perp_fill_age_ms"], errors="coerce")

    spot_ff = spot_age > 0.0
    perp_ff = perp_age > 0.0
    either_ff = spot_ff | perp_ff
    both_ff = spot_ff & perp_ff
    one_sided_ff = spot_ff ^ perp_ff

    threshold = 0.0 if stale_after_ms is None else float(stale_after_ms)
    spot_stale = spot_age > threshold
    perp_stale = perp_age > threshold
    either_stale = spot_stale | perp_stale
    both_stale = spot_stale & perp_stale
    one_sided_stale = spot_stale ^ perp_stale

    frame["diagnostic_basis"] = STALENESS_DIAGNOSTIC_BASIS
    frame["spot_forward_filled"] = spot_ff
    frame["perp_forward_filled"] = perp_ff
    frame["either_forward_filled"] = either_ff
    frame["both_forward_filled"] = both_ff
    frame["one_sided_forward_filled"] = one_sided_ff
    frame["spot_fill_stale"] = spot_stale
    frame["perp_fill_stale"] = perp_stale
    frame["either_fill_stale"] = either_stale
    frame["both_fill_stale"] = both_stale
    frame["one_sided_fill_stale"] = one_sided_stale
    # Backward-compatible row-level alias, now explicitly fill-age based.
    frame["both_stale"] = both_stale

    # Price-change metrics are retained as a separate diagnostic only. They do
    # not determine any of the fill-age stale flags above.
    if {"midpoint_spot", "midpoint_perp"}.issubset(frame.columns):
        spot_changed = frame["midpoint_spot"].ne(frame["midpoint_spot"].shift())
        perp_changed = frame["midpoint_perp"].ne(frame["midpoint_perp"].shift())
        spot_changed.iloc[0] = True
        perp_changed.iloc[0] = True
        frame["spot_changed"] = spot_changed
        frame["perp_changed"] = perp_changed
        frame["either_changed"] = spot_changed | perp_changed
        frame["neither_changed"] = ~(spot_changed | perp_changed)
    else:
        frame["spot_changed"] = False
        frame["perp_changed"] = False
        frame["either_changed"] = False
        frame["neither_changed"] = False

    try:
        times_ns = frame.index.as_unit("ns").asi8
    except AttributeError:
        times_ns = frame.index.asi8
    if len(frame) > 1:
        spacing_ms = np.diff(times_ns).astype(np.float64) / 1_000_000.0
        median_spacing_ms = float(np.median(spacing_ms))
    else:
        median_spacing_ms = np.nan

    def q(values: pd.Series, quantile: float) -> float:
        arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
        return float(np.nanquantile(arr, quantile)) if len(arr) else np.nan

    def longest_true_run(mask: pd.Series) -> tuple[int, float]:
        arr = np.asarray(mask, dtype=bool)
        if not arr.any():
            return 0, 0.0
        edges = np.diff(np.r_[False, arr, False].astype(np.int8))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        lengths = ends - starts
        winner = int(np.argmax(lengths))
        start_i = int(starts[winner])
        end_i = int(ends[winner])
        elapsed_ms = float(times_ns[end_i - 1] - times_ns[start_i]) / 1_000_000.0
        if not np.isnan(median_spacing_ms):
            elapsed_ms += median_spacing_ms
        return int(lengths[winner]), elapsed_ms

    longest_rows, longest_ms = longest_true_run(both_stale)

    spot_p50 = q(spot_age, 0.50)
    spot_p90 = q(spot_age, 0.90)
    spot_p95 = q(spot_age, 0.95)
    spot_p99 = q(spot_age, 0.99)
    perp_p50 = q(perp_age, 0.50)
    perp_p90 = q(perp_age, 0.90)
    perp_p95 = q(perp_age, 0.95)
    perp_p99 = q(perp_age, 0.99)
    spot_max = float(spot_age.max())
    perp_max = float(perp_age.max())

    summary = pd.DataFrame([{
        "diagnostic_basis": STALENESS_DIAGNOSTIC_BASIS,
        "n_rows": int(len(frame)),
        "median_spacing_ms": median_spacing_ms,
        "spot_forward_filled_share": float(spot_ff.mean()),
        "perp_forward_filled_share": float(perp_ff.mean()),
        "either_forward_filled_share": float(either_ff.mean()),
        "both_forward_filled_share": float(both_ff.mean()),
        "one_sided_forward_filled_share": float(one_sided_ff.mean()),
        "spot_fill_age_p50_ms": spot_p50,
        "spot_fill_age_p90_ms": spot_p90,
        "spot_fill_age_p95_ms": spot_p95,
        "spot_fill_age_p99_ms": spot_p99,
        "spot_fill_age_max_ms": spot_max,
        "perp_fill_age_p50_ms": perp_p50,
        "perp_fill_age_p90_ms": perp_p90,
        "perp_fill_age_p95_ms": perp_p95,
        "perp_fill_age_p99_ms": perp_p99,
        "perp_fill_age_max_ms": perp_max,
        "stale_after_ms": (
            np.nan if stale_after_ms is None else float(stale_after_ms)
        ),
        "spot_fill_stale_share": float(spot_stale.mean()),
        "perp_fill_stale_share": float(perp_stale.mean()),
        "either_fill_stale_share": float(either_stale.mean()),
        "both_fill_stale_share": float(both_stale.mean()),
        "one_sided_fill_stale_share": float(one_sided_stale.mean()),
        "longest_both_fill_stale_run_rows": longest_rows,
        "longest_both_fill_stale_run_ms": longest_ms,
        # Separate price-change diagnostics (not used to define staleness).
        "spot_change_share": float(frame["spot_changed"].mean()),
        "perp_change_share": float(frame["perp_changed"].mean()),
        "either_change_share": float(frame["either_changed"].mean()),
        "neither_change_share": float(frame["neither_changed"].mean()),
        # Backward-compatible aliases. These now refer to fill age.
        "spot_age_p50_ms": spot_p50,
        "spot_age_p90_ms": spot_p90,
        "spot_age_p95_ms": spot_p95,
        "spot_age_p99_ms": spot_p99,
        "spot_age_max_ms": spot_max,
        "perp_age_p50_ms": perp_p50,
        "perp_age_p90_ms": perp_p90,
        "perp_age_p95_ms": perp_p95,
        "perp_age_p99_ms": perp_p99,
        "perp_age_max_ms": perp_max,
        "both_stale_share": float(both_stale.mean()),
    }])

    if print_summary:
        with pd.option_context("display.max_columns", None, "display.width", 240):
            print(summary.to_string(index=False))

    return (summary, frame) if return_details else summary

class TradeData:
    def __init__(self, symbol, source, cm_um='um'):
        self.symbol = symbol
        self.source = source
        self.cm_um = cm_um
        self.convert_tz = None
        
    def get_funding_data(self, start_date, end_date):
        symbol = self.symbol
        num_entries = 3*(datetime.datetime.strptime(end_date, '%Y%m%d') - datetime.datetime.strptime(start_date, '%Y%m%d')).days

        frs = []
        if num_entries > 1000:
            init_day = datetime.datetime.strptime(end_date, '%Y%m%d')
            while init_day > datetime.datetime.strptime(start_date, '%Y%m%d'):
                binance_starttime = str(int(init_day.timestamp()*1000))
                if self.source == 'Binance':
                    prefix = 'fapi'
                    if self.cm_um == 'cm':
                        symbol = symbol.replace('USDT', 'USD_PERP')
                        prefix = 'dapi'
                    r = requests.get(f"https://{prefix}.binance.com/{prefix}/v1/fundingRate?symbol={symbol}&startTime={binance_starttime}&limit=1000")
                
                fr = pd.DataFrame(r.json())
                frs.append(fr)
                init_day = init_day - datetime.timedelta(days=333)

        else:
            init_day = datetime.datetime.strptime(start_date, '%Y%m%d')
            binance_starttime = str(int(init_day.timestamp()*1000))
            if self.source == 'Binance':
                prefix = 'fapi'
                if self.cm_um == 'cm':
                    symbol = symbol.replace('USDT', 'USD_PERP')
                    prefix = 'dapi'
                r = requests.get(f"https://{prefix}.binance.com/{prefix}/v1/fundingRate?symbol={symbol}&startTime={binance_starttime}&limit=1000")
            fr = pd.DataFrame(r.json())
            frs.append(fr)

        fr_binance = pd.concat(frs)
        fr_binance = fr_binance.drop_duplicates()

        fr_binance[['fundingRate','fundingTime']] = fr_binance[['fundingRate','fundingTime']].astype(float)
        fr_binance['fundingTime'] = fr_binance['fundingTime'].apply(lambda x: datetime.datetime.utcfromtimestamp(x/1000).replace(second=0, microsecond=0))
        fr_binance = fr_binance[(fr_binance['fundingTime']>=datetime.datetime.strptime(start_date,'%Y%m%d'))&(fr_binance['fundingTime']<=datetime.datetime.strptime(end_date,'%Y%m%d'))]

        return fr_binance

    def get_data_once_optimized(self, url):
        def looks_numeric(s: str) -> bool:
            try:
                float(s)
                return True
            except ValueError:
                return False

        is_perp = 'PERP' in url

        try:
            # Use session for connection reuse
            with requests.Session() as session:
                session.headers.update({'User-Agent': 'Mozilla/5.0'})
                response = session.get(url, timeout=30)
                
                if response.status_code == 200 and response.content[:2] == b'PK':
                    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                        with z.open(z.namelist()[0]) as f:
                            # Read and process more efficiently
                            first_line = f.readline().decode('utf-8').strip()
                            f.seek(0)
                            
                            tokens = first_line.split(',')
                            has_any_number = any(looks_numeric(t) for t in tokens)

                            if not has_any_number:
                                df = pl.read_csv(f)
                                if self.source == 'Kucoin':
                                    df.columns = ['id','time','price','qty','side']
                                elif self.source == 'OKX':
                                    df.columns = ['symbol', 'id', 'side', 'price', 'qty', 'time']
                            else:
                                if self.source == 'Binance':
                                    columns = ['id','price','qty','base_qty','time','is_buyer_maker','idk'] if len(tokens) == 7 else ['id','price','qty','base_qty','time','is_buyer_maker']
                                elif self.source == 'Kucoin':
                                    columns = ['id','time','price','qty','side']
                                elif self.source == 'OKX':
                                    df.columns = ['symbol', 'id', 'side', 'price', 'qty', 'time']
                                df = pl.read_csv(f, has_header=False, new_columns=columns)
                            
                            # Optimized processing using Polars expressions
                            if 'is_buyer_maker' in df.columns:
                                df = df.with_columns([
                                    pl.col('is_buyer_maker').alias('is_bid'),
                                    pl.when(pl.col('time') < 1e11)
                                    .then(pl.col('time') * 1000)
                                    .when(pl.col('time') < 1e14)
                                    .then(pl.col('time'))
                                    .otherwise(pl.col('time') / 1000)
                                    .cast(pl.Datetime(time_unit='ms'))
                                    .alias('timestamp')
                                ])
                            elif 'side' in df.columns:
                                df = df.with_columns([
                                    (pl.col('side').str.to_lowercase()=='sell').alias('is_bid'),
                                    pl.when(pl.col('time') < 1e11)
                                    .then(pl.col('time') * 1000)
                                    .when(pl.col('time') < 1e14)
                                    .then(pl.col('time'))
                                    .otherwise(pl.col('time') / 1000)
                                    .cast(pl.Datetime(time_unit='ms'))
                                    .alias('timestamp')
                                ])
                            
                            # Preserve exchange ordering when collapsing trades
                            # to the millisecond timestamp retained by this
                            # pipeline. This makes "last trade" deterministic
                            # and avoids arbitrary parallel group ordering.
                            df = df.with_columns(pl.col('timestamp').dt.truncate('1ms'))
                            if 'id' in df.columns:
                                df = (
                                    df.sort(['timestamp', 'is_bid', 'id'], maintain_order=True)
                                    .group_by(['timestamp', 'is_bid'], maintain_order=True)
                                    .agg(
                                        pl.col('price').last(),
                                        pl.col('id').last().alias('trade_id'),
                                    )
                                )
                                return df.select(['price', 'is_bid', 'timestamp', 'trade_id'])
                            df = (
                                df.sort(['timestamp', 'is_bid', 'price'], maintain_order=True)
                                .group_by(['timestamp', 'is_bid'], maintain_order=True)
                                .agg(pl.col('price').last())
                            )
                            return df.select(['price', 'is_bid', 'timestamp'])
                            
        except Exception as e:
            print(f"Error processing {url}: {e}")
            return None


    def get_all_data_optimized(self, end_date, days=30, kind='spot', n_jobs=8):
        symbol = self.symbol
        #dates_files = self.get_binance_data_optimized(end_date, days, kind=kind)
        
        if kind == 'perp':
            if self.source == 'Binance':
                if self.cm_um == 'cm':
                    symbol = symbol.replace('USDT', 'USD_PERP')
                urls = [f"https://data.binance.vision/data/futures/{self.cm_um}/daily/trades/{symbol}/{symbol}-trades-{datetime.datetime.strftime(end_date-datetime.timedelta(days=i), '%Y-%m-%d')}.zip" for i in range(days)]
            elif self.source == 'Kucoin':
                urls = [f"https://historical-data.kucoin.com/data/futures/daily/trades/{symbol}M/{symbol}M-trades-{datetime.datetime.strftime(end_date-datetime.timedelta(days=i), '%Y-%m-%d')}.zip" for i in range(days)] 
            elif self.source == 'OKX':
                urls = [f"https://static.okx.com/cdn/okex/traderecords/trades/daily/{datetime.datetime.strftime(end_date-datetime.timedelta(days=i),'%Y%m%d')}/{symbol}-SWAP-trades-{datetime.datetime.strftime(end_date-datetime.timedelta(days=i), '%Y-%m-%d')}.zip" for i in range(days)] 
        else:
            if self.source == 'Binance':
                urls = [f"https://data.binance.vision/data/spot/daily/trades/{symbol}/{symbol}-trades-{datetime.datetime.strftime(end_date-datetime.timedelta(days=i), '%Y-%m-%d')}.zip" for i in range(days)]
            elif self.source == 'Kucoin':
                urls = [f"https://historical-data.kucoin.com/data/spot/daily/trades/{symbol}/{symbol}-trades-{datetime.datetime.strftime(end_date-datetime.timedelta(days=i), '%Y-%m-%d')}.zip" for i in range(days)] 
            elif self.source == 'OKX':
                urls = [f"https://static.okx.com/cdn/okex/traderecords/trades/daily/{datetime.datetime.strftime(end_date-datetime.timedelta(days=i),'%Y%m%d')}/{symbol}-trades-{datetime.datetime.strftime(end_date-datetime.timedelta(days=i), '%Y-%m-%d')}.zip" for i in range(days)]

        print(f"Processing {len(urls)} files...")
        
        # Process in smaller batches to manage memory
        batch_size = min(20, len(urls))
        all_results = []
        
        for i in range(0, len(urls), batch_size):
            batch_urls = urls[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(urls)-1)//batch_size + 1}")
            
            # Use ThreadPoolExecutor for better control
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                batch_results = list(executor.map(self.get_data_once_optimized, batch_urls))
            
            # Filter valid results
            valid_batch = [r for r in batch_results if r is not None]
            all_results.extend(valid_batch)
        
        if all_results:
            print("Combining results...")
            df_combined = pl.concat(all_results)
            df_combined = df_combined.with_columns(pl.lit(kind).alias('type'))
            return df_combined.sort('timestamp')
        else:
            print("No valid data retrieved")
            return pl.DataFrame()
        
    def to_intervals_bidask(self, df, freq='1s', fill_gaps=False):
        """
        Resample trade data into time intervals (default: 1 second):
        - Takes the last price for bids and asks separately
        - Creates columns 'bid_price' and 'ask_price'
        - Optionally forward-fills missing intervals
        - 'freq' can be any valid Polars duration string ('500ms', '1s', '5s', etc.)
        
        Parameters
        ----------
        fill_gaps : bool, default False
            If True, creates continuous timeline and forward-fills (slower but complete).
            If False, returns only observed intervals (2-5x faster).
        """

        # Ensure timestamp is a Polars Datetime
        df = df.with_columns(pl.col("timestamp").cast(pl.Datetime))

        # Truncate timestamps to interval bins
        df = df.with_columns(pl.col("timestamp").dt.truncate(freq).alias("timestamp_bin"))

        # Aggregate by side and time bin (take last price per side)
        df_agg = (
            df
            .group_by(["timestamp_bin", "is_bid"], maintain_order=True)
            .agg(pl.col("price").last().alias("price"))
            .pivot(
                values="price",
                index="timestamp_bin",
                columns="is_bid"
            )
            .rename({
                "true": "bid_price",
                "false": "ask_price"
            })
            .sort("timestamp_bin")
        )

        if not fill_gaps:
            return df_agg

        # Build a continuous timeline (expensive)
        full_range = pl.DataFrame({
            "timestamp_bin": pl.datetime_range(
                start=df_agg["timestamp_bin"].min(),
                end=df_agg["timestamp_bin"].max(),
                interval=freq,
                eager=True
            )
        })

        # Join and forward-fill missing intervals
        df_filled = (
            full_range
            .join(df_agg, on="timestamp_bin", how="left")
            .fill_null(strategy="forward")
        )

        return df_filled
    

    def to_intervals_last_trade(self, df, freq='1s', fill_gaps=False):
        """Aggregate trades to the last retained transaction price per bin.

        This differs from :meth:`to_intervals_bidask`: trade direction is
        ignored, so each bin requires only one observed trade rather than one
        buyer-maker and one seller-maker trade.

        Parameters
        ----------
        df : polars.DataFrame
            Trade data containing ``timestamp`` and ``price``.
        freq : str, default '1s'
            Polars duration string such as ``'10ms'``, ``'100ms'``, or ``'1s'``.
        fill_gaps : bool, default False
            If True, construct a complete regular grid and forward-fill the
            last-trade price. If False, return only bins containing a trade.

        Returns
        -------
        polars.DataFrame
            Columns ``timestamp_bin`` and ``last_trade_price``.

        Notes
        -----
        The downloader currently retains at most one price per millisecond and
        trade side. If both sides have retained observations at the same exact
        millisecond, their original exchange ordering is no longer available;
        the last retained row is used as the tie-breaker.
        """
        required = {"timestamp", "price"}
        missing = required.difference(df.columns)
        if missing:
            raise KeyError(
                f"Trade data is missing required columns: {sorted(missing)}"
            )

        if df.height == 0:
            return pl.DataFrame({
                "timestamp_bin": pl.Series([], dtype=pl.Datetime("us")),
                "last_trade_price": pl.Series([], dtype=pl.Float64),
            })

        order_cols = ["timestamp"]
        if "trade_id" in df.columns:
            order_cols.append("trade_id")
        else:
            order_cols.extend([col for col in ("price", "is_bid") if col in df.columns])
        df_agg = (
            df
            .with_columns(pl.col("timestamp").cast(pl.Datetime))
            .sort(order_cols, maintain_order=True)
            .with_columns(
                pl.col("timestamp").dt.truncate(freq).alias("timestamp_bin")
            )
            .group_by("timestamp_bin", maintain_order=True)
            .agg(pl.col("price").last().alias("last_trade_price"))
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

        return (
            full_range
            .join(df_agg, on="timestamp_bin", how="left")
            .with_columns(pl.col("last_trade_price").forward_fill())
        )

    def grab_trades_data(self, end_date, days=30, n_jobs=10):
        df_trades_spots = self.get_all_data_optimized(end_date=end_date, days=days, kind='spot', n_jobs=n_jobs)
        df_trades_perps = self.get_all_data_optimized(end_date=end_date, days=days, kind='perp', n_jobs=n_jobs)
        self.df_trades_spots = df_trades_spots
        self.df_trades_perps = df_trades_perps
        return

    def agg_to_intervals(
        self,
        freq='1s',
        start=None,
        end=None,
        fill_gaps=False,
        max_fill_gap_ms: Optional[int] = None,
        drop_both_stale: bool = False,
        stale_after_ms: Optional[float] = None,
        return_fill_diagnostics: bool = False,
    ):
        """Aggregate trades to bid/ask midpoints at ``freq``.

        Observation timestamps are recorded for every bid/ask side before any
        forward fill. Consequently, fill age resets when a new observation
        arrives even if its numerical price is unchanged.

        Parameters
        ----------
        freq : str
            Aggregation frequency (``'10ms'``, ``'100ms'``, ``'1s'``, etc.).
        start, end : datetime-like, optional
            Time range to aggregate, interpreted as ``[start, end)``.
        fill_gaps : bool, default False
            Build a complete regular grid before forward-filling. When False,
            retain only the union of bins observed in spot or perpetual data.
        max_fill_gap_ms : int, optional
            Do not use an individual bid/ask input after it has been carried
            farther than this age. ``None`` imposes no age limit.
        drop_both_stale : bool, default False
            Remove rows where both market midpoints exceed
            ``stale_after_ms``. If no threshold is supplied, remove rows where
            both midpoint constructions use at least one forward-filled side.
        stale_after_ms : float, optional
            Fill-age threshold for ``drop_both_stale``.
        return_fill_diagnostics : bool, default False
            Return ``(model_frame, fill_details)``. ``fill_details`` measures
            time since actual side observations, not time since price changes.

        Returns
        -------
        pd.DataFrame or tuple[pd.DataFrame, pd.DataFrame]
            Model input frame, optionally accompanied by row-level fill ages.
        """
        if start is not None:
            start = pd.Timestamp(start).to_pydatetime()
        if end is not None:
            end = pd.Timestamp(end).to_pydatetime()
        if max_fill_gap_ms is not None and max_fill_gap_ms < 0:
            raise ValueError("max_fill_gap_ms must be non-negative.")
        if stale_after_ms is not None and stale_after_ms < 0:
            raise ValueError("stale_after_ms must be non-negative.")

        spots = self.df_trades_spots
        perps = self.df_trades_perps

        if start is not None:
            spots = spots.filter(pl.col("timestamp") >= pl.lit(start))
            perps = perps.filter(pl.col("timestamp") >= pl.lit(start))
        if end is not None:
            spots = spots.filter(pl.col("timestamp") < pl.lit(end))
            perps = perps.filter(pl.col("timestamp") < pl.lit(end))

        if spots.height == 0 or perps.height == 0:
            empty = pd.DataFrame()
            return (empty, empty.copy()) if return_fill_diagnostics else empty

        spots_bidask = self.to_intervals_bidask(spots, freq, fill_gaps=False)
        perps_bidask = self.to_intervals_bidask(perps, freq, fill_gaps=False)

        joined = (
            spots_bidask
            .join(perps_bidask, on="timestamp_bin", how="outer", suffix="_perp")
            .with_columns(
                pl.coalesce(["timestamp_bin", "timestamp_bin_perp"])
                .alias("timestamp_bin")
            )
            .drop("timestamp_bin_perp")
            .sort("timestamp_bin")
            .with_columns(pl.col("timestamp_bin").alias("timestamp"))
        )

        if fill_gaps:
            full_range = pl.DataFrame({
                "timestamp_bin": pl.datetime_range(
                    start=joined["timestamp_bin"].min(),
                    end=joined["timestamp_bin"].max(),
                    interval=freq,
                    eager=True,
                )
            })
            joined = (
                full_range
                .join(joined.drop("timestamp"), on="timestamp_bin", how="left")
                .with_columns(pl.col("timestamp_bin").alias("timestamp"))
            )

        value_cols = [
            "bid_price",
            "ask_price",
            "bid_price_perp",
            "ask_price_perp",
        ]
        side_labels = {
            "bid_price": "spot_bid",
            "ask_price": "spot_ask",
            "bid_price_perp": "perp_bid",
            "ask_price_perp": "perp_ask",
        }

        joined = joined.with_columns(
            pl.col("timestamp")
            .cast(pl.Datetime("ns"))
            .cast(pl.Int64)
            .alias("_timestamp_ns")
        )

        last_time_cols = {}
        for col in value_cols:
            last_col = f"_last_{col}_ns"
            last_time_cols[col] = last_col
            joined = joined.with_columns(
                pl.when(pl.col(col).is_not_null())
                .then(pl.col("_timestamp_ns"))
                .otherwise(pl.lit(None, dtype=pl.Int64))
                .forward_fill()
                .alias(last_col)
            )

        filled = joined.with_columns([
            pl.col(col).forward_fill().alias(col) for col in value_cols
        ])

        # Compute side ages before enforcing a fill limit. A same-price trade is
        # non-null in the original bin and therefore resets this age to zero.
        for col in value_cols:
            age_col = f"_{side_labels[col]}_fill_age_ms"
            last_col = last_time_cols[col]
            filled = filled.with_columns(
                pl.when(pl.col(last_col).is_not_null())
                .then(
                    (pl.col("_timestamp_ns") - pl.col(last_col))
                    .cast(pl.Float64) / 1_000_000.0
                )
                .otherwise(pl.lit(None, dtype=pl.Float64))
                .alias(age_col)
            )

        if max_fill_gap_ms is not None:
            threshold_ns = int(float(max_fill_gap_ms) * 1_000_000)
            for col in value_cols:
                last_col = last_time_cols[col]
                filled = filled.with_columns(
                    pl.when(
                        pl.col(last_col).is_null()
                        | ((pl.col("_timestamp_ns") - pl.col(last_col)) > threshold_ns)
                    )
                    .then(pl.lit(None))
                    .otherwise(pl.col(col))
                    .alias(col)
                )

        filled = filled.with_columns([
            ((pl.col("ask_price") + pl.col("bid_price")) / 2)
            .alias("midpoint_spot"),
            ((pl.col("ask_price_perp") + pl.col("bid_price_perp")) / 2)
            .alias("midpoint_perp"),
            pl.max_horizontal([
                pl.col("_spot_bid_fill_age_ms"),
                pl.col("_spot_ask_fill_age_ms"),
            ]).alias("spot_fill_age_ms"),
            pl.max_horizontal([
                pl.col("_perp_bid_fill_age_ms"),
                pl.col("_perp_ask_fill_age_ms"),
            ]).alias("perp_fill_age_ms"),
        ])

        diagnostic_columns = [
            "timestamp",
            "midpoint_spot",
            "midpoint_perp",
            "spot_fill_age_ms",
            "perp_fill_age_ms",
            "_spot_bid_fill_age_ms",
            "_spot_ask_fill_age_ms",
            "_perp_bid_fill_age_ms",
            "_perp_ask_fill_age_ms",
        ]
        bidask_meta = (
            filled
            .select(diagnostic_columns)
            .to_pandas()
            .set_index("timestamp")
            .dropna(subset=[
                "midpoint_spot",
                "midpoint_perp",
                "spot_fill_age_ms",
                "perp_fill_age_ms",
            ])
            .rename(columns={
                "_spot_bid_fill_age_ms": "spot_bid_fill_age_ms",
                "_spot_ask_fill_age_ms": "spot_ask_fill_age_ms",
                "_perp_bid_fill_age_ms": "perp_bid_fill_age_ms",
                "_perp_ask_fill_age_ms": "perp_ask_fill_age_ms",
            })
        )

        if bidask_meta.empty:
            empty = pd.DataFrame()
            return (empty, bidask_meta) if return_fill_diagnostics else empty

        threshold = 0.0 if stale_after_ms is None else float(stale_after_ms)
        both_fill_stale = (
            (bidask_meta["spot_fill_age_ms"] > threshold)
            & (bidask_meta["perp_fill_age_ms"] > threshold)
        )
        if drop_both_stale:
            bidask_meta = bidask_meta.loc[~both_fill_stale]

        bidask = bidask_meta[["midpoint_spot", "midpoint_perp"]]
        bidask_diff = find_first_diff(bidask).dropna().astype(np.float32)

        if not return_fill_diagnostics:
            return bidask_diff

        # Align diagnostics exactly to rows surviving the model-frame transform.
        fill_details = bidask_meta.loc[
            bidask_meta.index.intersection(bidask_diff.index)
        ].copy()
        fill_details = fill_details.reindex(bidask_diff.index)
        fill_details["diagnostic_basis"] = STALENESS_DIAGNOSTIC_BASIS
        return bidask_diff, fill_details



    def agg_last_trade_to_intervals(
        self,
        freq='1s',
        start=None,
        end=None,
        fill_gaps=False,
        max_fill_gap_ms: Optional[int] = None,
        return_fill_diagnostics: bool = False,
        rename_for_vecm: bool = False,
        retain_initial_grid_row: bool = False,
    ):
        """Aggregate spot and perpetual trades using last-trade prices.

        The last observed transaction price in each bin is used for each
        market. Missing market observations are forward-filled only after the
        spot and perpetual observed bins have been aligned.

        Parameters
        ----------
        freq : str
            Aggregation frequency such as ``'10ms'``, ``'50ms'``, or ``'1s'``.
        start, end : datetime-like, optional
            Half-open time range ``[start, end)``.
        fill_gaps : bool, default False
            If True, construct every calendar-time bin. If False, retain only
            the union of bins observed in either market.
        max_fill_gap_ms : int, optional
            Maximum age of a last-trade value that may be carried forward.
            ``None`` imposes no cutoff.
        return_fill_diagnostics : bool, default False
            If True, return ``(model_frame, fill_details)``. Fill ages measure
            time since an actual trade observation, including same-price trades.
        rename_for_vecm : bool, default False
            If True, rename the output columns to the legacy midpoint names
            expected by the current VECM code. This changes names only; the
            values remain last-trade prices.
        retain_initial_grid_row : bool, default False
            When returning fill diagnostics, retain the first valid price-grid
            row even though its log difference is undefined. This is useful
            for methods that consume price levels on a complete regular grid.

        Returns
        -------
        pandas.DataFrame or tuple[pandas.DataFrame, pandas.DataFrame]
            Price/log-difference frame, optionally with row-level fill ages.
        """
        if start is not None:
            start = pd.Timestamp(start).to_pydatetime()
        if end is not None:
            end = pd.Timestamp(end).to_pydatetime()
        if max_fill_gap_ms is not None and max_fill_gap_ms < 0:
            raise ValueError("max_fill_gap_ms must be non-negative.")

        spots = self.df_trades_spots
        perps = self.df_trades_perps

        if start is not None:
            spots = spots.filter(pl.col("timestamp") >= pl.lit(start))
            perps = perps.filter(pl.col("timestamp") >= pl.lit(start))
        if end is not None:
            spots = spots.filter(pl.col("timestamp") < pl.lit(end))
            perps = perps.filter(pl.col("timestamp") < pl.lit(end))

        if spots.height == 0 or perps.height == 0:
            empty = pd.DataFrame()
            return (empty, empty.copy()) if return_fill_diagnostics else empty

        spots_last = self.to_intervals_last_trade(
            spots,
            freq=freq,
            fill_gaps=False,
        ).rename({"last_trade_price": "last_trade_spot"})

        perps_last = self.to_intervals_last_trade(
            perps,
            freq=freq,
            fill_gaps=False,
        ).rename({"last_trade_price": "last_trade_perp"})

        joined = (
            spots_last
            .join(perps_last, on="timestamp_bin", how="outer", suffix="_perp")
            .with_columns(
                pl.coalesce(["timestamp_bin", "timestamp_bin_perp"])
                .alias("timestamp_bin")
            )
            .drop("timestamp_bin_perp")
            .sort("timestamp_bin")
            .with_columns(pl.col("timestamp_bin").alias("timestamp"))
        )

        if fill_gaps:
            full_range = pl.DataFrame({
                "timestamp_bin": pl.datetime_range(
                    start=joined["timestamp_bin"].min(),
                    end=joined["timestamp_bin"].max(),
                    interval=freq,
                    eager=True,
                )
            })
            joined = (
                full_range
                .join(joined.drop("timestamp"), on="timestamp_bin", how="left")
                .with_columns(pl.col("timestamp_bin").alias("timestamp"))
            )

        value_cols = ["last_trade_spot", "last_trade_perp"]
        age_cols = {
            "last_trade_spot": "spot_fill_age_ms",
            "last_trade_perp": "perp_fill_age_ms",
        }

        joined = joined.with_columns(
            pl.col("timestamp")
            .cast(pl.Datetime("ns"))
            .cast(pl.Int64)
            .alias("_timestamp_ns")
        )

        last_time_cols = {}
        for col in value_cols:
            last_col = f"_last_{col}_ns"
            last_time_cols[col] = last_col
            joined = joined.with_columns(
                pl.when(pl.col(col).is_not_null())
                .then(pl.col("_timestamp_ns"))
                .otherwise(pl.lit(None, dtype=pl.Int64))
                .forward_fill()
                .alias(last_col)
            )

        filled = joined.with_columns([
            pl.col(col).forward_fill().alias(col) for col in value_cols
        ])

        for col in value_cols:
            last_col = last_time_cols[col]
            filled = filled.with_columns(
                pl.when(pl.col(last_col).is_not_null())
                .then(
                    (pl.col("_timestamp_ns") - pl.col(last_col))
                    .cast(pl.Float64) / 1_000_000.0
                )
                .otherwise(pl.lit(None, dtype=pl.Float64))
                .alias(age_cols[col])
            )

        if max_fill_gap_ms is not None:
            threshold_ns = int(float(max_fill_gap_ms) * 1_000_000)
            for col in value_cols:
                last_col = last_time_cols[col]
                filled = filled.with_columns(
                    pl.when(
                        pl.col(last_col).is_null()
                        | (
                            (pl.col("_timestamp_ns") - pl.col(last_col))
                            > threshold_ns
                        )
                    )
                    .then(pl.lit(None))
                    .otherwise(pl.col(col))
                    .alias(col)
                )

        last_trade_meta = (
            filled
            .select([
                "timestamp",
                "last_trade_spot",
                "last_trade_perp",
                "spot_fill_age_ms",
                "perp_fill_age_ms",
            ])
            .to_pandas()
            .set_index("timestamp")
            .dropna(subset=[
                "last_trade_spot",
                "last_trade_perp",
                "spot_fill_age_ms",
                "perp_fill_age_ms",
            ])
        )

        if last_trade_meta.empty:
            empty = pd.DataFrame()
            return (empty, last_trade_meta) if return_fill_diagnostics else empty

        prices = last_trade_meta[["last_trade_spot", "last_trade_perp"]]
        model_frame = find_first_diff(prices).dropna().astype(np.float32)

        if rename_for_vecm:
            rename_map = {
                "last_trade_spot": "midpoint_spot",
                "last_trade_perp": "midpoint_perp",
                "log_last_trade_spot": "log_midpoint_spot",
                "log_last_trade_perp": "log_midpoint_perp",
            }
            model_frame = model_frame.rename(
                columns={
                    old: new
                    for old, new in rename_map.items()
                    if old in model_frame.columns
                }
            )

        if not return_fill_diagnostics:
            return model_frame

        if retain_initial_grid_row:
            fill_details = last_trade_meta.copy()
        else:
            fill_details = last_trade_meta.loc[
                last_trade_meta.index.intersection(model_frame.index)
            ].copy()
            fill_details = fill_details.reindex(model_frame.index)
        fill_details["diagnostic_basis"] = LAST_TRADE_DIAGNOSTIC_BASIS
        return model_frame, fill_details

    def get_klines(self, start_date, end_date, kind='spot', interval='1h', columns=[], n_jobs=10):
        symbol = self.symbol

        if kind=='spot':
            url = "https://api.binance.com/api/v3/klines"
        else:
            # symbol = symbol.replace('USDT', 'USD_PERP')
            url = "https://fapi.binance.com/fapi/v1/klines"
            
        limit = 1000
        interval_ms = {
            '1m': 60_000,
            '3m': 3 * 60_000,
            '5m': 5 * 60_000,
            '15m': 15 * 60_000,
            '30m': 30 * 60_000,
            '1h': 60 * 60_000,
            '2h': 2 * 60 * 60_000,
            '4h': 4 * 60 * 60_000,
            '6h': 6 * 60 * 60_000,
            '12h': 12 * 60 * 60_000,
            '1d': 24 * 60 * 60_000
        }[interval]

        start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
        end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)

        # Split into chunks
        total_hours = (end_ts - start_ts) // interval_ms
        n_chunks = int(np.ceil(total_hours / limit))

        timestamps = [start_ts + i * limit * interval_ms for i in range(n_chunks)]
        chunks = [(t, min(t + limit * interval_ms, end_ts)) for t in timestamps]

        def fetch_chunk(start_t, end_t):
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_t,
                "endTime": end_t,
                "limit": limit,
            }
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            return r.json()

        # Parallel fetch
        all_data = []
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(fetch_chunk, s, e): (s, e) for s, e in chunks}
            for future in as_completed(futures):
                try:
                    data = future.result()
                    if data:
                        all_data.extend(data)
                except Exception as e:
                    print(f"Chunk failed: {futures[future]} - {e}")

        if not all_data:
            raise ValueError("No data retrieved from Binance.")

        # Convert to DataFrame
        df_vol = pd.DataFrame(all_data, columns=[
            "Open time", "Open", "High", "Low", "Close", "Volume",
            "Close time", "Quote asset volume", "Number of trades",
            "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
        ])

        df_vol = df_vol.sort_values('Open time')
        df_vol[['Open', "Close", 'High', 'Low', 'Volume']] = df_vol[['Open', "Close", 'High', 'Low', 'Volume']].astype(float)
        df_vol["log_return"] = np.log(df_vol["Close"] / df_vol["Close"].shift(1)) # this is over 1 hour
        open_times = pd.to_datetime(df_vol["Open time"], unit="ms", utc=True)
        df_vol["Open time"] = open_times.dt.tz_convert(self.convert_tz).dt.tz_localize(None) # convert all the times to est
        close_times = pd.to_datetime(df_vol["Close time"], unit="ms", utc=True)
        df_vol["Close time"] = close_times.dt.tz_convert(self.convert_tz).dt.tz_localize(None) + datetime.timedelta(milliseconds=1)
        # Open time here is the start of the period

        if len(columns) == 0:
            columns = ["Open time", "Close time", 'Open', "Close", "log_return"]

        return df_vol[columns]

# Optimized timestamp conversion functions
def unix_to_timestamp_vectorized(timestamps):
    """Vectorized timestamp conversion"""
    return pl.from_pandas(pd.to_datetime(timestamps, unit='ms', utc=True).dt.tz_localize(None))

def unix_to_timestamp_us_vectorized(timestamps):
    """Vectorized microsecond timestamp conversion"""
    return pl.from_pandas(pd.to_datetime(timestamps, unit='us', utc=True).dt.tz_localize(None))


def save_load_data_optimized(file_names, vars=[], save=True, use_parquet=True):
    """Optimized save/load with Parquet support for better performance"""
    if save:
        for var, file in zip(vars, file_names):
            if use_parquet and hasattr(var, 'write_parquet'):
                # Use Parquet for Polars DataFrames - much faster and smaller
                var.write_parquet(f"{file}.parquet", compression='snappy')
            else:
                # Fallback to pickle
                with open(f"{file}.pkl", "wb") as f:
                    pickle.dump(var, f, protocol=pickle.HIGHEST_PROTOCOL)
        return
    else:
        ret_vars = []
        for file in file_names:
            if use_parquet and os.path.exists(f"{file}.parquet"):
                ret_vars.append(pl.read_parquet(f"{file}.parquet"))
            elif os.path.exists(f"{file}.pkl"):
                with open(f"{file}.pkl", "rb") as f:
                    ret_vars.append(pickle.load(f))
            else:
                print(f"File not found: {file}")
                ret_vars.append(None)
        return ret_vars

class DataTransformations:
    def __init__(self, interval='1D'):
        self.interval=interval

    def get_google(self, start, end):
        pytrends = TrendReq(hl='en-US', tz=0)

        pytrends.build_payload(
            kw_list=["Bitcoin"],
            timeframe=f"{datetime.datetime.strfptime(start, '%Y%m%d').strftime('%Y-%m-%d')} {datetime.datetime.strfptime(end, '%Y%m%d').strftime('%Y-%m-%d')}",
            geo=""
        )

        data = pytrends.interest_over_time()
        google_sentiment = data[["Bitcoin"]]

    def spread_basis(self, data):
        pass

    def trade_volume(self, data):
        pass

    def funding(self, data):
        pass

    def get_iv(self, source, start, end, instr='BTC', resolution=1):
        """
        source = ['block', 'deribit']
        """
        start = datetime.datetime.strptime(start, '%Y%m%d')
        end = datetime.datetime.strptime(end, '%Y%m%d')

        if source == 'block':
            r = requests.get('https://www.theblock.co/api/charts/chart/crypto-markets/options/btc-atm-implied-volatility')
            series_names = r.json()['chart']['jsonFile']['Series'].keys()

            df_vols = []
            for s in series_names:
                df_vol = pd.DataFrame(r.json()['chart']['jsonFile']['Series'][s]['Data'])
                df_vol['duration'] = s
                df_vols.append(df_vol)

            iv_block = pd.concat(df_vols)
            iv_block['Timestamp'] = iv_block['Timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x))
            iv_block['day'] = iv_block['Timestamp'].apply(lambda x: x.date())
            iv_block = iv_block[(iv_block['day'] >= start.date()) & (iv_block['day'] <= end.date())]
            return iv_block.set_index('Timestamp')
        elif source == 'deribit':
            base_url = "https://www.deribit.com/api/v2/public/get_volatility_index_data"

            if start is None:
                start = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=7)
            if end is None:
                end = datetime.datetime.now(datetime.timezone.utc)

            # Calculate chunk size based on resolution
            # For hourly data, ~1000 points = ~41 days
            points_per_request = 1000
            seconds_per_chunk = points_per_request * resolution
            chunk_duration = datetime.timedelta(seconds=seconds_per_chunk)

            all_data = []
            current_start = start

            while current_start < end:
                current_end = min(current_start + chunk_duration, end)
                
                start_ts = int(current_start.timestamp() * 1000)
                end_ts = int(current_end.timestamp() * 1000)

                params = {
                    "currency": instr.upper(),
                    "resolution": resolution,
                    "start_timestamp": start_ts,
                    "end_timestamp": end_ts,
                }

                r = requests.get(base_url, params=params)
                if r.status_code != 200:
                    raise RuntimeError(f"Deribit API error: {r.status_code} {r.text}")

                result = r.json().get("result", {})
                data_values = result.get("data", [])
                
                if data_values:
                    all_data.extend(data_values)
                    print(f"Fetched {len(data_values)} records for {current_start.date()} to {current_end.date()}. Total: {len(all_data)}")
                
                current_start = current_end

            if not all_data:
                print("⚠️ No data returned. Check your date range or resolution.")
                return pl.DataFrame()

            print(f"Total records fetched: {len(all_data)}")

            # Data format: [timestamp, open, high, low, close]
            df_vol = pl.DataFrame(
                all_data,
                schema=["timestamp", "open", "high", "low", "close"],
                strict=False
            )
            
            # Use close as the IV value
            df_vol = df_vol.select([
                pl.col("timestamp"),
                pl.col("close").alias("iv")
            ])

            # Convert timestamp to datetime (UTC)
            df_vol = df_vol.with_columns(
                pl.col("timestamp").cast(pl.Datetime("ms"))
            )

            # Convert UTC → EST
            if self.convert_tz is not None:
                df_vol = df_vol.with_columns(
                    pl.col("timestamp")
                    .dt.convert_time_zone(self.convert_tz)
                    .dt.replace_time_zone(None)
                )
            else:
                df_vol = df_vol.with_columns(
                    pl.col("timestamp")
                    .dt.replace_time_zone(None)
                )

            # Remove duplicates that might occur at chunk boundaries
            df_vol = df_vol.unique(subset=["timestamp"])

            return df_vol.sort("timestamp").to_pandas()
        
    def aggregate_vol(self, df_vol_klines, period, drop_incomplete=True):

        if period == '1H':
            return df_vol_klines[['Close time', 'realized_vol']].set_index('Close time')
        ann_factor = 1 #np.sqrt(365 * 24)
        agg_vol = df_vol_klines.set_index('Close time').rolling(window=period)["log_return"].std(ddof=0).reset_index().rename(columns={'log_return': 'realized_vol'}).set_index('Close time') * ann_factor
        
        if drop_incomplete:
            agg_vol.loc[:agg_vol.index[0] + pd.Timedelta(period)] = np.nan
            agg_vol = agg_vol.dropna()

        return agg_vol

class TradeDataMulti:
    def __init__(self, symbols, sources, cm_um='um'):
        if len(symbols) != len(sources):
            raise ValueError("symbols and sources must be same length (one-to-one pairing).")

        self.pairs: List[Tuple[str, str]] = list(zip(symbols, sources))
        self.convert_tz = None
        self.cm_um = cm_um
        self.trades: Dict[Tuple[str, str, str], pl.DataFrame] = {}

    def get_data_once_optimized(self, source, url):
        """Synchronous optimized version with better error handling"""
        def looks_numeric(s: str) -> bool:
            try:
                float(s)
                return True
            except ValueError:
                return False

        try:
            # Use session for connection reuse
            with requests.Session() as session:
                session.headers.update({'User-Agent': 'Mozilla/5.0'})
                response = session.get(url, timeout=30)
                if response.status_code == 200 and response.content[:2] == b'PK':
                    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                        with z.open(z.namelist()[0]) as f:
                            # Read and process more efficiently
                            first_line = f.readline().decode('utf-8').strip()
                            f.seek(0)
                            
                            tokens = first_line.split(',')
                            has_any_number = any(looks_numeric(t) for t in tokens)

                            if not has_any_number:
                                df = pl.read_csv(f)
                                if source == 'Kucoin':
                                    df.columns = ['id','time','price','qty','side']
                                elif source == 'OKX':
                                    df.columns = ['symbol', 'id', 'side', 'price', 'qty', 'time']
                            else:
                                if source == 'Binance':
                                    columns = ['id','price','qty','base_qty','time','is_buyer_maker','idk'] if len(tokens) == 7 else ['id','price','qty','base_qty','time','is_buyer_maker']
                                elif source == 'Kucoin':
                                    columns = ['id','time','price','qty','side']
                                elif source == 'OKX':
                                    df.columns = ['symbol', 'id', 'side', 'price', 'qty', 'time']
                                df = pl.read_csv(f, has_header=False, new_columns=columns)
                            
                            # Optimized processing using Polars expressions
                            if 'is_buyer_maker' in df.columns:
                                df = df.with_columns([
                                    pl.col('is_buyer_maker').alias('is_bid'),
                                    pl.when(pl.col('time') < 1e11)
                                    .then(pl.col('time') * 1000)
                                    .when(pl.col('time') < 1e14)
                                    .then(pl.col('time'))
                                    .otherwise(pl.col('time') / 1000)
                                    .cast(pl.Datetime(time_unit='ms'))
                                    .alias('timestamp')
                                ])
                            elif 'side' in df.columns:
                                df = df.with_columns([
                                    (pl.col('side').str.to_lowercase()=='sell').alias('is_bid'),
                                    pl.when(pl.col('time') < 1e11)
                                    .then(pl.col('time') * 1000)
                                    .when(pl.col('time') < 1e14)
                                    .then(pl.col('time'))
                                    .otherwise(pl.col('time') / 1000)
                                    .cast(pl.Datetime(time_unit='ms'))
                                    .alias('timestamp')
                                ])
                            
                            # Efficient aggregation
                            df = df.with_columns(pl.col('timestamp').dt.truncate('1ms'))
                            df = df.unique(subset=['timestamp', 'is_bid']).group_by(['timestamp', 'is_bid']).agg(pl.col('price').last())
                            return df.select(['price', 'is_bid', 'timestamp'])
                            
        except Exception as e:
            print(f"Error processing {url}: {e}")
            return None


    def get_all_data_optimized(self, symbol, source, end_date, days=30, kind='spot', n_jobs=8):
        #dates_files = self.get_binance_data_optimized(end_date, days, kind=kind)
        if kind == 'perp':
            if source == 'Binance':
                if self.cm_um == 'cm':
                    symbol = symbol.replace('USDT', 'USD_PERP')
                urls = [f"https://data.binance.vision/data/futures/{self.cm_um}/daily/trades/{symbol}/{symbol}-trades-{datetime.datetime.strftime(end_date-datetime.timedelta(days=i), '%Y-%m-%d')}.zip" for i in range(days)]
            elif source == 'Kucoin':
                urls = [f"https://historical-data.kucoin.com/data/futures/daily/trades/{symbol}M/{symbol}M-trades-{datetime.datetime.strftime(end_date-datetime.timedelta(days=i), '%Y-%m-%d')}.zip" for i in range(days)] 
            elif source == 'OKX':
                urls = [f"https://static.okx.com/cdn/okex/traderecords/trades/daily/{datetime.datetime.strftime(end_date-datetime.timedelta(days=i),'%Y%m%d')}/{symbol}-SWAP-trades-{datetime.datetime.strftime(end_date-datetime.timedelta(days=i), '%Y-%m-%d')}.zip" for i in range(-1,days+1)] 
        else:
            if source == 'Binance':
                urls = [f"https://data.binance.vision/data/spot/daily/trades/{symbol}/{symbol}-trades-{datetime.datetime.strftime(end_date-datetime.timedelta(days=i), '%Y-%m-%d')}.zip" for i in range(days)]
            elif source == 'Kucoin':
                urls = [f"https://historical-data.kucoin.com/data/spot/daily/trades/{symbol}/{symbol}-trades-{datetime.datetime.strftime(end_date-datetime.timedelta(days=i), '%Y-%m-%d')}.zip" for i in range(days)] 
            elif source == 'OKX':
                urls = [f"https://static.okx.com/cdn/okex/traderecords/trades/daily/{datetime.datetime.strftime(end_date-datetime.timedelta(days=i),'%Y%m%d')}/{symbol}-trades-{datetime.datetime.strftime(end_date-datetime.timedelta(days=i), '%Y-%m-%d')}.zip" for i in range(-1,days+1)]

        print(symbol)
        print(f"Processing {len(urls)} files...")
        
        # Process in smaller batches to manage memory
        batch_size = min(20, len(urls))
        all_results = []
        
        for i in range(0, len(urls), batch_size):
            batch_urls = urls[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(urls)-1)//batch_size + 1}")
            
            # Use ThreadPoolExecutor for better control
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                batch_results = list(
                    executor.map(lambda url: self.get_data_once_optimized(source, url), batch_urls)
                )
            # Filter valid results
            valid_batch = [r for r in batch_results if r is not None]
            all_results.extend(valid_batch)
        
        if all_results:
            print("Combining results...")
            df_combined = pl.concat(all_results)
            df_combined = df_combined.with_columns(pl.lit(kind).alias('type'))
            df_combined = df_combined.with_columns(pl.col("timestamp").is_between(end_date - datetime.timedelta(days=days), end_date).alias('is_between'))
            df_combined = df_combined.filter(pl.col("is_between") == True).drop("is_between")
            return df_combined.sort('timestamp')
        else:
            print("No valid data retrieved")
            return pl.DataFrame()

    def grab_trades_data(self, end_date: datetime.datetime, days: int = 30, n_jobs: int = 10):
        """
        Fetch spot+perp for each (symbol, source) pair.
        """
        for symbol, source in self.pairs:
            for kind in ("spot", "perp"):
                print(f"Fetching {symbol} {source} {kind} ...")
                df = self.get_all_data_optimized(
                    symbol=symbol,
                    source=source,
                    end_date=end_date,
                    days=days,
                    kind=kind,
                    n_jobs=n_jobs
                )
                self.trades[(symbol, source, kind)] = df

        
    def to_intervals_bidask(self, df, freq='1s', fill_gaps=False):
        """
        Resample trade data into time intervals (default: 1 second):
        - Takes the last price for bids and asks separately
        - Creates columns 'bid_price' and 'ask_price'
        - Optionally forward-fills missing intervals
        - 'freq' can be any valid Polars duration string ('500ms', '1s', '5s', etc.)
        
        Parameters
        ----------
        fill_gaps : bool, default False
            If True, creates continuous timeline and forward-fills (slower but complete).
            If False, returns only observed intervals (2-5x faster).
        """

        # Ensure timestamp is a Polars Datetime
        df = df.with_columns(pl.col("timestamp").cast(pl.Datetime))

        # Truncate timestamps to interval bins
        df = df.with_columns(pl.col("timestamp").dt.truncate(freq).alias("timestamp_bin"))

        # Aggregate by side and time bin (take last price per side)
        df_agg = (
            df
            .group_by(["timestamp_bin", "is_bid"], maintain_order=True)
            .agg(pl.col("price").last().alias("price"))
            .pivot(
                values="price",
                index="timestamp_bin",
                columns="is_bid"
            )
            .rename({
                "true": "bid_price",
                "false": "ask_price"
            })
            .sort("timestamp_bin")
        )

        if not fill_gaps:
            return df_agg

        # Build a continuous timeline (expensive)
        full_range = pl.DataFrame({
            "timestamp_bin": pl.datetime_range(
                start=df_agg["timestamp_bin"].min(),
                end=df_agg["timestamp_bin"].max(),
                interval=freq,
                eager=True
            )
        })

        # Join and forward-fill missing intervals
        df_filled = (
            full_range
            .join(df_agg, on="timestamp_bin", how="left")
            .fill_null(strategy="forward")
        )

        return df_filled

    def agg_to_intervals(
        self,
        freq: str = "1s",
        start=None,
        end=None,
        join: str = "inner",          # "inner" keeps only common timestamps; "outer" keeps union
        include_bidask: bool = False, # if True, also keep bid/ask columns per source
    ) -> pd.DataFrame:
        """
        Wide dataframe with a column for each spot/perp from each source.

        Output columns (default):
          midpoint_spot_{source}, midpoint_perp_{source}

        If include_bidask:
          bid_price_spot_{source}, ask_price_spot_{source},
          bid_price_perp_{source}, ask_price_perp_{source}
        """
        if start is not None:
            start = pd.Timestamp(start).to_pydatetime()
        if end is not None:
            end = pd.Timestamp(end).to_pydatetime()

        wide = None

        for symbol, source in self.pairs:
            spot = self.trades.get((symbol, source, "spot"), pl.DataFrame())
            perp = self.trades.get((symbol, source, "perp"), pl.DataFrame())

            if spot.is_empty() or perp.is_empty():
                # skip incomplete pair
                continue

            # filter early in polars
            if start is not None:
                spot = spot.filter(pl.col("timestamp") >= pl.lit(start))
                perp = perp.filter(pl.col("timestamp") >= pl.lit(start))
            if end is not None:
                spot = spot.filter(pl.col("timestamp") < pl.lit(end))
                perp = perp.filter(pl.col("timestamp") < pl.lit(end))

            if spot.height == 0 or perp.height == 0:
                continue

            # resample to bid/ask at freq
            spot_ba = self.to_intervals_bidask(spot, freq).to_pandas().set_index("timestamp_bin")
            perp_ba = self.to_intervals_bidask(perp, freq).to_pandas().set_index("timestamp_bin")

            # midpoints
            spot_mid = (spot_ba["ask_price"] + spot_ba["bid_price"]) / 2
            perp_mid = (perp_ba["ask_price"] + perp_ba["bid_price"]) / 2

            cols = {
                f"midpoint_spot_{source}": spot_mid,
                f"midpoint_perp_{source}": perp_mid,
            }

            if include_bidask:
                cols.update({
                    f"bid_price_spot_{source}": spot_ba["bid_price"],
                    f"ask_price_spot_{source}": spot_ba["ask_price"],
                    f"bid_price_perp_{source}": perp_ba["bid_price"],
                    f"ask_price_perp_{source}": perp_ba["ask_price"],
                })

            block = pd.DataFrame(cols, index=spot_ba.index.union(perp_ba.index))
            block = block.sort_index()
            block = block.ffill()

            if wide is None:
                wide = block
            else:
                wide = wide.join(block, how=join)

        if wide is None:
            return pd.DataFrame()

        return wide.sort_index().dropna()

class BinanceMetricsData:
    """
    Specialized class for pulling Binance Futures daily metrics data.
    Pulls from: https://data.binance.vision/?prefix=data/futures/um/daily/metrics/
    """

    def __init__(self, symbol: str, cm_um: str = 'um'):
        """
        Args:
            symbol: e.g., 'BTCUSDT'
            cm_um:  'um' for USDT‑M futures, 'cm' for COIN‑M futures
        """
        self.symbol = symbol
        self.cm_um = cm_um
        self.base_url = f"https://data.binance.vision/data/futures/{cm_um}/daily/metrics"
        self.convert_tz = None

    def get_metric_files(self,
                         start_date: datetime.datetime,
                         end_date: datetime.datetime) -> list:

        if isinstance(start_date, str):
            start_date = datetime.datetime.strptime(start_date, "%Y%m%d")
        if isinstance(end_date, str):
            end_date = datetime.datetime.strptime(end_date, "%Y%m%d")

        dates = pd.date_range(start=start_date.date(),
                              end=end_date.date(),
                              freq="D")
        urls: list = []
        for date in dates:
            ds = date.strftime("%Y-%m-%d")
            url = f"{self.base_url}/{self.symbol}/{self.symbol}-metrics-{ds}.zip"
            urls.append((url, date))
        return urls

    def fetch_metric_file(self, url: str) -> pl.DataFrame:
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200 and response.content[:2] == b'PK':
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    with z.open(z.namelist()[0]) as f:
                        df = pl.read_csv(f)
                return df
            else:
                print(f"Failed to fetch {url}: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    def grab_metrics_data(self,
                          start_date: datetime.datetime,
                          end_date: datetime.datetime,
                          n_jobs: int = 8) -> pl.DataFrame:

        urls = self.get_metric_files(start_date, end_date)
        print(f"Fetching {len(urls)} metric files for {self.symbol}...")

        all_results = []
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(self.fetch_metric_file, url): (url, date)
                for url, date in urls
            }
            for future in as_completed(futures):
                url, date = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        all_results.append(result)
                        print(f"✓ Fetched {date.strftime('%Y-%m-%d')}")
                except Exception as e:
                    print(f"✗ Failed {date.strftime('%Y-%m-%d')}: {e}")

        if not all_results:
            print(f"No metric data retrieved for {self.symbol}")
            return pl.DataFrame()

        df_combined = pl.concat(all_results, how='diagonal')
        df_combined = df_combined.sort('create_time')
        print(f"Successfully loaded {len(df_combined)} metric records")
        return df_combined

    def process_metrics(self, df: pl.DataFrame) -> pd.DataFrame:
        # Convert to pandas for easier manipulation
        df_pd = df.to_pandas().drop('symbol', axis=1)
        
        df_pd['create_time'] = pd.to_datetime(df_pd['create_time'], errors='coerce')
        
        # Convert numeric columns
        numeric_cols = [col for col in df_pd.columns if col!='create_time']
        for col in numeric_cols:
            try:
                df_pd[col] = pd.to_numeric(df_pd[col], errors='coerce')
            except:
                pass        
        return df_pd.sort_values('create_time').reset_index(drop=True)

    def get_all_metrics(self,
                        start_date: datetime.datetime,
                        end_date: datetime.datetime,
                        n_jobs: int = 8,
                        process: bool = True) -> pl.DataFrame:
        
        df = self.grab_metrics_data(start_date, end_date, n_jobs)
        if process and not df.is_empty():
            return self.process_metrics(df)
        return df
