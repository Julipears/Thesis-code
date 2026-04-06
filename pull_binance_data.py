"""
This file was originally for pulling data from Binance and processing them, 
but I ended up writing a new version in trade_data_pull.py.

However, the new file imports this file and I didn't want to screw up any dependencies.
"""
import pandas as pd
import numpy as np
import math
import string
import requests
import datetime
from matplotlib import pyplot as plt

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from statsmodels.tsa.stattools import grangercausalitytests
import statsmodels.api as sm

import polars as pl

import warnings
warnings.filterwarnings('ignore')

import seaborn as sns
import re
import zipfile
import io
import pickle
import time
import os

import multiprocessing as mp
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import aiohttp
import pytz


class BinanceData:
    """
    This class is only for pulling data for the specific ticker
    """
    def __init__(self, symbol, convert_tz=None):
        self.symbol = symbol
        self.convert_tz = convert_tz

    def grab_trade_data(self, end, days=30, kind='spot', n_jobs=8):
        def get_binance_data_optimized(end_date, days=30):
            """Optimized version with caching and faster parsing"""
            
            # Cache file list to avoid repeated web scraping
            cache_file = f"binance_files_cache_{end_date.strftime('%Y%m%d')}_{days}days.pkl"
            
            if os.path.exists(cache_file) and (time.time() - os.path.getmtime(cache_file)) < 3600:  # 1 hour cache
                print("Using cached file list...")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            
            print("Fetching file list from Binance...")
            url = 'https://data.binance.vision/?prefix=data/futures/cm/daily/trades/BTCUSD_PERP/'
            
            # Use headless browser for faster execution
            options = webdriver.EdgeOptions()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            
            browser = webdriver.Edge(options=options)
            
            try:
                browser.get(url)
                WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.XPATH, "/html/body/div/table/tbody/tr[1]/td[1]/a")))
                
                files = browser.execute_script('return jQuery(\'a:not(:contains("CHECKSUM")):contains(".zip")\')')
                files = [f.text for f in files]
                
            finally:
                browser.quit()

            # Optimized date parsing using vectorized operations
            date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}')
            matches = [date_pattern.search(x) for x in files]
            
            # Filter and create DataFrame more efficiently
            valid_files = []
            target_start = end_date - datetime.timedelta(days=days)
            
            for match, file in zip(matches, files):
                if match:
                    file_date = datetime.datetime.strptime(match.group(), '%Y-%m-%d')
                    if target_start <= file_date <= end_date:
                        valid_files.append((file_date, file))
            
            dates_files = pd.DataFrame(valid_files, columns=['date', 'file'])
            
            # Cache the results
            with open(cache_file, 'wb') as f:
                pickle.dump(dates_files, f)
            
            return dates_files
        
        def get_data_once_optimized(url):
            """Synchronous optimized version with better error handling"""
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
                                    df_vol = pl.read_csv(f)
                                else:
                                    columns = ['id','price','qty','base_qty','time','is_buyer_maker','idk'] if len(tokens) == 7 else ['id','price','qty','base_qty','time','is_buyer_maker']
                                    df_vol = pl.read_csv(f, has_header=False, new_columns=columns)
                                
                                # Optimized processing using Polars expressions
                                df_vol = df_vol.with_columns([
                                    pl.col('is_buyer_maker').alias('is_bid'), pl.col('qty'),
                                    pl.when(pl.col('time') < 1e11)
                                    .then(pl.col('time') * 1000)
                                    .when(pl.col('time') < 1e14)
                                    .then(pl.col('time'))
                                    .otherwise(pl.col('time') / 1000)
                                    .cast(pl.Datetime(time_unit='ms'))
                                    .alias('timestamp')
                                ])

                                if self.convert_tz is not None:
                                    df_vol = df_vol.with_columns([
                                        pl.col('timestamp')
                                        .dt.convert_time_zone(convert_tz)
                                        .dt.replace_time_zone(None)
                                        .alias('timestamp')
                                    ])
                                
                                # Efficient aggregation
                                df_vol = df_vol.with_columns(pl.col('timestamp').dt.truncate('1ms'))
                                df_vol = df_vol.unique(subset=['timestamp', 'is_bid']).group_by(['timestamp', 'is_bid']).agg(pl.col('price').last(), pl.col('qty').sum())
                                
                                return df_vol.select(['price', 'is_bid', 'qty', 'timestamp'])
                                
            except Exception as e:
                print(f"Error processing {url}: {e}")
                return None
            
        dates_files = get_binance_data_optimized(end, days)
        
        if len(dates_files) == 0:
            print("No files found for the specified date range")
            return pl.DataFrame()
        
        perp_base = "https://data.binance.vision/data/futures/cm/daily/trades/BTCUSD_PERP/"
        spot_base = "https://data.binance.vision/data/spot/daily/trades/BTCUSDT/"
        
        # Prepare URLs
        if kind == 'perp':
            urls = [perp_base + file for file in dates_files['file'].values]
        else:
            urls = [spot_base + file.replace('BTCUSD_PERP', 'BTCUSDT') for file in dates_files['file'].values]
        
        print(f"Processing {len(urls)} files...")
        
        # Process in smaller batches to manage memory
        batch_size = min(20, len(urls))
        all_results = []
        
        for i in range(0, len(urls), batch_size):
            batch_urls = urls[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(urls)-1)//batch_size + 1}")
            
            # Use ThreadPoolExecutor for better control
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                batch_results = list(executor.map(get_data_once_optimized, batch_urls))
            
            # Filter valid results
            valid_batch = [r for r in batch_results if r is not None]
            all_results.extend(valid_batch)
        
        if all_results:
            print("Combining results...")
            df_vol_combined = pl.concat(all_results)
            df_vol_combined = df_vol_combined.with_columns(pl.lit(kind).alias('type'))
            return df_vol_combined.sort('timestamp')
        else:
            print("No valid data retrieved")
            return pl.DataFrame()

    def grab_funding_fees(self, start, end):
        start_dt = datetime.datetime.strptime(start, '%Y%m%d')
        end_dt = datetime.datetime.strptime(end, '%Y%m%d')

        num_entries = 3 * (end_dt - start_dt).days
        frs = []

        if num_entries > 1000:
            init_day = end_dt
            while init_day > start_dt:
                binance_starttime = str(int(init_day.timestamp() * 1000))
                r = requests.get(
                    f"https://fapi.binance.com/fapi/v1/fundingRate",
                    params={"symbol": self.symbol, "startTime": binance_starttime, "limit": 1000}
                )
                fr = pd.DataFrame(r.json())
                frs.append(fr)
                init_day -= datetime.timedelta(days=333)
        else:
            binance_starttime = str(int(start_dt.timestamp() * 1000))
            r = requests.get(
                f"https://fapi.binance.com/fapi/v1/fundingRate",
                params={"symbol": self.symbol, "startTime": binance_starttime, "limit": 1000}
            )
            fr = pd.DataFrame(r.json())
            frs.append(fr)

        # Combine and clean
        fr_binance = pd.concat(frs, ignore_index=True).drop_duplicates()

        # Convert columns
        fr_binance[['fundingRate', 'fundingTime']] = fr_binance[['fundingRate', 'fundingTime']].astype(float)

        if self.convert_tz is not None:
            fr_binance['fundingTime_utc'] = pd.to_datetime(fr_binance['fundingTime'], unit='ms', utc=True)
            fr_binance['fundingTime'] = fr_binance['fundingTime_utc'].dt.tz_convert(self.convert_tz).dt.tz_localize(None)
        else:
            fr_binance['fundingTime'] = fr_binance['fundingTime'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000).replace(second=0, microsecond=0))
        fr_binance = fr_binance[(fr_binance['fundingTime'] >= start_dt) & (fr_binance['fundingTime'] <= end_dt)]

        # Sort and set index
        fr_binance = fr_binance.sort_values('fundingTime').reset_index(drop=True)

        return fr_binance
    
    def get_iv(self, source, start, end, instr='BTC', resolution=3600):
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
    
def clean_ff(funding_fees):
    funding_fees['fundingTime'] = funding_fees['fundingTime'].apply(lambda x: x.replace(second=0, microsecond=0, minute=0, hour=x.hour))
    funding_fees = funding_fees.sort_values('fundingTime')
    return funding_fees

def combine_fees_vol(funding_fees, df_vol_klines):
    merged_df_vol = funding_fees.merge(df_vol_klines, how='inner', left_on='fundingTime', right_on='Open time')
    # fundingTime -> end of period, but we want the value at exactly that time, which is Open time
    merged_df_vol['fundingFee'] = merged_df_vol['fundingRate'] * merged_df_vol['Open']
    return merged_df_vol

def aggregate_ff(funding_fees, period, drop_incomplete=True):
    """
    Original funding fee times refer to the amount paid out at that time, aka 
    the funding fee listed for t=1 refers to the aggregated amount over t=0->1
    
    After aggregating, the new timestamp should be the timestamp of the last row, or 
    the "ending time"
    """
    if period == '8H':
        return funding_fees[['fundingTime', 'fundingFee']].set_index('fundingTime')
    
    # index of rolling gives the index of the ending row
    agg_ff = funding_fees.set_index('fundingTime').rolling(window=period)['fundingFee'].sum()
    
    if drop_incomplete:
        agg_ff.loc[:agg_ff.index[0] + pd.Timedelta(period)] = np.nan
        agg_ff = agg_ff.dropna()

    return agg_ff

def aggregate_vol(df_vol_klines, period, drop_incomplete=True):

    if period == '1H':
        return df_vol_klines[['Close time', 'realized_vol']].set_index('Close time')
    ann_factor = 1 #np.sqrt(365 * 24)
    agg_vol = df_vol_klines.set_index('Close time').rolling(window=period)["log_return"].std(ddof=0).reset_index().rename(columns={'log_return': 'realized_vol'}).set_index('Close time') * ann_factor
    
    if drop_incomplete:
        agg_vol.loc[:agg_vol.index[0] + pd.Timedelta(period)] = np.nan
        agg_vol = agg_vol.dropna()

    return agg_vol

def aggregate_max_drawdown(df_vol_klines, period, price_col='Close', drop_incomplete=True):
    """
    Rolling max drawdown over a time-based window `period` (e.g., '8H', '1D').
    Returns a DataFrame indexed by 'Close time' with a single column 'max_drawdown'
    expressed as a positive fraction (e.g., 0.25 for -25%).
    """
    # Work on price series indexed by time
    s = df_vol_klines.set_index('Close time')[price_col].astype(float).sort_index()

    # For a 1-step window the drawdown is always 0
    if period == '1H':
        mdd = s.copy() * 0.0
        mdd = mdd.to_frame(name='max_drawdown')
        return mdd

    # Rolling apply: compute max drawdown within each window
    def _max_dd_np(a):
        peak = a[0]
        max_dd = 0.0  # positive number, e.g. 0.25 for a -25% drop
        for x in a:
            if x > peak:
                peak = x
            # drawdown from current peak
            dd = (peak - x) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return float(max_dd)

    mdd = s.rolling(window=period).apply(lambda x: _max_dd_np(x.values), raw=False)
    mdd = mdd.to_frame(name='max_drawdown')

    if drop_incomplete:
        # drop the initial incomplete window range to mimic your other helpers
        mdd.loc[:mdd.index[0] + pd.Timedelta(period)] = np.nan
        mdd = mdd.dropna()

    return mdd

def aggregate_momentum(df_vol_klines, lookback_days=252, price_col='Close', drop_incomplete=True, log=False):
    """
    Computes daily momentum factor using intraday price data.
    Momentum_t = (P_t / P_{t-L}) - 1, where L = lookback_days.

    Parameters
    ----------
    df_vol_klines : pd.DataFrame
        DataFrame containing 'Close time' and price columns (e.g. hourly or 30min data).
    lookback_days : int
        Lookback period in days (default 252 ~ 12 months).
    price_col : str
        Name of the price column (default 'Close').
    drop_incomplete : bool
        Whether to drop incomplete lookback windows.
    log : bool
        If True, compute log(P_t / P_{t-L}) instead of percentage change.

    Returns
    -------
    pd.DataFrame
        Daily time series with one column 'momentum', indexed by day end.
    """
    # Ensure time is datetime and sorted
    df = df_vol_klines.copy()
    df['Close time'] = pd.to_datetime(df['Close time'])
    df = df.sort_values('Close time')

    # Resample to daily close price
    daily_prices = df.set_index('Close time')[price_col].resample('1D').last()

    # Compute momentum over lookback_days
    if log:
        momentum = np.log(daily_prices / daily_prices.shift(lookback_days))
    else:
        momentum = daily_prices / daily_prices.shift(lookback_days) - 1

    mom_df = momentum.to_frame(name='momentum')

    if drop_incomplete:
        mom_df = mom_df.dropna()

    return mom_df



def vwap_price(df_klines, period, kind='typical', start=None, rolling=False, drop_incomplete=True):
    agg_price = df_klines.set_index('Close time')
    
    # Calculate average price
    if kind == 'typical':
        agg_price['Avg'] = agg_price[['High', 'Low', 'Close']].sum(1)/3
    else:
        agg_price['Avg'] = agg_price[kind]
    
    agg_price['Avg*Volume'] = agg_price['Avg'] * agg_price['Volume']
    
    if rolling:
        # Rolling VWAP: calculate over past 'period' window at each timestamp
        agg_price = agg_price.sort_index()
        
        # Rolling sum of Avg*Volume and Volume
        rolling_avg_vol = agg_price['Avg*Volume'].rolling(period).sum()
        rolling_vol = agg_price['Volume'].rolling(period).sum()
        
        # Calculate rolling VWAP
        agg_price['vwap'] = rolling_avg_vol / rolling_vol
        agg_price['volume'] = rolling_vol
        
        # Keep only vwap and volume columns
        agg_price = agg_price[['vwap', 'volume']]
        
    else:
        # Original resampling logic (non-overlapping periods)
        if start is not None:
            agg_price = agg_price.sort_index().resample(period, origin=start)[['Avg*Volume', 'Volume']].sum()
        else:
            agg_price = agg_price.sort_index().resample(period)[['Avg*Volume', 'Volume']].sum()
        
        agg_price['vwap'] = agg_price['Avg*Volume'] / agg_price['Volume']
        agg_price = agg_price.rename(columns={'Volume': 'volume'}).drop('Avg*Volume', axis=1)

    if drop_incomplete:
        if rolling:
            # For rolling, drop rows where window isn't complete
            agg_price = agg_price.iloc[pd.Timedelta(period) // agg_price.index.to_series().diff().median():]
        else:
            # Original logic for resampled data
            agg_price.loc[:agg_price.index[0] + pd.Timedelta(period)] = np.nan
            agg_price = agg_price.dropna()

    return agg_price

def calc_basis(perp_df, spot_df):
    
    df = perp_df.merge(spot_df, how='inner', left_index=True, right_index=True, suffixes=['_perp', '_spot'])
    
    # Calculate basis
    df['basis'] = (df['vwap_perp'] - df['vwap_spot']) / df['vwap_spot'] * 100
    df['basis_bps'] = df['basis'] * 100  # basis points
    df['basis_abs'] = df['basis'].abs()
    
    # Calculate returns
    df['perp_return'] = df['vwap_perp'].pct_change()
    df['spot_return'] = df['vwap_spot'].pct_change()
    
    return df.dropna()

def calc_periods(funding_fees, df_vol_klines, periods=[], drop_incomplete=True):
    period_dfs = {}
    for p in periods:
        df_vol_agg = aggregate_vol(df_vol_klines, p, drop_incomplete)
        df_ff_agg = aggregate_ff(funding_fees, p, drop_incomplete)
        df_agg_combined = pd.DataFrame(df_ff_agg).merge(pd.DataFrame(df_vol_agg), how='left', left_index=True, right_index=True)
        period_dfs[p] = df_agg_combined
    return period_dfs

def smooth_data(df_full, periods=[]):
    '''
    df must have a datetime index
    '''
    df = df_full.copy()
    for c in df.columns:
        for p in periods:
            df[c+'_ewma_' + p] = df[c].ewm(halflife=p, times=df.index).mean()
    return df

def add_transformations(df_1, lags=[], means=[], stds=[]):
    df = df_1.copy()
    targets = ['realized_vol', 'log_vol'] + [f'vol_lag{str(l)}' for l in lags] + [f'log_vol_lag{str(l)}' for l in lags] + ['norm_vol']
    features = ['fundingFee', 'funding_abs'] + [f'funding_lag{str(l)}' for l in lags] + ['funding_squared', 'funding_cubed'] + [f'funding_roll_mean{str(m)}' for m in means] + [f'funding_roll_std{str(s)}' for s in stds]

    # --- Preprocessing ---
    # volatility transformations
    df['log_vol'] = np.log(df['realized_vol']) # log values to remove right skew
    # df['delta_log_vol'] = df['log_vol'].diff() # change in volatility between levels might have an effect
    for l in lags:
        df[f'vol_lag{str(l)}'] = df['realized_vol'].shift(l) # to see if volatility preceeds ff
        df[f'log_vol_lag{str(l)}'] = df['log_vol'].shift(l)

    # normalized values as comparison to a baseline
    df['norm_vol'] = df['realized_vol'].sub(df['realized_vol'].mean()).div(df['realized_vol'].std())
    
    df['funding_abs'] = df['fundingFee'].abs() # ff can be positive or negative, so here we take the magnitude
    
    for l in lags:
        df[f'funding_lag{str(l)}'] = df['fundingFee'].shift(l) # see if funding fee preceeds volatility

    df['funding_squared'] = df['fundingFee'] ** 2 # for gauging effect of outliers
    df['funding_cubed'] = df['fundingFee'] ** 3 # same, except preserves sign
    
    for m in means:
        df[f'funding_roll_mean{str(m)}'] = df['fundingFee'].rolling(m).mean() # smooths changes in ff
    
    for s in stds:
        df[f'funding_roll_std{str(s)}'] = df['fundingFee'].rolling(s).std() # finds changes in ff over past 3 periods

    df.dropna(inplace=True)

    results = []

    # --- Compare against both realized_vol and log_vol ---
    for target in targets:
        for feat in features:
            corr = df[target].corr(df[feat])

            X = sm.add_constant(df[feat])
            y = df[target]
            model = sm.OLS(y, X).fit()
            r2 = model.rsquared

            results.append({
                'target': target,
                'feature': feat,
                'corr': corr,
                'R_squared': r2
            })

    results_df = pd.DataFrame(results)

    # Pivot for easier comparison
    results_df = results_df.pivot(index='feature', columns='target', values=['corr', 'R_squared'])
    results_df.columns = [f"{a}_{b}" for a, b in results_df.columns]
    results_df = results_df.reset_index().sort_values(by='R_squared_realized_vol', ascending=False)

    # --- Optional rolling correlation ---
    rolling_corr = None
    return results_df, rolling_corr, df

def granger_causality_matrix(df, p_threshold=0.05, maxlag=4, verbose=False):
    """
    Compute pairwise Granger causality p-values for all combinations of columns in df.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data (each column is a variable).
    maxlag : int
        Maximum lag to test in Granger causality.
    verbose : bool
        Whether to print test details.

    Returns
    -------
    pd.DataFrame
        DataFrame of p-values where entry (i, j) is the min p-value for
        testing whether column j Granger-causes column i.
    """
    variables = df.columns
    n = len(variables)
    result = pd.DataFrame(np.ones((n, n)), columns=variables, index=variables)
    records = []

    for cause in variables:
        for effect in variables:
            if (('vol' in cause and 'vol' in effect) or ('funding' in cause and 'funding' in effect)):
                result.loc[effect, cause] = np.nan
                continue
            sub = df[[effect, cause]].dropna()
            if len(sub) < maxlag + 1:
                continue
            try:
                test_result = grangercausalitytests(sub, maxlag=maxlag, verbose=verbose)
                p_values = [test_result[i + 1][0]['ssr_chi2test'][1] for i in range(maxlag)]
                result.loc[effect, cause] = min(p_values)
                min_p = min(p_values)
                if min_p < p_threshold:
                    records.append({
                        "cause": cause,
                        "effect": effect,
                        "p_value": min_p
                    })
            except Exception as e:
                result.loc[effect, cause] = np.nan
                if verbose:
                    print(f"Error testing {cause} → {effect}: {e}")

    long_table = pd.DataFrame(records).sort_values("p_value") if records else pd.DataFrame(columns=["cause", "effect", "p_value"])

    return result, long_table

def find_first_diff(df, log=True):
    df_copy = df.copy()
    for col in df.columns:
        if log:
            df_copy['log_'+col] = df_copy[col].apply(np.log)
            df_copy['dlog_'+col] = df_copy['log_'+col].diff()
        else:
            df_copy['d_'+col] = df_copy[col].diff()
    return df_copy