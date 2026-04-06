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
import re
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

import os
import time
from pull_binance_data import *
    
from typing import Dict, Tuple, Optional, List

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
                            
                            # Efficient aggregation
                            df = df.with_columns(pl.col('timestamp').dt.truncate('1ms'))
                            df = df.unique(subset=['timestamp', 'is_bid']).group_by(['timestamp', 'is_bid']).agg(pl.col('price').last())
                            
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
        
    def to_intervals_bidask(self, df, freq='1s'):
        """
        Resample trade data into time intervals (default: 1 second):
        - Takes the last price for bids and asks separately
        - Creates columns 'bid_price' and 'ask_price'
        - Forward-fills missing intervals
        - 'freq' can be any valid Polars duration string ('500ms', '1s', '5s', etc.)
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

        # Build a continuous timeline
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
    
    def grab_trades_data(self, end_date, days=30, n_jobs=10):
        df_trades_spots = self.get_all_data_optimized(end_date=end_date, days=days, kind='spot', n_jobs=n_jobs)
        df_trades_perps = self.get_all_data_optimized(end_date=end_date, days=days, kind='perp', n_jobs=n_jobs)
        self.df_trades_spots = df_trades_spots
        self.df_trades_perps = df_trades_perps
        return

    def agg_to_intervals(self, freq='1s', start=None, end=None):
        """
        Aggregate trades to bid/ask midpoints at `freq`, optionally for only [start, end).

        start/end can be:
        - pandas Timestamp / datetime
        - string parseable by pd.Timestamp
        - None (use full range)
        """
        # --- parse times ---
        if start is not None:
            start = pd.Timestamp(start).to_pydatetime()
        if end is not None:
            end = pd.Timestamp(end).to_pydatetime()

        # --- filter raw trades early (Polars) ---
        spots = self.df_trades_spots
        perps = self.df_trades_perps

        if start is not None:
            spots = spots.filter(pl.col("timestamp") >= pl.lit(start))
            perps = perps.filter(pl.col("timestamp") >= pl.lit(start))
        if end is not None:
            spots = spots.filter(pl.col("timestamp") < pl.lit(end))
            perps = perps.filter(pl.col("timestamp") < pl.lit(end))

        # if empty, return empty df early
        if spots.height == 0 or perps.height == 0:
            return pd.DataFrame()

        # --- aggregate to bid/ask intervals ---
        spots_bidask = self.to_intervals_bidask(spots, freq).to_pandas().set_index('timestamp_bin')
        perps_bidask = self.to_intervals_bidask(perps, freq).to_pandas().set_index('timestamp_bin')

        # --- merge + midpoint + first diff ---
        bidask = (
            spots_bidask
            .merge(perps_bidask, how='inner', left_index=True, right_index=True, suffixes=['_spot', '_perp'])
            .dropna()
        )

        bidask['midpoint_spot'] = (bidask['ask_price_spot'] + bidask['bid_price_spot']) / 2
        bidask['midpoint_perp'] = (bidask['ask_price_perp'] + bidask['bid_price_perp']) / 2

        bidask_diff = find_first_diff(bidask[['midpoint_spot', 'midpoint_perp']]).dropna()
        return bidask_diff

    
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

        
    def to_intervals_bidask(self, df, freq='1s'):
        """
        Resample trade data into time intervals (default: 1 second):
        - Takes the last price for bids and asks separately
        - Creates columns 'bid_price' and 'ask_price'
        - Forward-fills missing intervals
        - 'freq' can be any valid Polars duration string ('500ms', '1s', '5s', etc.)
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

        # Build a continuous timeline
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