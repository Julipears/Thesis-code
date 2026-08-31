"""All external data access for the final survival-analysis pipeline.

This module contains raw trade, funding, kline, Fear & Greed, and Binance
futures-metrics retrieval. Market-data transformations live in
survival_analysis_data_processing_final.py.
"""

from __future__ import annotations

import datetime as dt
import io
import time
import zipfile
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import requests


SOURCE_METRIC_COLUMNS = {
    "sum_open_interest_value": "open_interest",
    "sum_toptrader_long_short_ratio": "trader_long_short",
    "sum_taker_long_short_vol_ratio": "taker_long_short",
}
OUTPUT_METRIC_COLUMNS = list(SOURCE_METRIC_COLUMNS.values())


class TradeData:
    """Raw Binance trade/funding/kline loader used by the final pipeline.

    The trade-download behavior intentionally follows the existing thesis
    TradeData implementation so shock/event inputs are not silently redefined.
    """

    def __init__(self, symbol: str, source: str = "Binance", cm_um: str = "um"):
        if source != "Binance":
            raise NotImplementedError("The final thesis pipeline is frozen to Binance data.")
        if cm_um not in {"um", "cm"}:
            raise ValueError("cm_um must be 'um' or 'cm'")
        self.symbol = symbol
        self.source = source
        self.cm_um = cm_um
        self.convert_tz = None
        self.df_trades_spots = pl.DataFrame()
        self.df_trades_perps = pl.DataFrame()

    def _perp_symbol(self) -> str:
        if self.cm_um == "cm":
            return self.symbol.replace("USDT", "USD_PERP")
        return self.symbol

    def get_funding_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Pull funding observations in the same format as the legacy loader."""
        symbol = self._perp_symbol()
        prefix = "dapi" if self.cm_um == "cm" else "fapi"
        start_dt = dt.datetime.strptime(start_date, "%Y%m%d")
        end_dt = dt.datetime.strptime(end_date, "%Y%m%d")
        num_entries = 3 * (end_dt - start_dt).days

        frames = []
        if num_entries > 1000:
            # Preserve the legacy pagination convention for compatibility.
            init_day = end_dt
            while init_day > start_dt:
                start_ms = str(int(init_day.timestamp() * 1000))
                response = requests.get(
                    f"https://{prefix}.binance.com/{prefix}/v1/fundingRate",
                    params={"symbol": symbol, "startTime": start_ms, "limit": 1000},
                    timeout=30,
                )
                response.raise_for_status()
                frames.append(pd.DataFrame(response.json()))
                init_day -= dt.timedelta(days=333)
        else:
            start_ms = str(int(start_dt.timestamp() * 1000))
            response = requests.get(
                f"https://{prefix}.binance.com/{prefix}/v1/fundingRate",
                params={"symbol": symbol, "startTime": start_ms, "limit": 1000},
                timeout=30,
            )
            response.raise_for_status()
            frames.append(pd.DataFrame(response.json()))

        nonempty = [frame for frame in frames if not frame.empty]
        if not nonempty:
            return pd.DataFrame(columns=["fundingRate", "fundingTime"])

        funding = pd.concat(nonempty, ignore_index=True).drop_duplicates()
        funding[["fundingRate", "fundingTime"]] = funding[["fundingRate", "fundingTime"]].astype(float)
        funding["fundingTime"] = funding["fundingTime"].apply(
            lambda x: dt.datetime.utcfromtimestamp(x / 1000).replace(second=0, microsecond=0)
        )
        funding = funding[
            (funding["fundingTime"] >= start_dt)
            & (funding["fundingTime"] <= end_dt)
        ]
        return funding.reset_index(drop=True)

    @staticmethod
    def _looks_numeric(value: str) -> bool:
        try:
            float(value)
            return True
        except ValueError:
            return False

    def get_data_once_optimized(self, url: str):
        """Download and parse one legacy daily trade ZIP."""
        try:
            with requests.Session() as session:
                session.headers.update({"User-Agent": "Mozilla/5.0"})
                response = session.get(url, timeout=30)
                if response.status_code != 200 or response.content[:2] != b"PK":
                    return None

                with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                    with archive.open(archive.namelist()[0]) as stream:
                        first_line = stream.readline().decode("utf-8").strip()
                        stream.seek(0)
                        tokens = first_line.split(",")
                        has_any_number = any(self._looks_numeric(token) for token in tokens)

                        if not has_any_number:
                            frame = pl.read_csv(stream)
                        else:
                            columns = (
                                ["id", "price", "qty", "base_qty", "time", "is_buyer_maker", "idk"]
                                if len(tokens) == 7
                                else ["id", "price", "qty", "base_qty", "time", "is_buyer_maker"]
                            )
                            frame = pl.read_csv(stream, has_header=False, new_columns=columns)

                frame = frame.with_columns([
                    pl.col("is_buyer_maker").alias("is_bid"),
                    pl.when(pl.col("time") < 1e11)
                    .then(pl.col("time") * 1000)
                    .when(pl.col("time") < 1e14)
                    .then(pl.col("time"))
                    .otherwise(pl.col("time") / 1000)
                    .cast(pl.Datetime(time_unit="ms"))
                    .alias("timestamp"),
                ])
                frame = frame.with_columns(pl.col("timestamp").dt.truncate("1ms"))
                if "id" in frame.columns:
                    frame = (
                        frame.sort(["timestamp", "is_bid", "id"], maintain_order=True)
                        .group_by(["timestamp", "is_bid"], maintain_order=True)
                        .agg(
                            pl.col("price").last(),
                            pl.col("id").last().alias("trade_id"),
                        )
                    )
                    return frame.select(["price", "is_bid", "timestamp", "trade_id"])
                frame = (
                    frame.sort(["timestamp", "is_bid", "price"], maintain_order=True)
                    .group_by(["timestamp", "is_bid"], maintain_order=True)
                    .agg(pl.col("price").last())
                )
                return frame.select(["price", "is_bid", "timestamp"])
        except Exception as exc:
            print(f"Error processing {url}: {exc}")
            return None

    def get_all_data_optimized(self, end_date: dt.datetime, days: int = 30, kind: str = "spot", n_jobs: int = 8):
        """Fetch the same daily files as the legacy TradeData workflow."""
        symbol = self.symbol
        if kind == "perp":
            if self.cm_um == "cm":
                symbol = symbol.replace("USDT", "USD_PERP")
            urls = [
                f"https://data.binance.vision/data/futures/{self.cm_um}/daily/trades/"
                f"{symbol}/{symbol}-trades-{(end_date - dt.timedelta(days=i)):%Y-%m-%d}.zip"
                for i in range(days)
            ]
        elif kind == "spot":
            urls = [
                f"https://data.binance.vision/data/spot/daily/trades/"
                f"{symbol}/{symbol}-trades-{(end_date - dt.timedelta(days=i)):%Y-%m-%d}.zip"
                for i in range(days)
            ]
        else:
            raise ValueError("kind must be 'spot' or 'perp'")

        print(f"Processing {len(urls)} {kind} files for {symbol}...")
        batch_size = min(20, len(urls)) if urls else 0
        all_results = []
        for start in range(0, len(urls), batch_size or 1):
            batch = urls[start:start + batch_size]
            print(f"Processing batch {start // batch_size + 1}/{(len(urls) - 1) // batch_size + 1}")
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                results = list(executor.map(self.get_data_once_optimized, batch))
            all_results.extend(result for result in results if result is not None)

        if not all_results:
            print(f"No valid {kind} data retrieved")
            return pl.DataFrame()

        combined = pl.concat(all_results).with_columns(pl.lit(kind).alias("type"))
        order = ["timestamp"]
        if "trade_id" in combined.columns:
            order.append("trade_id")
        return combined.sort(order, maintain_order=True)

    def grab_trades_data(self, end_date: dt.datetime, days: int = 30, n_jobs: int = 10) -> None:
        self.df_trades_spots = self.get_all_data_optimized(end_date, days, kind="spot", n_jobs=n_jobs)
        self.df_trades_perps = self.get_all_data_optimized(end_date, days, kind="perp", n_jobs=n_jobs)

    def get_klines(
        self,
        start_date: str,
        end_date: str,
        kind: str = "spot",
        interval: str = "1h",
        columns=None,
        n_jobs: int = 10,
    ) -> pd.DataFrame:
        """Pull Binance klines using the existing thesis timestamp convention."""
        symbol = self.symbol
        if kind == "spot":
            url = "https://api.binance.com/api/v3/klines"
        else:
            url = "https://fapi.binance.com/fapi/v1/klines"

        interval_ms = {
            "1m": 60_000,
            "3m": 3 * 60_000,
            "5m": 5 * 60_000,
            "15m": 15 * 60_000,
            "30m": 30 * 60_000,
            "1h": 60 * 60_000,
            "2h": 2 * 60 * 60_000,
            "4h": 4 * 60 * 60_000,
            "6h": 6 * 60 * 60_000,
            "12h": 12 * 60 * 60_000,
            "1d": 24 * 60 * 60_000,
        }[interval]
        limit = 1000
        start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
        end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
        total_intervals = (end_ts - start_ts) // interval_ms
        n_chunks = int(np.ceil(total_intervals / limit))
        starts = [start_ts + i * limit * interval_ms for i in range(n_chunks)]
        chunks = [(value, min(value + limit * interval_ms, end_ts)) for value in starts]

        def fetch_chunk(chunk_start, chunk_end):
            response = requests.get(
                url,
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": chunk_start,
                    "endTime": chunk_end,
                    "limit": limit,
                },
                timeout=10,
            )
            response.raise_for_status()
            return response.json()

        all_data = []
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(fetch_chunk, left, right): (left, right) for left, right in chunks}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        all_data.extend(result)
                except Exception as exc:
                    print(f"Kline chunk failed {futures[future]}: {exc}")

        if not all_data:
            raise ValueError("No kline data retrieved from Binance")

        frame = pd.DataFrame(all_data, columns=[
            "Open time", "Open", "High", "Low", "Close", "Volume",
            "Close time", "Quote asset volume", "Number of trades",
            "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore",
        ])
        frame = frame.sort_values("Open time")
        frame[["Open", "Close", "High", "Low", "Volume"]] = frame[["Open", "Close", "High", "Low", "Volume"]].astype(float)
        frame["log_return"] = np.log(frame["Close"] / frame["Close"].shift(1))

        open_times = pd.to_datetime(frame["Open time"], unit="ms", utc=True)
        close_times = pd.to_datetime(frame["Close time"], unit="ms", utc=True)
        if self.convert_tz is None:
            frame["Open time"] = open_times.dt.tz_localize(None)
            frame["Close time"] = close_times.dt.tz_localize(None) + dt.timedelta(milliseconds=1)
        else:
            frame["Open time"] = open_times.dt.tz_convert(self.convert_tz).dt.tz_localize(None)
            frame["Close time"] = close_times.dt.tz_convert(self.convert_tz).dt.tz_localize(None) + dt.timedelta(milliseconds=1)

        if columns is None:
            columns = ["Open time", "Close time", "Open", "Close", "log_return"]
        return frame[columns]


@lru_cache(maxsize=1)
def _fear_greed_cache() -> pd.DataFrame:
    """Return the complete Alternative.me Fear & Greed history by UTC date."""
    response = requests.get("https://api.alternative.me/fng/?limit=0", timeout=30)
    response.raise_for_status()
    frame = pd.DataFrame(response.json()["data"])
    frame["covariate_date"] = (
        pd.to_datetime(frame["timestamp"].astype(int), unit="s", utc=True)
        .dt.tz_localize(None)
        .dt.date
    )
    frame["fear_greed"] = frame["value"].astype(int)
    return frame[["covariate_date", "fear_greed"]].drop_duplicates("covariate_date", keep="last")


def get_fear_greed_history() -> pd.DataFrame:
    """Return a copy of the cached complete Fear & Greed history."""
    return _fear_greed_cache().copy()


def archive_symbol(symbol: str, cm_um: str) -> str:
    return symbol.replace("USDT", "USD_PERP") if cm_um == "cm" else symbol


def metric_url(symbol: str, cm_um: str, day) -> str:
    archived = archive_symbol(symbol, cm_um)
    day_string = pd.Timestamp(day).strftime("%Y-%m-%d")
    return (
        f"https://data.binance.vision/data/futures/{cm_um}/daily/metrics/"
        f"{archived}/{archived}-metrics-{day_string}.zip"
    )


def fetch_one_metric_day(
    session: requests.Session,
    symbol: str,
    cm_um: str,
    day,
    retries: int = 3,
    timeout_seconds: int = 45,
):
    """Fetch the three external liquidity covariates from one metrics archive."""
    # this is only for the daily metrics
    url = metric_url(symbol, cm_um, day)
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, timeout=timeout_seconds)
            if response.status_code == 404:
                print(f"Missing metrics archive: {archive_symbol(symbol, cm_um)} {pd.Timestamp(day).date()}")
                return None
            response.raise_for_status()
            if response.content[:2] != b"PK":
                raise ValueError("Metrics response was not a ZIP file")

            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                files = [name for name in archive.namelist() if not name.endswith("/")]
                if not files:
                    raise ValueError("Metrics ZIP contained no CSV")
                with archive.open(files[0]) as stream:
                    frame = pd.read_csv(stream, dtype=str, low_memory=False)

            required = ["create_time", *SOURCE_METRIC_COLUMNS.keys()]
            # create_time enforces that the metrics are strictly before the event time
            missing = set(required).difference(frame.columns)
            if missing:
                raise KeyError(f"Metrics archive missing columns: {sorted(missing)}")

            frame = frame[required].copy()
            frame["create_time"] = pd.to_datetime(frame["create_time"], utc=True, errors="coerce")
            for source_col, output_col in SOURCE_METRIC_COLUMNS.items():
                frame[output_col] = pd.to_numeric(frame[source_col], errors="coerce").astype("float64")
            return frame[["create_time", *OUTPUT_METRIC_COLUMNS]]
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(attempt * 2)

    print(
        f"Failed metrics: {archive_symbol(symbol, cm_um)} {pd.Timestamp(day).date()} "
        f"after {retries} attempts: {last_error!r}"
    )
    return None


def find_event_range(event_files) -> tuple[pd.Timestamp, pd.Timestamp]:
    minimum = None
    maximum = None
    for path in event_files:
        timestamps = pd.to_datetime(
            pd.read_parquet(path, columns=["start_ts"])["start_ts"],
            utc=True,
            errors="coerce",
        ).dropna()
        if timestamps.empty:
            continue
        local_min = timestamps.min()
        local_max = timestamps.max()
        minimum = local_min if minimum is None else min(minimum, local_min)
        maximum = local_max if maximum is None else max(maximum, local_max)
    if minimum is None or maximum is None:
        raise ValueError("No valid start_ts values found in event files")
    return minimum.floor("D") - pd.Timedelta(days=1), maximum.ceil("D")


def pull_or_load_market_metrics(
    run_dir,
    symbol: str,
    cm_um: str,
    event_files,
    redownload: bool = False,
    cache_subdir: str = "metrics_cache",
) -> pd.DataFrame:
    """Load cached Binance metrics or fetch only the days needed by the events."""
    run_dir = Path(run_dir)
    cache_dir = run_dir / cache_subdir
    cache_dir.mkdir(parents=True, exist_ok=True)
    archived = archive_symbol(symbol, cm_um)
    cache_path = cache_dir / f"{archived}_{cm_um}_liquidity_metrics.csv.gz"

    if cache_path.exists() and not redownload:
        print(f"Reading metrics cache: {cache_path}")
        metrics = pd.read_csv(cache_path, compression="infer")
        metrics["create_time"] = pd.to_datetime(metrics["create_time"], utc=True, errors="coerce")
    else:
        start_day, end_day = find_event_range(event_files) # find the range of days needed for the events
        days = pd.date_range(start=start_day, end=end_day, freq="D", tz="UTC") # binance files are utc
        print(
            f"Pulling metrics for {archived} ({cm_um}), {len(days)} days: "
            f"{start_day.date()} through {end_day.date()}"
        )
        frames = []
        with requests.Session() as session:
            session.headers.update({"User-Agent": "Mozilla/5.0"})
            for index, day in enumerate(days, start=1):
                result = fetch_one_metric_day(session, symbol, cm_um, day)
                # this gets the daily binance files
                if result is not None and not result.empty:
                    frames.append(result)
                if index == 1 or index % 50 == 0 or index == len(days):
                    print(f"Processed {index}/{len(days)} metric days")
        if not frames:
            raise RuntimeError(f"No metrics loaded for {archived}")
        metrics = pd.concat(frames, ignore_index=True, sort=False)
        metrics = (
            metrics.dropna(subset=["create_time"])
            .sort_values("create_time")
            .drop_duplicates("create_time", keep="last")
            .reset_index(drop=True)
        )
        metrics.to_csv(cache_path, index=False, compression="gzip")
        print(f"Saved metrics cache: {cache_path}")

    for column in OUTPUT_METRIC_COLUMNS:
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce").astype("float64")
    return (
        metrics.dropna(subset=["create_time"])
        .sort_values("create_time")
        .drop_duplicates("create_time", keep="last")
        .reset_index(drop=True)
    )
