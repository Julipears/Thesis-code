from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import polars as pl

from final_code.event_processing import (
    augment_liquidity_metrics,
    calculate_basis_resolution_events,
)
from final_code.survival_data import OUTPUT_METRIC_COLUMNS, TradeData
from final_code.survival_utils import COVARIATE_TIMING_VERSION, as_timestamp, read_event_parquet
from final_code.section_5_2 import build_aft_covariates
from final_code.vecm_hasbrouck3 import SimpleMVAR


class TimingAndPrecisionTests(unittest.TestCase):
    def test_liquidity_join_normalizes_utc_and_skips_equal_timestamps(self):
        events = pd.DataFrame({
            "start_ts": pd.to_datetime(["2021-01-02T00:00:00", "2021-01-02T00:00:01"]).as_unit("us"),
            "prev_ts": ["2021-01-01T23:59:59Z", "2021-01-02T00:00:00Z"],
            "covariate_5min_ts": "2021-01-01T23:55:00Z",
            "daily_covariate_date": "2021-01-01",
            "covariate_timing_version": COVARIATE_TIMING_VERSION,
        })
        metrics = pd.DataFrame({
            "create_time": ["2021-01-01 23:55:00", "2021-01-01T19:00:00-05:00"],
            **{column: [1.0, 2.0] for column in OUTPUT_METRIC_COLUMNS},
        })
        result = augment_liquidity_metrics(events, metrics)
        self.assertEqual(result["open_interest"].tolist(), [1.0, 2.0])
        self.assertEqual(str(result["create_time"].dtype), "datetime64[ns, UTC]")
        self.assertEqual(as_timestamp("2021-01-01T19:00:00-05:00"), pd.Timestamp("2021-01-02"))

    def test_event_timestamps_preserve_integer_nanoseconds(self):
        start = pd.Timestamp("2025-01-01T00:00:00.000000001").value
        ticks = [start + offset for offset in (0, 1_000_013, 2_000_027, 3_000_043, 4_000_051, 10_000_059)]
        prices = pl.DataFrame({
            "timestamp": ticks,
            "p_spot": [100.0] * len(ticks),
            "p_perp": [100.0, 101.0, 100.0, 102.0, 102.0, 102.0],
        }).with_columns(pl.col("timestamp").cast(pl.Datetime("ns")))
        result = calculate_basis_resolution_events(
            prices, [1, 3], max_duration_seconds=0.004, require_full_horizon=False,
        )
        self.assertEqual(result["Status"].to_list(), [1, 0])
        self.assertEqual(result["start_ts"].cast(pl.Int64).to_list(), [ticks[1], ticks[3]])
        self.assertEqual(result["prev_ts"].cast(pl.Int64).to_list(), [ticks[0], ticks[2]])
        self.assertEqual(result["resolution_ts"].cast(pl.Int64).to_list(), [ticks[2], None])

    def test_funding_request_uses_utc_midnight(self):
        epoch_ms = 1609459200000
        response = Mock()
        response.json.return_value = [{"fundingRate": "0.0001", "fundingTime": epoch_ms}]
        with patch("final_code.survival_data.requests.get", return_value=response) as get:
            result = TradeData("BTCUSDT").get_funding_data("20210101", "20210102")
        self.assertEqual(get.call_args.kwargs["params"]["startTime"], str(epoch_ms))
        self.assertEqual(result["fundingTime"].iloc[0], pd.Timestamp("2021-01-01"))

    def test_event_reader_accepts_old_float_resolution_timestamps(self):
        timestamp = pd.Timestamp("2021-01-01T00:00:01")
        legacy = pd.DataFrame({"resolution_ts": [float(timestamp.value), np.nan]})
        modern = pd.DataFrame({"resolution_ts": [timestamp + pd.Timedelta(nanoseconds=1)]})
        with patch("pandas.read_parquet", side_effect=[legacy, modern]):
            pooled = pd.concat([read_event_parquet("old.parquet"), read_event_parquet("new.parquet")], ignore_index=True)
        self.assertEqual(str(pooled["resolution_ts"].dtype), "datetime64[ns]")
        self.assertEqual(pooled["resolution_ts"].iloc[0], timestamp)
        self.assertTrue(pd.isna(pooled["resolution_ts"].iloc[1]))
        self.assertEqual(pooled["resolution_ts"].iloc[2].value, timestamp.value + 1)

    def test_aft_open_interest_join_preserves_event_order_and_unmatched_rows(self):
        events = pd.DataFrame({
            "start_ts": pd.to_datetime(["2021-01-02 00:01:00", None, "2021-01-02 00:00:00"]).as_unit("us"),
            "volume": 1.0,
            "taker_long_short": 1.0,
        })
        metrics = pd.DataFrame({
            "create_time": pd.to_datetime(["2021-01-01 23:55:00", "2021-01-02 00:00:00"], utc=True),
            "open_interest": [1.0, 2.0],
        })
        with (
            patch.object(build_aft_covariates.pd, "read_parquet", return_value=events),
            patch.object(build_aft_covariates, "_load_oi_metrics", return_value=metrics),
            patch.object(build_aft_covariates, "atomic_pandas_parquet") as save,
        ):
            build_aft_covariates.reconstruct_file(Path("events.parquet"), Path("out.parquet"), "btc_um")
        result = save.call_args.args[0]
        np.testing.assert_allclose(result["open_interest"], [2.0, np.nan, 1.0], equal_nan=True)
        self.assertTrue(pd.isna(result.loc[1, "create_time"]))

    def test_price_discovery_extracts_single_long_run_denominator(self):
        result = SimpleMVAR._price_discovery_from_outputs(
            ["spot", "perp"], np.zeros((2, 2, 1)), np.eye(2), np.array([[-0.02], [0.01]])
        )
        np.testing.assert_allclose(result["cs"], [1 / 3, 2 / 3])
        np.testing.assert_allclose(result["his_mid"], [0.2, 0.8])
        np.testing.assert_allclose(result["ils_mid"], [0.2, 0.8])


if __name__ == "__main__":
    unittest.main()
