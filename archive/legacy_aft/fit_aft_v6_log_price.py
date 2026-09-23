"""Fit V6 log-logistic AFT models: V5 covariates plus unstandardized log price."""

import fit_aft_v3_all_covariates as base


base.INPUT_DIR_NAME = "aft_data_liquidity_v6_log_price"
base.RESULTS_DIR_NAME = "aft_results_v6_log_price"
base.COVARIATES = (
    "basis_5min",
    "fundingRate_bps",
    "log_open_interest",
    "shock_size_bps",
    "spread_spot_5min",
    "log_taker_long_short",
    "vol_spot_5min_pct",
    "log_volume",
    "daily_log_price",
)


if __name__ == "__main__":
    base.main()
