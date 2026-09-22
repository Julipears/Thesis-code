"""Run the log-logistic AFT model on the raw V5 log covariates."""

import fit_aft_v3_all_covariates as base


base.INPUT_DIR_NAME = "aft_data_liquidity_v5_log_covariates"
base.RESULTS_DIR_NAME = "aft_results_v5_log_covariates"
base.COVARIATES = (
    "basis_5min",
    "fundingRate_bps",
    "log_open_interest",
    "shock_size_bps",
    "spread_spot_5min",
    "log_taker_long_short",
    "vol_spot_5min_pct",
    "log_volume",
)


if __name__ == "__main__":
    base.main()
