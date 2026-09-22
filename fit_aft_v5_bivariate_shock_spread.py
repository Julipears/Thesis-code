"""Fit the V5 log-logistic AFT model with shock size and spread only."""

import fit_aft_v3_all_covariates as base


base.INPUT_DIR_NAME = "aft_data_liquidity_v5_log_covariates"
base.RESULTS_DIR_NAME = "aft_results_v5_bivariate_shock_spread"
base.COVARIATES = (
    "shock_size_bps",
    "spread_spot_5min",
)


if __name__ == "__main__":
    base.main()
