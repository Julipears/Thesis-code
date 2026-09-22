"""Generate reduced V8 AFT coefficient heatmaps and concordance plots."""

from pathlib import Path

import plot_aft_v8_log_covariate_results as base


base.INPUT_DIR = Path("sa_results/km_v8_final_01/aft_data_liquidity_v5_log_covariates")
base.RESULTS_DIR = Path("sa_results/km_v8_final_01/aft_results_v5_without_spot_volatility")
base.OUTPUT = Path("sa_results/km_v8_final_01/analysis_outputs/aft_v5_without_spot_volatility_results")
base.COVARIATES = [
    "basis_5min",
    "fundingRate_bps",
    "log_open_interest",
    "shock_size_bps",
    "spread_spot_5min",
    "log_taker_long_short",
    "log_volume",
]


if __name__ == "__main__":
    base.main()
