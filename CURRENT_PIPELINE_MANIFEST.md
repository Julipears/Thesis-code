# Current thesis pipeline: keep list and outputs

This manifest covers the current production path used for the VECM, KM, and
V8-event AFT analyses. It is an inventory only; it does not delete or move
anything.

## 1. Hasbrouck VECM

### Required code

- `vecm_hasbrouck3.py`: estimator, lag buckets, interval fitting, alpha/beta,
  price-discovery shares, and ILS.
- `trade_data_pull.py`: market-data access used by `vecm_hasbrouck3.py`.
- `timeout.py`: timeout helper imported by the Hasbrouck implementation.
- `run_hasbrouck3_btc_full.py`: BTC-UM/BTC-CM production runner.

The saved ETH VECM result folders exist, but no current standalone ETH
Hasbrouck-3 production runner was found in the repository. Preserve the ETH
result folders until that provenance is documented or a runner is restored.

### VECM result data

- `vecm_hasbrouck3_btc_um/`
- `vecm_hasbrouck3_btc_cm/`
- `vecm_hasbrouck3_eth_um/`
- `vecm_hasbrouck3_eth_cm/`

These contain the chunked hourly CSV estimates consumed by the plotting and
regime-summary scripts.

### VECM plots and plot-only code

- `plot_hasbrouck3_alpha_over_time.py`
- `plot_hasbrouck3_alpha_separate.py`
- `regenerate_vecm_regime_heatmaps_april2023.py`
- output: `vecm_hasbrouck3_alpha_plots/`
- output: `vecm_regime_heatmaps_april2023/`

`regenerate_v8_km_vecm_correlations.py` is an optional legacy comparison, not
part of the clean VECM3 path. Its BTC configuration reads `vecm_hasbrouck2_um/`
and `vecm_hasbrouck2_cm/`. The generated `sa_results/km_v8_final_01/
km_vecm_correlations_v8/` folder is not consumed by the current KM or AFT
pipeline. It can be archived or deleted if those old KM--VECM correspondence
figures are not used in the thesis.

## 2. KM event identification and covariates

### Required code

- `km_v8_events_and_covariates_single_pull.ipynb`: production configuration
  and execution notebook.
- `km_v8_regular_lm_pilot.py`: event detection, KM construction, covariate
  timing, and KM plotting functions.
- `survival_analysis_data_processing_final.py`: event-time construction,
  resolution rules, and liquidity-covariate augmentation.
- `survival_analysis_data_pull_final.py`: funding, klines, Fear & Greed, and
  Binance metric access.
- `survival_analysis_utils_final.py`: shared CSV/parquet/timing helpers.
- `trade_data_pull.py`: `TradeData` used by the V8 KM implementation.
- `KM_V8_METHODOLOGY_FINAL.txt`: methodology record.

The production notebook currently sets `GRID = '1s'`, `SIGNIFICANCE = 0.01`,
the date range 2021-01-01 through 2025-12-31, and output root
`sa_results/km_v8_final_01/`.

### KM data outputs

- `sa_results/km_v8_final_01/events/`: daily event parquet files.
- `sa_results/km_v8_final_01/aft_data/`: daily base covariate files.
- `sa_results/km_v8_final_01/aft_data_monthly/`: monthly pooled base files.
- `sa_results/km_v8_final_01/metric_runs/`: cached liquidity metrics.
- `sa_results/km_v8_final_01/aft_data_liquidity_monthly/`: monthly augmented
  covariates before the V5 log transformation.
- `sa_results/km_v8_final_01/aft_data_liquidity_v5_log_covariates/`: current
  AFT input files used by the V8-event AFT refits.
- `sa_results/km_v8_final_01/comparison_summary.csv` and
  `sa_results/km_v8_final_01/invalid_days.csv`: KM run manifests/checkpoints.

### KM plots

- primary output: `sa_results/km_v8_final_01/plots/`
- `regenerate_km_all_decembers.py`: optional December-only regeneration using
  the saved V8 event files.
- `regenerate_v8_km_vecm_correlations.py` and
  `sa_results/km_v8_final_01/km_vecm_correlations_v8/`: optional legacy
  comparison outputs only. They are not required for the current KM panels,
  VECM3 figures, or AFT fits.

## 3. AFT models and figures

The current AFT fits use the V8 event set and the V5-style covariates. The
`v5` name identifies the covariate specification; it does not mean these fits
use the older V5 event files under `sa_*`.

### Required fitting code

- `fit_aft_v8_log_covariates.py`: eight-variable log-logistic fit.
- `fit_aft_v8_without_spot_volatility.py`: seven-variable refit used in the
  latest reduced specification.
- `fit_aft_v8_univariate_models.py`: shock-size, spread, log-volume, basis,
  and null fits.
- `survival_analysis_utils_final.py`: atomic result writing.

The stopped experimental file `fit_aft_v8_bivariate_shock_spread.py` produced
partial output and should be ignored or deleted after checking that no desired
result came from it.

### AFT result data

- `sa_results/km_v8_final_01/aft_results_v5_log_covariates/`
- `sa_results/km_v8_final_01/aft_results_v5_without_spot_volatility/`
- `sa_results/km_v8_final_01/aft_results_v8_univariate_shock_size/`
- `sa_results/km_v8_final_01/aft_results_v8_univariate_spread/`
- `sa_results/km_v8_final_01/aft_results_v8_univariate_log_volume/`
- `sa_results/km_v8_final_01/aft_results_v8_univariate_basis/`
- `sa_results/km_v8_final_01/aft_results_v8_null/`

### AFT figure and table code

- `plot_aft_v8_log_covariate_results.py`: eight-variable heatmaps,
  concordance, and significance figures.
- `plot_aft_v8_without_spot_volatility_results.py`: current seven-variable
  wrapper that points the shared plotting code at the reduced result root.
- `plot_aft_no_spot_volatility_results.py`: reduced seven-variable heatmaps
  and concordance figures.
- `plot_aft_v8_log_covariate_diagnostics.py` and
  `plot_aft_v8_without_volatility_diagnostics.py`: correlations and VIFs.
- current output roots:
  - `sa_results/km_v8_final_01/analysis_outputs/aft_v5_log_covariate_results/`
  - `sa_results/km_v8_final_01/analysis_outputs/aft_v5_without_spot_volatility_results/`
  - `sa_results/km_v8_final_01/analysis_outputs/aft_v8_univariate_results/`

## Do not confuse with the older path

The following are older or separate pipelines and should not be mixed into
the current review without a deliberate comparison:

- `sa_btc_um/`, `sa_btc_cm/`, `sa_eth_um/`, `sa_eth_cm/` AFT results.
- `open_interest_figures/native/aft_v5_*` outputs.
- `aft_results_no_spot_volatility*` and `aft_results_btc_*` under
  `sa_results/km_v8_final_01/`.
- `vecm_hasbrouck2_*` and notebooks importing `vecm_hasbrouck2.py`.
- old `km_v5_*`, `km_v7_*`, and saved-survival-analysis notebooks.

The older AFT outputs use a different saved-event pipeline. They are not
interchangeable with the V8 event files merely because both specifications
contain similarly named covariates.
