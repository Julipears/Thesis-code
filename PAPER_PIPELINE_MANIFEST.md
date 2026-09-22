# Paper reproduction manifest

This is the live code set for the figures and tables in the thesis report. The
remaining scripts are kept in `archive/` so the cleanup is reversible.

All source paths below are relative to `final_code/`. Shared data and utility
modules are directly under `final_code/`; section-specific code is grouped in
the three section folders.

## VECM figures and tables (`section_4/`)

- `trade_data_pull.py`, `pull_binance_data.py`, `timeout.py`: exchange-data access used by the VECM and Johansen routines.
- `vecm_hasbrouck3.py`: Hasbrouck multi-resolution VECM estimation.
- `run_hasbrouck3_full.py`: standard BTC/ETH contract-grid run configuration.
- `vecm_plotting.py`: alpha-over-time plots, regime heatmaps, and spot/perpetual heatmaps.
- `section_4/appendix/johansen_rotating_hour_audit.py`, `run_johansen_all_markets_parallel.py`: rotating-hour Johansen diagnostics (Appendix Tables 4–5).
- `section_4/appendix/run_hasbrouck_random_full_grid.py`, `audit_hasbrouck_one_hour.py`: complete-grid robustness run (Appendix Figures 12–13 and Table 6).
- `section_4/appendix/plot_random_full_grid_heatmaps.py`, `summarize_random_full_grid_ils.py`: robustness heatmaps and the LaTeX comparison table.
- `section_4/appendix/trade_activity_diagnostics.py`: trade/update-frequency diagnostics used for Appendix Figure 11.

## Event identification and market-data figures (`section_5_1/`)

- `survival_analysis_data_pull_final.py`, `survival_analysis_data_processing_final.py`, `survival_analysis_utils_final.py`: shared event and covariate construction.
- `km_v8_events_and_covariates_single_pull.ipynb`, `km_v8_regular_lm_pilot.py`, `KM_V8_METHODOLOGY_FINAL.txt`: KM-V8 event identification and method record.
- `regenerate_km_all_decembers.py`: December KM figures (main Figures 5–6).
- `section_5_2/appendix/plot_open_interest_native.py`: unit-matched open-interest positions (Appendix Figure 9).
- `section_5_2/appendix/plot_daily_perpetual_volumes.py`: daily contract-quantity perpetual volumes (Appendix Figure 10).

## AFT models, diagnostics, and tables (`section_5_2/`)

- `survival_analysis_pipeline_final.py`: reusable AFT pipeline and Weibull/lognormal/log-logistic family fitting.
- `reconstruct_aft_v8_log_covariates.py`: builds the shared V8 log-covariate input files.
- `fit_aft_v8_without_spot_volatility.py`: seven-variable paper specification.
- `fit_aft_v8_univariate_models.py`: null and univariate specifications.
- `plot_aft_no_spot_volatility_results.py`, `plot_aft_v8_without_spot_volatility_results.py`, and the shared plotting helper `plot_aft_v8_log_covariate_results.py`: coefficient heatmaps and concordance outputs.

The standalone complete eight-variable V8 fit is archived at
`archive/legacy_aft/v8_with_volatility/fit_aft_v8_log_covariates.py`.
- `section_5_2/appendix/plot_aft_v8_log_covariate_diagnostics.py`, `plot_aft_v8_without_volatility_diagnostics.py`: correlation heatmaps, VIF outputs, and the covariate distribution summary table (Appendix Figures 14–15 and Tables 8–10).
- `summarize_aft_v8_univariate_results.py`: univariate likelihood and coefficient tables (Appendix Tables 3 and 17).
- `section_5_2/appendix/summarize_aft_v8_without_volatility_coefficients.py`, `summarize_aft_v5_complete_coefficients_monthly.py`, `summarize_aft_v5_complete_coefficients_vecm_periods.py`: period-specific coefficient tables (Appendix Tables 12–16).
- `section_5_2/appendix/average_midperiod_comparisons_by_currency.py`, `combine_spread_shock_entire_period_tables.py`, `compare_midperiod_to_entire_adjacent.py`, `compare_shock_size_coefficient_vecm_midperiod.py`, `compare_spread_coefficient_vecm_midperiod.py`, `concat_midperiod_coefficient_tables.py`, `make_midperiod_coefficient_side_comparisons.py`: mid-period comparison tables (Appendix Tables 18–19).

## Repository outputs

The scripts write to the existing result/output folders (`vecm_hasbrouck3_*`,
`vecm_regime_heatmaps_april2023`, `hasbrouck_spec_audit`, `sa_results`,
`daily_perpetual_volume_figures`, `open_interest_figures`, and `latex_tables`).
Those data and generated artifacts remain ignored by Git; only source code and
documentation are kept in the live tree.

## Shared implementation and timing rules

- `vecm_hasbrouck3.py` contains `VECMHasbrouck2` and `SimpleMVAR`, including
  lag buckets, alpha/beta estimation, price-discovery shares, and ILS.
- `survival_analysis_data_processing_final.py` contains the event-time,
  resolution, covariate, liquidity-augmentation, and timing-validation logic.
- `survival_analysis_data_pull_final.py` contains funding, kline, Fear & Greed,
  and Binance futures-metric retrieval.
- `survival_analysis_pipeline_final.py` contains the reusable saved-KM AFT
  pipeline and Weibull/lognormal/log-logistic family fitting.

The saved KM events are treated as immutable inputs. Timing rules are:

- event-time volatility and instantaneous basis use event tick minus one;
- 5-minute spread/basis observations are matched no later than the event
  predecessor (`covariate_5min_ts <= prev_ts < start_ts`);
- daily funding, volume, and Fear & Greed use the previous UTC calendar day;
- Binance liquidity metrics use the latest observation strictly before the
  event (`create_time < start_ts`);
- no interpolation is used for event-time prices; previous-tick filling is
  used only to form the regular one-second bid/ask grid.

The validation functions `validate_base_covariate_timing` and
`validate_augmented_timing` enforce these rules. The current notebook run uses
`SIGNIFICANCE = 0.01`; the methodology text records `0.001`, so the paper
should state explicitly which threshold is intended.

The older `sa_*` event/AFT outputs and the complete V8 model with spot
volatility are historical paths. They are retained under `archive/` and should
not be mixed with the current V8 reduced, univariate, and null results.
