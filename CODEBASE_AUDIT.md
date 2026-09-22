# Thesis code audit

This is the working inventory for the cleanup. It separates the code that
generated the current results from older experiments and from plotting code.
No result directories are changed by this inventory.

## Current result provenance

### VECM / Hasbrouck

The current Hasbrouck implementation is `vecm_hasbrouck3.py`. The active
model class is `VECMHasbrouck2`, which builds a `SimpleMVAR` for each saved
window. Its `SimpleMVAR` is the code that actually estimates the VECM and
computes alpha, beta, price-discovery shares, and ILS. The model is configured
with `intercept=True` and `ecm=True` in the production call path.

Core reusable pieces:

- `SimpleMVAR.__post_init__`
- `SimpleMVAR._parse_lag_structure`
- `SimpleMVAR._build_ec_terms`
- `SimpleMVAR._build_bucketed_lags`
- `SimpleMVAR._build_design_for_prices`
- `SimpleMVAR.fit`
- `SimpleMVAR.fit_by_interval`
- `SimpleMVAR._fit_single_interval`
- `SimpleMVAR.gamma_matrix`, `beta_matrix`, `phi_matrices`, `irf`
- `SimpleMVAR._price_discovery_from_outputs`
- `SimpleMVAR.shares_table`, `interval_summary_table`, `all_interval_summaries`
- `VECMHasbrouck2._get_and_parse_data`
- `VECMHasbrouck2.get_data_multiperiod`
- `read_files`, `price_discovery_shares`, `shares_table`
- `generate_lag_buckets`, `generate_multiple_lags`, `find_period_mean`

Execution wrappers are separate from the model:

- `run_hasbrouck3_btc_full.py` runs the complete BTC-UM/BTC-CM production
  folders.
- `run_hasbrouck_random_full_grid.py` runs the stratified/random-grid audit.
- `audit_hasbrouck_*.py`, `benchmark_hasbrouck_full_grid.py`, and
  `validate_hasbrouck_random_supervisor.py` are validation/benchmark tools.

The four result folders are `vecm_hasbrouck3_btc_um`,
`vecm_hasbrouck3_btc_cm`, `vecm_hasbrouck3_eth_um`, and
`vecm_hasbrouck3_eth_cm`. The alpha-over-time scripts consume these folders;
they are plotting clients and should become a notebook.

The older `vecm_hasbrouck2.py`, `vecm_hasbrouck2_old.py`, and
`vecm_analysis2.py` are not part of this current VECM3 result path. The old
`hasbrouck_vecm.ipynb` imports `vecm_hasbrouck2.py`, so it should be retained
as a legacy reproduction notebook until the VECM3 notebook is made explicit.

### KM event identification and covariates

The current full KM run is driven by
`km_v8_events_and_covariates_single_pull.ipynb`, with implementation in
`km_v8_regular_lm_pilot.py`. The notebook sets:

- `OUTPUT_ROOT = sa_results/km_v8_final_01`
- `GRID = 1s`
- `SIGNIFICANCE = 0.01`
- `KM_PLOT_MONTHS = 2021-12, 2022-12, 2023-12, 2024-12, 2025-12`

**Audit flag:** `KM_V8_METHODOLOGY_FINAL.txt` documents a Lee--Mykland
significance threshold of `0.001`, while the current full-run notebook sets
`SIGNIFICANCE = 0.01`. This must be resolved and recorded as the thesis
specification before archiving either version. It changes event identification,
so it is not a plotting-only difference.

The KM panel files under `sa_results/km_v8_final_01/plots/` are produced by
`plot_asset_km_month_comparison` (called through the compatibility wrapper
`plot_btc_km_month_comparison`). This is the source of the current V8 graphs,
not the older V5/V7 comparison notebooks.

Event/KM core functions in `km_v8_regular_lm_pilot.py`:

- `grid_seconds`
- `regular_lee_mykland`
- `map_bins_to_origin_trade_ticks`
- `load_v5_events`
- `run_one`
- `run_date_block`
- `km_resolved_by`
- `overlap_metrics`

Plotting functions currently mixed into that file and destined for a notebook:

- `plot_pooled_km_comparison`
- `_weighted_km_inputs` (a small KM aggregation helper can remain reusable)
- `plot_yearly_btc_km`
- `plot_asset_km_month_comparison`
- `plot_btc_km_month_comparison`

The V8 implementation deliberately calls reusable event/covariate functions
from `survival_analysis_data_processing_final.py`:

- event-time construction: `to_intervals_bidask`, `build_event_time_prices`
- resolution/event rules: `greedy_refractory_seconds`,
  `_first_sustained_entry_index`, `calculate_basis_resolution_events`,
  `detect_basis_events_from_event_time`
- covariates: `build_aft_covariates`
- saved-event validation: `reconstruct_saved_event_ticks`,
  `assert_saved_km_outcomes_unchanged`

`survival_analysis_data_pull_final.py` is the final data-access module for
funding, klines, Fear & Greed, and Binance metrics. It contains `TradeData`,
`pull_or_load_market_metrics`, and the metric archive helpers.

There is one cleanup hazard: V8 currently imports `TradeData` from the older
`trade_data_pull.py`, while the final covariate module imports `TradeData` from
`survival_analysis_data_pull_final.py`. These implementations must be compared
and regression-tested before replacing the V8 import.

### AFT setup

There are two valid AFT entry points, serving different purposes:

1. The saved-KM reconstruction workflow is
   `survival_analysis_pipeline_final.py` plus
   `survival_analysis_workflow_final.ipynb`. Its main entry point is
   `run_market_pipeline_from_saved_km`; it attaches leakage-safe covariates and
   can fit Weibull, log-normal, and log-logistic models.
2. The latest V5 model family uses the generic log-logistic fitter in
   `fit_aft_v3_all_covariates.py`. The V5 wrappers select the data/specification:
   `fit_aft_v5_log_covariates.py`,
   `fit_aft_v5_bivariate_shock_spread.py`, and
   `fit_aft_v5_univariate_baselines.py`.

Reusable AFT fitting pieces:

- `fit_aft_v3_all_covariates.fit_one`
- `fit_aft_v3_all_covariates.save_manifest_retry`
- `fit_aft_v3_all_covariates.main` (runner only; move orchestration out of the
  reusable module)
- `survival_analysis_pipeline_final.fit_one_aft_file`
- `survival_analysis_pipeline_final.fit_market_aft_models`
- `survival_analysis_pipeline_final.run_market_pipeline_from_saved_km`
- `survival_analysis_pipeline_final.run_market_pipeline`
- `survival_analysis_data_processing_final.prepare_aft_data`
- `survival_analysis_data_processing_final.augment_liquidity_metrics`

The latest V5 log specification is eight variables:

`basis_5min`, `fundingRate_bps`, `log_open_interest`, `shock_size_bps`,
`spread_spot_5min`, `log_taker_long_short`, `vol_spot_5min_pct`, and
`log_volume`.

The latest V5 model outputs are under each `sa_*` market directory in
`aft_results_v5_log_covariates`, with bivariate and univariate ablations in
their separately named result directories. The source event/covariate rows are
the V5 reconstruction outputs under `aft_data_liquidity_v5_log_covariates`.

**Provenance flag:** the current V5 fitter reads
`sa_*/aft_data_liquidity_v5_log_covariates`, which is derived from
`sa_*/aft_data_liquidity_unit_agreeing_oi` and ultimately from the saved-KM
event files in each `sa_*` directory. It does not read the separate
`sa_results/km_v8_final_01/aft_data_liquidity_monthly` tree. Thus the V8 KM
panels and the latest V5 AFT coefficients are currently two related but
different event pipelines. They should be labelled separately in the cleaned
layout until a deliberate reconciliation is performed.

For example, in BTC-UM spot, January 2021 contains 139,194 rows in
`sa_btc_um/events/spot_2021-01.parquet` (the source of the current V5 AFT
rows), but 5,294 rows across the V8 regular-grid files under
`sa_results/km_v8_final_01/events/btc_um/spot_2021-01-*`. Only 10 event
timestamps overlap in that sample. This is a real event-set difference, not
just a covariate transformation.

The V8-consistent refit now has its own isolated paths:

- inputs: `sa_results/km_v8_final_01/aft_data_liquidity_v5_log_covariates`
- fits: `sa_results/km_v8_final_01/aft_results_v5_log_covariates`
- reconstruction: `reconstruct_aft_v8_log_covariates.py`
- fitting: `fit_aft_v8_log_covariates.py`

The primary V8-consistent eight-variable log-logistic refit currently uses the
executed V8 output threshold (`0.01`) and writes figures to
`sa_results/km_v8_final_01/analysis_outputs/aft_v5_log_covariate_results`.

## Plotting code to move into notebooks

These scripts should become notebooks that read saved CSV/Parquet results and
write figures/tables without re-running event detection or model fitting:

- `plot_hasbrouck3_alpha_over_time.py`
- `plot_hasbrouck3_alpha_separate.py`
- `generate_btc_eth_vif_outputs.py`
- `plot_aft_v5_log_covariate_diagnostics.py`
- `plot_aft_v5_log_covariate_results.py`
- `plot_aft_no_spot_volatility_results.py`
- `plot_aft_v3_all_covariates_results.py`
- `plot_aft_unit_agreeing_oi_results.py`
- `plot_aft_unit_agreeing_oi_correlations.py`
- `plot_random_full_grid_heatmaps.py`
- `regenerate_vecm_regime_heatmaps_april2023.py`
- `regenerate_v8_km_vecm_correlations.py`
- `regenerate_km_all_decembers.py`

The notebooks should have no data-download or fitting side effects. They should
only load explicit result roots, make figures, and save tables/plots.

## Legacy/stale candidates

These files are older implementations or superseded experiment runners and
should be moved under a `legacy/` directory only after their output folders are
confirmed unnecessary:

- `vecm_hasbrouck2.py`, `vecm_hasbrouck2_old.py`, `vecm_analysis2.py`
- `survival_analysis.py`, `survival_analysis_basis.py`,
  `survival_analysis_basis_monthly.py`
- `trade_data_pull_old.py`
- `fit_aft_v2_without_open_interest.py`, `refit_aft_without_spot_volatility.py`
- old KM notebooks (`km_v5_*`, `km_v7_*`) and old survival-analysis notebooks

The four `survival_analysis_*_final.py` modules are still needed by the V8 and
saved-KM workflows. They should be split into focused modules before any files
are removed:

```text
vecm_core.py              # VECM/Hasbrouck estimator and lag utilities
km_events.py              # LM shocks, event mapping, refractory/resolution rules
market_data.py            # TradeData and external metric downloads
aft_covariates.py         # event tick reconstruction and covariate timing
aft_models.py             # AFT fit/manifest logic
run_vecm.py, run_km.py,
run_aft.py                # thin command-line runners
notebooks/                 # all plotting and table-generation notebooks
legacy/                    # frozen old implementations and old notebooks
```

Before archiving a file, run a repository-wide import search and compare one
small deterministic sample against the saved result manifest. The cleanup must
preserve the V8 methodology version, the timing contract, and the current AFT
row counts before and after the module split.
