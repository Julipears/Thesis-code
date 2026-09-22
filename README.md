# Thesis code

This repository contains the reproducible source for the thesis analysis of
price discovery and convergence between cryptocurrency spot and perpetual
markets.

## Installation

```bash
pip install -r requirements.txt
pip install jupyter
```

## Source layout

The active source is under `final_code/`. Shared modules are directly under
that folder; section-specific code follows the paper structure.

```text
final_code/
|-- section_4/                 # VECM and price-discovery analysis
|   `-- appendix/              # Johansen and complete-grid robustness code
|-- section_5_1/               # KM event identification and covariates
`-- section_5_2/               # AFT models, figures, and tables
    `-- appendix/              # diagnostics and appendix tables/figures
```

## Running the pipeline

Run modules from the repository root with Python's module syntax so output
paths continue to point to the existing result folders:

```powershell
# Section 4: VECM plots from saved results
python -m final_code.section_4.vecm_plotting

# Section 5.1: regenerate the December KM figures
python -m final_code.section_5_1.regenerate_km_all_decembers

# Section 5.2: reduced AFT diagnostics
python -m final_code.section_5_2.appendix.plot_aft_v8_without_volatility_diagnostics
```

The production runners and fitting scripts are in the corresponding section
folders. Appendix scripts are nested under `appendix/` beside the method they
extend.

## Paper code map

### Section 4 — VECM

- `final_code/vecm_hasbrouck3.py`: `VECMHasbrouck2` and `SimpleMVAR`, including
  lag buckets, alpha/beta estimation, price-discovery shares, and ILS.
- `final_code/section_4/run_hasbrouck3_full.py`: BTC/ETH linear and inverse
  production runner.
- `final_code/section_4/vecm_plotting.py`: alpha plots and regime heatmaps.
- `final_code/section_4/appendix/`: Johansen diagnostics, complete-grid
  robustness, and trade-activity analyses.

### Section 5.1 — KM events and covariates

- `final_code/section_5_1/km_v8_events_and_covariates_single_pull.ipynb`:
  production configuration and execution record.
- `final_code/section_5_1/km_v8_regular_lm_pilot.py`: regular-grid event
  detection, KM construction, covariate timing, and KM plotting functions.
- `final_code/survival_analysis_data_processing_final.py`: event-time,
  resolution, covariate, and timing-validation logic.
- `final_code/survival_analysis_data_pull_final.py`: funding, kline, Fear &
  Greed, and Binance futures-metric retrieval.

### Section 5.2 — AFT models

- `final_code/section_5_2/reconstruct_aft_v8_log_covariates.py`: constructs the
  unit-matched, log-transformed AFT inputs from the V8 event set.
- `final_code/section_5_2/fit_aft_v8_without_spot_volatility.py`: current
  seven-variable log-logistic specification.
- `final_code/section_5_2/fit_aft_v8_univariate_models.py`: null and univariate
  models.
- `final_code/section_5_2/plot_aft_v8_without_spot_volatility_results.py`:
  coefficient heatmaps and concordance plots.
- `final_code/section_5_2/appendix/`: VIF/correlation diagnostics,
  open-interest and volume figures, and appendix coefficient tables.

The standalone V8 model containing spot volatility is archived at
`archive/legacy_aft/v8_with_volatility/`.

## Timing contract

Saved KM events are immutable inputs. Covariates follow these rules:

- Event-time volatility and instantaneous basis use event tick minus one.
- Five-minute spread/basis observations satisfy
  `covariate_5min_ts <= prev_ts < start_ts`.
- Daily funding, volume, and Fear & Greed use the previous UTC calendar day.
- Binance liquidity metrics satisfy `create_time < start_ts`.
- No interpolation is used for event-time prices; previous-tick filling is
  used only to form the regular one-second bid/ask grid.

`validate_base_covariate_timing` and `validate_augmented_timing` enforce these
conditions. The current notebook run uses `SIGNIFICANCE = 0.01`.

## Data sources and dependencies

The pipeline uses Binance spot/futures trades, klines, funding rates, and
futures metrics. Core dependencies include pandas, NumPy, Polars, PyArrow,
Matplotlib, Seaborn, Statsmodels, Lifelines, and Requests.

See `LICENSE` for licensing information.
