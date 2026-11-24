# Paper code

This repository contains the reproducible source for the paper analysis of
price discovery and convergence between cryptocurrency spot and perpetual
markets.

> **Reproduction note.** This repository includes the data behind the paper's
> figures, tables, and appendices. [Paper data index](paper_data/README.md)
>
> Running the full 2021–2025 event pipeline took multiple days on the author's
> setup. The smaller test run refits the January 2021 BTC KM and AFT models
> offline using the included checkpoints. [One-month test run](#one-month-test-run)

## Installation

Use Python 3.12 or 3.13. The scientific packages in `requirements.txt` are
pinned to the versions used for the paper analysis.

### Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### macOS or Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
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

## Reproducing the paper outputs

Run these commands from the repository root using the virtual environment's
Python. On Windows, use `.\.venv\Scripts\python.exe` in place of `python`.
The commands use the bundled data and work offline after installation.

### One-month test run

```sh
python one_month_test_run.py
```

The script refits the January 2021 KM and seven-variable log-logistic AFT models
for BTC spot versus linear perpetuals, in both shock-origin directions. It
uses the repo's model code and the included one-month checkpoints and reference
results. It writes `km.csv`, `aft_spot.csv`, `aft_perp.csv`, and `aft_fits.csv`
to `results/test/`. [Test script](one_month_test_run.py)

### Paper figures and tables

```sh
# Figures 1–2 and the VECM adjustment-coefficient plots
python -m final_code.section_4.plot_vecm
python -m final_code.section_4.appendix.corrected_ils.plots

# Figures 3–4 and Appendix E.2's monthly KM figure rows
python -m final_code.section_5_1.plot_december_km
python -m final_code.section_5_1.appendix.export_km_rows

# Figure 5, the appendix coefficient heatmaps, and AFT summaries
python -m final_code.section_5_2.plot_aft_results --perp-only
python -m final_code.section_5_2.plot_aft_results
python -m final_code.section_5_2.appendix.plot_diagnostics
python -m final_code.section_5_2.appendix.plot_diagnostics --with-spot-volatility
python -m final_code.section_5_2.appendix.coefficient_tables
python -m final_code.section_5_2.univariate_tables
python -m final_code.section_5_2.appendix.midperiod_tables

# Figures 6–7
python -m final_code.section_5_2.appendix.plot_open_interest
python -m final_code.section_5_2.appendix.plot_perp_volume

# UTC and numerical-precision regression tests
python -m unittest discover -s tests
```

The [paper data index](paper_data/README.md) maps tables and figures to their
CSV inputs. The bundle includes the regime-level price-discovery shares,
daily V8 event counts, plotted KM curves, and AFT table and figure inputs.
The commands save generated files by analysis under `results/`, which Git
ignores. The full rebuild uses local checkpoints and Binance downloads.
[Analysis guide](final_code/README.md)

## Paper code map

### Section 4 — VECM

- `VECMHasbrouck2` and `SimpleMVAR` implement lag buckets, alpha and beta
  estimation, price-discovery shares, and ILS. [vecm_hasbrouck3.py](final_code/vecm_hasbrouck3.py)
- The production runner fits BTC and ETH linear and inverse markets. [run_vecm.py](final_code/section_4/run_vecm.py)
- The plot script draws alpha plots and regime heatmaps. [plot_vecm.py](final_code/section_4/plot_vecm.py)
- The appendix covers Johansen diagnostics, complete-grid robustness, and
  trade-activity analyses. [Section 4 appendix](final_code/section_4/appendix/)

### Section 5.1 — KM events and covariates

- The notebook records the production configuration and execution. [km_events.ipynb](final_code/section_5_1/km_events.ipynb)
- The event script handles regular-grid detection, KM construction,
  covariate timing, and KM plotting. [detect_events.py](final_code/section_5_1/detect_events.py)
- Shared processing covers event time, resolution, covariates, and timing
  validation. [event_processing.py](final_code/event_processing.py)
- Data retrieval covers funding, klines, the Fear & Greed Index, and Binance
  futures metrics. [survival_data.py](final_code/survival_data.py)

### Section 5.2 — AFT models

- Build the unit-matched, log-transformed AFT inputs from the V8 event set. [build_aft_covariates.py](final_code/section_5_2/build_aft_covariates.py)
- Fit the seven-variable log-logistic specification. [fit_aft.py](final_code/section_5_2/fit_aft.py)
- Fit the null and univariate models. [fit_univariate.py](final_code/section_5_2/fit_univariate.py)
- Render coefficient heatmaps and concordance plots. [plot_aft_results.py](final_code/section_5_2/plot_aft_results.py)
- Render VIF and correlation diagnostics and the appendix tables. [plot_diagnostics.py](final_code/section_5_2/appendix/plot_diagnostics.py), [coefficient_tables.py](final_code/section_5_2/appendix/coefficient_tables.py), [univariate_tables.py](final_code/section_5_2/univariate_tables.py)
- The appendix also contains the open-interest and volume figures. [Section 5.2 appendix](final_code/section_5_2/appendix/)

## Timing contract

Saved KM events are immutable inputs. Covariates follow these rules:

- Event-time volatility and instantaneous basis use event tick minus one.
- Five-minute spread/basis observations satisfy
  `covariate_5min_ts <= prev_ts < start_ts`.
- Daily funding, volume, and Fear & Greed use the previous UTC calendar day.
- Binance liquidity metrics satisfy `create_time < start_ts`.
- No interpolation is used for event-time prices; previous-tick filling is
  used only to form the regular one-second bid/ask grid.

The current notebook run uses `SIGNIFICANCE = 0.01`.

## Data sources and dependencies

The pipeline uses Binance spot/futures trades, klines, funding rates, and
futures metrics. Core dependencies include pandas, NumPy, Polars, PyArrow,
Matplotlib, Seaborn, Statsmodels, Lifelines, and Requests.

See `LICENSE` for licensing information.
