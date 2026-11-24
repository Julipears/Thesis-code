# Final paper code

The source used for the paper is grouped by paper section:

- Section 4 contains Hasbrouck VECM estimation and the main VECM figures.
  [run_vecm.py](section_4/run_vecm.py), [plot_vecm.py](section_4/plot_vecm.py)
- `section_4/appendix/`: Johansen diagnostics, complete-grid robustness
  figures and tables, and trade-activity analysis.
- `section_4/appendix/corrected_ils/`: corrected ILS figures and Table 4.
- Section 5.1 contains KM-V8 event identification, covariate construction,
  and December KM figures. [detect_events.py](section_5_1/detect_events.py), [km_events.ipynb](section_5_1/km_events.ipynb), [plot_december_km.py](section_5_1/plot_december_km.py)
- `section_5_1/appendix/`: monthly KM curves and 120-second panels for
  Appendix E.2, Figures 11–20.
- Section 5.2 contains AFT fitting and the main AFT figures and tables.
  [fit_aft.py](section_5_2/fit_aft.py), [fit_univariate.py](section_5_2/fit_univariate.py), [plot_aft_results.py](section_5_2/plot_aft_results.py), [univariate_tables.py](section_5_2/univariate_tables.py)
- The Section 5.2 appendix contains diagnostics, open-interest and volume
  figures, and coefficient and comparison tables. [plot_diagnostics.py](section_5_2/appendix/plot_diagnostics.py), [coefficient_tables.py](section_5_2/appendix/coefficient_tables.py), [Section 5.2 appendix](section_5_2/appendix/)
- Shared modules cover data access, timing, covariates, and pipeline functions.
  [survival_data.py](survival_data.py), [event_processing.py](event_processing.py), [survival_utils.py](survival_utils.py), [survival_pipeline.py](survival_pipeline.py)

Run modules from the repository root with the Python environment from the
[installation instructions](../README.md). The root README contains the
offline figure and table commands and the [paper data index](../paper_data/README.md)
lists their inputs.

## Rebuilding the estimates

A reviewer can refit January 2021 KM and AFT models offline with
`python one_month_test_run.py` from the repository root. To start from Binance
trades for the same month, limit the production runners to BTC linear and a
1-second sampling interval. Both commands download the trade archives they need.

```sh
python -m final_code.section_4.run_vecm --markets btc_um --start 2021-01-01 --end 2021-02-01 --latencies 1s --output results/raw/vecm
python -m final_code.section_5_1.detect_events --markets btc_um --start 2021-01-01 --end 2021-02-01 --grids 1s --significance 0.01 --output results/raw/km
```

The UTC end date is exclusive. The VECM runner defaults to all four markets,
all paper latencies and 2021–2025 when those options are omitted.
`plot_vecm --from-run` reads
the hourly result files under `vecm_hasbrouck3_*`. The corrected ILS appendix
renderer reads the included summary CSVs and writes to
`results/vecm/appendix/`. To recompute those summaries,
the separate `final_code.section_4.appendix.corrected_ils.ils` module needs
the corrected interval inputs in `hasbrouck_spec_audit/ils_correction/` and
the historical VECM and full-grid audit results.

Run the KM events notebook for the V8 production configuration.
It builds the daily event files and monthly AFT inputs using Binance trade,
funding, candle, and futures-metric data. The notebook can be run in an
editor with Jupyter support or from the command line. [km_events.ipynb](section_5_1/km_events.ipynb)

```sh
python -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir results final_code/section_5_1/km_events.ipynb
```

`plot_december_km --from-run` refits the December curves from the daily Parquets.
`plot_km_curves` draws the monthly and annual KM panels from saved curves by
default. To refit them, pass `--input` with your full run directory, which
contains the production manifest, daily events, and monthly Parquets. A fresh
checkout uses `results/full/` for the checkpoints. The panels go in
`results/km/appendix/`.

For Section 5.2, `plot_open_interest --refresh` fetches the raw metrics and
`build_aft_covariates` uses them with the monthly inputs to reconstruct the
unit-matched open-interest and log covariates. Then run
`fit_aft` and `fit_univariate` to
refit the monthly models. `plot_aft_results --from-run` and
`plot_diagnostics --from-run` use those local estimates or inputs.
The coefficient, univariate, and midperiod table commands accept `--input`
with a local run directory.

The full 2021–2025 rebuild needs the original multi-gigabyte event and
covariate checkpoints or access to the underlying data sources.
