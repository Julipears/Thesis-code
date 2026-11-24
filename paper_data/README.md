# Paper data

The CSVs in this directory are the saved values used to reproduce the paper's
tables and figures. Files ending in `.csv.gz` are gzip-compressed CSVs. Pandas
can read them directly with `read_csv`. Market codes use `um` for linear
perpetuals and `cm` for inverse perpetuals. Dates and sample boundaries use UTC.

| Paper output | Data files |
| --- | --- |
| Figures 1–2, regime ILS | [Corrected spot ILS](section_4/spot_ils.csv) and [spot and perpetual ILS](section_4/ils_by_series.csv) |
| Regime summaries of all three price-discovery measures | [HIS, CS and ILS by regime](section_4/shares.csv) |
| Figures 3–4 and 11–20, KM curves | [Monthly KM curves](section_5_1/monthly_km.csv.gz) and [monthly event coverage](section_5_1/monthly_coverage.csv) |
| Figures 5 and 23, AFT coefficient heatmaps | [Standardized monthly coefficients](section_5_2/standardized_coefs.csv) |
| Figure 6, open interest | [Daily open interest in base units](section_5_2/daily_open_interest.csv) |
| Figure 7, perpetual volume | [Daily perpetual volumes](section_5_2/daily_volume.csv) |
| Figures 8–10 and Table 4, corrected ILS appendix | [regression.csv](../final_code/section_4/appendix/corrected_ils/data/regression.csv), [full_grid.csv](../final_code/section_4/appendix/corrected_ils/data/full_grid.csv), and [robustness.csv](../final_code/section_4/appendix/corrected_ils/data/robustness.csv) |
| Figures 21–22, Table 8 and Table 9, correlations and VIF | [Full diagnostics](section_5_2/diagnostics/full/) and [reduced diagnostics](section_5_2/diagnostics/reduced/) |
| Table 1, significance at 1% | [Significance percentages](section_5_2/significance.csv) |
| Tables 2–3, Johansen cointegration ranks | [Rank distribution](section_4/johansen_ranks.csv) |
| Table 5, events per day | [Daily event counts](section_5_1/daily_event_counts.csv.gz) and [mean event counts](section_5_1/mean_events_per_day.csv) |
| Table 7, covariate distributions | [Descriptive statistics](section_5_2/diagnostics/stats_wide.csv) |
| Table 10, AFT specification comparison | [Comparison summary](section_5_2/model_comparison.csv) and [monthly AIC and concordance](section_5_2/model_fits.csv) |
| Tables 11–19, AFT coefficients and likelihood comparisons | [Monthly fit coefficients and log-likelihoods](section_5_2/monthly_coefs.csv.gz) |

Table 6 defines the covariates. See the [analysis guide](../final_code/README.md)
for their construction.

## Export details

- The price-discovery summaries use the saved observed-bin hourly estimates
  and their corrected ILS values. The share summary shows means and medians
  for each market, latency, regime and series. `HIS_lower_bound` and
  `HIS_upper_bound` give the bounds across the two Cholesky orderings.
- The event counts come from the V8 production manifest at the one-second
  detection grid and 1% significance. The KM exports show the probability
  of remaining unresolved at 0.25-second steps through 120 seconds for the
  90% basis-resolution definition. The first 20 seconds supply the shorter
  panels. Annual curves use events pooled within each year.
- The AFT exports use the saved monthly log-logistic fits. The univariate
  likelihood comparisons use the null model evaluated on the matching rows.
  Table 10 uses all successful fits for concordance and fit counts and
  compares AIC within market, origin and month when all three models converged.

## Reviewer checkpoint

The two Parquet files contain January 2021 BTC linear events, durations,
censoring indicators, model covariates and the timestamps used to check
covariate timing. They contain the columns used from the saved monthly
covariate files for the paper's BTC linear fits. [Spot checkpoint](reviewer_sample/btc_um/spot_2021-01.parquet), [perp checkpoint](reviewer_sample/btc_um/perp_2021-01.parquet)
The reference CSVs contain the matching January rows from the paper's archived
KM curves and monthly AFT fits. [km_reference.csv](reviewer_sample/btc_um/km_reference.csv), [aft_reference.csv](reviewer_sample/btc_um/aft_reference.csv)

Run `python one_month_test_run.py` from the repository root to refit the KM and
AFT models and check the results. [Test script](../one_month_test_run.py)

A full re-estimation uses the trade archives and checkpoints described in the
[analysis guide](../final_code/README.md).
