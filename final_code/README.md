# Final paper code

The source used for the paper is grouped by paper section:

- `section_4/`: Hasbrouck VECM estimation and main VECM figures.
- `section_4/appendix/`: Johansen diagnostics, complete-grid robustness, and
  Appendix Figure 11–13/table generators.
- `section_5_1/`: KM-V8 event identification, covariate construction, and
  December KM figures.
- `section_5_1/appendix/`: monthly KM curves and 120-second panels for
  Appendix E.2, Figures 11–20.
- `section_5_2/`: AFT fitting and main AFT figures/tables.
- `section_5_2/appendix/`: AFT diagnostics, open-interest/volume figures, and
  appendix coefficient/comparison tables.
- The files directly under `final_code/` are shared data-access, timing,
  covariate, and AFT-pipeline functions.

Run modules from the repository root so their relative output paths continue
to point to the existing result folders. For example:

```powershell
python -m final_code.section_4.plot_vecm
python -m final_code.section_5_1.plot_december_km
python -m final_code.section_5_1.appendix.plot_km_curves
python -m final_code.section_5_1.appendix.export_km_rows
python -m final_code.section_5_2.plot_aft_results
python -m final_code.section_5_2.appendix.plot_aft_diagnostics
```

The generated data and figures remain in the existing Git-ignored output
folders described in the repository `README.md`.
The Appendix E.2 commands run in order and write to
`sa_results/km_v8_final_01/appendix_km/` by default.
Keep the KM run's `comparison_summary.csv`, the daily `events/` files,
and one complete set of the monthly event Parquets in
`aft_data_monthly/`, `aft_data_liquidity_monthly/`, or
`aft_data_liquidity_v5_log_covariates/`.
