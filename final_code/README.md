# Final paper code

The source used for the paper is grouped by paper section:

- `section_4/`: Hasbrouck VECM estimation and main VECM figures.
- `section_4/appendix/`: Johansen diagnostics, complete-grid robustness, and
  Appendix Figure 11–13/table generators.
- `section_5_1/`: KM-V8 event identification, covariate construction, and
  December KM figures.
- `section_5_2/`: AFT fitting and main AFT figures/tables.
- `section_5_2/appendix/`: AFT diagnostics, open-interest/volume figures, and
  appendix coefficient/comparison tables.
- The files directly under `final_code/` are shared data-access, timing,
  covariate, and AFT-pipeline functions.

Run modules from the repository root so their relative output paths continue
to point to the existing result folders. For example:

```powershell
python -m final_code.section_4.vecm_plotting
python -m final_code.section_5_1.regenerate_km_all_decembers
python -m final_code.section_5_2.plot_aft_v8_log_covariate_diagnostics
```

The generated data and figures remain in the existing Git-ignored output
folders described in `PAPER_PIPELINE_MANIFEST.md`.
