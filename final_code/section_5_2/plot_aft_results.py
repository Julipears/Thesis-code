"""Generate reduced V8 AFT coefficient heatmaps and concordance plots."""

import argparse
from pathlib import Path

import pandas as pd

from final_code import FULL_RUN, PAPER_DATA, RESULTS
from final_code.section_5_2 import aft_plot_utils as base


base.INPUT_DIR = FULL_RUN / "aft_data_liquidity_v5_log_covariates"
base.RESULTS_DIR = FULL_RUN / "aft_results_v5_without_spot_volatility"
base.OUTPUT = RESULTS / "aft"
base.COVARIATES = [
    "basis_5min",
    "fundingRate_bps",
    "log_open_interest",
    "shock_size_bps",
    "spread_spot_5min",
    "log_taker_long_short",
    "log_volume",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perp-only", action="store_true", help="Render Figure 5 from saved standardized coefficients.")
    parser.add_argument("--from-run", action="store_true", help="Standardize the coefficients using the local fit and covariate files.")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    source = PAPER_DATA / "section_5_2"
    if args.from_run:
        manifest = base.load_manifest()
        coefficients = base.coefficient_rows(manifest)
    else:
        manifest = pd.read_csv(source / "aft_fit_manifest.csv.gz", float_precision="round_trip").query("specification == 'full'")
        coefficients = pd.read_csv(source / "standardized_coefs.csv", float_precision="round_trip")
    if args.output_dir is not None:
        base.OUTPUT = args.output_dir
    if args.perp_only:
        base.plot_heatmaps(coefficients, perp_only=True)
    else:
        base.write_results(coefficients, manifest)


if __name__ == "__main__":
    main()
