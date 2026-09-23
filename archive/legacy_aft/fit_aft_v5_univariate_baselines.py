"""Fit V5 univariate and intercept-only log-logistic AFT baselines.

The repeated ``shock size`` request is represented once.  The specifications
are fit on the existing V5 input files and write to separate result folders.
"""

from __future__ import annotations

import sys
import argparse

import fit_aft_v3_all_covariates as base


SPECS = {
    "shock_size": ("aft_results_v5_univariate_shock_size", ("shock_size_bps",)),
    "volatility": ("aft_results_v5_univariate_volatility", ("vol_spot_5min_pct",)),
    "spread": ("aft_results_v5_univariate_spread", ("spread_spot_5min",)),
    "null": ("aft_results_v5_null", ()),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", choices=tuple(SPECS), nargs="+", default=list(SPECS))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    # Run one specification at a time so each has an independent manifest and
    # likelihood output.  Workers=1 avoids the numerical/thread contention
    # observed in the full V5 refit.
    for name in args.spec:
        results_dir, covariates = SPECS[name]
        base.INPUT_DIR_NAME = "aft_data_liquidity_v5_log_covariates"
        base.RESULTS_DIR_NAME = results_dir
        base.COVARIATES = covariates
        sys.argv = ["fit_aft_v5_univariate_baselines.py", "--workers", str(args.workers)]
        print(f"[start] {name}: {covariates or 'intercept-only'}", flush=True)
        base.main()
        print(f"[complete] {name}", flush=True)


if __name__ == "__main__":
    main()
