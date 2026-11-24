"""Refit and validate the January 2021 BTC linear KM and AFT models offline."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from final_code import PAPER_DATA, PROJECT_ROOT, RESULTS
from final_code.section_5_1.detect_events import _weighted_km_inputs
from final_code.section_5_2.fit_aft import fit_one
from final_code.survival_utils import atomic_pandas_csv, validate_augmented_timing


MARKET = "btc_um"
MONTH = "2021-01"
SAMPLE_DIR = PAPER_DATA / "reviewer_sample" / MARKET
DEFAULT_OUTPUT = PROJECT_ROOT / RESULTS / "test"


def run(output: Path) -> None:
    km_reference = pd.read_csv(SAMPLE_DIR / "km_reference.csv", float_precision="round_trip")
    aft_reference = pd.read_csv(SAMPLE_DIR / "aft_reference.csv", float_precision="round_trip")

    curves, fits = [], []
    for first in ("spot", "perp"):
        source = SAMPLE_DIR / f"{first}_{MONTH}.parquet"
        validate_augmented_timing(pd.read_parquet(source), source.name)

        expected = km_reference.loc[km_reference["first"].eq(first)].sort_values("timeline")
        durations, observed, weights = _weighted_km_inputs([source], horizon=120.0)
        n_events = int(weights.sum())
        if not expected["n_events"].eq(n_events).all() or len(expected) != 481:
            raise ValueError(f"KM checkpoint coverage differs from the paper for {first} in {MONTH}.")
        km = KaplanMeierFitter().fit(durations, observed, weights=weights)
        survival = km.survival_function_at_times(expected["timeline"]).to_numpy()
        np.testing.assert_allclose(survival, expected["survival"], atol=1e-12, rtol=0)
        curves.append(expected.assign(survival=survival))

        coefficient_path = output / f"aft_{first}.csv"
        fit = fit_one(source, MARKET, overwrite=True, coefficient_path=coefficient_path)
        if fit["status"] != "complete":
            raise RuntimeError(f"AFT refit for {first} in {MONTH} failed: {fit}")
        actual = pd.read_csv(coefficient_path).set_index(["param", "covariate"]).sort_index()
        reference = aft_reference.loc[aft_reference["first"].eq(first)].set_index(["param", "covariate"]).sort_index()
        pd.testing.assert_index_equal(actual.index, reference.index)
        np.testing.assert_allclose(actual["coef"], reference["coef"], atol=1e-5, rtol=1e-4)
        np.testing.assert_allclose(fit["log_likelihood"], reference["log_likelihood"], rtol=1e-7)
        fits.append({**fit, "coefficient_file": coefficient_path.name})
        print(f"{first.capitalize()} origin used {n_events:,} KM events and {fit['complete_rows']:,} AFT rows. Both fits matched the paper.")

    atomic_pandas_csv(pd.concat(curves, ignore_index=True), output / "km.csv")
    atomic_pandas_csv(pd.DataFrame(fits), output / "aft_fits.csv")
    saved_to = output.relative_to(PROJECT_ROOT) if output.is_relative_to(PROJECT_ROOT) else output
    print(f"Saved results to {saved_to.as_posix()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Directory for the refitted curves and coefficients.")
    run(parser.parse_args().output)


if __name__ == "__main__":
    main()
