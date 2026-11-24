"""Regenerate the BTC and ETH KM panels for December 2021–2023."""
import argparse

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from final_code import FULL_RUN, PAPER_DATA, RESULTS
from final_code.section_5_1.detect_events import plot_asset_km_month_comparison

MONTHS = ("2021-12", "2022-12", "2023-12")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-run", action="store_true", help="Fit the curves from the local daily event files.")
    args = parser.parse_args()
    if args.from_run:
        saved_curves = None
        coverage = pd.DataFrame([
            {"market": market, "first": first, "month": month,
             "daily_files": len(list((FULL_RUN / "events" / market).glob(f"{first}_{month}-*_1s.parquet")))}
            for market in ("btc_um", "btc_cm", "eth_um", "eth_cm")
            for first in ("spot", "perp") for month in MONTHS
        ])
    else:
        saved_curves = pd.read_csv(PAPER_DATA / "section_5_1/monthly_km.csv.gz", float_precision="round_trip")
        coverage = saved_curves.loc[
            saved_curves["period"].isin(MONTHS), ["market", "first", "period", "available_days"]
        ].drop_duplicates().rename(columns={"period": "month", "available_days": "daily_files"})
    if len(coverage) != 24 or not coverage["daily_files"].eq(31).all():
        raise ValueError("December KM panels require 31 days for every month, market and origin.")

    output = RESULTS / "km" / "december"
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for asset in ("btc", "eth"):
        for horizon in (120.0, 20.0):
            path = plot_asset_km_month_comparison(
                FULL_RUN, asset=asset, months=MONTHS, horizon=horizon,
                saved_curves=saved_curves, output_path=output / f"{asset}_{horizon:g}s.png",
            )
            manifest.append({"asset": asset, "horizon_seconds": horizon, "path": path.name})
            print(f"Saved {path}", flush=True)
    coverage.to_csv(output / "coverage.csv", index=False)
    pd.DataFrame(manifest).to_csv(output / "plots.csv", index=False)


if __name__ == "__main__":
    main()
