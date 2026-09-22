"""Regenerate the BTC and ETH KM panels with every December, 2021--2025."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import pandas as pd
from km_v8_regular_lm_pilot import plot_asset_km_month_comparison

ROOT = Path("sa_results/km_v8_final_01")
MONTHS = tuple(f"{year}-12" for year in range(2021, 2026))
coverage = []
for market in ["btc_um", "btc_cm", "eth_um", "eth_cm"]:
    for first in ["spot", "perp"]:
        for month in MONTHS:
            files = list((ROOT / "events" / market).glob(f"{first}_{month}-*_1s.parquet"))
            coverage.append(dict(market=market, first=first, month=month, daily_files=len(files)))
            if len(files) != 31:
                raise RuntimeError(f"Incomplete December: {market} {first} {month}: {len(files)}/31 daily files")
pd.DataFrame(coverage).to_csv(ROOT / "analysis_outputs/km_december_coverage.csv", index=False)
manifest = []
for asset in ["btc", "eth"]:
    for horizon in [120.0, 20.0]:
        print(f"Plotting {asset} at {horizon:g}s, Decembers 2021--2025", flush=True)
        path = plot_asset_km_month_comparison(ROOT, asset=asset, months=MONTHS,
            grid="1s", horizon=horizon, plot_step_seconds=.25, require_complete=True)
        curves = pd.read_csv(path.with_suffix(".csv"))
        assert set(curves.period) == set(MONTHS)
        assert curves.groupby("period").apply(lambda x: len(x[["market", "first"]].drop_duplicates()), include_groups=False).eq(4).all()
        manifest.append(dict(asset=asset, horizon_seconds=horizon, path=str(path)))
        print(f"Saved {path}", flush=True)
pd.DataFrame(manifest).to_csv(ROOT / "analysis_outputs/km_graph_manifest.csv", index=False)
