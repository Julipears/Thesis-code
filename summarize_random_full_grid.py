"""Summarize random-hour full-grid results and match saved observed-bin fits."""
from pathlib import Path
import json
import sqlite3
import numpy as np
import pandas as pd

OUT = Path("hasbrouck_spec_audit/random_10_days_month_full_grid")
with sqlite3.connect(OUT / "results.sqlite") as db:
    payloads = [json.loads(x[0]) for x in db.execute("SELECT payload FROM fits")]
results = pd.DataFrame([r for p in payloads for r in p["rows"]])
perp = results.loc[results.series.eq("log_midpoint_perp")].copy()
perp["interval"] = pd.to_datetime(perp.hour_start_utc)
assert not perp.duplicated(["market", "interval", "latency"]).any()
assert np.isfinite(perp.his_mid).all() and perp.his_mid.between(0, 1).all()
perp["perp_leads"] = perp.his_mid > .5
summary = perp.groupby(["market", "latency"]).agg(
    hours=("his_mid", "size"), mean_perp_is=("his_mid", "mean"),
    median_perp_is=("his_mid", "median"), perp_leader_fraction=("perp_leads", "mean"),
    mean_lower=("his_lower_corrected", "mean"), mean_upper=("his_upper_corrected", "mean"),
).reset_index()
summary.to_csv(OUT / "full_grid_summary.csv", index=False)
paired_parts = []
coverage = []
for (market, latency), sample in perp.groupby(["market", "latency"]):
    folder = Path(f"vecm_hasbrouck3_{market}")
    margin = market.split("_")[1]
    files = sorted(folder.glob(f"*_alignedv2_1h_{latency}_10_{margin}_results_*.csv"))
    frames = []
    wanted = set(sample.interval)
    for file in files:
        frame = pd.read_csv(file, usecols=["interval", "series", "HIS_mid"])
        frame = frame.loc[frame.series.eq("log_midpoint_perp")].copy()
        frame["interval"] = pd.to_datetime(frame.interval)
        frame = frame.loc[frame.interval.isin(wanted)]
        frames.append(frame)
    if not frames:
        coverage.append(dict(market=market, latency=latency, matched=0, available=len(sample)))
        continue
    baseline = pd.concat(frames, ignore_index=True)
    spread = baseline.groupby("interval").HIS_mid.agg(lambda x: x.max()-x.min())
    ambiguous = spread.index[~(spread < 1e-6)]
    if len(ambiguous):
        print(f"EXCLUDED {market} {latency}: {len(ambiguous)} hours with conflicting/nonfinite baseline duplicates", flush=True)
        baseline = baseline.loc[~baseline.interval.isin(ambiguous)]
    baseline = baseline.drop_duplicates("interval").rename(columns={"HIS_mid": "observed_perp_is"})
    pair = sample.merge(baseline[["interval", "observed_perp_is"]], on="interval", validate="one_to_one")
    coverage.append(dict(market=market, latency=latency, matched=len(pair), available=len(sample), ambiguous_baseline_hours=len(ambiguous)))
    pair["delta_pp"] = 100*(pair.his_mid - pair.observed_perp_is)
    pair["abs_delta_pp"] = pair.delta_pp.abs()
    pair["leader_flip"] = (pair.his_mid > .5) != (pair.observed_perp_is > .5)
    paired_parts.append(pair)
paired = pd.concat(paired_parts, ignore_index=True)
paired.to_csv(OUT / "paired_observed_vs_full_grid.csv", index=False)
comparison = paired.groupby(["market", "latency"]).agg(
    hours=("his_mid", "size"), mean_observed_perp_is=("observed_perp_is", "mean"),
    mean_full_grid_perp_is=("his_mid", "mean"), mean_delta_pp=("delta_pp", "mean"),
    mean_absolute_delta_pp=("abs_delta_pp", "mean"), median_absolute_delta_pp=("abs_delta_pp", "median"),
    p95_absolute_delta_pp=("abs_delta_pp", lambda x: x.quantile(.95)),
    leader_flips=("leader_flip", "sum"), leader_flip_fraction=("leader_flip", "mean"),
).reset_index()
comparison.to_csv(OUT / "paired_summary.csv", index=False)
pd.DataFrame(coverage).to_csv(OUT / "comparison_coverage.csv", index=False)
print("FULL GRID\n" + summary.to_string(index=False))
print("PAIRED COMPARISON\n" + comparison.to_string(index=False))
print("MATCHED", len(paired), "OF", len(perp))
