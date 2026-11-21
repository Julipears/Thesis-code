"""Matched ILS robustness table for the random-hour full-grid audit."""
import json
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd

OUT = Path("hasbrouck_spec_audit/random_10_days_month_full_grid")
with sqlite3.connect(OUT / "results.sqlite") as db:
    data = pd.DataFrame([row for (payload,) in db.execute("SELECT payload FROM fits")
                         for row in json.loads(payload)["rows"]])
data = data.loc[data.series.eq("log_midpoint_perp")].copy()
data["interval"] = pd.to_datetime(data.hour_start_utc)
paired = []
for (market, latency), sample in data.groupby(["market", "latency"]):
    frames = []
    for path in sorted(Path(f"vecm_hasbrouck3_{market}").glob(
        f"*_alignedv2_1h_{latency}_10_{market.split('_')[1]}_results_*.csv"
    )):
        frame = pd.read_csv(path, usecols=["interval", "series", "ILS_mid"])
        frame = frame.loc[frame.series.eq("log_midpoint_perp")].copy()
        frame["interval"] = pd.to_datetime(frame.interval)
        frames.append(frame.loc[frame.interval.isin(sample.interval)])
    baseline = pd.concat(frames, ignore_index=True)
    stats = baseline.groupby("interval").ILS_mid.agg(["min", "max", "count", "size"])
    valid = stats.index[(stats["max"]-stats["min"] < 1e-6)
                        & (stats["count"] == stats["size"])
                        & stats["min"].ge(0) & stats["max"].le(1)]
    baseline = baseline.loc[baseline.interval.isin(valid)].drop_duplicates("interval")
    pair = sample.merge(baseline[["interval", "ILS_mid"]], on="interval", validate="one_to_one")
    pair = pair.rename(columns={"ILS_mid": "observed_ils", "information_leadership_share": "full_grid_ils"})
    assert np.isfinite(pair[["observed_ils", "full_grid_ils"]]).all().all()
    pair["absolute_difference_pp"] = 100*(pair.full_grid_ils-pair.observed_ils).abs()
    pair["tie"] = pair.full_grid_ils.eq(.5) | pair.observed_ils.eq(.5)
    pair["leadership_flip"] = ((pair.full_grid_ils-.5)*(pair.observed_ils-.5) < 0)
    paired.append(pair)
paired = pd.concat(paired, ignore_index=True)
assert not paired.duplicated(["market", "latency", "interval"]).any()
assert not paired.tie.any(), "Ties require explicit handling in the table"
paired.to_csv(OUT / "paired_ils_observed_vs_full_grid.csv", index=False)
summary = paired.groupby(["market", "latency"]).agg(
    matched_hours=("interval", "size"), mean_absolute_difference_pp=("absolute_difference_pp", "mean"),
    leadership_flips=("leadership_flip", "sum"), leadership_flip_fraction=("leadership_flip", "mean"),
).reset_index()
summary["leadership_flip_pct"] = 100*summary.leadership_flip_fraction
summary.to_csv(OUT / "paired_ils_summary.csv", index=False)
lines = [r"% Requires \usepackage{booktabs}", r"\begin{table}[htbp]", r"\centering",
         r"\caption{Sensitivity of information leadership shares (ILS) to filling empty sampling bins.}",
         r"\label{tab:ils-gap-filling}", r"\begin{tabular}{llrrr}", r"\toprule",
         r"Contract & Interval & Hours & \shortstack{Mean absolute\\$\Delta$ILS (pp)} & \shortstack{Leadership\\flips (\%)} \\",
         r"\midrule"]
for market in ["btc_um", "btc_cm", "eth_um", "eth_cm"]:
    for latency in ["1s", "100ms", "10ms"]:
        row = summary.loc[summary.market.eq(market) & summary.latency.eq(latency)].iloc[0]
        label = market.upper().replace("_", " ")
        spacing = {"1s": "1 s", "100ms": "100 ms", "10ms": "10 ms"}[latency]
        lines.append(f"{label} & {spacing} & {row.matched_hours:d} & {row.mean_absolute_difference_pp:.2f} & {row.leadership_flip_pct:.2f} " + r"\\")
    if market != "eth_cm":
        lines.append(r"\addlinespace")
lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
          r"\begin{minipage}{\textwidth}\footnotesize",
          r"\textit{Notes:} Each comparison uses the same contract, date, UTC hour, and sampling interval with gap filling enabled and disabled. Mean absolute differences are expressed in percentage points. A leadership flip occurs when the perpetual ILS crosses 0.5 between specifications. Dates comprise ten randomly sampled days per month in 2021--2025, with one random hour per date. Only valid, unambiguous matched results are included.",
          r"\end{minipage}", r"\end{table}"]
(OUT / "ils_gap_filling_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary.to_string(index=False))
print("Matched fits:", len(paired))
