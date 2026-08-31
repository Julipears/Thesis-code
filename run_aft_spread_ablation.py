"""Matched-sample log-logistic AFT ablation for spread_spot_5min."""

from __future__ import annotations

import gc
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import LogLogisticAFTFitter

from survival_analysis_data_processing_final import prepare_aft_data
from survival_analysis_utils_final import atomic_pandas_csv


ROOT = Path("sa_results/km_v8_final_01")
INPUTS = ROOT / "aft_data_liquidity_monthly"
BASELINE = ROOT / "aft_shock_size_ablation" / "monthly_matched_ablation.csv"
OUTPUT = ROOT / "aft_spread_ablation"
FULL_COVARIATES = [
    "basis_5min", "fundingRate_bps", "open_interest", "shock_size_bps",
    "spread_spot_5min", "taker_long_short", "volume",
]
REDUCED_COVARIATES = [c for c in FULL_COVARIATES if c != "spread_spot_5min"]


def fit_reduced(frame: pd.DataFrame) -> float:
    columns = ["Length", "Status", *REDUCED_COVARIATES]
    model = frame[columns].copy()
    for column in REDUCED_COVARIATES:
        scale = model[column].std(ddof=0)
        if scale > 0:
            model[column] = (model[column] - model[column].mean()) / scale
    fitter = LogLogisticAFTFitter()
    fitter.fit(model, duration_col="Length", event_col="Status", ancillary=False)
    return float(fitter.concordance_index_)


def summarize(result: pd.DataFrame) -> pd.DataFrame:
    complete = result[result.status.eq("complete")].copy()
    complete["period_date"] = pd.to_datetime(complete.period)
    complete["sample_period"] = complete.period_date.le("2023-05-01").map(
        {True: "2021--May 2023", False: "June 2023--2025"}
    )
    return (
        complete.groupby(["market", "first", "sample_period"], as_index=False)
        .agg(
            months=("period", "nunique"),
            mean_full_concordance=("concordance_full", "mean"),
            mean_without_spread=("concordance_without_spread", "mean"),
            mean_concordance_loss=("concordance_loss", "mean"),
            median_concordance_loss=("concordance_loss", "median"),
        )
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT / "monthly_matched_ablation.csv"
    baseline = pd.read_csv(BASELINE)
    baseline = baseline[baseline.status.eq("complete")].copy()
    full_lookup = {
        (r.market, r.first, r.period): r.concordance_full
        for r in baseline.itertuples()
    }
    prior = pd.read_csv(output_path) if output_path.exists() else pd.DataFrame()
    retained = prior[~prior.get("status", pd.Series(dtype=str)).eq("failed")].copy()
    completed = set(zip(retained.get("market", []), retained.get("first", []),
                        retained.get("period", [])))
    rows = retained.to_dict("records")
    files = sorted(INPUTS.glob("*/*.parquet"))

    for i, path in enumerate(files, 1):
        market = path.parent.name
        first, period = path.stem.split("_", 1)
        key = (market, first, period)
        if key not in full_lookup or key in completed:
            continue
        record = {"market": market, "first": first, "period": period,
                  "source_file": str(path),
                  "concordance_full": full_lookup[key]}
        try:
            sample, diagnostics = prepare_aft_data(path, FULL_COVARIATES)
            record.update(diagnostics)
            record["concordance_without_spread"] = fit_reduced(sample)
            record["concordance_loss"] = (
                record["concordance_full"] - record["concordance_without_spread"]
            )
            record["status"] = "complete"
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = repr(exc)
        rows.append(record)
        atomic_pandas_csv(pd.DataFrame(rows).sort_values(
            ["market", "first", "period"]), output_path)
        print(f"[{i}/{len(files)}] {market} {first} {period} {record['status']}",
              flush=True)
        gc.collect()

    result = pd.DataFrame(rows)
    summary = summarize(result)
    atomic_pandas_csv(summary, OUTPUT / "ablation_summary_by_period.csv")

    complete = result[result.status.eq("complete")].copy()
    complete["period_date"] = pd.to_datetime(complete.period)
    styles = {("um", "spot"): ("Linear: spot", "tab:blue", "-"),
              ("um", "perp"): ("Linear: perp", "tab:blue", "--"),
              ("cm", "spot"): ("Inverse: spot", "tab:orange", "-"),
              ("cm", "perp"): ("Inverse: perp", "tab:orange", "--")}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    for ax, asset in zip(axes, ("btc", "eth")):
        for (suffix, first), (label, color, ls) in styles.items():
            z = complete[(complete.market == f"{asset}_{suffix}") &
                         (complete["first"] == first)].sort_values("period_date")
            ax.plot(z.period_date, z.concordance_loss, label=label, color=color,
                    linestyle=ls, linewidth=1.4)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(pd.Timestamp("2023-06-01"), color="grey", linestyle=":")
        ax.set_title(asset.upper())
        ax.set_xlabel("Month")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Concordance loss after removing spot spread")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Matched-sample spot-spread ablation", y=1.13)
    fig.tight_layout()
    fig.savefig(OUTPUT / "spread_ablation_concordance_loss.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
