"""Matched-sample log-logistic AFT ablation for shock_size_bps."""

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
OUTPUT = ROOT / "aft_shock_size_ablation"
FULL_COVARIATES = [
    "basis_5min",
    "fundingRate_bps",
    "open_interest",
    "shock_size_bps",
    "spread_spot_5min",
    "taker_long_short",
    "volume",
]
REDUCED_COVARIATES = [c for c in FULL_COVARIATES if c != "shock_size_bps"]


def fit_concordance(frame: pd.DataFrame, covariates: list[str]) -> float:
    columns = ["Length", "Status", *covariates]
    frame = frame[columns].copy()
    # Scaling is rank-preserving and improves numerical conditioning without
    # changing the fitted model's concordance in exact arithmetic.
    for column in covariates:
        scale = frame[column].std(ddof=0)
        if scale > 0:
            frame[column] = (frame[column] - frame[column].mean()) / scale
    fitter = LogLogisticAFTFitter()
    fitter.fit(frame, duration_col="Length", event_col="Status", ancillary=False)
    return float(fitter.concordance_index_)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    monthly_path = OUTPUT / "monthly_matched_ablation.csv"
    prior = pd.read_csv(monthly_path) if monthly_path.exists() else pd.DataFrame()
    retained = prior[~prior.get("status", pd.Series(dtype=str)).eq("failed")].copy()
    completed = set(zip(retained.get("market", []), retained.get("first", []),
                        retained.get("period", [])))
    rows = retained.to_dict("records")
    files = sorted(INPUTS.glob("*/*.parquet"))

    for i, path in enumerate(files, 1):
        market = path.parent.name
        first, period = path.stem.split("_", 1)
        key = (market, first, period)
        if key in completed:
            continue
        record = {"market": market, "first": first, "period": period,
                  "source_file": str(path)}
        try:
            # Construct the sample with the full covariate list first. Both models
            # therefore use exactly the same observations.
            sample, diagnostics = prepare_aft_data(path, FULL_COVARIATES)
            record.update(diagnostics)
            if len(sample) < 100:
                record["status"] = "skipped_too_few_observations"
            else:
                record["concordance_full"] = fit_concordance(sample, FULL_COVARIATES)
                record["concordance_without_shock_size"] = fit_concordance(
                    sample, REDUCED_COVARIATES
                )
                record["concordance_loss"] = (
                    record["concordance_full"]
                    - record["concordance_without_shock_size"]
                )
                record["status"] = "complete"
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = repr(exc)
        rows.append(record)
        atomic_pandas_csv(
            pd.DataFrame(rows).sort_values(["market", "first", "period"]), monthly_path
        )
        print(f"[{i}/{len(files)}] {market} {first} {period} {record['status']}", flush=True)
        gc.collect()

    result = pd.DataFrame(rows)
    complete = result[result["status"].eq("complete")].copy()
    complete["period_date"] = pd.to_datetime(complete["period"])
    complete["sample_period"] = complete["period_date"].lt("2023-05-01").map(
        {True: "2021--April 2023", False: "May 2023--2025"}
    )
    summary = (
        complete.groupby(["market", "first", "sample_period"], as_index=False)
        .agg(
            months=("period", "nunique"),
            mean_full_concordance=("concordance_full", "mean"),
            mean_without_shock_size=("concordance_without_shock_size", "mean"),
            mean_concordance_loss=("concordance_loss", "mean"),
            median_concordance_loss=("concordance_loss", "median"),
        )
    )
    atomic_pandas_csv(summary, OUTPUT / "ablation_summary_by_period.csv")

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
        ax.axvline(pd.Timestamp("2023-05-01"), color="grey", linestyle=":")
        ax.set_title(asset.upper())
        ax.set_xlabel("Month")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Concordance loss after removing shock size")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Matched-sample shock-size ablation", y=1.13)
    fig.tight_layout()
    fig.savefig(OUTPUT / "shock_size_ablation_concordance_loss.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
