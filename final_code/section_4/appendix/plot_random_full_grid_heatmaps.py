"""Render full-grid sampled results using the original regime heatmap functions."""
import json
from pathlib import Path
import sqlite3

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import vecm_plotting as plots


def main():
    source = Path("hasbrouck_spec_audit/random_10_days_month_full_grid")
    plots.HEATMAP_OUT = source / "regime_heatmaps"
    plots.HEATMAP_OUT.mkdir(parents=True, exist_ok=True)
    plots.LATENCIES = ["10ms", "100ms", "1s"]
    with sqlite3.connect(source / "results.sqlite") as db:
        rows = [row for (payload,) in db.execute("SELECT payload FROM fits")
                for row in json.loads(payload)["rows"]]
    data = pd.DataFrame(rows).rename(columns={"information_leadership_share": "ILS_mid"})
    data["interval"] = pd.to_datetime(data.hour_start_utc)
    assert not data.duplicated(["market", "interval", "latency", "series"]).any()
    assert np.isfinite(data.ILS_mid).all()
    assert data.ILS_mid.between(0, 1).all()
    sums = data.groupby(["market", "interval", "latency"]).ILS_mid.sum()
    assert np.allclose(sums, 1), "Spot and perpetual shares must be complementary"
    summary = plots.summarize(data)
    perspectives = plots.summarize_perspectives(data)
    assert (perspectives.model_intervals > 0).all()
    summary.to_csv(plots.HEATMAP_OUT / "spot_ils_summary.csv", index=False)
    perspectives.to_csv(plots.HEATMAP_OUT / "spot_perp_ils_summary.csv", index=False)
    data.groupby(["market", "latency"])["interval"].agg(
        first_interval="min", last_interval="max", model_intervals="nunique"
    ).reset_index().to_csv(plots.HEATMAP_OUT / "input_coverage.csv", index=False)
    suffix = "\nFull grid: 10 sampled days per month, one random hour per day"
    for markets, name, title in [
        (list(plots.CONFIGS), "all_contracts_spot", "VECM price leadership by regime"),
        (["btc_um", "btc_cm"], "btc_spot", "BTC VECM price leadership by regime"),
        (["eth_um", "eth_cm"], "eth_spot", "ETH VECM price leadership by regime"),
    ]:
        plots.draw(summary, markets, name, title + suffix)
    plots.draw_spot_perp(perspectives, list(plots.CONFIGS), "all_contracts_spot_perp",
                         "VECM price leadership by regime: spot and perpetual" + suffix)
    for market in plots.CONFIGS:
        plots.draw_spot_perp(perspectives, [market],
            f"{plots.FILE_LABELS[market]}_spot_perp_heatmap",
            f"{plots.CONFIGS[market][3]} VECM price leadership by regime" + suffix)
    (plots.HEATMAP_OUT / "README.md").write_text(
        "# Full-grid regime heatmaps\n\n"
        "Same plotting functions, regime boundaries, median ILS statistic, coolwarm palette, "
        "and symmetric colour normalization around 0.5 as the original observed-bin heatmaps. "
        "As in the original code, colour limits are calculated separately for each figure; "
        "compare annotated numbers when comparing separate figures.\n\n"
        "Only the sampled latencies (10ms, 100ms, 1s) are shown. Data comprise 10 sampled "
        "dates per month in 2021-2025, one random UTC hour per date, shared across contracts. "
        "There are 7,197 successful fits. BTC CM on 2024-06-12 at 22:00 UTC is missing at "
        "all three latencies because data retrieval failed. CSV files include cell counts. "
        "Original observed-bin heatmaps use a different, larger sample, so differences between "
        "those figures and these figures cannot be attributed solely to gap filling.\n",
        encoding="utf-8",
    )
    print(f"Created 8 PNGs, 8 PDFs and summary tables in {plots.HEATMAP_OUT}")
    print(perspectives.groupby("market").model_intervals.sum().to_string())


if __name__ == "__main__":
    main()
