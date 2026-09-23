"""Plot monthly medians of saved hourly Hasbrouck 3 adjustment coefficients."""
from pathlib import Path
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

OUT = Path("vecm_hasbrouck3_alpha_plots")
LATENCIES = ["10ms", "50ms", "100ms", "200ms", "500ms", "1s"]


def main():
    OUT.mkdir(exist_ok=True)
    summaries, coverage = [], []
    for asset in ["btc", "eth"]:
        for margin in ["um", "cm"]:
            market = f"{asset}_{margin}"
            folder = Path(f"vecm_hasbrouck3_{market}")
            for latency in LATENCIES:
                pattern = re.compile(rf"hasbrouck3_{asset}_alignedv2_1h_{latency}_10_{margin}_results_\d{{8}}_\d{{8}}\.csv$")
                paths = sorted(p for p in folder.glob("*results*.csv") if pattern.fullmatch(p.name))
                frames = [pd.read_csv(p, usecols=["interval", "series", "alpha"]) for p in paths]
                if not frames:
                    raise RuntimeError(f"No data for {market} {latency}")
                data = pd.concat(frames, ignore_index=True)
                data["interval"] = pd.to_datetime(data.interval, errors="coerce")
                data["alpha"] = pd.to_numeric(data.alpha, errors="coerce")
                data = data.loc[data.interval.ge("2021-01-01") & data.interval.lt("2026-01-01")]
                keys = ["interval", "series"]
                # Exclude overlapping saved estimates that disagree materially.
                stats = data.groupby(keys).alpha.agg(["min", "max", "count", "size"])
                valid = (stats["max"]-stats["min"] <= 1e-10+1e-6*stats[["min", "max"]].abs().max(axis=1))
                valid &= stats["count"].eq(stats["size"]) & np.isfinite(stats["min"]) & np.isfinite(stats["max"])
                good_keys = stats.index[valid]
                data = data.set_index(keys).loc[lambda x: x.index.isin(good_keys)].reset_index().drop_duplicates(keys)
                data["month"] = data.interval.dt.to_period("M").dt.to_timestamp()
                monthly = data.groupby(["month", "series"]).alpha.agg(
                    median_alpha="median", mean_alpha="mean", hourly_models="size").reset_index()
                monthly["market"] = market
                monthly["latency"] = latency
                summaries.append(monthly)
                coverage.append(dict(market=market, latency=latency, files=len(paths),
                                     valid_hourly_rows=len(data), excluded_ambiguous_or_invalid_keys=int((~valid).sum())))
            print(f"Loaded {market}", flush=True)
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(OUT / "monthly_alpha_summary.csv", index=False)
    pd.DataFrame(coverage).to_csv(OUT / "input_coverage.csv", index=False)
    months = pd.date_range("2021-01-01", "2025-12-01", freq="MS")
    for asset in ["btc", "eth"]:
        fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True, constrained_layout=True)
        for ax, latency in zip(axes.flat, LATENCIES):
            for margin, color in [("um", "tab:blue"), ("cm", "tab:orange")]:
                for series, style in [("log_midpoint_spot", "-"), ("log_midpoint_perp", "--")]:
                    values = summary.loc[summary.market.eq(f"{asset}_{margin}") & summary.latency.eq(latency)
                                         & summary.series.eq(series)].set_index("month").median_alpha.reindex(months)
                    assert values.notna().any()
                    ax.plot(values.index, values.values, color=color, linestyle=style, linewidth=1.7)
            ax.axhline(0, color="0.45", linewidth=.7)
            ax.set_title(f"Sampling interval: {latency}")
            ax.set_ylabel(r"Adjustment coefficient $\alpha$")
            ax.grid(alpha=.18)
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.set_xlim(pd.Timestamp("2021-01-01"), pd.Timestamp("2026-01-01"))
        handles = [Line2D([0], [0], color=color, linestyle=style, linewidth=1.8, label=f"{contract} — {series}")
                   for contract, color in [("Linear", "tab:blue"), ("Inverse", "tab:orange")]
                   for series, style in [("Spot", "-"), ("Perpetual", "--")]]
        fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
        fig.suptitle(f"{asset.upper()}: error-correction coefficients over time\nMonthly median of hourly models; basis = log spot − log perpetual", fontsize=14)
        fig.savefig(OUT / f"{asset}_alpha_over_time.png", dpi=200, bbox_inches="tight")
        fig.savefig(OUT / f"{asset}_alpha_over_time.pdf", bbox_inches="tight")
        plt.close(fig)
    (OUT / "README.md").write_text(
        "# Alpha over time\n\nMonthly medians of saved hourly alpha estimates, taken directly over "
        "hourly models (no daily averaging). Separate panels retain each sampling interval's raw "
        "coefficient units; y-axis scales vary by panel. Source: alignedv2 results with lag base 10 "
        "in the four vecm_hasbrouck3 folders. Blue: linear; orange: inverse; solid: spot; dashed: "
        "perpetual. Basis orientation is log spot minus log perpetual.\n\n"
        "Duplicate hourly keys with conflicting alpha values are excluded; consistent duplicates "
        "are counted once. input_coverage.csv records exclusions, and monthly_alpha_summary.csv "
        "contains medians, means and observation counts.\n", encoding="utf-8")
    print(f"Saved figures and tables to {OUT}")


if __name__ == "__main__":
    main()
