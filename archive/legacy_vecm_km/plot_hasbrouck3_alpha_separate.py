"""Separate asset/latency plots from the saved monthly alpha summary."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

BASE = Path("vecm_hasbrouck3_alpha_plots")
OUT = BASE / "separate"
OUT.mkdir(exist_ok=True)
summary = pd.read_csv(BASE / "monthly_alpha_summary.csv", parse_dates=["month"])
months = pd.date_range("2021-01-01", "2025-12-01", freq="MS")
for asset in ["btc", "eth"]:
    for latency in ["10ms", "100ms", "1s"]:
        fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
        for margin, contract, color in [("um", "Linear", "tab:blue"), ("cm", "Inverse", "tab:orange")]:
            for series, label, style in [("log_midpoint_spot", "Spot", "-"), ("log_midpoint_perp", "Perpetual", "--")]:
                selected = summary.loc[summary.market.eq(f"{asset}_{margin}") & summary.latency.eq(latency)
                                       & summary.series.eq(series)]
                values = selected.set_index("month").median_alpha.reindex(months)
                assert values.notna().any()
                ax.plot(values.index, values, color=color, linestyle=style, linewidth=1.8,
                        label=f"{contract} — {label}")
        ax.axhline(0, color="0.45", linewidth=.7)
        ax.set_title(f"{asset.upper()} adjustment coefficients — {latency}\nMonthly median of hourly estimates", fontsize=13)
        ax.set_ylabel(r"Adjustment coefficient $\alpha$")
        ax.set_xlabel("Year")
        ax.grid(alpha=.18)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_xlim(pd.Timestamp("2021-01-01"), pd.Timestamp("2026-01-01"))
        ax.legend(loc="best", ncol=2, framealpha=.9)
        for extension in ["png", "pdf"]:
            fig.savefig(OUT / f"{asset}_alpha_{latency}.{extension}", dpi=200, bbox_inches="tight")
        plt.close(fig)
print(f"Saved six separate graphs as PNG and PDF to {OUT}")
