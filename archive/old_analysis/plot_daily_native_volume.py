"""Four-panel native traded quantity: coin units for linear, contracts for inverse."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

OUT = Path("daily_perpetual_volume_figures")
data = pd.read_csv(OUT / "daily_perpetual_volumes_2021_2025.csv", parse_dates=["date"])
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=False)
for row, asset in enumerate(["btc", "eth"]):
    for col, (margin, label, color) in enumerate([
        ("um", "Linear", "tab:blue"), ("cm", "Inverse", "tab:orange")
    ]):
        ax = axes[row, col]
        frame = data.loc[data.market.eq(f"{asset}_{margin}")].sort_values("date")
        expected_unit = asset.upper() if margin == "um" else "contracts"
        assert frame.native_unit.eq(expected_unit).all()
        assert len(frame) == 1826 and frame.date.nunique() == 1826
        assert np.isfinite(frame.native_volume).all() and frame.native_volume.ge(0).all()
        ax.plot(frame.date, frame.native_volume, color=color, linewidth=.8)
        ax.set_title(f"{asset.upper()} — {label}")
        ax.set_ylabel(f"Daily traded quantity ({expected_unit})")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.set_ylim(0, float(frame.native_volume.max()) * 1.05)
        ax.set_xlim(pd.Timestamp("2021-01-01"), pd.Timestamp("2026-01-01"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(alpha=.2)
        if row == 1:
            ax.set_xlabel("Year")
fig.suptitle("Daily perpetual traded quantity")
fig.text(.5, .01, "Daily totals, UTC; unsmoothed. Native exchange units: BTC/ETH for linear, contract counts for inverse.",
         ha="center", fontsize=9)
fig.tight_layout(rect=(0, .035, 1, .96))
for extension in ["png", "pdf"]:
    path = OUT / f"all_contracts_daily_volume_native_2x2.{extension}"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(path)
plt.close(fig)
