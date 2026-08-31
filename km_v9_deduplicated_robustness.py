"""V9 robustness check: deduplicate same-bin V8 spot/perp LM events.

The Lee--Mykland statistic was estimated on a one-second grid, so detections in
the same one-second bin cannot be ordered reliably.  Such detections are saved
once as ``joint`` and excluded from the uniquely attributed ``spot`` and
``perp`` samples.  No subsecond origin is inferred from mapped trade times.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter


SOURCE_ROOT = Path("sa_results/km_v8_final_01")
OUTPUT_ROOT = Path("sa_results/km_v9_deduplicated_december_2025")
MONTH = "2025-12"
MARKETS = ("btc_um", "btc_cm", "eth_um", "eth_cm")
GRID = "1s"
METHODOLOGY_VERSION = "v9_same_1s_bin_joint_v1"


def _read_day(market: str, origin: str, day: str) -> pd.DataFrame:
    path = SOURCE_ROOT / "events" / market / f"{origin}_{day}_{GRID}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _normalize_bin(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if not out.empty:
        out["timestamp_bin"] = pd.to_datetime(out["timestamp_bin"])
        out = out.sort_values("start_ts").drop_duplicates("timestamp_bin", keep="first")
    return out


def deduplicate_day(market: str, day: str) -> dict[str, pd.DataFrame]:
    spot = _normalize_bin(_read_day(market, "spot", day))
    perp = _normalize_bin(_read_day(market, "perp", day))
    spot_bins = set(spot.get("timestamp_bin", []))
    perp_bins = set(perp.get("timestamp_bin", []))
    joint_bins = spot_bins & perp_bins

    spot_only = spot[~spot["timestamp_bin"].isin(joint_bins)].copy()
    perp_only = perp[~perp["timestamp_bin"].isin(joint_bins)].copy()

    # Retain a joint event once.  The earlier constituent row is used only to
    # carry its already-computed basis-resolution outcome; it is not treated as
    # evidence that the corresponding market originated the shock.
    spot_joint = spot[spot["timestamp_bin"].isin(joint_bins)].copy()
    perp_joint = perp[perp["timestamp_bin"].isin(joint_bins)].copy()
    joint = pd.concat([spot_joint, perp_joint], ignore_index=True)
    if not joint.empty:
        joint["_constituent_origin"] = joint["first"]
        joint = joint.sort_values(["timestamp_bin", "start_ts"]).drop_duplicates(
            "timestamp_bin", keep="first"
        )
        joint["first"] = "joint"
        joint["origin_at_resolution"] = "joint_at_1s"
        joint["origin_confidence"] = "unidentified_within_1s"

    for origin, frame in (("spot", spot_only), ("perp", perp_only)):
        frame["first"] = origin
        frame["origin_at_resolution"] = origin
        frame["origin_confidence"] = "unique_1s_detection"

    return {"spot": spot_only, "perp": perp_only, "joint": joint}


def _fit_curve(frame: pd.DataFrame, timeline: np.ndarray) -> np.ndarray:
    clean = frame.dropna(subset=["Length", "Status"])
    if clean.empty:
        return np.full(len(timeline), np.nan)
    kmf = KaplanMeierFitter().fit(
        clean["Length"].astype(float), clean["Status"].astype(int)
    )
    return kmf.survival_function_at_times(timeline).to_numpy()


def plot_comparison(asset: str, pooled: dict[tuple[str, str], pd.DataFrame]) -> Path:
    timeline = np.arange(0, 120.0001, 0.25)
    panels = ((f"{asset}_um", "spot", "Linear: spot"),
              (f"{asset}_um", "perp", "Linear: perp"),
              (f"{asset}_cm", "spot", "Inverse: spot"),
              (f"{asset}_cm", "perp", "Inverse: perp"))
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharex=True, sharey=True)
    for ax, (market, origin, title) in zip(axes, panels):
        original = pd.concat(
            [_read_day(market, origin, f"{MONTH}-{d:02d}") for d in range(1, 32)],
            ignore_index=True,
        )
        unique = pooled[(market, origin)]
        ax.plot(timeline, _fit_curve(original, timeline), color="0.55", ls="--",
                label=f"V8 original (n={len(original):,})")
        ax.plot(timeline, _fit_curve(unique, timeline), color="tab:blue",
                label=f"V9 unique (n={len(unique):,})")
        ax.set_title(title)
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Probability basis remains unresolved")
    for ax in axes:
        ax.set_xlabel("Seconds since event")
    fig.suptitle(f"{asset.upper()}: effect of same-bin event deduplication, December 2025")
    fig.tight_layout()
    path = OUTPUT_ROOT / "plots" / f"km_v9_dedup_comparison_{asset}_2025-12.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_v8_format_side_by_side(
    pooled: dict[tuple[str, str], pd.DataFrame], horizon: float = 120.0
) -> Path:
    """V8 thesis styling, with BTC and ETH replacing the four month panels."""
    timeline = np.arange(0, horizon + 0.125, 0.25)
    styles = {
        ("um", "spot"): ("Linear: spot", "tab:blue", "-"),
        ("um", "perp"): ("Linear: perp", "tab:blue", "--"),
        ("cm", "spot"): ("Inverse: spot", "tab:orange", "-"),
        ("cm", "perp"): ("Inverse: perp", "tab:orange", "--"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharex=True, sharey=True)
    for ax, asset in zip(axes, ("btc", "eth")):
        for (contract, origin), (label, color, linestyle) in styles.items():
            frame = pooled[(f"{asset}_{contract}", origin)]
            ax.plot(
                timeline,
                _fit_curve(frame, timeline),
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=1.7,
            )
        ax.set_title(asset.upper())
        ax.set_xlim(0, horizon)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Seconds since basis displacement")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Probability unresolved")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, 0.91),
    )
    fig.suptitle("Kaplan–Meier curves for 90% basis resolution", y=0.98)
    fig.subplots_adjust(top=0.75, bottom=0.15, left=0.08, right=0.98, wspace=0.08)
    suffix = "20s" if horizon == 20 else "120s"
    path = OUTPUT_ROOT / "plots" / f"survival_all_graphs_btc_eth_v9_2025-12_{suffix}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    rows = []
    pooled: dict[tuple[str, str], list[pd.DataFrame]] = {
        (market, origin): [] for market in MARKETS for origin in ("spot", "perp", "joint")
    }
    for market in MARKETS:
        for d in range(1, 32):
            day = f"{MONTH}-{d:02d}"
            groups = deduplicate_day(market, day)
            for origin, frame in groups.items():
                frame = frame.copy()
                frame["methodology_version"] = METHODOLOGY_VERSION
                out = OUTPUT_ROOT / "events" / market / f"{origin}_{day}_{GRID}.parquet"
                out.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(out, index=False)
                pooled[(market, origin)].append(frame)
            original_spot = len(_read_day(market, "spot", day))
            original_perp = len(_read_day(market, "perp", day))
            rows.append({
                "market": market, "date": day,
                "v8_spot": original_spot, "v8_perp": original_perp,
                "v9_spot_only": len(groups["spot"]),
                "v9_perp_only": len(groups["perp"]),
                "v9_joint": len(groups["joint"]),
                "v9_unknown": 0,
                "methodology_version": METHODOLOGY_VERSION,
            })
    pooled_frames = {
        key: pd.concat(parts, ignore_index=True) for key, parts in pooled.items()
    }
    summary = pd.DataFrame(rows)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_ROOT / "deduplication_summary_daily.csv", index=False)
    totals = summary.groupby("market", as_index=False).sum(numeric_only=True)
    totals.to_csv(OUTPUT_ROOT / "deduplication_summary_totals.csv", index=False)
    paths = [plot_comparison(asset, pooled_frames) for asset in ("btc", "eth")]
    paths.extend([
        plot_v8_format_side_by_side(pooled_frames, horizon=120.0),
        plot_v8_format_side_by_side(pooled_frames, horizon=20.0),
    ])
    print(totals.to_string(index=False))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
