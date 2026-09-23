"""Regenerate monthly KM--VECM comparisons using finalized V8 KM events."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter
from scipy.stats import pearsonr, spearmanr


ROOT = Path("sa_results/km_v8_final_01")
OUT = ROOT / "km_vecm_correlations_v8"
LATENCIES = ["1s", "500ms", "200ms", "100ms"]
LAG = 10
HORIZON = 10.0
VECM_FOLDERS = {
    ("btc", "um"): Path("vecm_hasbrouck2_um"),
    ("btc", "cm"): Path("vecm_hasbrouck2_cm"),
    ("eth", "um"): Path("vecm_hasbrouck3_eth_um"),
    ("eth", "cm"): Path("vecm_hasbrouck3_eth_cm"),
}
STYLES = {
    ("um", "spot"): ("Linear: spot", "tab:blue", "-", "o"),
    ("um", "perp"): ("Linear: perp", "tab:blue", "--", "s"),
    ("cm", "spot"): ("Inverse: spot", "tab:orange", "-", "o"),
    ("cm", "perp"): ("Inverse: perp", "tab:orange", "--", "s"),
}


def monthly_v8_km() -> pd.DataFrame:
    rows = []
    for market in ("btc_um", "btc_cm", "eth_um", "eth_cm"):
        for first in ("spot", "perp"):
            files = sorted((ROOT / "events" / market).glob(f"{first}_????-??-??_1s.parquet"))
            by_month: dict[str, list[Path]] = {}
            for path in files:
                match = re.search(r"_(\d{4}-\d{2})-\d{2}_1s$", path.stem)
                if match:
                    by_month.setdefault(match.group(1), []).append(path)
            for period, month_files in sorted(by_month.items()):
                frames = [pd.read_parquet(p, columns=["Length", "Status"]) for p in month_files]
                data = pd.concat(frames, ignore_index=True).dropna()
                if data.empty:
                    continue
                kmf = KaplanMeierFitter().fit(data["Length"], data["Status"])
                survival = float(kmf.survival_function_at_times([HORIZON]).iloc[0])
                rows.append({
                    "asset": market.split("_")[0],
                    "market": market,
                    "contract": market.split("_")[1],
                    "first": first,
                    "period": period,
                    "n_events": len(data),
                    "survival_at_10s": survival,
                    "resolved_share_10s": 1.0 - survival,
                    "resolved_percent_10s": 100.0 * (1.0 - survival),
                    "methodology_version": "v8_regular_lm_daily_v4",
                })
    return pd.DataFrame(rows)


def _vecm_files(folder: Path, latency: str, contract: str) -> list[Path]:
    pattern = re.compile(
        rf"_1h_{re.escape(latency)}_{LAG}_{contract}_results_\d{{8}}_\d{{8}}\.csv$",
        re.IGNORECASE,
    )
    return [p for p in folder.glob("*.csv") if pattern.search(p.name)]


def monthly_vecm() -> pd.DataFrame:
    frames = []
    for (asset, contract), folder in VECM_FOLDERS.items():
        for latency in LATENCIES:
            files = _vecm_files(folder, latency, contract)
            parts = []
            for path in files:
                part = pd.read_csv(path, usecols=lambda c: c in {"interval", "series", "alpha"})
                parts.append(part)
            if not parts:
                continue
            data = pd.concat(parts, ignore_index=True)
            data["interval"] = pd.to_datetime(data["interval"], errors="coerce")
            data["alpha"] = pd.to_numeric(data["alpha"], errors="coerce")
            data = data.dropna(subset=["interval", "series", "alpha"])
            data = data.drop_duplicates(["interval", "series"], keep="last")
            data["period"] = data["interval"].dt.to_period("M").astype(str)
            wide = (
                data.groupby(["period", "series"], observed=True)["alpha"].median()
                .unstack("series")
                .reset_index()
                .rename(columns={
                    "log_midpoint_spot": "alpha_spot",
                    "log_midpoint_perp": "alpha_perp",
                })
            )
            if not {"alpha_spot", "alpha_perp"}.issubset(wide.columns):
                continue
            wide["asset"] = asset
            wide["contract"] = contract
            wide["latency"] = latency
            wide["lag_base"] = LAG
            frames.append(wide)
    return pd.concat(frames, ignore_index=True)


def merge_results(km: pd.DataFrame, vecm: pd.DataFrame) -> pd.DataFrame:
    comparison = km.merge(vecm, on=["asset", "contract", "period"], how="inner")
    comparison["responding_alpha"] = np.where(
        comparison["first"].eq("spot"), comparison["alpha_perp"], comparison["alpha_spot"]
    )
    comparison["abs_responding_alpha"] = comparison["responding_alpha"].abs()
    comparison["conventional_responding_sign"] = np.where(
        comparison["first"].eq("spot"),
        comparison["responding_alpha"] > 0,
        comparison["responding_alpha"] < 0,
    )
    return comparison


def correlation_table(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["asset", "contract", "first", "latency", "lag_base"]
    for key, group in comparison.groupby(keys, observed=True):
        valid = group[["abs_responding_alpha", "resolved_share_10s"]].dropna()
        if len(valid) < 3:
            continue
        pr, pp = pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
        sr, sp = spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])
        rows.append(dict(zip(keys, key)) | {
            "n_months": len(valid), "pearson_r": pr, "pearson_p": pp,
            "spearman_rho": sr, "spearman_p": sp,
        })
    return pd.DataFrame(rows)


def plot_primary_scatter(comparison: pd.DataFrame, latency: str = "100ms") -> None:
    for asset in ("btc", "eth"):
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), sharey=True, constrained_layout=True)
        for ax, ((contract, first), (label, color, _, marker)) in zip(axes, STYLES.items()):
            data = comparison[
                comparison["asset"].eq(asset)
                & comparison["contract"].eq(contract)
                & comparison["first"].eq(first)
                & comparison["latency"].eq(latency)
            ].dropna(subset=["abs_responding_alpha", "resolved_percent_10s"])
            ax.scatter(data["abs_responding_alpha"], data["resolved_percent_10s"],
                       color=color, marker=marker, alpha=0.65, s=28)
            if len(data) >= 3:
                x = data["abs_responding_alpha"].to_numpy()
                y = data["resolved_percent_10s"].to_numpy()
                order = np.argsort(x)
                slope, intercept = np.polyfit(x, y, 1)
                ax.plot(x[order], intercept + slope * x[order], color="0.2", linewidth=1)
                rho = spearmanr(x, y).statistic
                ax.text(0.04, 0.95, rf"$\rho_s={rho:.2f}$", transform=ax.transAxes,
                        va="top")
            ax.set_title(label)
            ax.set_xlabel(r"$|\alpha|$ of responding market")
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("Resolved within 10 seconds (%)")
        fig.suptitle(f"{asset.upper()} V8 KM resolution versus VECM error correction ({latency})")
        fig.savefig(OUT / f"km_vecm_scatter_{asset}_{latency}_lag{LAG}.png", dpi=180,
                    bbox_inches="tight")
        plt.close(fig)


def _latency_seconds(value: str) -> float:
    return float(value[:-2]) / 1000 if value.endswith("ms") else float(value[:-1])


def plot_across_latency(correlations: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, asset in zip(axes, ("btc", "eth")):
        for (contract, first), (label, color, linestyle, marker) in STYLES.items():
            data = correlations[
                correlations["asset"].eq(asset)
                & correlations["contract"].eq(contract)
                & correlations["first"].eq(first)
            ].copy()
            data["latency_seconds"] = data["latency"].map(_latency_seconds)
            data = data.sort_values("latency_seconds")
            ax.plot(data["latency_seconds"], data["spearman_rho"], label=label,
                    color=color, linestyle=linestyle, marker=marker)
        ax.axhline(0, color="0.4", linestyle=":")
        ax.set_xscale("log")
        ax.set_title(asset.upper())
        ax.set_xlabel("VECM aggregation interval (seconds)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Spearman correlation")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("V8 KM–VECM correspondence across aggregation intervals", y=1.12)
    fig.savefig(OUT / "km_vecm_spearman_across_latency_btc_eth.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)


def plot_heatmaps(correlations: pd.DataFrame) -> None:
    for asset in ("btc", "eth"):
        table = correlations[correlations["asset"].eq(asset)].copy()
        table["group"] = table.apply(
            lambda r: STYLES[(r["contract"], r["first"])][0], axis=1
        )
        pivot = table.pivot(index="group", columns="latency", values="spearman_rho")
        pivot = pivot.reindex(
            [v[0] for v in STYLES.values()], columns=LATENCIES
        )
        fig, ax = plt.subplots(figsize=(7, 3.5))
        sns.heatmap(pivot, vmin=-1, vmax=1, center=0, cmap="coolwarm", annot=True,
                    fmt=".2f", ax=ax, cbar_kws={"label": "Spearman correlation"})
        ax.set(xlabel="VECM aggregation interval", ylabel="", title=f"{asset.upper()} V8 KM–VECM correlations")
        fig.tight_layout()
        fig.savefig(OUT / f"km_vecm_correlation_heatmap_{asset}.png", dpi=180,
                    bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    km = monthly_v8_km()
    vecm = monthly_vecm()
    comparison = merge_results(km, vecm)
    correlations = correlation_table(comparison)
    km.to_csv(OUT / "v8_monthly_km_10s.csv", index=False)
    vecm.to_csv(OUT / "monthly_vecm_medians.csv", index=False)
    comparison.to_csv(OUT / "v8_km_vecm_monthly_merged.csv", index=False)
    correlations.to_csv(OUT / "v8_km_vecm_correlations.csv", index=False)
    plot_primary_scatter(comparison)
    plot_across_latency(correlations)
    plot_heatmaps(correlations)
    print(correlations.to_string(index=False))
    print("plots", *sorted(OUT.glob("*.png")), sep="\n")


if __name__ == "__main__":
    main()
