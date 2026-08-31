"""V9 December robustness using confirmed one-second lead--response sequences."""

import csv
import gc
import io
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from lifelines import KaplanMeierFitter

from km_v9_deduplicated_robustness import deduplicate_day


MONTH = "2025-12"
OUT = Path("sa_results/km_v9_confirmed_lead_december_2025")
MARKETS = ("btc_um", "btc_cm", "eth_um", "eth_cm")
VERSION = "v9_joint1s_confirmed_next1s_reversal50_next5_v1"


@dataclass
class CompactTrades:
    seconds: np.ndarray
    prices: np.ndarray
    next_five: dict[int, np.ndarray]

    def price_at_end(self, second: int) -> float:
        i = int(np.searchsorted(self.seconds, second, side="right") - 1)
        return float(self.prices[i]) if i >= 0 else np.nan

    def one_second_return(self, second: int) -> float:
        before = self.price_at_end(second - 1)
        after = self.price_at_end(second)
        if not np.isfinite(before) or not np.isfinite(after) or before <= 0 or after <= 0:
            return np.nan
        return float(np.log(after) - np.log(before))


def _url(asset: str, contract: str, day: str) -> str:
    symbol = asset.upper() + "USDT"
    if contract == "spot":
        return f"https://data.binance.vision/data/spot/daily/trades/{symbol}/{symbol}-trades-{day}.zip"
    if contract == "um":
        return f"https://data.binance.vision/data/futures/um/daily/trades/{symbol}/{symbol}-trades-{day}.zip"
    symbol = asset.upper() + "USD_PERP"
    return f"https://data.binance.vision/data/futures/cm/daily/trades/{symbol}/{symbol}-trades-{day}.zip"


def _millis(raw: str) -> int:
    value = int(float(raw))
    if value < 100_000_000_000:
        return value * 1000
    if value < 100_000_000_000_000:
        return value
    return value // 1000


def stream_compact_trades(url: str, event_seconds: set[int]) -> CompactTrades:
    """Stream a Binance zip and retain only 1s lasts plus five post-bin trades."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = Path(tmp_dir) / "trades.zip"
        for attempt in range(1, 5):
            try:
                with requests.get(url, stream=True, timeout=(30, 180)) as response:
                    response.raise_for_status()
                    with zip_path.open("wb") as download:
                        for chunk in response.iter_content(1024 * 1024):
                            if chunk:
                                download.write(chunk)
                break
            except requests.RequestException:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        last_by_second: dict[int, float] = {}
        next_five: dict[int, list[float]] = {s: [] for s in event_seconds}
        pending = sorted(event_seconds)
        active: list[int] = []
        pending_i = 0
        with zipfile.ZipFile(zip_path) as archive:
            with archive.open(archive.namelist()[0]) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
                reader = csv.reader(text)
                first = next(reader)
                header = not all(_is_number(x) for x in first[:5])
                if header:
                    names = [x.strip().lower() for x in first]
                    price_i = names.index("price")
                    time_i = names.index("time")
                    rows = reader
                else:
                    price_i, time_i = 1, 4
                    rows = _prepend(first, reader)
                for row in rows:
                    try:
                        ms = _millis(row[time_i])
                        price = float(row[price_i])
                    except (ValueError, IndexError):
                        continue
                    sec = ms // 1000
                    last_by_second[sec] = price
                    while pending_i < len(pending) and pending[pending_i] + 1 <= sec:
                        active.append(pending[pending_i])
                        pending_i += 1
                    if active:
                        still = []
                        for event_sec in active:
                            values = next_five[event_sec]
                            if len(values) < 5:
                                values.append(price)
                            if len(values) < 5:
                                still.append(event_sec)
                        active = still
    seconds = np.array(sorted(last_by_second), dtype=np.int64)
    prices = np.array([last_by_second[s] for s in seconds], dtype=float)
    return CompactTrades(seconds, prices, {k: np.asarray(v) for k, v in next_five.items()})


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _prepend(first, iterator):
    yield first
    yield from iterator


def _event_seconds(frame: pd.DataFrame) -> set[int]:
    if frame.empty:
        return set()
    # Parquet may restore this column as datetime64[ms]. Explicitly promote to
    # ns before converting to epoch seconds; otherwise the keys are 1000x low.
    timestamps = pd.to_datetime(frame["timestamp_bin"]).astype("datetime64[ns]")
    return set((timestamps.astype("int64") // 1_000_000_000).astype(int))


def classify(
    unique: pd.DataFrame,
    origin: str,
    origin_data: CompactTrades,
    other_data: CompactTrades,
) -> pd.DataFrame:
    out = unique.copy()
    records = []
    for idx, event in out.iterrows():
        sec = int(pd.Timestamp(event["timestamp_bin"]).value // 1_000_000_000)
        lead_return = float(event["return"])
        sign, magnitude = np.sign(lead_return), abs(lead_return)
        other_same = other_data.one_second_return(sec)
        other_next = other_data.one_second_return(sec + 1)
        same_material = (
            np.isfinite(other_same) and magnitude > 0
            and np.sign(other_same) == sign and abs(other_same) >= 0.5 * magnitude
        )
        next_confirmed = (
            not same_material and np.isfinite(other_next) and magnitude > 0
            and np.sign(other_next) == sign and abs(other_next) >= 0.5 * magnitude
        )
        post_price = origin_data.price_at_end(sec)
        following = origin_data.next_five.get(sec, np.array([]))
        retrace_fraction = np.nan
        reversal = False
        mapping_ok = np.isfinite(post_price) and len(following) == 5 and magnitude > 0
        if mapping_ok:
            retrace = -sign * (np.log(following) - np.log(post_price))
            retrace_fraction = max(0.0, float(np.max(retrace, initial=0.0))) / magnitude
            reversal = retrace_fraction >= 0.5
        if same_material:
            classification = "same_second_ambiguous"
        elif next_confirmed:
            classification = f"{origin}_leading_confirmed"
        else:
            classification = "unconfirmed"
        records.append({
            "_idx": idx,
            "classification": classification,
            "other_market_return_same_second": other_same,
            "other_market_return_next_second": other_next,
            "same_second_material_response": same_material,
            "next_second_response_confirmed": next_confirmed,
            "origin_trade_mapping_ok": mapping_ok,
            "max_origin_retracement_fraction": retrace_fraction,
            "rapid_same_market_reversal": reversal,
        })
    if not records:
        return out
    return out.join(pd.DataFrame(records).set_index("_idx"))


def _km(frame: pd.DataFrame, timeline: np.ndarray) -> np.ndarray:
    if frame.empty:
        return np.full(len(timeline), np.nan)
    fit = KaplanMeierFitter().fit(frame["Length"], frame["Status"])
    return fit.survival_function_at_times(timeline).to_numpy()


def plot(pooled: dict[tuple[str, str], pd.DataFrame], horizon: int) -> Path:
    timeline = np.arange(0, horizon + 0.125, 0.25)
    styles = {
        ("um", "spot"): ("Linear: spot", "tab:blue", "-"),
        ("um", "perp"): ("Linear: perp", "tab:blue", "--"),
        ("cm", "spot"): ("Inverse: spot", "tab:orange", "-"),
        ("cm", "perp"): ("Inverse: perp", "tab:orange", "--"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), sharex=True, sharey=True)
    for ax, asset in zip(axes, ("btc", "eth")):
        for (contract, origin), (label, color, ls) in styles.items():
            ax.plot(timeline, _km(pooled[(f"{asset}_{contract}", origin)], timeline),
                    label=label, color=color, linestyle=ls, linewidth=1.7)
        ax.set(title=asset.upper(), xlim=(0, horizon), ylim=(0, 1),
               xlabel="Seconds since basis displacement")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Probability unresolved")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.91))
    fig.suptitle("Kaplan–Meier curves for 90% basis resolution", y=0.98)
    fig.subplots_adjust(top=0.75, bottom=0.15, left=0.08, right=0.98, wspace=0.08)
    path = OUT / "plots" / f"survival_btc_eth_v9_confirmed_2025-12_{horizon}s.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    audit_path = OUT / "screening_audit_daily.csv"
    if audit_path.exists():
        existing = pd.read_csv(audit_path)
        audit_rows = existing.to_dict("records")
    else:
        audit_rows = []
    completed = {
        (asset, day)
        for asset in ("btc", "eth")
        for day in [f"{MONTH}-{d:02d}" for d in range(1, 32)]
        if sum(r["date"] == day and str(r["market"]).startswith(asset) for r in audit_rows) == 4
        and all(
            (OUT / "events" / f"{asset}_{contract}" / f"{origin}_{day}_1s.parquet").exists()
            for contract in ("um", "cm") for origin in ("spot", "perp")
        )
    }
    for asset in ("btc", "eth"):
        for day_num in range(1, 32):
            day = f"{MONTH}-{day_num:02d}"
            if (asset, day) in completed:
                print(f"[skip complete] {asset} {day}", flush=True)
                continue
            print(f"[day] {asset} {day}", flush=True)
            groups = {market: deduplicate_day(market, day)
                      for market in (f"{asset}_um", f"{asset}_cm")}
            spot_bins = set().union(*[_event_seconds(g["spot"]) for g in groups.values()])
            spot = stream_compact_trades(_url(asset, "spot", day), spot_bins)
            perps = {}
            for contract in ("um", "cm"):
                market = f"{asset}_{contract}"
                perps[contract] = stream_compact_trades(
                    _url(asset, contract, day), _event_seconds(groups[market]["perp"])
                )
            for contract in ("um", "cm"):
                market = f"{asset}_{contract}"
                joint_n = len(groups[market]["joint"])
                for origin, own, other in (
                    ("spot", spot, perps[contract]),
                    ("perp", perps[contract], spot),
                ):
                    classified = classify(groups[market][origin], origin, own, other)
                    confirmed_label = f"{origin}_leading_confirmed"
                    final = classified[
                        classified["classification"].eq(confirmed_label)
                        & classified["origin_trade_mapping_ok"]
                        & ~classified["rapid_same_market_reversal"]
                    ].copy()
                    final["first"] = origin
                    final["methodology_version"] = VERSION
                    path = OUT / "events" / market / f"{origin}_{day}_1s.parquet"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    final.to_parquet(path, index=False)
                    audit_rows.append({
                        "market": market, "date": day, "origin": origin,
                        "joint_lm_same_second": joint_n,
                        "unique_lm_candidates": len(classified),
                        "same_second_ambiguous": int(classified["classification"].eq("same_second_ambiguous").sum()),
                        "next_second_response_confirmed": int(classified["classification"].eq(confirmed_label).sum()),
                        "unconfirmed": int(classified["classification"].eq("unconfirmed").sum()),
                        "rapid_reversals_removed_from_confirmed": int((classified["classification"].eq(confirmed_label) & classified["rapid_same_market_reversal"]).sum()),
                        "mapping_unknown": int((~classified["origin_trade_mapping_ok"]).sum()),
                        "final_v9_events": len(final),
                        "censored_final": int((final["Status"] == 0).sum()),
                        "methodology_version": VERSION,
                    })
            pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
            del spot, perps, groups
            gc.collect()

    audit = pd.DataFrame(audit_rows)
    totals = audit.groupby(["market", "origin"], as_index=False).sum(numeric_only=True)
    totals.to_csv(OUT / "screening_audit_totals.csv", index=False)
    pooled = {}
    for market in MARKETS:
        for origin in ("spot", "perp"):
            files = sorted((OUT / "events" / market).glob(f"{origin}_2025-12-*_1s.parquet"))
            pooled[(market, origin)] = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    paths = [plot(pooled, 120), plot(pooled, 20)]
    print(totals.to_string(index=False), flush=True)
    for path in paths:
        print(path, flush=True)


if __name__ == "__main__":
    main()
