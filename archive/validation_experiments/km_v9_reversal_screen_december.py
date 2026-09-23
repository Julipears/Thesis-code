"""December 2025 V9 robustness: joint-bin and origin-reversal screens.

Starting from the reproducible V8 LM detections, this script reloads raw trades
only to apply path-dependent robustness screens. Same-second spot/perp LM
detections are joint. A unique-origin event is excluded when its origin price
retraces at least 50% of the one-second shock within the next five origin trades.
Cross-market response within 120 seconds is recorded but is not a sample filter.
"""

import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from lifelines import KaplanMeierFitter

from km_v8_regular_lm_pilot import MARKETS
from km_v9_deduplicated_robustness import deduplicate_day
from trade_data_pull import TradeData


MONTH = "2025-12"
OUTPUT_ROOT = Path("sa_results/km_v9_reversal_screen_december_2025")
METHODOLOGY_VERSION = "v9_joint_1s_reversal50_next5_v1"
BLOCK_DAYS = 1
RETRACEMENT_THRESHOLD = 0.50
RESPONSE_THRESHOLD = 0.50
RESPONSE_HORIZON_SECONDS = 120.0


def _trades(frame: pl.DataFrame) -> pd.DataFrame:
    cols = ["timestamp", "price"] + (["trade_id"] if "trade_id" in frame.columns else [])
    out = frame.select(cols).to_pandas()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    order = ["timestamp"] + (["trade_id"] if "trade_id" in out else [])
    return out.sort_values(order, kind="stable").reset_index(drop=True)


def _screen_unique_events(
    events: pd.DataFrame,
    origin: str,
    origin_trades: pd.DataFrame,
    reacting_trades: pd.DataFrame,
) -> pd.DataFrame:
    out = events.copy()
    if out.empty:
        return out.assign(
            rapid_same_market_reversal=pd.Series(dtype="bool"),
            max_origin_retracement_fraction=pd.Series(dtype="float64"),
            cross_market_response_50pct=pd.Series(dtype="bool"),
        )

    ots = origin_trades["timestamp"].to_numpy(dtype="datetime64[ns]")
    opx = origin_trades["price"].to_numpy(dtype=float)
    rts = reacting_trades["timestamp"].to_numpy(dtype="datetime64[ns]")
    rpx = reacting_trades["price"].to_numpy(dtype=float)
    records = []

    for idx, event in out.iterrows():
        bin_start = np.datetime64(pd.Timestamp(event["timestamp_bin"]), "ns")
        bin_end = bin_start + np.timedelta64(1, "s")
        shock_return = float(event["return"])
        shock_abs = abs(shock_return)
        shock_sign = np.sign(shock_return)

        post_idx = int(np.searchsorted(ots, bin_end, side="left") - 1)
        pre_idx = int(np.searchsorted(ots, bin_start, side="left") - 1)
        mapping_ok = pre_idx >= 0 and post_idx >= 0 and post_idx + 1 < len(ots)
        max_retrace = np.nan
        rapid_reversal = False
        response = False
        response_ts = pd.NaT

        if mapping_ok and shock_abs > 0:
            next_end = min(post_idx + 6, len(ots))
            next_prices = opx[post_idx + 1:next_end]
            retracement = -shock_sign * (np.log(next_prices) - np.log(opx[post_idx]))
            max_retrace = max(0.0, float(np.max(retracement, initial=0.0))) / shock_abs
            rapid_reversal = max_retrace >= RETRACEMENT_THRESHOLD

            # The lagging market response starts from its last price known by
            # the end of the detected one-second bin and is evaluated over the
            # same 120-second horizon used by the KM outcome.
            react_base_idx = int(np.searchsorted(rts, bin_end, side="left") - 1)
            react_end = bin_end + np.timedelta64(int(RESPONSE_HORIZON_SECONDS * 1000), "ms")
            react_stop = int(np.searchsorted(rts, react_end, side="right"))
            if react_base_idx >= 0 and react_stop > react_base_idx + 1:
                moves = shock_sign * (
                    np.log(rpx[react_base_idx + 1:react_stop]) - np.log(rpx[react_base_idx])
                )
                hits = np.flatnonzero(moves >= RESPONSE_THRESHOLD * shock_abs)
                if len(hits):
                    response = True
                    response_ts = pd.Timestamp(rts[react_base_idx + 1 + int(hits[0])])

        records.append({
            "_row": idx,
            "origin_trade_mapping_ok": mapping_ok,
            "max_origin_retracement_fraction": max_retrace,
            "rapid_same_market_reversal": rapid_reversal,
            "cross_market_response_50pct": response,
            "cross_market_response_ts": response_ts,
            "origin_screen_market": origin,
        })

    flags = pd.DataFrame(records).set_index("_row")
    return out.join(flags)


def _fit(frame: pd.DataFrame, timeline: np.ndarray) -> np.ndarray:
    clean = frame.dropna(subset=["Length", "Status"])
    if clean.empty:
        return np.full(len(timeline), np.nan)
    kmf = KaplanMeierFitter().fit(clean["Length"].astype(float), clean["Status"].astype(int))
    return kmf.survival_function_at_times(timeline).to_numpy()


def plot_v8_style(pooled: dict[tuple[str, str], pd.DataFrame], horizon: float) -> Path:
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
            ax.plot(timeline, _fit(frame, timeline), label=label, color=color,
                    linestyle=linestyle, linewidth=1.7)
        ax.set_title(asset.upper())
        ax.set_xlim(0, horizon)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Seconds since basis displacement")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Probability unresolved")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.91))
    fig.suptitle("Kaplan–Meier curves for 90% basis resolution", y=0.98)
    fig.subplots_adjust(top=0.75, bottom=0.15, left=0.08, right=0.98, wspace=0.08)
    suffix = "20s" if horizon == 20 else "120s"
    path = OUTPUT_ROOT / "plots" / f"survival_all_graphs_btc_eth_v9_screened_2025-12_{suffix}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    days = pd.date_range("2025-12-01", "2025-12-31", freq="D")
    pooled_parts = {(market, origin): [] for market in MARKETS for origin in ("spot", "perp")}
    audit_rows = []

    for market in MARKETS:
        symbol, cm_um, _ = MARKETS[market]
        for block_start_idx in range(0, len(days), BLOCK_DAYS):
            block = days[block_start_idx:block_start_idx + BLOCK_DAYS]
            # One archive day at a time is sufficient for the five-trade
            # screen and keeps BTC spot decompression within memory. Events at
            # the day boundary without a pre/post price are explicitly audited
            # as mapping-unknown rather than inferred from incomplete data.
            archive_end = block[0]
            archive_days = 1
            print(f"[load] {market} {block[0]:%Y-%m-%d}..{block[-1]:%Y-%m-%d}", flush=True)
            data = TradeData(symbol, source="Binance", cm_um=cm_um)
            data.grab_trades_data(archive_end.to_pydatetime(), archive_days, n_jobs=1)
            spot_trades = _trades(data.df_trades_spots)
            perp_trades = _trades(data.df_trades_perps)

            for day_ts in block:
                day = day_ts.strftime("%Y-%m-%d")
                groups = deduplicate_day(market, day)
                for origin, own, other in (
                    ("spot", spot_trades, perp_trades),
                    ("perp", perp_trades, spot_trades),
                ):
                    screened = _screen_unique_events(groups[origin], origin, own, other)
                    unknown = (~screened["origin_trade_mapping_ok"]).sum() if len(screened) else 0
                    reversal = screened["rapid_same_market_reversal"].sum() if len(screened) else 0
                    final = screened[
                        screened["origin_trade_mapping_ok"]
                        & ~screened["rapid_same_market_reversal"]
                    ].copy() if len(screened) else screened.copy()
                    final["methodology_version"] = METHODOLOGY_VERSION
                    out = OUTPUT_ROOT / "events" / market / f"{origin}_{day}_1s.parquet"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    final.to_parquet(out, index=False)
                    pooled_parts[(market, origin)].append(final)
                    audit_rows.append({
                        "market": market, "date": day, "origin": origin,
                        "v8_events": len(groups[origin]) + len(groups["joint"]),
                        "joint_same_second": len(groups["joint"]),
                        "unique_origin_before_reversal": len(screened),
                        "rapid_same_market_reversals_removed": int(reversal),
                        "mapping_unknown_removed": int(unknown),
                        "final_v9_events": len(final),
                        "cross_market_response_50pct": int(final["cross_market_response_50pct"].sum()) if len(final) else 0,
                        "censored_final": int((final["Status"] == 0).sum()) if len(final) else 0,
                        "methodology_version": METHODOLOGY_VERSION,
                    })
            del spot_trades, perp_trades, data
            gc.collect()

    pooled = {key: pd.concat(parts, ignore_index=True) for key, parts in pooled_parts.items()}
    audit = pd.DataFrame(audit_rows)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUTPUT_ROOT / "screening_audit_daily.csv", index=False)
    totals = audit.groupby(["market", "origin"], as_index=False).sum(numeric_only=True)
    totals.to_csv(OUTPUT_ROOT / "screening_audit_totals.csv", index=False)
    paths = [plot_v8_style(pooled, 120.0), plot_v8_style(pooled, 20.0)]
    print(totals.to_string(index=False), flush=True)
    for path in paths:
        print(path, flush=True)


if __name__ == "__main__":
    main()
