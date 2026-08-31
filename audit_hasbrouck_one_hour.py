"""Small, non-destructive audit of the high-resolution Hasbrouck specification.

The script downloads one Binance trading day, retains a single UTC hour, and
fits the same two-market VECM under four increasingly MATLAB-like variants.
All outputs are written below ``hasbrouck_spec_audit``; production results are
never read or overwritten.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trade_data_pull import TradeData
from vecm_hasbrouck3 import SimpleMVAR, generate_multiple_lags


MARKET = "btc_um"
SYMBOL = "BTCUSDT"
DAY = pd.Timestamp("2025-12-06")
HOUR_START = DAY
HOUR_END = HOUR_START + pd.Timedelta(hours=1)
LATENCIES = ["10ms", "100ms", "1s"]
OUTPUT_DIR = Path("hasbrouck_spec_audit")


def stable_fit(
    prices: pd.DataFrame,
    latency: str,
    lag_structure: dict,
    *,
    intercept: bool,
    demean_ec: bool,
) -> tuple[dict, int, float]:
    """Fit in float64 with least squares and return price-discovery outputs."""
    prices = prices.astype(np.float64).sort_index()
    names = list(prices.columns)
    dp = prices.diff()

    y = dp.copy()
    y.columns = [f"d{c}" for c in names]
    parts: list[pd.DataFrame] = []

    if intercept:
        parts.append(pd.DataFrame({"const": 1.0}, index=prices.index))

    buckets = lag_structure[latency]
    for start, end in buckets:
        block = sum((dp.shift(lag) for lag in range(start, end + 1)))
        parts.append(block)

    ec = (prices[names[0]] - prices[names[1]]).to_frame("ec")
    if demean_ec:
        ec = ec - ec.mean()
    parts.append(ec.shift(1))

    xy = pd.concat([y, *parts], axis=1).dropna()
    Y = xy[y.columns].to_numpy(dtype=np.float64)
    X = xy.iloc[:, len(y.columns):].to_numpy(dtype=np.float64)
    b, _, _, singular_values = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ b
    omega = resid.T @ resid / len(Y)

    row_start = 1 if intercept else 0
    max_lag = max(end for _, end in buckets)
    phi = np.zeros((2, 2, max_lag), dtype=np.float64)
    for bucket_idx, (start, end) in enumerate(buckets):
        block = b[row_start + 2 * bucket_idx:row_start + 2 * (bucket_idx + 1), :].T
        for lag in range(start, end + 1):
            phi[:, :, lag - 1] = block

    alpha = b[-1:, :].T
    out = SimpleMVAR._price_discovery_from_outputs(names, phi, omega, alpha)
    condition = float(singular_values[0] / singular_values[-1])
    return out, len(Y), condition


def current_fit(prices: pd.DataFrame, latency: str, lag_structure: dict) -> tuple[dict, int, float]:
    """Fit the production specification: float32, intercept, explicit inverse."""
    model = SimpleMVAR(
        prices=prices,
        lag_structure=lag_structure,
        latency=latency,
        interval="1H",
        intercept=True,
        ticker=SYMBOL,
        source="Binance",
        cm_um="um",
    ).fit()
    out = model._price_discovery_from_outputs(
        model.price_names,
        model.phi_matrices(),
        model.e_cov,
        model.gamma_matrix(),
    )
    return out, len(model.Y), float(np.linalg.cond(model.X.astype(np.float64)))


def rows_for_result(latency: str, specification: str, out: dict, n_obs: int, condition: float, n_grid: int) -> list[dict]:
    rows = []
    order_1 = np.asarray(out["his_lower"], dtype=float)
    order_2 = np.asarray(out["his_upper"], dtype=float)
    true_lower = np.minimum(order_1, order_2)
    true_upper = np.maximum(order_1, order_2)
    for i, series in enumerate(out["series"]):
        rows.append({
            "market": MARKET,
            "hour_start_utc": HOUR_START,
            "latency": latency,
            "specification": specification,
            "series": series,
            "n_price_rows": n_grid,
            "n_model_obs": n_obs,
            "design_condition_number": condition,
            "alpha": float(out["alpha"][i]),
            "component_share": float(out["cs"][i]),
            "his_order_1": float(order_1[i]),
            "his_order_2": float(order_2[i]),
            "his_lower_corrected": float(true_lower[i]),
            "his_upper_corrected": float(true_upper[i]),
            "his_mid": float(out["his_mid"][i]),
            "information_leadership_share": float(out["ils_mid"][i]),
        })
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = TradeData(SYMBOL, "Binance", "um")
    # ``grab_trades_data`` is day-file based and includes DAY when end_date=DAY.
    data.grab_trades_data(DAY.to_pydatetime(), days=1, n_jobs=2)
    lag_structure = generate_multiple_lags(10, LATENCIES, max_length="10s")

    all_rows: list[dict] = []
    grid_diagnostics: list[dict] = []
    for latency in LATENCIES:
        frames = {}
        for fill_gaps in (False, True):
            frame = data.agg_last_trade_to_intervals(
                freq=latency,
                start=HOUR_START,
                end=HOUR_END,
                fill_gaps=fill_gaps,
                rename_for_vecm=True,
                retain_initial_grid_row=True,
            )
            if frame.empty:
                raise RuntimeError(f"No usable {latency} data for fill_gaps={fill_gaps}")
            price_cols = ["log_midpoint_spot", "log_midpoint_perp"]
            frames[fill_gaps] = frame[price_cols]

        observed = frames[False]
        full = frames[True]
        expected = int(pd.Timedelta(hours=1) / pd.Timedelta(latency))
        grid_diagnostics.append({
            "latency": latency,
            "expected_calendar_bins": expected,
            "observed_union_bins": len(observed),
            "complete_grid_bins": len(full),
            "omitted_bins_current": expected - len(observed),
            "omitted_pct_current": 100.0 * (expected - len(observed)) / expected,
        })

        specs = [
            ("current_observed_intercept_float32", observed, "current"),
            ("observed_intercept_float64", observed, "stable_intercept"),
            ("full_grid_intercept_float32", full, "current"),
            ("full_grid_intercept_float64", full, "stable_intercept"),
            ("full_grid_no_intercept_float64", full, "stable"),
            ("full_grid_demeaned_ec_no_intercept_float64", full, "demeaned"),
        ]
        for label, frame, method in specs:
            if method == "current":
                out, n_obs, condition = current_fit(frame, latency, lag_structure)
            else:
                out, n_obs, condition = stable_fit(
                    frame,
                    latency,
                    lag_structure,
                    intercept=(method == "stable_intercept"),
                    demean_ec=(method == "demeaned"),
                )
            all_rows.extend(rows_for_result(latency, label, out, n_obs, condition, len(frame)))
            print(f"[done] {latency} {label}: n={n_obs:,}")

    results = pd.DataFrame(all_rows)
    baseline = (
        results[results["specification"].eq("current_observed_intercept_float32")]
        [["latency", "series", "his_mid"]]
        .rename(columns={"his_mid": "baseline_his_mid"})
    )
    results = results.merge(baseline, on=["latency", "series"], how="left")
    results["his_mid_change_pp"] = 100 * (results["his_mid"] - results["baseline_his_mid"])
    results.to_csv(OUTPUT_DIR / "one_hour_specification_results.csv", index=False)
    pd.DataFrame(grid_diagnostics).to_csv(OUTPUT_DIR / "one_hour_grid_diagnostics.csv", index=False)

    spot = results[results["series"].str.contains("spot")].copy()
    display_cols = ["latency", "specification", "n_model_obs", "his_mid", "his_mid_change_pp", "component_share", "design_condition_number"]
    print("\nSPOT PRICE-DISCOVERY COMPARISON")
    print(spot[display_cols].to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"\nSaved audit outputs to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
