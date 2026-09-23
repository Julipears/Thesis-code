"""Merge and validate the completed one-hour-per-day Hasbrouck audit."""

from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("hasbrouck_spec_audit")
ROOTS = [
    BASE / "full_sample_daily_hours",
    BASE / "full_sample_daily_hours_2024_2025",
    BASE / "full_sample_daily_hours_2025_june_dec",
    BASE / "full_sample_daily_hours_2025_sep_dec",
    BASE / "full_sample_daily_hours_2025_nov_dec",
]
OUT = BASE / "full_sample_daily_hours_combined"
CURRENT = "current_observed_intercept_float32"
CORRECTED = "full_grid_demeaned_ec_no_intercept_float64"


def read_all(filename: str) -> pd.DataFrame:
    frames = []
    for priority, root in enumerate(ROOTS):
        path = root / filename
        if path.exists():
            frame = pd.read_csv(path)
            frame["_source_priority"] = priority
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    results = read_all("daily_results.csv")
    results["day"] = pd.to_datetime(results["day"], format="mixed").dt.normalize()
    result_keys = ["day", "latency", "specification", "series"]
    results = (
        results.sort_values(result_keys + ["_source_priority"])
        .drop_duplicates(result_keys, keep="first")
        .drop(columns="_source_priority")
        .sort_values(result_keys)
        .reset_index(drop=True)
    )

    grids = read_all("daily_grid_diagnostics.csv")
    grids["day"] = pd.to_datetime(grids["day"], format="mixed").dt.normalize()
    grid_keys = ["day", "latency"]
    grids = (
        grids.sort_values(grid_keys + ["_source_priority"])
        .drop_duplicates(grid_keys, keep="first")
        .drop(columns="_source_priority")
        .sort_values(grid_keys)
        .reset_index(drop=True)
    )

    errors = read_all("errors.csv")
    if not errors.empty:
        errors["day"] = pd.to_datetime(errors["day"], format="mixed").dt.normalize()
        errors = errors.drop(columns="_source_priority").drop_duplicates("day")

    expected_days = pd.date_range("2021-01-01", "2025-12-31", freq="D")
    actual_days = pd.DatetimeIndex(results["day"].unique()).sort_values()
    missing_days = expected_days.difference(actual_days)
    unexpected_days = actual_days.difference(expected_days)
    counts = results.groupby(result_keys).size()

    assert missing_days.empty, f"Missing dates: {missing_days.tolist()}"
    assert unexpected_days.empty, f"Unexpected dates: {unexpected_days.tolist()}"
    assert len(actual_days) == 1826
    assert len(results) == 1826 * 3 * 2 * 2
    assert counts.eq(1).all()
    assert len(grids) == 1826 * 3
    assert errors.empty, f"Authoritative shards contain {len(errors)} errors"
    assert np.isfinite(results["his_mid"]).all()
    assert results["his_mid"].between(0, 1).all()

    results.to_csv(OUT / "daily_results.csv", index=False)
    grids.to_csv(OUT / "daily_grid_diagnostics.csv", index=False)
    errors.to_csv(OUT / "errors.csv", index=False)

    perp = results[results["series"].eq("log_midpoint_perp")].copy()
    index = ["day", "latency"]
    wide = perp.pivot(index=index, columns="specification", values="his_mid").reset_index()
    wide["current_leader"] = np.where(wide[CURRENT] > 0.5, "perp", "spot")
    wide["corrected_leader"] = np.where(wide[CORRECTED] > 0.5, "perp", "spot")
    wide["leader_flip"] = wide["current_leader"] != wide["corrected_leader"]
    wide["delta_corrected_minus_current_pp"] = 100 * (wide[CORRECTED] - wide[CURRENT])
    wide.to_csv(OUT / "paired_perp_is_daily.csv", index=False)

    rows = []
    for latency, group in wide.groupby("latency", sort=False):
        flips = group[group["leader_flip"]]
        rows.append({
            "latency": latency,
            "days": len(group),
            "mean_current_perp_is": group[CURRENT].mean(),
            "mean_corrected_perp_is": group[CORRECTED].mean(),
            "mean_delta_corrected_minus_current_pp": group["delta_corrected_minus_current_pp"].mean(),
            "median_abs_delta_pp": group["delta_corrected_minus_current_pp"].abs().median(),
            "p95_abs_delta_pp": group["delta_corrected_minus_current_pp"].abs().quantile(0.95),
            "paired_correlation": group[[CURRENT, CORRECTED]].corr().iloc[0, 1],
            "current_perp_leader_days": int((group["current_leader"] == "perp").sum()),
            "corrected_perp_leader_days": int((group["corrected_leader"] == "perp").sum()),
            "leader_flip_days": int(group["leader_flip"].sum()),
            "leader_flip_pct": 100 * group["leader_flip"].mean(),
            "spot_to_perp_flips": int(((flips["current_leader"] == "spot") & (flips["corrected_leader"] == "perp")).sum()),
            "perp_to_spot_flips": int(((flips["current_leader"] == "perp") & (flips["corrected_leader"] == "spot")).sum()),
        })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "is_leadership_comparison.csv", index=False)

    bounds = perp.copy()
    bounds["bound_class"] = np.select(
        [bounds["his_lower_corrected"] > 0.5, bounds["his_upper_corrected"] < 0.5],
        ["perp", "spot"],
        default="ambiguous",
    )
    bound_summary = (
        bounds.groupby(["latency", "specification", "bound_class"])
        .size().unstack(fill_value=0).reset_index()
    )
    for column in ["perp", "spot", "ambiguous"]:
        if column not in bound_summary:
            bound_summary[column] = 0
    bound_summary["days"] = bound_summary[["perp", "spot", "ambiguous"]].sum(axis=1)
    for column in ["perp", "spot", "ambiguous"]:
        bound_summary[f"{column}_pct"] = 100 * bound_summary[column] / bound_summary["days"]
    bound_summary.to_csv(OUT / "bound_leadership_summary.csv", index=False)

    print("VALIDATED_DAYS", len(actual_days))
    print("VALIDATED_RESULT_ROWS", len(results))
    print("VALIDATED_GRID_ROWS", len(grids))
    print("ERRORS", len(errors))
    print("\nIS_AND_MIDPOINT_LEADERSHIP")
    print(comparison.to_string(index=False))
    print("\nBOUND_LEADERSHIP")
    print(bound_summary.to_string(index=False))


if __name__ == "__main__":
    main()
