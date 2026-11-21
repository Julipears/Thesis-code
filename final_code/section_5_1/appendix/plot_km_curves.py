"""Export monthly and event-pooled annual KM curves from saved V8 events.

This script reads saved event columns, including their unchanged monthly
copies in the base AFT exports. It does not detect new events, download trades,
change the research repository, or alter old figures.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from lifelines import KaplanMeierFitter


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = PROJECT_ROOT / "sa_results/km_v8_final_01"
DEFAULT_OUTPUT = DEFAULT_INPUT / "appendix_km"
MONTHS = tuple(pd.period_range("2021-01", "2025-12", freq="M").astype(str))
YEARS = tuple(str(year) for year in range(2021, 2026))
MARKETS = ("btc_um", "btc_cm", "eth_um", "eth_cm")
FIRSTS = ("spot", "perp")
HORIZONS = (20, 120)
STEP = 0.25
MAX_HORIZON = 120.0
PATTERN = re.compile(r"^(spot|perp)_(\d{4}-\d{2}-\d{2})_1s\.parquet$")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def collect_one(job, expected_rows, cache, manifest_hash, root):
    market, first, month, files = job
    stem = f"{market}_{first}_{month}"
    cache_file = cache / f"{stem}.parquet"
    record_file = cache / f"{stem}.json"
    available = sorted(path.name.split("_")[1] for path in files)
    expected_dates = pd.date_range(month + "-01", periods=pd.Period(month).days_in_month).strftime("%Y-%m-%d").tolist()
    previous_record = json.loads(record_file.read_text(encoding="utf-8")) if record_file.exists() else {}
    if cache_file.exists() and previous_record.get("cache_version") == 2 and previous_record.get("manifest_sha256") == manifest_hash and previous_record.get("available_dates") == available:
        source = Path(previous_record["source_event_columns_file"])
        if source.exists() and source.stat().st_size == previous_record["source_file_bytes"] and source.stat().st_mtime_ns == previous_record["source_file_mtime_ns"]:
            return previous_record, pd.read_parquet(cache_file)
    # The base monthly AFT exports retain the original event columns before
    # model-specific complete-case selection. They are saved copies of these
    # events, not fitted AFT predictions or a covariate-filtered event subset.
    attempts = []
    for folder in ("aft_data_monthly", "aft_data_liquidity_monthly", "aft_data_liquidity_v5_log_covariates"):
        source = root / folder / market / f"{first}_{month}.parquet"
        try:
            frame = pd.read_parquet(source, columns=["Length", "Status", "resolution_pct", "start_ts", "market", "first", "grid", "methodology_version"])
        except Exception as exc:
            attempts.append(f"{source}: {type(exc).__name__}: {exc}")
            continue
        break
    else:
        raise RuntimeError("Could not read any saved monthly event-column copy:\n" + "\n".join(attempts))
    assert frame.market.eq(market).all() and frame["first"].eq(first).all(), source
    assert frame.grid.eq("1s").all(), source
    assert frame.methodology_version.eq("v8_regular_lm_daily_v4").all(), source
    assert frame.resolution_pct.eq(90).all(), source
    dates = pd.to_datetime(frame.start_ts, utc=True).dt.strftime("%Y-%m-%d")
    observed = dates.value_counts().sort_index()
    expected = pd.Series({date: expected_rows[(market, date, first)] for date in available}).sort_index()
    assert observed.index.isin(expected.index).all(), source
    # A saved, successfully processed day can legitimately contain zero events.
    # It remains a covered day even though it contributes no timestamp rows.
    observed = observed.reindex(expected.index, fill_value=0)
    assert observed.equals(expected), f"Daily event counts in the saved monthly copy disagree with the production manifest: {source}"
    raw_rows = len(frame)
    frame["Length"] = pd.to_numeric(frame["Length"], errors="coerce")
    frame["Status"] = pd.to_numeric(frame["Status"], errors="coerce")
    valid = frame[["Length", "Status"]].notna().all(axis=1)
    dropped = int((~valid).sum())
    frame = frame.loc[valid, ["Length", "Status"]].copy()
    assert np.isfinite(frame.Length).all() and not frame.Length.lt(0).any(), source
    assert frame.Status.isin([0, 1]).all(), source
    beyond = frame.Length.gt(MAX_HORIZON)
    rows_over_120 = int(beyond.sum())
    frame.loc[beyond, "Length"] = MAX_HORIZON
    frame.loc[beyond, "Status"] = 0
    aggregated = frame.groupby(["Length", "Status"], sort=False).size().rename("weight").reset_index()
    aggregated = aggregated.sort_values(["Length", "Status"]).reset_index(drop=True)
    daily_cache_verified = previous_record.get("validated_against_daily_event_cache", False)
    if cache_file.exists() and previous_record.get("manifest_sha256") == manifest_hash and "source_event_columns_file" not in previous_record:
        previous_counts = pd.read_parquet(cache_file)
        check = previous_counts.merge(aggregated, on=["Length", "Status"], how="outer", suffixes=("_daily", "_monthly"))
        assert check.weight_daily.eq(check.weight_monthly).all(), f"Saved monthly event columns do not reproduce the directly read daily event files: {source}"
        daily_cache_verified = True
    missing = sorted(set(expected_dates) - set(available))
    record = dict(market=market, first=first, period=month, expected_days=len(expected_dates), available_days=len(available), missing_days=len(missing), missing_dates=missing, available_dates=available, raw_event_rows=raw_rows, excluded_missing_value_rows=dropped, n_events=int(aggregated.weight.sum()), durations_above_120s_censored=rows_over_120, manifest_sha256=manifest_hash, cache_version=2, source_event_columns_file=str(source), source_file_bytes=source.stat().st_size, source_file_mtime_ns=source.stat().st_mtime_ns, validated_against_daily_event_cache=daily_cache_verified)
    assert record["n_events"] == raw_rows - dropped
    aggregated.to_parquet(cache_file, index=False)
    atomic_json(record_file, record)
    return record, aggregated


def styles(asset):
    return ((f"{asset}_um", "spot", "Linear: spot", "tab:blue", "-"),
            (f"{asset}_um", "perp", "Linear: perp", "tab:blue", "--"),
            (f"{asset}_cm", "spot", "Inverse: spot", "tab:orange", "-"),
            (f"{asset}_cm", "perp", "Inverse: perp", "tab:orange", "--"))


def draw_panel(ax, asset, period, aggregation, horizon, curves):
    incomplete = False
    for market, first, label, color, linestyle in styles(asset):
        result = curves[(aggregation, market, first, period)]
        end = int(horizon / STEP) + 1
        ax.step(result["timeline"][:end], result["survival"][:end], where="post", color=color, linestyle=linestyle, linewidth=1.8, label=label)
        incomplete |= result["missing_days"] > 0
    ax.set_title(period + (" *" if incomplete else ""))
    ax.set_xlim(0, horizon)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Seconds since basis displacement")
    ax.grid(alpha=0.3)
    return incomplete


def decorate(fig, axes, asset, title, incomplete, compact=False):
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2 if compact else 4, frameon=False)
    fig.suptitle(f"{asset.upper()}: {title}\nKaplan–Meier curves for 90% basis resolution", y=1.20 if compact else 1.13, fontsize=12)
    if incomplete:
        fig.text(0.01, -0.055 if compact else -0.025, "* Some daily event files are missing; curves use available days. See coverage CSV.", ha="left", fontsize=8)
    fig.tight_layout()


def save_pair(fig, base):
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=100, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")


def verify_decembers(root, frames):
    checks = []
    for asset in ("btc", "eth"):
        for horizon in HORIZONS:
            suffix = "" if horizon == 120 else "_20s"
            old = pd.read_csv(root / "plots" / f"survival_all_graphs_{asset}_v8{suffix}.csv")
            current = frames[("monthly", horizon)]
            merged = old.merge(current, on=["period", "market", "first", "timeline"], suffixes=("_original", "_generated"), validate="one_to_one")
            assert len(merged) == len(old), (asset, horizon, len(merged), len(old))
            maximum = float(np.max(np.abs(merged.survival_original - merged.survival_generated)))
            assert maximum < 1e-12, (asset, horizon, maximum)
            assert merged.n_events_original.eq(merged.n_events_generated).all()
            checks.append(dict(asset=asset, horizon_seconds=horizon, matched_rows=len(merged), matched_curves=20, maximum_survival_difference=maximum))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_INPUT / "km_all_curves_work")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    started = time.monotonic()
    root, out, cache = args.input.resolve(), args.output.resolve(), args.cache.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    data_dir = out / "data"
    data_dir.mkdir(exist_ok=True)
    manifest_path = root / "comparison_summary.csv"
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest = pd.read_csv(manifest_path)
    manifest = manifest.loc[manifest.grid.eq("1s") & manifest.date.ge("2021-01-01") & manifest.date.le("2025-12-31")].copy()
    assert not manifest.duplicated(["market", "date", "first"]).any()
    expected_rows = {(row.market, row.date, row.first): int(row.v8_events) for row in manifest.itertuples()}
    found = {}
    for market in MARKETS:
        for path in (root / "events" / market).iterdir():
            match = PATTERN.fullmatch(path.name)
            if match and "2021-01-01" <= match.group(2) <= "2025-12-31":
                found[(market, match.group(2), match.group(1))] = path
    assert set(found) == set(expected_rows), "Run manifest and saved event filenames do not agree."
    jobs = []
    for market in MARKETS:
        for first in FIRSTS:
            for month in MONTHS:
                files = sorted(path for (key_market, date, key_first), path in found.items() if key_market == market and key_first == first and date.startswith(month))
                assert files, (market, first, month)
                jobs.append((market, first, month, files))
    assert manifest.methodology_version.eq("v8_regular_lm_daily_v4").all()
    print(f"Reading saved monthly copies of {len(found):,} daily event series; checking per-day counts for {len(jobs)} month-series with {args.jobs} concurrent reads.", flush=True)
    monthly_counts, records = {}, []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(collect_one, job, expected_rows, cache, manifest_hash, root) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            record, counts = future.result()
            monthly_counts[(record["market"], record["first"], record["period"])] = counts
            records.append(record)
            if done % 20 == 0 or done == len(jobs):
                print(f"Read and checked {done}/{len(jobs)} monthly series ({sum(row['raw_event_rows'] for row in records):,} event rows); elapsed {time.monotonic()-started:.0f}s.", flush=True)
    records.sort(key=lambda row: (row["market"], row["first"], row["period"]))
    coverage = pd.DataFrame([{key: value for key, value in row.items() if key not in {"available_dates", "manifest_sha256"}} for row in records])
    coverage["missing_dates"] = coverage.missing_dates.map(lambda dates: "; ".join(dates))
    coverage.to_csv(data_dir / "monthly_event_coverage.csv", index=False)
    monthly_record = {(row["market"], row["first"], row["period"]): row for row in records}
    annual_record = {}
    missing_rows = []
    for row in records:
        for date in row["missing_dates"]:
            missing_rows.append(dict(market=row["market"], first=row["first"], date=date, period=row["period"]))
    pd.DataFrame(missing_rows).to_csv(data_dir / "missing_daily_event_files.csv", index=False)
    assert int(coverage.raw_event_rows.sum()) == int(manifest.v8_events.sum())

    frames = {(aggregation, horizon): [] for aggregation in ("monthly", "annual") for horizon in HORIZONS}
    curves = {}
    summary_rows = []
    max_product_error = 0.0
    n_models = 0
    exact_writers = {}
    try:
        for aggregation, periods in (("monthly", MONTHS), ("annual", YEARS)):
            for period in periods:
                for market in MARKETS:
                    for first in FIRSTS:
                        if aggregation == "monthly":
                            counts = monthly_counts[(market, first, period)]
                            record = monthly_record[(market, first, period)]
                        else:
                            month_keys = [(market, first, month) for month in MONTHS if month.startswith(period)]
                            counts = pd.concat([monthly_counts[key] for key in month_keys], ignore_index=True).groupby(["Length", "Status"], as_index=False).weight.sum()
                            record = dict(market=market, first=first, period=period, expected_days=sum(monthly_record[key]["expected_days"] for key in month_keys), available_days=sum(monthly_record[key]["available_days"] for key in month_keys), missing_days=sum(monthly_record[key]["missing_days"] for key in month_keys), missing_dates=sorted(date for key in month_keys for date in monthly_record[key]["missing_dates"]))
                            assert int(counts.weight.sum()) == sum(monthly_record[key]["n_events"] for key in month_keys)
                            annual_record[(market, first, period)] = record
                        n_events = int(counts.weight.sum())
                        model = KaplanMeierFitter().fit(counts.Length.to_numpy(), event_observed=counts.Status.to_numpy(), weights=counts.weight.to_numpy(dtype=np.int64), label="survival")
                        product = (1.0 - model.event_table.observed / model.event_table.at_risk).cumprod().to_numpy()
                        discrepancy = float(np.max(np.abs(product - model.survival_function_.survival.to_numpy())))
                        max_product_error = max(max_product_error, discrepancy)
                        assert discrepancy < 1e-12, (aggregation, market, first, period, discrepancy)
                        timeline = np.arange(0.0, MAX_HORIZON + STEP / 2.0, STEP)
                        survival = model.survival_function_at_times(timeline).to_numpy()
                        assert np.isfinite(survival).all() and (survival >= -1e-14).all() and (survival <= 1 + 1e-14).all()
                        assert (np.diff(survival) <= 1e-12).all()
                        curves[(aggregation, market, first, period)] = dict(timeline=timeline, survival=survival, n_events=n_events, missing_days=record["missing_days"])
                        for horizon in HORIZONS:
                            end = int(horizon / STEP) + 1
                            frame = pd.DataFrame(dict(period=period, timeline=timeline[:end], survival=survival[:end], market=market, first=first, resolution_pct=90, n_events=n_events, available_days=record["available_days"], expected_days=record["expected_days"], missing_days=record["missing_days"]))
                            frames[(aggregation, horizon)].append(frame)
                            summary_rows.append(dict(aggregation=aggregation, period=period, market=market, first=first, horizon_seconds=horizon, n_events=n_events, survival_at_horizon=float(survival[end-1]), fraction_resolved_by_horizon=float(1-survival[end-1]), median_resolution_seconds=float(model.median_survival_time_) if float(model.median_survival_time_) <= horizon else np.nan, median_reached_by_horizon=bool(float(model.median_survival_time_) <= horizon), available_days=record["available_days"], expected_days=record["expected_days"], missing_days=record["missing_days"]))
                        exact = model.event_table.reset_index().rename(columns={"event_at": "timeline"})
                        exact["survival"] = model.survival_function_.survival.to_numpy()
                        exact.insert(0, "period", period)
                        exact.insert(1, "market", market)
                        exact.insert(2, "first", first)
                        exact["resolution_pct"] = 90
                        exact["n_events"] = n_events
                        table = pa.Table.from_pandas(exact, preserve_index=False)
                        if aggregation not in exact_writers:
                            exact_writers[aggregation] = pq.ParquetWriter(data_dir / f"{aggregation}_km_exact_event_times_120s.parquet", table.schema, compression="zstd")
                        exact_writers[aggregation].write_table(table)
                        n_models += 1
                if aggregation == "annual" or int(period[-2:]) == 12:
                    print(f"Fitted {aggregation} curves through {period}; {n_models}/520 fits checked against the KM product formula.", flush=True)
    finally:
        for writer in exact_writers.values():
            writer.close()
    for key, parts in frames.items():
        frames[key] = pd.concat(parts, ignore_index=True)
        frames[key].to_csv(data_dir / f"{key[0]}_km_curves_{key[1]}s.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(data_dir / "km_curve_summary.csv", index=False)
    annual_coverage = pd.DataFrame(annual_record.values())
    annual_coverage["missing_dates"] = annual_coverage.missing_dates.map(lambda dates: "; ".join(dates))
    annual_coverage.to_csv(data_dir / "annual_event_coverage.csv", index=False)
    december_checks = verify_decembers(root, frames)
    print("All existing December curves reproduced, including event counts and both display horizons.", flush=True)

    # Individual plots retain the original blue/orange, solid/dashed styles,
    # 1.8-point line width, 100-DPI PNG output, and 0.25-second display grid.
    n_individual = 0
    for aggregation, periods in (("monthly", MONTHS), ("annual", YEARS)):
        for asset in ("btc", "eth"):
            for period in periods:
                for horizon in HORIZONS:
                    base = out / "figures" / aggregation / asset / period[:4] / f"km_{asset}_{period}_{horizon}s"
                    if not (base.with_suffix(".png").exists() and base.with_suffix(".pdf").exists()):
                        fig, ax = plt.subplots(figsize=(4.5, 3.5))
                        incomplete = draw_panel(ax, asset, period, aggregation, horizon, curves)
                        ax.set_ylabel("Probability unresolved")
                        decorate(fig, [ax], asset, "Monthly" if aggregation == "monthly" else "Annual", incomplete, compact=True)
                        save_pair(fig, base)
                        plt.close(fig)
                    n_individual += 1
                if (aggregation == "monthly" and period.endswith("-12")) or aggregation == "annual":
                    print(f"Rendered {aggregation} {asset.upper()} through {period}; {n_individual}/260 individual plot pairs completed.", flush=True)

    for horizon in HORIZONS:
        with PdfPages(out / f"KM_monthly_overview_{horizon}s.pdf") as book:
            for asset in ("btc", "eth"):
                for year in YEARS:
                    fig, axes = plt.subplots(3, 4, figsize=(18, 10.5), sharex=True, sharey=True)
                    missing = False
                    for ax, month in zip(axes.flat, (month for month in MONTHS if month.startswith(year))):
                        missing |= draw_panel(ax, asset, month, "monthly", horizon, curves)
                        if ax in list(axes[:, 0]):
                            ax.set_ylabel("Probability unresolved")
                    decorate(fig, list(axes.flat), asset, f"Monthly, {year}", missing)
                    save_pair(fig, out / "figures" / "monthly_overviews" / f"km_{asset}_{year}_all_months_{horizon}s")
                    book.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
            assert book.get_pagecount() == 10
        with PdfPages(out / f"KM_annual_overview_{horizon}s.pdf") as book:
            for asset in ("btc", "eth"):
                fig, axes = plt.subplots(1, 5, figsize=(22.5, 3.5), sharex=True, sharey=True)
                missing = False
                for ax, year in zip(axes, YEARS):
                    missing |= draw_panel(ax, asset, year, "annual", horizon, curves)
                axes[0].set_ylabel("Probability unresolved")
                decorate(fig, list(axes), asset, "Annual, 2021–2025", missing)
                save_pair(fig, out / "figures" / "annual_overviews" / f"km_{asset}_annual_comparison_{horizon}s")
                book.savefig(fig, bbox_inches="tight")
                plt.close(fig)
            assert book.get_pagecount() == 2
        print(f"Completed the {horizon}s monthly and annual overview PDF books.", flush=True)

    pngs = sorted((out / "figures").rglob("*.png"))
    pdfs = sorted((out / "figures").rglob("*.pdf"))
    assert len(pngs) == len(pdfs) == 284, (len(pngs), len(pdfs))
    assert len(curves) == n_models == 520
    assert sum(len(frame) for frame in frames.values()) == 292240
    validation = dict(source=str(root), source_run_manifest_sha256=manifest_hash, source_daily_event_files=len(found), source_event_rows=int(coverage.raw_event_rows.sum()), source_event_columns="Unchanged saved event columns in monthly base AFT exports; exact daily event counts, dates, markets, origins, grid and methodology verified.", monthly_series_also_validated_against_direct_daily_event_reads=sum(row["validated_against_daily_event_cache"] for row in records), excluded_missing_value_rows=int(coverage.excluded_missing_value_rows.sum()), basis_resolution_pct=90, shock_detection_grid="1s", source_significance_values=sorted(manifest.significance.unique().tolist()), period_range=["2021-01", "2025-12"], monthly_series=480, annual_series=40, annual_estimator="Pool events within calendar year, retaining integer event-count weights; do not average monthly curves.", display_horizons_seconds=list(HORIZONS), display_step_seconds=STEP, monthly_curve_rows_120s=len(frames[("monthly", 120)]), monthly_curve_rows_20s=len(frames[("monthly", 20)]), annual_curve_rows_120s=len(frames[("annual", 120)]), annual_curve_rows_20s=len(frames[("annual", 20)]), individual_monthly_plots=240, individual_annual_plots=20, monthly_overview_figures=20, annual_overview_figures=4, png_figures=len(pngs), vector_pdf_figures=len(pdfs), overview_pdf_books=4, max_absolute_error_against_KM_product_formula=max_product_error, existing_december_curve_reproduction=december_checks, missing_day_market_direction_records=len(missing_rows), elapsed_seconds=round(time.monotonic()-started, 1))
    atomic_json(out / "validation.json", validation)
    readme = """# Monthly and annual Kaplan–Meier curves

Saved V8 events, January 2021–December 2025. These outputs use the existing
90% basis-resolution event definition and the 1-second shock-detection grid.
The saved run manifest records a significance threshold of 0.01.

## Start here

- [Monthly overview, 20 s](./KM_monthly_overview_20s.pdf): 10 pages, one
  BTC/ETH asset-year per page, with all 12 months in a 3-by-4 grid.
- [Monthly overview, 120 s](./KM_monthly_overview_120s.pdf): the same pages
  with the full display horizon.
- [Annual overview, 20 s](./KM_annual_overview_20s.pdf) and
  [annual overview, 120 s](./KM_annual_overview_120s.pdf): BTC and ETH, each
  with separate curves for 2021, 2022, 2023, 2024 and 2025.

Each panel has four curves: linear spot (blue solid), linear perp (blue
dashed), inverse spot (orange solid), and inverse perp (orange dashed).
Individual PNG and vector PDF plots are under [figures/](./figures/).
The two horizons are views of the same fitted survival function.

## Data and estimation

The four `data/*_km_curves_*s.csv` files contain the evaluated KM curves on
the same 0.25-second display grid as the existing thesis plots. Each row
identifies the period, market, direction, event count and day coverage.
The exact event-time KM estimates and risk/event/censor counts through 120 s
are also exported as two Parquet files in [data/](./data/).

Some individual event files are OneDrive placeholders that cannot be opened.
The saved base monthly AFT exports retain their original event durations,
censoring indicators, timestamps and methodology fields. Those unchanged
event columns are used here, before any model-specific covariate filtering.
Every daily event count and date was verified against the production run
manifest. All directly readable, previously cached monthly event counts were
also compared exactly with these saved monthly copies. Source filenames and
cross-check indicators are recorded in the monthly coverage CSV.

Monthly estimates pool all usable saved events in each calendar month.
Annual estimates pool all saved events in each calendar year. Months are
therefore weighted by their event counts, rather than being averaged with
equal weight. Duration/status frequency counts retain exact integer weights.
Events extending beyond 120 seconds are administratively censored at 120 s,
matching the existing KM plotting code. The 20-second figures show the same
survival estimates restricted to 0–20 seconds.

The summaries report an unreached median as a blank value and set
`median_reached_by_horizon` to false. A blank median is not zero.

## Coverage

Every month is represented. BTC linear has all 1,826 days. BTC inverse is
missing 2024-06-12 and 2024-06-13; ETH linear is missing 2025-10-30,
2025-10-31 and 2025-11-03; ETH inverse is missing 2022-10-17, 2023-09-25
and 2023-09-26. These gaps affect both event-origin directions. Asterisks
mark affected plot panels. Curves use the available days without imputing
events for missing dates. Coverage CSVs list every month/year and missing date.

## Validation and reproduction

All per-day event counts in the monthly saved copies were checked against the
saved production manifest.
Every fitted monthly and annual curve was checked against the KM product
formula, bounds, and monotonicity. All five previously saved December panels
for each asset were reproduced numerically, including event counts, at both
horizons. See [validation.json](./validation.json).

The standalone generator [plot_km_curves.py](./plot_km_curves.py)
accepts `--input`, `--output`, `--cache`, and `--jobs`. Its input is the saved
`sa_results/km_v8_final_01` directory. It requires NumPy, pandas, PyArrow,
Matplotlib and lifelines. It reads saved events; it does not rerun event
identification or edit the research repository.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    if Path(__file__).resolve() != out / Path(__file__).name:
        shutil.copyfile(__file__, out / Path(__file__).name)
    exports = sorted(path for path in out.rglob("*") if path.is_file() and path.suffix.lower() != ".zip" and path.name != "file_manifest.json")
    file_manifest = [{"file":path.relative_to(out).as_posix(), "bytes":path.stat().st_size, "sha256":hashlib.sha256(path.read_bytes()).hexdigest()} for path in exports]
    atomic_json(out / "file_manifest.json", file_manifest)
    archive_path = out / "KM_monthly_and_annual.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in exports + [out / "file_manifest.json"]:
            archive.write(path, path.relative_to(out))
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(exports) + 1
    print(json.dumps(validation, indent=2), flush=True)
    print(f"Finished {len(exports)+2} deliverable files including the verified ZIP: {archive_path}", flush=True)


if __name__ == "__main__":
    main()
