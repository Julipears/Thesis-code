"""
Memory-safe, checkpointed month-by-month wrappers for survival_analysis_basis.py.

Each calendar month is processed independently and written to disk immediately.
Completed months are skipped when resume=True. No all-period event dataframe is
kept in memory unless the user explicitly loads the saved checkpoint files.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl

from survival_analysis_basis import (
    BasisSurvivalAnalysis,
    _format_yyyymmdd,
    build_periods,
    fit_km_grid,
    fit_loglogistic_aft_by_period,
    legacy_find_covariates,
)

PathLike = Union[str, Path]


def _atomic_polars_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.write_parquet(temp)
    os.replace(temp, path)


def _atomic_pandas_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def _atomic_pandas_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temp, index=False)
    os.replace(temp, path)


def _safe_label(label: str) -> str:
    return label.replace("/", "-").replace(" ", "_")


def _write_manifest_row(manifest_path: Path, row: dict) -> None:
    """Upsert one period row in the manifest and save atomically."""
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        if "period" in manifest.columns and "first" in manifest.columns:
            mask = (
                manifest["period"].astype(str).eq(str(row["period"]))
                & manifest["first"].astype(str).eq(str(row["first"]))
            )
            manifest = manifest.loc[~mask]
    else:
        manifest = pd.DataFrame()

    manifest = pd.concat([manifest, pd.DataFrame([row])], ignore_index=True)
    if {"period", "first"}.issubset(manifest.columns):
        manifest = manifest.sort_values(["first", "period"]).reset_index(drop=True)
    _atomic_pandas_csv(manifest, manifest_path)


class MonthlyBasisSurvivalAnalysis(BasisSurvivalAnalysis):
    """Checkpointed survival-analysis workflow that processes one month at a time."""

    def fit_basis_month_by_month(
        self,
        start,
        end,
        checkpoint_dir: PathLike,
        shock_significance: float = 0.001,
        method: str = "lee_mykland",
        resolution_pcts: Iterable[int] = (90,),
        max_duration_seconds: float = 120.0,
        refractory_seconds: float = 5.0,
        min_hold_seconds: float = 0.0,
        basis_tolerance_bps: float = 0.0,
        min_basis_shock_bps: float = 0.0,
        first: str = "spot",
        timeline_step_seconds: float = 1.0,
        resume: bool = True,
        load_followup_days: int = 1,
    ) -> pd.DataFrame:
        """
        Detect basis-resolution episodes and fit KM curves month by month.

        Files written after every successfully completed month:
          events/{first}_{YYYY-MM}.parquet
          km/{first}_{YYYY-MM}.csv
          done/{first}_{YYYY-MM}.done
          manifest.csv

        The done marker is written last, so a crash during a month causes only
        that month to be retried. Completed months remain intact.
        """
        root = Path(checkpoint_dir)
        events_dir = root / "events"
        km_dir = root / "km"
        done_dir = root / "done"
        manifest_path = root / "manifest.csv"
        for directory in (events_dir, km_dir, done_dir):
            directory.mkdir(parents=True, exist_ok=True)

        for period_start, period_end, label in build_periods(start, end, "MS"):
            safe = _safe_label(label)
            stem = f"{first}_{safe}"
            event_path = events_dir / f"{stem}.parquet"
            km_path = km_dir / f"{stem}.csv"
            done_path = done_dir / f"{stem}.done"

            if resume and done_path.exists() and event_path.exists() and km_path.exists():
                print(f"[skip] {label} ({first}) already completed")
                continue

            print(f"[start] {label} ({first})")
            data = None
            event_table = None
            event_pd = None
            km_grid = None

            try:
                data, _, event_table = self.fit_basis_events(
                    start=period_start,
                    end=period_end,
                    shock_significance=shock_significance,
                    method=method,
                    resolution_pcts=resolution_pcts,
                    max_duration_seconds=max_duration_seconds,
                    refractory_seconds=refractory_seconds,
                    min_hold_seconds=min_hold_seconds,
                    basis_tolerance_bps=basis_tolerance_bps,
                    min_basis_shock_bps=min_basis_shock_bps,
                    first=first,
                    load_followup_days=load_followup_days,
                )

                if event_table.height:
                    event_table = event_table.with_columns(pl.lit(label).alias("period"))
                    event_pd = event_table.to_pandas()
                    km_grid, _ = fit_km_grid(
                        event_pd,
                        timeline_step_seconds=timeline_step_seconds,
                        max_duration_seconds=max_duration_seconds,
                    )
                else:
                    # Preserve a valid, typed empty checkpoint.
                    event_table = event_table.with_columns(
                        pl.lit(label).cast(pl.String).alias("period")
                    )
                    km_grid = pd.DataFrame(
                        columns=[
                            "time_s", "survival", "ci_lower", "ci_upper",
                            "period", "first", "shock_significance",
                            "resolution_pct", "n_episodes", "n_resolved",
                            "censoring_share",
                        ]
                    )

                _atomic_polars_parquet(event_table, event_path)
                _atomic_pandas_csv(km_grid, km_path)

                n_episodes = int(event_table.height)
                if event_pd is not None and "event_tick" in event_pd.columns:
                    # One episode appears once per resolution threshold.
                    n_unique_episodes = int(event_pd["event_tick"].nunique())
                    n_resolved = int(event_pd["Status"].sum())
                else:
                    n_unique_episodes = 0
                    n_resolved = 0

                manifest_row = {
                    "period": label,
                    "first": first,
                    "status": "complete",
                    "period_start": pd.Timestamp(period_start).strftime("%Y-%m-%d"),
                    "period_end": pd.Timestamp(period_end).strftime("%Y-%m-%d"),
                    "event_rows": n_episodes,
                    "unique_episodes": n_unique_episodes,
                    "resolved_rows": n_resolved,
                    "resolution_pcts": ",".join(map(str, resolution_pcts)),
                    "max_duration_seconds": max_duration_seconds,
                    "events_file": str(event_path),
                    "km_file": str(km_path),
                }
                _write_manifest_row(manifest_path, manifest_row)
                done_path.write_text("complete\n", encoding="utf-8")
                print(f"[saved] {label}: {n_unique_episodes:,} unique episodes")

            except Exception as exc:
                _write_manifest_row(
                    manifest_path,
                    {
                        "period": label,
                        "first": first,
                        "status": "failed",
                        "period_start": pd.Timestamp(period_start).strftime("%Y-%m-%d"),
                        "period_end": pd.Timestamp(period_end).strftime("%Y-%m-%d"),
                        "error": repr(exc),
                    },
                )
                print(f"[failed] {label}: {exc!r}")
                raise
            finally:
                # Release the month's raw trade frames before loading the next month.
                del data, event_table, event_pd, km_grid
                gc.collect()

        return pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()

    def build_aft_data_month_by_month(
        self,
        start,
        end,
        checkpoint_dir: PathLike,
        shock_significance: float = 0.001,
        method: str = "lee_mykland",
        primary_resolution_pct: int = 90,
        max_duration_seconds: float = 120.0,
        refractory_seconds: float = 5.0,
        min_hold_seconds: float = 0.0,
        basis_tolerance_bps: float = 0.0,
        min_basis_shock_bps: float = 0.0,
        first: str = "spot",
        resume: bool = True,
        load_followup_days: int = 1,
    ) -> pd.DataFrame:
        """
        Build and save one covariate-enriched AFT input file per calendar month.

        This deliberately does not concatenate all months in memory.
        """
        root = Path(checkpoint_dir)
        aft_dir = root / "aft_data"
        done_dir = root / "aft_done"
        manifest_path = root / "aft_manifest.csv"
        aft_dir.mkdir(parents=True, exist_ok=True)
        done_dir.mkdir(parents=True, exist_ok=True)

        for period_start, period_end, label in build_periods(start, end, "MS"):
            safe = _safe_label(label)
            stem = f"{first}_{safe}"
            output_path = aft_dir / f"{stem}.parquet"
            done_path = done_dir / f"{stem}.done"

            if resume and done_path.exists() and output_path.exists():
                print(f"[skip AFT data] {label} ({first})")
                continue

            print(f"[start AFT data] {label} ({first})")
            data = None
            events = None
            period_model = None

            try:
                data, _, events = self.fit_basis_events(
                    start=period_start,
                    end=period_end,
                    shock_significance=shock_significance,
                    method=method,
                    resolution_pcts=[primary_resolution_pct],
                    max_duration_seconds=max_duration_seconds,
                    refractory_seconds=refractory_seconds,
                    min_hold_seconds=min_hold_seconds,
                    basis_tolerance_bps=basis_tolerance_bps,
                    min_basis_shock_bps=min_basis_shock_bps,
                    first=first,
                    load_followup_days=load_followup_days,
                )

                if events.height:
                    event_pd = events.select(
                        [
                            "event_tick", "start_ts", "prev_ts", "Status",
                            "Length", "shock_size",
                        ]
                    ).to_pandas()
                    event_pd["Shock"] = f"{primary_resolution_pct}%"
                    wrapped = {0.001: {"90%": pl.from_pandas(event_pd)}}
                    period_model = legacy_find_covariates(
                        _format_yyyymmdd(period_start),
                        _format_yyyymmdd(period_end),
                        data,
                        wrapped,
                    )
                    period_model["period"] = label
                    period_model["first"] = first
                    period_model["resolution_pct"] = primary_resolution_pct
                    period_model["shock_significance"] = shock_significance
                else:
                    period_model = pd.DataFrame()

                _atomic_pandas_parquet(period_model, output_path)
                _write_manifest_row(
                    manifest_path,
                    {
                        "period": label,
                        "first": first,
                        "status": "complete",
                        "rows": len(period_model),
                        "aft_data_file": str(output_path),
                    },
                )
                done_path.write_text("complete\n", encoding="utf-8")
                print(f"[saved AFT data] {label}: {len(period_model):,} rows")

            except Exception as exc:
                _write_manifest_row(
                    manifest_path,
                    {
                        "period": label,
                        "first": first,
                        "status": "failed",
                        "error": repr(exc),
                    },
                )
                print(f"[failed AFT data] {label}: {exc!r}")
                raise
            finally:
                del data, events, period_model
                gc.collect()

        return pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()


def fit_loglogistic_aft_month_by_month(
    aft_data_dir: PathLike,
    output_dir: PathLike,
    covariates: Sequence[str],
    resolution_pct: int = 90,
    max_duration_seconds: float = 120.0,
    timeline_step_seconds: float = 1.0,
    percentile_step: int = 1,
    min_episodes: int = 100,
    ancillary: bool = False,
    resume: bool = True,
) -> pd.DataFrame:
    """Fit and checkpoint one Log-Logistic AFT model per saved monthly file."""
    source = Path(aft_data_dir)
    root = Path(output_dir)
    coef_dir = root / "coefficients"
    curve_dir = root / "curves"
    pct_dir = root / "percentiles"
    done_dir = root / "done"
    manifest_path = root / "manifest.csv"
    for directory in (coef_dir, curve_dir, pct_dir, done_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for input_path in sorted(source.glob("*.parquet")):
        stem = input_path.stem
        done_path = done_dir / f"{stem}.done"
        coef_path = coef_dir / f"{stem}.csv"
        curve_path = curve_dir / f"{stem}.csv"
        pct_path = pct_dir / f"{stem}.csv"

        if resume and done_path.exists() and coef_path.exists() and curve_path.exists():
            print(f"[skip AFT fit] {stem}")
            continue

        monthly = None
        coefficients = None
        curves = None
        percentiles = None
        try:
            monthly = pd.read_parquet(input_path)
            if monthly.empty:
                coefficients = pd.DataFrame()
                curves = pd.DataFrame()
                percentiles = pd.DataFrame()
            else:
                _, coefficients, curves, percentiles = fit_loglogistic_aft_by_period(
                    monthly,
                    covariates=covariates,
                    period_col="period",
                    resolution_pct=resolution_pct,
                    max_duration_seconds=max_duration_seconds,
                    timeline_step_seconds=timeline_step_seconds,
                    percentile_step=percentile_step,
                    min_episodes=min_episodes,
                    ancillary=ancillary,
                )

            _atomic_pandas_csv(coefficients, coef_path)
            _atomic_pandas_csv(curves, curve_path)
            _atomic_pandas_csv(percentiles, pct_path)
            _write_manifest_row(
                manifest_path,
                {
                    "period": stem,
                    "first": stem.split("_")[0],
                    "status": "complete",
                    "input_rows": len(monthly),
                    "coefficient_rows": len(coefficients),
                    "curve_rows": len(curves),
                    "percentile_rows": len(percentiles),
                },
            )
            done_path.write_text("complete\n", encoding="utf-8")
            print(f"[saved AFT fit] {stem}")
        except Exception as exc:
            _write_manifest_row(
                manifest_path,
                {
                    "period": stem,
                    "first": stem.split("_")[0],
                    "status": "failed",
                    "error": repr(exc),
                },
            )
            print(f"[failed AFT fit] {stem}: {exc!r}")
            raise
        finally:
            del monthly, coefficients, curves, percentiles
            gc.collect()

    return pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()


def load_monthly_km(checkpoint_dir: PathLike, first: Optional[str] = None) -> pd.DataFrame:
    """Load only the small saved KM grids; does not load event-level checkpoints."""
    km_dir = Path(checkpoint_dir) / "km"
    pattern = f"{first}_*.csv" if first else "*.csv"
    frames = [pd.read_csv(path) for path in sorted(km_dir.glob(pattern))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_monthly_events(
    checkpoint_dir: PathLike,
    first: Optional[str] = None,
    periods: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load selected monthly event files only; avoid loading the whole sample by default."""
    events_dir = Path(checkpoint_dir) / "events"
    pattern = f"{first}_*.parquet" if first else "*.parquet"
    paths = sorted(events_dir.glob(pattern))
    if periods is not None:
        wanted = set(periods)
        paths = [path for path in paths if any(period in path.stem for period in wanted)]
    frames = [pd.read_parquet(path) for path in paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
