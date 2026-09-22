"""Resumable full-grid audit with isolated workers and a ten-minute watchdog."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import calendar
import csv
import datetime as dt
import json
import os
from pathlib import Path
import random
import sqlite3
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "hasbrouck_spec_audit/random_10_days_month_full_grid"
CONTRACTS = {"btc_um": ("BTCUSDT", "um"), "btc_cm": ("BTCUSDT", "cm"),
             "eth_um": ("ETHUSDT", "um"), "eth_cm": ("ETHUSDT", "cm")}
LATENCIES = ("1s", "100ms", "10ms")
SEED = 20260914
CHECK_SECONDS = 600
MAX_ATTEMPTS = 3


@contextmanager
def database():
    db = sqlite3.connect(OUT / "results.sqlite", timeout=30)
    try:
        with db:
            yield db
    finally:
        db.close()


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    # Windows readers or file-sync software can briefly hold the destination.
    for attempt in range(20):
        try:
            temp.replace(path)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.1 * (attempt + 1))


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUT / "sample_plan.json"
    if not plan_path.exists():
        rng = random.Random(SEED)
        dates = []
        for year in range(2021, 2026):
            for month in range(1, 13):
                for day in sorted(rng.sample(range(1, calendar.monthrange(year, month)[1] + 1), 10)):
                    dates.append(dict(day=f"{year:04d}-{month:02d}-{day:02d}", hour_utc=rng.randrange(24)))
        write_json(plan_path, dict(seed=SEED, fill_gaps=True, latencies=LATENCIES,
                                  contracts=CONTRACTS, dates=dates))
    plan = json.loads(plan_path.read_text())
    with database() as db:
        db.execute("CREATE TABLE IF NOT EXISTS fits (task TEXT, latency TEXT, payload TEXT, PRIMARY KEY(task, latency))")
        db.execute("CREATE TABLE IF NOT EXISTS failures (task TEXT PRIMARY KEY, attempts INTEGER, last_error TEXT)")
    return plan


def tasks(plan):
    for date in plan["dates"]:
        for market in CONTRACTS:
            yield dict(**date, market=market, task=f"{date['day']}_{date['hour_utc']:02d}_{market}")


def progress(plan):
    with database() as db:
        complete = db.execute("SELECT COUNT(*) FROM fits").fetchone()[0]
        failed = db.execute("SELECT task, attempts, last_error FROM failures ORDER BY task").fetchall()
    total = len(plan["dates"]) * len(CONTRACTS) * len(LATENCIES)
    return dict(completed_fits=complete, total_fits=total,
                failed_tasks=[dict(task=x[0], attempts=x[1], last_error=x[2]) for x in failed])


def active(task, stage, **extra):
    write_json(OUT / "active.json", dict(**task, stage=stage, updated_at=now(), updated_epoch=time.time(), **extra))


def worker(plan, limit=12):
    # Large scientific imports and trade data stay out of the supervisor.
    import gc
    import numpy as np
    import pandas as pd
    from audit_hasbrouck_one_hour import current_fit, rows_for_result
    from trade_data_pull import TradeData
    from vecm_hasbrouck3 import generate_multiple_lags

    lags = generate_multiple_lags(10, list(LATENCIES), max_length="10s")
    processed = 0
    for task in tasks(plan):
        key = task["task"]
        with database() as db:
            done = {r[0] for r in db.execute("SELECT latency FROM fits WHERE task=?", (key,))}
            failure = db.execute("SELECT attempts FROM failures WHERE task=?", (key,)).fetchone()
        needed = [latency for latency in LATENCIES if latency not in done]
        if not needed or (failure and failure[0] >= MAX_ATTEMPTS):
            continue
        active(task, "download")
        print(f"{now()} START {key} {needed}", flush=True)
        started = time.perf_counter()
        symbol, margin = CONTRACTS[task["market"]]
        day = pd.Timestamp(task["day"])
        start = day + pd.Timedelta(hours=task["hour_utc"])
        data = TradeData(symbol, "Binance", margin)
        data.grab_trades_data(day.to_pydatetime(), days=1, n_jobs=2)
        load_seconds = time.perf_counter() - started
        if data.df_trades_spots.is_empty() or data.df_trades_perps.is_empty():
            raise RuntimeError("Daily archive missing, empty, or download failed")
        for latency in needed:
            active(task, "aggregate", latency=latency)
            began = time.perf_counter()
            frame = data.agg_last_trade_to_intervals(
                freq=latency, start=start, end=start + pd.Timedelta(hours=1),
                fill_gaps=True, rename_for_vecm=True, retain_initial_grid_row=True,
            )[["log_midpoint_spot", "log_midpoint_perp"]]
            if frame.empty or not np.isfinite(frame.to_numpy()).all():
                raise ValueError("Empty or nonfinite price grid")
            aggregation_seconds = time.perf_counter() - began
            active(task, "fit", latency=latency)
            began = time.perf_counter()
            fit, n_obs, condition = current_fit(frame, latency, lags)
            fit_seconds = time.perf_counter() - began
            rows = rows_for_result(latency, "full_grid_intercept_float32", fit, n_obs, condition, len(frame))
            for row in rows:
                row.update(market=task["market"], day=task["day"], hour_utc=task["hour_utc"],
                           hour_start_utc=start.isoformat())
            payload = dict(task=task, latency=latency, load_seconds=load_seconds,
                           aggregation_seconds=aggregation_seconds, fit_seconds=fit_seconds,
                           completed_at=now(), rows=rows)
            encoded = json.dumps(payload, allow_nan=False)
            with database() as db:
                db.execute("INSERT INTO fits VALUES (?, ?, ?)", (key, latency, encoded))
            active(task, "saved", latency=latency)
            print(f"{now()} SAVED {key} {latency} load={load_seconds:.2f}s aggregate={aggregation_seconds:.2f}s fit={fit_seconds:.2f}s", flush=True)
            del frame, fit, rows, payload
        del data
        gc.collect()
        processed += 1
        if processed >= limit:
            return 10  # Planned recycle to release all retained process memory.
    return 0


def export_results():
    with database() as db:
        records = db.execute("SELECT payload FROM fits ORDER BY task, latency").fetchall()
    rows = [row for record in records for row in json.loads(record[0])["rows"]]
    if rows:
        temp = OUT / "daily_results.csv.tmp"
        with temp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(OUT / "daily_results.csv")


def log(message):
    print(f"{now()} {message}", flush=True)
    with (OUT / "monitor.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{now()} {message}\n")


def record_failure(task, error):
    if task:
        with database() as db:
            db.execute("INSERT INTO failures VALUES (?, 1, ?) ON CONFLICT(task) DO UPDATE SET attempts=attempts+1, last_error=excluded.last_error", (task, error))
    log(f"ERROR task={task} {error}; restarting worker")


def supervise(plan):
    # Lock is released by Windows even if the supervisor crashes.
    import msvcrt
    lock = (OUT / "supervisor.lock").open("a+b")
    lock.seek(0)
    if lock.read(1) == b"":
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    try:
        import psutil
    except ImportError:
        psutil = None
    started = time.time()
    initial = progress(plan)["completed_fits"]
    worker_process = None
    restarts = 0
    last_check = 0
    startup_failures = 0
    try:
        while True:
            write_json(OUT / "active.json", dict(stage="starting", updated_epoch=time.time(), updated_at=now()))
            with (OUT / "worker.log").open("a", encoding="utf-8") as output:
                worker_process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--worker"],
                    cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                log(f"Worker started pid={worker_process.pid}")
                forced_error = None
                while worker_process.poll() is None:
                    stamp = time.time()
                    if stamp - last_check >= CHECK_SECONDS:
                        last_check = stamp
                        state = json.loads((OUT / "active.json").read_text())
                        status = progress(plan)
                        gained = status["completed_fits"] - initial
                        remaining = status["total_fits"] - status["completed_fits"]
                        status.update(status="running", checked_at=now(), supervisor_pid=os.getpid(),
                                      worker_pid=worker_process.pid, restarts=restarts, active=state,
                                      elapsed_hours=(stamp-started)/3600,
                                      estimated_remaining_hours=((stamp-started)*remaining/gained/3600 if gained else None))
                        if psutil:
                            try:
                                rss = psutil.Process(worker_process.pid).memory_info().rss
                                available = psutil.virtual_memory().available
                                status.update(worker_rss_gb=rss/2**30, available_memory_gb=available/2**30)
                                if available < 0.75*2**30 and rss > 2*2**30:
                                    forced_error = "Critical memory pressure"
                            except psutil.NoSuchProcess:
                                pass
                        if stamp - state["updated_epoch"] > 1200:
                            forced_error = "No stage progress for more than 20 minutes"
                        write_json(OUT / "status.json", status)
                        export_results()
                        log(f"CHECK {status['completed_fits']}/{status['total_fits']} fits; restarts={restarts}; active={state.get('task')} {state['stage']}")
                        if forced_error:
                            worker_process.kill()
                            worker_process.wait()
                            break
                    time.sleep(2)
            code = worker_process.wait()
            if code == 0:
                break
            if code == 10:
                log("Planned worker recycle after 12 contract-days")
                continue
            state = json.loads((OUT / "active.json").read_text())
            record_failure(state.get("task"), forced_error or f"Worker exit code {code}; see worker.log")
            restarts += 1
            if not state.get("task"):
                startup_failures += 1
                if startup_failures >= 3:
                    raise RuntimeError("Three worker startup failures; inspect worker.log")
            else:
                startup_failures = 0
            time.sleep(5)
        export_results()
        status = progress(plan)
        status.update(status="complete" if status["completed_fits"] == status["total_fits"] else "incomplete_errors",
                      checked_at=now(), supervisor_pid=os.getpid(), restarts=restarts,
                      elapsed_hours=(time.time()-started)/3600)
        write_json(OUT / "status.json", status)
        log(f"FINISHED {status['status']} {status['completed_fits']}/{status['total_fits']}")
    except BaseException:
        if worker_process and worker_process.poll() is None:
            worker_process.kill()
            worker_process.wait()
        write_json(OUT / "status.json", dict(**progress(plan), status="supervisor_error", checked_at=now(), error=traceback.format_exc()))
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    plan = prepare()
    if args.prepare_only:
        print(json.dumps(progress(plan)))
    elif args.worker:
        try:
            sys.exit(worker(plan))
        except Exception:
            traceback.print_exc()
            sys.exit(1)
    else:
        supervise(plan)
