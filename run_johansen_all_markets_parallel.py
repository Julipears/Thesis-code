"""Run/resume each market's rotating-hour Johansen audit in parallel."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


MARKETS = ("btc_um", "btc_cm", "eth_um", "eth_cm")
LOG_DIR = Path("hasbrouck_spec_audit/johansen_rotating_hour/run_logs")


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    processes = []
    handles = []
    try:
        for market in MARKETS:
            log_path = LOG_DIR / f"{market}.log"
            handle = log_path.open("a", encoding="utf-8", buffering=1)
            handles.append(handle)
            command = [
                sys.executable,
                "johansen_rotating_hour_audit.py",
                "--markets", market,
                "--start", "2021-01-01",
                "--end", "2025-12-31",
            ]
            process = subprocess.Popen(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                cwd=Path(__file__).resolve().parent,
            )
            processes.append((market, process, log_path))
            print(f"[started] {market} pid={process.pid} log={log_path}", flush=True)

        failures = []
        for market, process, log_path in processes:
            return_code = process.wait()
            print(f"[finished] {market} exit={return_code} log={log_path}", flush=True)
            if return_code != 0:
                failures.append((market, return_code))
        if failures:
            raise SystemExit(f"Johansen market runners failed: {failures}")
    finally:
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()
