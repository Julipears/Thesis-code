"""Exercise checkpoint planning and restart handling without downloading data."""
import collections
import json
from pathlib import Path
import sqlite3
import tempfile
from unittest.mock import patch

import run_hasbrouck_random_full_grid as audit


with tempfile.TemporaryDirectory(dir=audit.ROOT) as temp:
    audit.OUT = Path(temp)
    plan = audit.prepare()
    assert len(plan["dates"]) == 600
    assert len({x["day"] for x in plan["dates"]}) == 600
    assert set(collections.Counter(x["day"][:7] for x in plan["dates"]).values()) == {10}
    assert all(0 <= x["hour_utc"] < 24 for x in plan["dates"])
    assert audit.prepare() == plan
    assert audit.progress(plan)["total_fits"] == 7200
    first = next(audit.tasks(plan))
    launches = []

    class FakeWorker:
        pid = 987654321

        def __init__(self, *args, **kwargs):
            launches.append(1)
            self.code = 1 if len(launches) == 1 else 0
            audit.active(first, "fit", latency="10ms")

        def poll(self):
            return self.code

        def wait(self):
            return self.code

    with patch.object(audit.subprocess, "Popen", FakeWorker), patch.object(audit.time, "sleep"):
        audit.supervise(plan)
    assert len(launches) == 2, "An error must restart the worker"
    status = json.loads((audit.OUT / "status.json").read_text())
    assert status["status"] == "incomplete_errors", "Missing fits must never be reported complete"
    assert status["restarts"] == 1
    assert status["failed_tasks"][0]["attempts"] == 1
    with audit.database() as db:
        db.execute("INSERT INTO fits VALUES (?, ?, ?)", (first["task"], "1s", '{"rows": []}'))
    audit.prepare()
    assert audit.progress(plan)["completed_fits"] == 1, "Reopening must preserve checkpoints"

print("PASS: 600-date reproducible plan, restart after failure, incomplete reporting, durable checkpoint")
