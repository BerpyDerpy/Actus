"""The run log: one JSON object per line, one line per sequencer event.

This is the problem statement's "timestamped, structured, lightweight text
file of the conducted steps with outcomes". JSONL rather than a single JSON
document so a crash mid-run still leaves every line written so far readable,
and so a GUI or a ground-side tool can tail it.

Each line carries both clocks: `t` is seconds on the run's own timeline (the
video position when replaying a recording), `wall` is when it was written.
The final `run_end` line carries a per-step summary with statuses,
durations and flags, whether or not the task was completed.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path


class EventLog:
    def __init__(self, path: Path | str, meta: dict | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("a", encoding="utf-8")
        self._write({"kind": "run_start", **(meta or {})})

    def write(self, event) -> None:
        record = {"t": round(event.t, 3), "kind": event.kind, "step": event.step}
        if event.message:
            record["message"] = event.message
        if event.detail:
            record.update(event.detail)
        self._write(record)

    def note(self, record: dict) -> None:
        """A line that is not a sequencer event, e.g. where an alert's snapshot was saved."""
        self._write(dict(record))

    def close(self, final: dict | None = None) -> None:
        """`final` is the sequencer snapshot, so an unfinished run still records
        which steps were done, missed or never reached."""
        if not self._f.closed:
            self._write({"kind": "run_end", **(final or {})})
            self._f.close()

    def _write(self, record: dict) -> None:
        record = {"wall": dt.datetime.now().isoformat(timespec="milliseconds"), **record}
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()
