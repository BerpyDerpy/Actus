"""Where step observations come from, before a trained model exists.

Both sources hand the sequencer the same thing: (time, step id) pairs.

    ScriptSource   replays a CSV of timed steps against a recorded video, so
                   the demo runs identically every time with nobody at the keys
    HotkeySource   a hidden operator presses 0-5 as the performer works, and
                   every press is kept so the session can be saved as a script

The trained classifier becomes a third source later. The sequencer does not
change.

Script CSV format, one row per step start (header required, frame optional):

    frame,time_s,step
    62,2.067,s1
    118,3.933,s2

`time_s` is seconds from the start of the video. `step` is a step id or its
hotkey number. End the script with a row for s0 (idle): that is what closes
the last step and completes the task.

Two more kinds of row script the system being unsure:

    ?s2      the system cannot tell whether s2 happened, and asks (`?2` works too)
    yes/no   the operator's answer, when the answer is scripted as well

Leave the answer out of the script to have whoever is at the keyboard answer
live (Y / N in the GUI).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from step_graph import StepGraph


class ScriptSource:
    def __init__(self, path: Path | str, graph: StepGraph):
        self.path = Path(path)
        self._rows: list[tuple[float, str]] = []
        with self.path.open(newline="", encoding="utf-8") as f:
            for n, row in enumerate(csv.DictReader(f), start=2):
                step = _resolve(row["step"].strip(), graph)
                if step is None:
                    raise ValueError(f"{self.path}:{n}: unknown step {row['step']!r}")
                self._rows.append((float(row["time_s"]), step))
        self._rows.sort(key=lambda r: r[0])
        self._cursor = 0

    def due(self, now: float) -> list[tuple[float, str]]:
        """Every scripted step whose time has come, oldest first."""
        out = []
        while self._cursor < len(self._rows) and self._rows[self._cursor][0] <= now:
            out.append(self._rows[self._cursor])
            self._cursor += 1
        return out

    @property
    def rows(self) -> list[tuple[float, str]]:
        """The whole script, in time order, regardless of how far replay has got."""
        return list(self._rows)

    @property
    def exhausted(self) -> bool:
        return self._cursor >= len(self._rows)


class HotkeySource:
    def __init__(self, graph: StepGraph):
        self.graph = graph
        self.presses: list[tuple[int, float, str]] = []  # (frame, time_s, step id)

    def key(self, key: int, frame: int, now: float) -> Optional[str]:
        """Feed a cv2.waitKey code. Returns the step id if it was a step hotkey."""
        if not (ord("0") <= key <= ord("9")):
            return None
        step = self.graph.by_hotkey(key - ord("0"))
        if step is None:
            return None
        self.presses.append((frame, now, step.id))
        return step.id

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "time_s", "step"])
            for frame, now, step in self.presses:
                writer.writerow([frame, f"{now:.3f}", step])


ANSWERS = ("yes", "no")


def _resolve(value: str, graph: StepGraph) -> Optional[str]:
    """A step id, "?<step id>" for a question, or "yes"/"no"; None if invalid."""
    if value.lower() in ANSWERS:
        return value.lower()
    if value.startswith("?"):
        step = _resolve(value[1:], graph)
        return f"?{step}" if step and step not in ANSWERS else None
    if value.isdigit():
        step = graph.by_hotkey(int(value))
        return step.id if step else None
    try:
        return graph.get(value).id
    except KeyError:
        return None
