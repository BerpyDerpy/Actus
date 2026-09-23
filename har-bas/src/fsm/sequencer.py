"""The step-sequence state machine: observed steps in, protocol events out.

Input is a stream of "step X is happening now" observations, each with a
timestamp. Where they come from does not matter here -- a hidden operator's
hotkeys, a replayed script, or (later) the trained classifier after it has
been debounced. Output is a list of `Event`s per call that the voice, the log
and the GUI all consume.

Rules:
    * A new observation ends the active step. The active step is then done,
      or flagged too_fast if it lasted less than its min_duration.
    * Observing idle only ends the active step. Once the last task step
      has ended this way, the task is complete.
    * A step whose requirements are all done is accepted and becomes active,
      and the step after it is suggested as next.
    * A step whose requirements are not done is a skip. It is NOT accepted:
      the protocol holds where it was, every unmet requirement is marked
      missed, and the earliest one is suggested as next. Doing it clears the
      miss, and the sequence carries on normally from there.
    * Observing a step that is already done is out of order. It is alerted
      and otherwise ignored.
    * Repeating the previous observation does nothing, so a source may
      report the same step every frame.
    * `tick()` raises `stuck` once when the active step exceeds its
      max_duration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from step_graph import Step, StepGraph

PENDING = "pending"
ACTIVE = "active"
DONE = "done"
MISSED = "missed"

ALERT_KINDS = frozenset({"skipped", "out_of_order", "too_fast", "stuck"})


@dataclass(frozen=True)
class Event:
    t: float
    kind: str  # ready|started|done|next|skipped|out_of_order|too_fast|stuck|complete
    step: Optional[str] = None
    message: Optional[str] = None  # text to speak and show, if any
    detail: dict = field(default_factory=dict)

    @property
    def is_alert(self) -> bool:
        return self.kind in ALERT_KINDS


@dataclass
class StepRecord:
    status: str = PENDING
    started: Optional[float] = None
    ended: Optional[float] = None
    too_fast: bool = False
    was_missed: bool = False

    @property
    def duration(self) -> Optional[float]:
        if self.started is None or self.ended is None:
            return None
        return self.ended - self.started


class StepSequencer:
    def __init__(self, graph: StepGraph):
        self.graph = graph
        self.records: dict[str, StepRecord] = {s.id: StepRecord() for s in graph.task_steps}
        self.active: Optional[str] = None
        self.active_since: Optional[float] = None
        self.complete = False
        self._last_observed: Optional[str] = None
        self._stuck_warned = False

    # --- queries -----------------------------------------------------------

    @property
    def expected(self) -> Optional[Step]:
        """The next step the protocol wants: the earliest one not yet done."""
        for step in self.graph.task_steps:
            if self.records[step.id].status in (PENDING, MISSED):
                return step
        return None

    def snapshot(self) -> dict:
        """Everything a display needs to draw the current state."""
        expected = self.expected
        return {
            "active": self.active,
            "expected": expected.id if expected else None,
            "complete": self.complete,
            "steps": [
                {
                    "id": s.id,
                    "name": s.name,
                    "status": self.records[s.id].status,
                    "too_fast": self.records[s.id].too_fast,
                    "was_missed": self.records[s.id].was_missed,
                    "duration": (
                        None if self.records[s.id].duration is None
                        else round(self.records[s.id].duration, 3)
                    ),
                }
                for s in self.graph.task_steps
            ],
        }

    # --- inputs ------------------------------------------------------------

    def start(self, t: float) -> list[Event]:
        first = self.graph.task_steps[0]
        return [Event(t, "ready", first.id, self.graph.idle.prompt)]

    def observe(self, step_id: str, t: float) -> list[Event]:
        if self.complete or step_id == self._last_observed:
            return []
        self._last_observed = step_id

        events = self._close_active(t)
        if step_id == self.graph.idle.id or self.complete:
            return events

        step = self.graph.get(step_id)
        record = self.records[step.id]

        if record.status == DONE:
            expected = self.expected
            events.append(
                self._alert(
                    t, "out_of_order", step,
                    expected=expected.name if expected else "nothing",
                )
            )
            return events

        unmet = self._unmet_requirements(step)
        if unmet:
            for missing in unmet:
                self.records[missing.id].status = MISSED
                self.records[missing.id].was_missed = True
            events.append(
                self._alert(t, "skipped", step, missing=unmet[0].name,
                            detail={"missing": [m.id for m in unmet]})
            )
            events.append(Event(t, "next", unmet[0].id, unmet[0].prompt))
            return events

        record.status = ACTIVE
        record.started = t
        self.active = step.id
        self.active_since = t
        self._stuck_warned = False
        events.append(Event(t, "started", step.id))

        following = self._next_after(step)
        if following is not None:
            events.append(Event(t, "next", following.id, following.prompt))
        return events

    def tick(self, t: float) -> list[Event]:
        if self.active is None or self._stuck_warned:
            return []
        step = self.graph.get(self.active)
        if step.max_duration is not None and t - self.active_since > step.max_duration:
            self._stuck_warned = True
            return [self._alert(t, "stuck", step)]
        return []

    # --- internals ---------------------------------------------------------

    def _close_active(self, t: float) -> list[Event]:
        if self.active is None:
            return []
        step = self.graph.get(self.active)
        record = self.records[step.id]
        record.status = DONE
        record.ended = t
        self.active = None
        self.active_since = None

        detail = {"duration": round(record.duration, 3)}
        if record.duration < step.min_duration:
            record.too_fast = True
            events = [self._alert(t, "too_fast", step, detail=detail)]
        else:
            events = [Event(t, "done", step.id, detail=detail)]

        if all(r.status == DONE for r in self.records.values()):
            self.complete = True
            events.append(
                Event(t, "complete", None, self.graph.messages["complete"],
                      detail={"summary": self.snapshot()["steps"]})
            )
        return events

    def _unmet_requirements(self, step: Step) -> list[Step]:
        """Every not-done step `step` depends on, directly or not, in protocol order."""
        unmet: set[str] = set()
        stack = list(step.requires)
        seen: set[str] = set()
        while stack:
            sid = stack.pop()
            if sid in seen or sid == self.graph.idle.id:
                continue
            seen.add(sid)
            if self.records[sid].status != DONE:
                unmet.add(sid)
            stack.extend(self.graph.get(sid).requires)
        return [s for s in self.graph.task_steps if s.id in unmet]

    def _next_after(self, step: Step) -> Optional[Step]:
        after = self.graph.task_steps[self.graph.task_steps.index(step) + 1:]
        for candidate in after:
            if self.records[candidate.id].status != DONE:
                return candidate
        return None

    def _alert(self, t: float, kind: str, step: Step, detail: Optional[dict] = None,
               **fields: str) -> Event:
        message = self.graph.messages[kind].format(step=step.name, **fields)
        return Event(t, kind, step.id, message, detail or {})
