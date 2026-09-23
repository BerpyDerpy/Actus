"""One run of the task: sequencer + step source + log + voice, driven by a clock.

Both front ends (tools/run_fsm.py and the GUI) drive a Session the same way:

    session.start()
    every frame:     session.advance(t)       scripted steps due by t, then tick
    on a keypress:   session.key(code, frame, t)   hotkey mode only
    on Y / N:        session.answer(yes, t)        when a question is pending
    at end of video: session.end_of_video(t)
    finally:         session.close()

Being unsure: a script row `?s2` means the system cannot tell whether s2
happened. It asks the step's yes/no question and holds it as `pending`.
A yes counts the step, timed from when the question was asked; a no repeats
the step's instruction. If the next step arrives before anyone answers, the
question lapses unanswered. The sequencer itself never sees any of this,
only the step observations that result.

`t` is always the video's own timeline. Listeners get every batch of events
after they are logged and spoken, which is how a display finds out.

The log and the voice are passed in rather than built here, so this module
depends on nothing but the sequencer: anything with `write(event)` /
`close(final)` and `say(text)` will do.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from sequencer import Event, StepSequencer
from sources import HotkeySource, ScriptSource
from step_graph import StepGraph

Listener = Callable[[list[Event]], None]


class Session:
    def __init__(self, graph: StepGraph, log, voice, script: Optional[ScriptSource] = None):
        self.graph = graph
        self.seq = StepSequencer(graph)
        self.log = log
        self.voice = voice
        self.script = script
        self.hotkeys = None if script else HotkeySource(graph)
        self.listeners: list[Listener] = []
        self.history: list[Event] = []
        self.next_prompt = ""
        self.last_alert: Optional[Event] = None
        self.pending: Optional[tuple[str, float]] = None  # (step id, asked at)
        self._closed = False

    @property
    def mode(self) -> str:
        return "script" if self.script else "hotkeys"

    def start(self) -> None:
        self._handle(self.seq.start(0.0))

    def advance(self, t: float) -> None:
        if self.script:
            for ts, value in self.script.due(t):
                if value.startswith("?"):
                    self.ask(value[1:], ts)
                elif value in ("yes", "no"):
                    self.answer(value == "yes", ts)
                else:
                    self._observe(value, ts)
        self._handle(self.seq.tick(t))

    def ask(self, step_id: str, t: float) -> None:
        """The system is unsure whether `step_id` happened: ask, and wait for an answer."""
        if self.seq.complete:
            return
        self._lapse(t)
        step = self.graph.get(step_id)
        self.pending = (step_id, t)
        message = self.graph.messages["ask"].format(question=step.question, step=step.name)
        self._handle([Event(t, "ask", step_id, message, {"question": step.question})])

    def answer(self, yes: bool, t: float) -> bool:
        """The operator's yes / no to the pending question. False if nothing was asked."""
        if self.pending is None:
            return False
        step_id, asked = self.pending
        self.pending = None
        step = self.graph.get(step_id)
        if yes:
            self._handle([Event(t, "confirmed", step_id, self.graph.messages["confirmed"],
                                {"asked_at": round(asked, 3)})])
            self._handle(self.seq.observe(step_id, asked))
        else:
            message = self.graph.messages["denied"].format(prompt=step.prompt, step=step.name)
            self._handle([Event(t, "denied", step_id, message, {"asked_at": round(asked, 3)})])
            self.next_prompt = step.prompt
        return True

    def key(self, code: int, frame: int, t: float) -> Optional[str]:
        """A hotkey press. Returns the step id it mapped to, if any."""
        if self.hotkeys is None:
            return None
        step = self.hotkeys.key(code, frame, t)
        if step:
            self._observe(step, t)
        return step

    def end_of_video(self, t: float) -> None:
        """The recording ran out: whatever step was running is over."""
        self._observe(self.graph.idle.id, t)

    def save_hotkeys(self, path: Path) -> Optional[Path]:
        """Write this session's presses as a replayable script, keeping any old one as .bak."""
        if self.hotkeys is None or not self.hotkeys.presses:
            return None
        if path.exists():
            path.replace(path.with_suffix(path.suffix + ".bak"))
        self.hotkeys.save(path)
        return path

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.log.close(final=self.seq.snapshot())

    def _observe(self, step_id: str, t: float) -> None:
        self._lapse(t)
        self._handle(self.seq.observe(step_id, t))

    def _lapse(self, t: float) -> None:
        """Drop an unanswered question: events have moved on without an answer."""
        if self.pending is not None:
            step_id, asked = self.pending
            self.pending = None
            self._handle([Event(t, "unanswered", step_id, None, {"asked_at": round(asked, 3)})])

    def _handle(self, events: list[Event]) -> None:
        if not events:
            return
        for e in events:
            self.log.write(e)
            if e.message:
                self.voice.say(e.message)
            if e.kind in ("ready", "next", "complete"):
                self.next_prompt = e.message
            if e.is_alert:
                self.last_alert = e
        self.history.extend(events)
        for listener in self.listeners:
            listener(events)
