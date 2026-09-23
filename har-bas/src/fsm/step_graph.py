"""Loads configs/step_graph.yaml into typed, validated objects.

The YAML is the contract every other part reads (recorder, labelling, FSM,
voice, GUI), so a typo in it should fail loudly here at startup, not as a
KeyError halfway through a demo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "step_graph.yaml"

MESSAGE_KEYS = ("skipped", "out_of_order", "too_fast", "stuck", "complete",
                "ask", "confirmed", "denied")


@dataclass(frozen=True)
class Step:
    id: str
    name: str
    prompt: str
    requires: tuple[str, ...]
    min_duration: float
    max_duration: Optional[float]
    hotkey: Optional[int]
    question: str  # yes/no check for when the system is unsure this step happened


@dataclass(frozen=True)
class StepGraph:
    steps: tuple[Step, ...]
    messages: dict[str, str]
    confirm_seconds: float
    min_confidence: float

    @property
    def idle(self) -> Step:
        """The resting state. Seeing it closes whatever step was active."""
        return self.steps[0]

    @property
    def task_steps(self) -> tuple[Step, ...]:
        """Every step except idle, in protocol order."""
        return self.steps[1:]

    def get(self, step_id: str) -> Step:
        for step in self.steps:
            if step.id == step_id:
                return step
        raise KeyError(f"unknown step id {step_id!r}")

    def by_hotkey(self, key: int) -> Optional[Step]:
        for step in self.steps:
            if step.hotkey == key:
                return step
        return None

    def index(self, step_id: str) -> int:
        return [s.id for s in self.steps].index(step_id)


def load_step_graph(path: Path | str = DEFAULT_PATH) -> StepGraph:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    steps = tuple(
        Step(
            id=str(s["id"]),
            name=str(s["name"]),
            prompt=str(s["prompt"]),
            requires=tuple(s.get("requires") or ()),
            min_duration=float(s.get("min_duration") or 0.0),
            max_duration=(
                None if s.get("max_duration") is None else float(s["max_duration"])
            ),
            hotkey=s.get("hotkey"),
            question=str(s.get("question") or f"Did you {s['name']}?"),
        )
        for s in raw["steps"]
    )
    graph = StepGraph(
        steps=steps,
        messages={k: str(v) for k, v in (raw.get("messages") or {}).items()},
        confirm_seconds=float(raw.get("confirm_seconds", 0.0)),
        min_confidence=float(raw.get("min_confidence", 0.0)),
    )
    _validate(graph)
    return graph


def _validate(graph: StepGraph) -> None:
    ids = [s.id for s in graph.steps]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate step ids in step graph: {ids}")
    if len(graph.steps) < 2:
        raise ValueError("step graph needs an idle step and at least one task step")
    if graph.idle.requires:
        raise ValueError(f"first step {graph.idle.id!r} is idle and must require nothing")

    seen = set()
    for step in graph.steps:
        for req in step.requires:
            if req not in ids:
                raise ValueError(f"{step.id} requires unknown step {req!r}")
            if req not in seen:
                raise ValueError(
                    f"{step.id} requires {req!r}, which is listed after it; "
                    "steps must be listed in protocol order"
                )
        seen.add(step.id)

    hotkeys = [s.hotkey for s in graph.steps if s.hotkey is not None]
    if len(hotkeys) != len(set(hotkeys)):
        raise ValueError(f"duplicate hotkeys in step graph: {hotkeys}")

    missing = [k for k in MESSAGE_KEYS if k not in graph.messages]
    if missing:
        raise ValueError(f"step graph is missing messages: {missing}")
