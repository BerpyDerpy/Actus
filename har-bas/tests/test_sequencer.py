"""Sequencer, step graph and step source tests, against the real step_graph.yaml.

Run from har-bas/:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "fsm"))

import yaml  # noqa: E402
from sequencer import DONE, MISSED, PENDING, StepSequencer  # noqa: E402
from sources import HotkeySource, ScriptSource  # noqa: E402
from step_graph import DEFAULT_PATH, load_step_graph  # noqa: E402

GRAPH = load_step_graph()


def kinds(events):
    return [e.kind for e in events]


def run(seq, observations):
    """Feed (time, step) pairs, return every event emitted."""
    out = []
    for t, step in observations:
        out += seq.observe(step, t)
    return out


HAPPY = [(1, "s1"), (3, "s2"), (6, "s3"), (10, "s4"), (12, "s5"), (16, "s0")]


class HappyPath(unittest.TestCase):
    def test_start_speaks_idle_prompt(self):
        [ready] = StepSequencer(GRAPH).start(0.0)
        self.assertEqual(ready.kind, "ready")
        self.assertEqual(ready.message, GRAPH.idle.prompt)

    def test_full_run_completes_without_alerts(self):
        seq = StepSequencer(GRAPH)
        events = run(seq, HAPPY)
        self.assertFalse([e for e in events if e.is_alert])
        self.assertTrue(seq.complete)
        self.assertEqual(events[-1].kind, "complete")
        self.assertEqual(events[-1].message, GRAPH.messages["complete"])
        self.assertTrue(all(r.status == DONE for r in seq.records.values()))

    def test_each_start_suggests_the_following_step(self):
        seq = StepSequencer(GRAPH)
        events = seq.observe("s1", 1)
        self.assertEqual(kinds(events), ["started", "next"])
        self.assertEqual(events[1].step, "s2")
        self.assertEqual(events[1].message, GRAPH.get("s2").prompt)

    def test_last_step_has_no_next_and_needs_idle_to_complete(self):
        seq = StepSequencer(GRAPH)
        run(seq, HAPPY[:-1])
        self.assertFalse(seq.complete)
        self.assertEqual(seq.active, "s5")
        self.assertEqual(kinds(seq.observe("s0", 16)), ["done", "complete"])

    def test_durations_recorded(self):
        seq = StepSequencer(GRAPH)
        run(seq, HAPPY)
        self.assertAlmostEqual(seq.records["s3"].duration, 4.0)

    def test_repeated_observation_is_ignored(self):
        seq = StepSequencer(GRAPH)
        seq.observe("s1", 1)
        self.assertEqual(seq.observe("s1", 1.5), [])

    def test_idle_between_steps_closes_the_step_and_waits(self):
        seq = StepSequencer(GRAPH)
        run(seq, [(1, "s1"), (3, "s0")])
        self.assertIsNone(seq.active)
        self.assertEqual(seq.expected.id, "s2")
        self.assertEqual(kinds(seq.observe("s2", 5)), ["started", "next"])

    def test_everything_after_complete_is_ignored(self):
        seq = StepSequencer(GRAPH)
        run(seq, HAPPY)
        self.assertEqual(seq.observe("s2", 20), [])


class Skips(unittest.TestCase):
    def test_pour_without_unscrewing(self):
        """The demo's mistake run: tilt the bottle with the cap still on."""
        seq = StepSequencer(GRAPH)
        run(seq, [(1, "s1")])
        events = seq.observe("s3", 3)

        self.assertEqual(kinds(events), ["done", "skipped", "next"])
        self.assertEqual(
            events[1].message,
            "Wait. You started pour into container without unscrew cap.",
        )
        self.assertEqual(events[1].detail["missing"], ["s2"])
        self.assertEqual(events[2].step, "s2")
        self.assertEqual(events[2].message, "Unscrew the cap.")

        # The skipped-to step is not accepted: the protocol holds at s2.
        self.assertIsNone(seq.active)
        self.assertEqual(seq.records["s3"].status, PENDING)
        self.assertEqual(seq.records["s2"].status, MISSED)
        self.assertEqual(seq.expected.id, "s2")

    def test_recovering_from_a_skip_carries_on_normally(self):
        seq = StepSequencer(GRAPH)
        run(seq, [(1, "s1"), (3, "s3")])
        events = run(seq, [(5, "s2"), (8, "s3"), (12, "s4"), (14, "s5"), (18, "s0")])
        self.assertFalse([e for e in events if e.is_alert])
        self.assertTrue(seq.complete)
        self.assertTrue(seq.records["s2"].was_missed)
        self.assertEqual(seq.records["s2"].status, DONE)

    def test_multi_step_skip_names_the_earliest_missing(self):
        seq = StepSequencer(GRAPH)
        run(seq, [(1, "s1")])
        events = seq.observe("s4", 3)
        skipped = next(e for e in events if e.kind == "skipped")
        self.assertEqual(skipped.detail["missing"], ["s2", "s3"])
        self.assertIn("without unscrew cap", skipped.message)
        self.assertEqual(events[-1].step, "s2")

    def test_skip_from_the_very_start(self):
        seq = StepSequencer(GRAPH)
        events = seq.observe("s3", 1)
        self.assertEqual(kinds(events), ["skipped", "next"])
        self.assertEqual(events[0].detail["missing"], ["s1", "s2"])

    def test_the_same_skip_twice_alerts_once(self):
        seq = StepSequencer(GRAPH)
        run(seq, [(1, "s1"), (3, "s3")])
        self.assertEqual(seq.observe("s3", 3.5), [])

    def test_skip_again_after_idle_alerts_again(self):
        seq = StepSequencer(GRAPH)
        run(seq, [(1, "s1"), (3, "s3"), (4, "s0")])
        self.assertIn("skipped", kinds(seq.observe("s3", 5)))


class OutOfOrder(unittest.TestCase):
    def test_redoing_a_done_step(self):
        seq = StepSequencer(GRAPH)
        run(seq, [(1, "s1"), (3, "s2"), (6, "s3")])
        events = seq.observe("s2", 9)
        self.assertEqual(kinds(events), ["done", "out_of_order"])
        self.assertEqual(
            events[1].message, "Wait. unscrew cap is already done. Next, screw cap back on."
        )
        self.assertEqual(seq.expected.id, "s4")


class Timing(unittest.TestCase):
    def test_too_fast_step_is_flagged_but_counts(self):
        seq = StepSequencer(GRAPH)
        events = run(seq, [(1, "s1"), (1.1, "s2")])
        self.assertIn("too_fast", kinds(events))
        self.assertTrue(seq.records["s1"].too_fast)
        self.assertEqual(seq.records["s1"].status, DONE)
        self.assertEqual(seq.active, "s2")

    def test_stuck_fires_once_past_max_duration(self):
        seq = StepSequencer(GRAPH)
        seq.observe("s1", 0)
        limit = GRAPH.get("s1").max_duration
        self.assertEqual(seq.tick(limit - 1), [])
        [stuck] = seq.tick(limit + 1)
        self.assertEqual(stuck.kind, "stuck")
        self.assertEqual(stuck.message, "Still on pick up bottle. Do you need help?")
        self.assertEqual(seq.tick(limit + 5), [])

    def test_stuck_rearms_on_the_next_step(self):
        seq = StepSequencer(GRAPH)
        seq.observe("s1", 0)
        seq.tick(100)
        seq.observe("s2", 101)
        self.assertEqual(kinds(seq.tick(101 + GRAPH.get("s2").max_duration + 1)), ["stuck"])

    def test_no_stuck_while_idle(self):
        seq = StepSequencer(GRAPH)
        self.assertEqual(seq.tick(1000), [])


class Sources(unittest.TestCase):
    def test_script_accepts_ids_and_hotkeys_and_replays_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "script.csv"
            path.write_text("time_s,step\n3.0,s2\n1.0,1\n5.0,0\n", encoding="utf-8")
            src = ScriptSource(path, GRAPH)
            self.assertEqual(src.due(0.5), [])
            self.assertEqual(src.due(3.0), [(1.0, "s1"), (3.0, "s2")])
            self.assertFalse(src.exhausted)
            self.assertEqual(src.due(99), [(5.0, "s0")])
            self.assertTrue(src.exhausted)

    def test_script_rejects_unknown_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "script.csv"
            path.write_text("time_s,step\n1.0,s9\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                ScriptSource(path, GRAPH)

    def test_hotkey_session_saves_a_replayable_script(self):
        keys = HotkeySource(GRAPH)
        self.assertEqual(keys.key(ord("1"), 30, 1.0), "s1")
        self.assertIsNone(keys.key(ord("q"), 40, 1.3))
        self.assertIsNone(keys.key(ord("9"), 45, 1.5))  # no step on 9
        self.assertEqual(keys.key(ord("0"), 90, 3.0), "s0")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "take.csv"
            keys.save(path)
            self.assertEqual(ScriptSource(path, GRAPH).due(99), [(1.0, "s1"), (3.0, "s0")])


class GraphValidation(unittest.TestCase):
    def _load(self, mutate):
        raw = yaml.safe_load(DEFAULT_PATH.read_text(encoding="utf-8"))
        mutate(raw)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            return load_step_graph(path)

    def test_real_graph_loads(self):
        self.assertEqual([s.id for s in GRAPH.task_steps], ["s1", "s2", "s3", "s4", "s5"])

    def test_requirement_listed_later_is_rejected(self):
        def mutate(raw):
            raw["steps"][1]["requires"] = ["s3"]
        with self.assertRaises(ValueError):
            self._load(mutate)

    def test_missing_message_is_rejected(self):
        with self.assertRaises(ValueError):
            self._load(lambda raw: raw["messages"].pop("stuck"))


if __name__ == "__main__":
    unittest.main()
