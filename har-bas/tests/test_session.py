"""Session: scripted replay and the "unsure" questions, with a fake log and voice.

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

from session import Session  # noqa: E402
from sources import ScriptSource  # noqa: E402
from step_graph import load_step_graph  # noqa: E402

GRAPH = load_step_graph()


class FakeLog:
    def __init__(self):
        self.events, self.final = [], None

    def write(self, event):
        self.events.append(event)

    def close(self, final=None):
        self.final = final


class FakeVoice:
    def __init__(self):
        self.said = []

    def say(self, text):
        self.said.append(text)


def scripted(rows: str) -> Session:
    tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8")
    tmp.write("time_s,step\n" + rows)
    tmp.close()
    session = Session(GRAPH, FakeLog(), FakeVoice(), ScriptSource(tmp.name, GRAPH))
    Path(tmp.name).unlink()
    session.start()
    return session


def kinds(session):
    return [e.kind for e in session.history]


class Unsure(unittest.TestCase):
    def test_script_parses_questions_and_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.csv"
            path.write_text("time_s,step\n1,?s2\n2,YES\n3,?3\n4,no\n", encoding="utf-8")
            self.assertEqual([v for _, v in ScriptSource(path, GRAPH).rows],
                             ["?s2", "yes", "?s3", "no"])

    def test_script_rejects_a_question_about_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.csv"
            path.write_text("time_s,step\n1,?s9\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                ScriptSource(path, GRAPH)

    def test_ask_speaks_the_steps_question_and_waits(self):
        s = scripted("1,s1\n3,?s2\n")
        s.advance(3)
        ask = s.history[-1]
        self.assertEqual(ask.kind, "ask")
        self.assertEqual(ask.message, "I can't see clearly. Is the cap off?")
        self.assertEqual(s.voice.said[-1], ask.message)
        self.assertEqual(s.pending, ("s2", 3.0))
        self.assertEqual(s.seq.active, "s1")  # nothing counted yet

    def test_yes_counts_the_step_from_when_it_was_asked(self):
        s = scripted("1,s1\n3,?s2\n")
        s.advance(3)
        self.assertTrue(s.answer(True, 5.0))
        self.assertIsNone(s.pending)
        self.assertEqual(s.seq.active, "s2")
        self.assertEqual(s.seq.records["s2"].started, 3.0)
        self.assertEqual(s.seq.records["s1"].ended, 3.0)
        self.assertIn("Thanks. Noted.", s.voice.said)
        self.assertEqual(s.next_prompt, GRAPH.get("s3").prompt)

    def test_no_repeats_the_instruction(self):
        s = scripted("1,s1\n3,?s2\n")
        s.advance(3)
        s.answer(False, 5.0)
        self.assertEqual(s.history[-1].kind, "denied")
        self.assertEqual(s.history[-1].message, "Okay. Unscrew the cap.")
        self.assertEqual(s.next_prompt, "Unscrew the cap.")
        self.assertEqual(s.seq.active, "s1")

    def test_scripted_answer(self):
        s = scripted("1,s1\n3,?s2\n4,yes\n8,s3\n")
        s.advance(10)
        after_ask = [(e.kind, e.step) for e in s.history[kinds(s).index("ask"):]]
        self.assertEqual(after_ask[:4], [("ask", "s2"), ("confirmed", "s2"),
                                         ("done", "s1"), ("started", "s2")])
        self.assertEqual(s.seq.records["s2"].started, 3.0)
        self.assertEqual(s.seq.records["s2"].status, "done")
        self.assertEqual(s.seq.active, "s3")

    def test_unanswered_question_lapses_when_the_next_step_arrives(self):
        s = scripted("1,s1\n3,?s2\n6,s2\n")
        s.advance(10)
        self.assertIn("unanswered", kinds(s))
        self.assertIsNone(s.pending)
        self.assertEqual(s.seq.active, "s2")
        self.assertEqual(s.seq.records["s2"].started, 6.0)

    def test_answer_with_nothing_asked_is_ignored(self):
        s = scripted("1,s1\n")
        s.advance(2)
        self.assertFalse(s.answer(True, 2))
        self.assertNotIn("confirmed", kinds(s))

    def test_questions_are_logged_not_alerted(self):
        s = scripted("1,s1\n3,?s2\n4,no\n")
        s.advance(5)
        logged = [e.kind for e in s.log.events]
        self.assertIn("ask", logged)
        self.assertIn("denied", logged)
        self.assertIsNone(s.last_alert)


if __name__ == "__main__":
    unittest.main()
