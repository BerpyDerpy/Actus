"""End-to-end GUI runs, offscreen and unpaced, against a synthetic video.

Run from har-bas/:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
for sub in ("fsm", "io", "gui"):
    sys.path.insert(0, str(ROOT / "src" / sub))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

try:
    from PyQt6.QtWidgets import QApplication
except ImportError:  # pragma: no cover
    QApplication = None

FPS = 30
SECONDS = 22


def make_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (320, 180))
    for i in range(FPS * SECONDS):
        img = np.full((180, 320, 3), 30, np.uint8)
        cv2.putText(img, f"{i / FPS:5.2f}", (90, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(img)
    writer.release()


def video_frame_rgb(path: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, image = cap.read()
    cap.release()
    assert ok
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def qimage_to_rgb(image) -> np.ndarray:
    from PyQt6.QtGui import QImage

    image = image.convertToFormat(QImage.Format.Format_RGB888)
    w, h, stride = image.width(), image.height(), image.bytesPerLine()
    ptr = image.constBits()
    ptr.setsize(h * stride)
    return np.frombuffer(ptr, np.uint8).reshape(h, stride)[:, : w * 3].reshape(h, w, 3).copy()


@unittest.skipIf(QApplication is None, "PyQt6 not installed")
class GuiRuns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from main_window import MainWindow
        from step_graph import load_step_graph
        from voice import Voice

        cls.app = QApplication.instance() or QApplication([])
        cls.graph = load_step_graph()
        cls.Voice = Voice
        cls.MainWindow = MainWindow

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.video = self.dir / "take.mp4"
        make_video(self.video)

    def tearDown(self):
        self.window.close()
        self.tmp.cleanup()

    def open(self, window_cls=None, script_rows=None):
        if script_rows is not None:
            with (self.dir / "take_steps.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["time_s", "step"])
                w.writerows(script_rows)
        cls = window_cls or self.MainWindow
        self.window = cls(self.graph, self.dir / "logs", self.Voice(enabled=False), paced=False)
        self.window.show()
        self.window.open_take(self.video)
        deadline = time.monotonic() + 60
        while self.window.state not in ("COMPLETE", "INCOMPLETE", "STOPPED"):
            self.app.processEvents()
            self.assertLess(time.monotonic(), deadline, "run did not finish")
        return self.window

    def glyphs(self):
        return [self.window.steps.rows[s.id][0].text() for s in self.graph.task_steps]

    def test_scripted_mistake_run(self):
        w = self.open(script_rows=[
            (2.0, "s1"), (4.0, "s3"), (6.0, "s0"), (8.0, "s2"),
            (11.0, "s3"), (14.5, "s4"), (18.5, "s5"), (20.5, "s0"),
        ])
        self.assertEqual(w.state, "COMPLETE")
        self.assertEqual(w.session.mode, "script")
        self.assertEqual(w.alerts.count(), 1)
        card = w.alerts.cards[0]
        self.assertIn("without unscrew cap", card.message)
        self.assertEqual(w.events.rowCount(), len(w.session.history))
        self.assertEqual(self.glyphs(), ["✓"] * 5)
        self.assertEqual(w.next_card.text.text(), "Task complete.")
        self.assertEqual(w.next_card.caption.text(), "COMPLETE")

        records = [json.loads(line) for line in
                   w.session.log.path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(records[-1]["kind"], "run_end")
        self.assertTrue(records[-1]["complete"])

        # The alert carries the frame it fired on: shown, saved, and logged.
        self.assertIsNotNone(card.image)
        [snap] = [r for r in records if r["kind"] == "snapshot"]
        self.assertEqual((snap["alert"], snap["step"], snap["t"]), ("skipped", "s3", 4.0))
        saved = w.session.log.path.parent / snap["path"]
        self.assertTrue(saved.exists())
        # It is the frame the alert fired on (t=4.0 -> frame 120), not the one before.
        shot = qimage_to_rgb(card.image)
        self.assertTrue(np.array_equal(shot, video_frame_rgb(self.video, 120)))
        self.assertFalse(np.array_equal(shot, video_frame_rgb(self.video, 119)))

    def test_unfinished_run_ends_incomplete(self):
        w = self.open(script_rows=[(2.0, "s1"), (4.0, "s3"), (7.0, "s5")])
        self.assertEqual(w.state, "INCOMPLETE")
        self.assertEqual(w.alerts.count(), 2)
        self.assertEqual(self.glyphs(), ["✓", "✗", "✗", "✗", "○"])
        self.assertEqual(w.next_card.text.text(), "Unscrew the cap.")

    def test_timeline_follows_the_run(self):
        w = self.open(script_rows=[
            (2.0, "s1"), (4.0, "s3"), (6.0, "s0"), (8.0, "s2"),
            (11.0, "s3"), (14.5, "s4"), (18.5, "s5"), (20.5, "s0"),
        ])
        bar = w.timeline
        self.assertAlmostEqual(bar.duration, SECONDS, places=1)
        self.assertEqual([seg[0] for seg in bar.segments], ["s1", "s2", "s3", "s4", "s5"])
        self.assertTrue(all(seg[3] == "done" for seg in bar.segments))
        self.assertEqual(bar.segments[1][1:3], [8.0, 11.0])
        self.assertEqual([t for t, _, _ in bar.ticks], [4.0])  # the skip alert
        self.assertEqual(bar.complete_at, 20.5)

    def test_unsure_question_answered_yes_by_the_operator(self):
        seen = {}

        class AnsweringWindow(self.MainWindow):
            def _on_frame(self, image, t, index):
                super()._on_frame(image, t, index)
                if index == 140:  # t = 4.67s, while the question is up
                    seen["banner"] = not self.video.question.isHidden()
                    seen["card"] = self.next_card.caption.text()
                    seen["glyph"] = self.steps.rows["s2"][0].text()
                    self.answer(True)

        w = self.open(window_cls=AnsweringWindow, script_rows=[
            (2.0, "s1"), (4.0, "?s2"), (11.0, "s3"), (14.5, "s4"), (18.5, "s5"), (20.5, "s0"),
        ])
        self.assertEqual(seen, {"banner": True, "card": "CONFIRM", "glyph": "?"})
        self.assertEqual(w.state, "COMPLETE")
        self.assertEqual(w.session.seq.records["s2"].started, 4.0)
        self.assertTrue(w.video.question.isHidden())
        self.assertEqual(w.next_card.caption.text(), "COMPLETE")
        self.assertEqual([t for t, _, _ in w.timeline.ticks], [4.0])  # the question, amber

    def test_upside_down_flips_the_frames_shown(self):
        class FlippedWindow(self.MainWindow):
            def restart(self):
                self.flip_button.setChecked(True)
                super().restart()

        w = self.open(window_cls=FlippedWindow, script_rows=[(2.0, "s1")])
        last = FPS * SECONDS - 1
        shown = qimage_to_rgb(w.video.current_image())
        self.assertTrue(np.array_equal(shown, np.rot90(video_frame_rgb(self.video, last), 2)))

    def test_hotkeys_are_saved_and_restart_replays_them(self):
        presses = {60: 1, 120: 2, 240: 3, 360: 4, 480: 5, 600: 0}

        class KeyedWindow(self.MainWindow):
            def _on_frame(self, image, t, index):
                super()._on_frame(image, t, index)
                if index in presses:
                    self.step_key(presses[index])

        w = self.open(window_cls=KeyedWindow)
        self.assertEqual(w.session.mode, "hotkeys")
        self.assertEqual(w.state, "COMPLETE")
        script = self.dir / "take_steps.csv"
        self.assertTrue(script.exists())
        with script.open() as f:
            self.assertEqual([r["step"] for r in csv.DictReader(f)],
                             ["s1", "s2", "s3", "s4", "s5", "s0"])

        # Restart now finds the saved script and replays it with no keys.
        presses.clear()
        w.restart()
        deadline = time.monotonic() + 60
        while w.state not in ("COMPLETE", "INCOMPLETE"):
            self.app.processEvents()
            self.assertLess(time.monotonic(), deadline)
        self.assertEqual(w.session.mode, "script")
        self.assertEqual(w.state, "COMPLETE")
        self.assertEqual(w.alerts.count(), 0)


if __name__ == "__main__":
    unittest.main()
