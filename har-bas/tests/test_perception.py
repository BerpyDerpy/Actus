"""Perception overlays: the three layers, layer filtering, and the GUI wiring.

Needs the model files (yolo11m.pt, models/hands/hand_landmarker.task); the
whole module skips without them. The real-frame checks also skip when the
git-ignored takes/ folder is absent.

Run from har-bas/:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
for sub in ("fsm", "io", "gui", "perception"):
    sys.path.insert(0, str(ROOT / "src" / sub))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from perception import HAND_MODEL, LAYERS, YOLO_MODEL, Perception  # noqa: E402

TAKE1 = ROOT / "takes" / "take1.mp4"
HAVE_MODELS = YOLO_MODEL.exists() and HAND_MODEL.exists()


def marker_frame(marker_id: int) -> np.ndarray:
    """A 1280x720 white frame with one printed-size marker in it."""
    frame = np.full((720, 1280, 3), 255, np.uint8)
    marker = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), marker_id, 120)
    frame[300:420, 580:700] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    return frame


def take1_frame(index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(TAKE1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    assert ok
    return cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)


@unittest.skipUnless(HAVE_MODELS, "perception model files not present")
class Layers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = Perception()
        cls.p.load()

    @classmethod
    def tearDownClass(cls):
        cls.p.close()

    def test_loads(self):
        self.assertIsNone(self.p.error)
        self.assertTrue(self.p.ready)

    def test_blank_frame_finds_nothing(self):
        stats = self.p.annotate(np.zeros((720, 1280, 3), np.uint8), set(LAYERS))
        self.assertEqual((stats.objects, stats.hands, stats.markers), (0, 0, 0))

    def test_marker_found_and_drawn(self):
        frame = marker_frame(2)
        before = frame.copy()
        stats = self.p.annotate(frame, {"markers"})
        self.assertEqual(stats.markers, 1)
        self.assertFalse(np.array_equal(frame, before), "nothing was drawn")

    def test_disabled_layers_do_not_run(self):
        frame = marker_frame(2)
        before = frame.copy()
        stats = self.p.annotate(frame, {"objects"})
        self.assertEqual(stats.markers, 0)
        self.assertTrue(np.array_equal(frame, before))

    def test_no_layers_is_a_no_op(self):
        frame = marker_frame(2)
        before = frame.copy()
        self.p.annotate(frame, frozenset())
        self.assertTrue(np.array_equal(frame, before))

    @unittest.skipUnless(TAKE1.exists(), "takes/take1.mp4 not present")
    def test_take1_hand_on_bottle(self):
        """Frame 250: one hand grips the bottle, the other lifts the cap."""
        from perception import MARKER_NAMES

        frame = take1_frame(250)
        boxes = self.p._detect(frame)
        self.assertIn("bottle", {name for name, _, _ in boxes})
        self.assertIn("cup", {name for name, _, _ in boxes})
        marker_ids = {i for i, _ in self.p._markers(frame)}
        self.assertLessEqual({0, 2}, marker_ids)
        self.assertEqual((MARKER_NAMES[0], MARKER_NAMES[2]), ("container", "bottle"))
        self.assertGreaterEqual(len(self.p._hands(frame)), 1)


@unittest.skipUnless(HAVE_MODELS, "perception model files not present")
class GuiOverlays(unittest.TestCase):
    def test_overlays_reach_the_window(self):
        from main_window import MainWindow
        from PyQt6.QtWidgets import QApplication
        from step_graph import load_step_graph
        from voice import Voice

        app = QApplication.instance() or QApplication([])
        perception = Perception()
        perception.load()
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "take.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
            for _ in range(60):
                writer.write(marker_frame(3))
            writer.release()

            w = MainWindow(load_step_graph(), Path(tmp) / "logs", Voice(enabled=False),
                           paced=False, perception=perception)
            w.show()
            deadline = time.monotonic() + 10
            while not all(b.isEnabled() for b in w.layer_buttons.values()):
                app.processEvents()
                self.assertLess(time.monotonic(), deadline)

            # Markers only: the stats readout names them, and the frame shows the outline.
            w.layer_buttons["objects"].setChecked(False)
            w.layer_buttons["hands"].setChecked(False)
            w.open_take(video)
            deadline = time.monotonic() + 60
            while w.state not in ("COMPLETE", "INCOMPLETE"):
                app.processEvents()
                self.assertLess(time.monotonic(), deadline)
            self.assertIn("1 mk", w.perception_label.text())
            self.assertNotIn("obj", w.perception_label.text())
            shown = w.video.current_image()
            self.assertIsNotNone(shown)
            w.close()
        perception.close()


if __name__ == "__main__":
    unittest.main()
