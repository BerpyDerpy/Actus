"""Detection, hand and marker overlays for the monitoring GUI.

One object owns the three stages and draws their results onto a frame:

    objects   YOLO11m, stock COCO weights, filtered to the classes the task
              involves (bottle, cup, wine glass, bowl)
    hands     MediaPipe hand landmarker, up to two hands, 21 points each
    markers   ArUco DICT_4X4_50, ids named by what they are stuck on

What these outputs do NOT do yet is drive the step sequencer. The steps still
come from the step source (script or hotkeys); these layers show what the
perception stack sees on the same frames, and are the inputs the trained
step classifier will be built on.

Hands, not body pose: the takes are framed on the table, so the camera sees
forearms and hands and almost never a head or shoulders. BlazePose (the
33-point body model in pose_test.py) needs the upper body to anchor on; on
take1 it returned a skeleton on 65% of frames but often a wrong one, e.g.
drawn along the bottle mid-pour. The hand landmarker found hands on 56% of
frames and placed them correctly, and draws nothing rather than guessing.
It also matches the problem statement's "hand-object interaction".

Cost per 1280x720 frame on the RTX 4070 laptop, measured on take1: YOLO
~21 ms (GPU), hands ~20 ms (CPU), markers ~3 ms (CPU). YOLO and hands run
concurrently, since one waits on the GPU while the other uses the CPU, which
brings the frame to roughly the slower of the two (~25 ms).

Markers use STOCK detector parameters, not marker_test.py's tuned ones. The
tuned set exists for markers shown on a phone screen (inverted, glare); on
these printed markers it measured 2.7x slower AND found fewer: 356 vs 394
detections over 275 frames of take1 + the error take.

Orientation: detection is relative to the payload, not the camera. The
problem statement's optional requirement is that there is no fixed "up" in
orbit. Hands and markers do not care (measured on take1 flipped 180 degrees:
hands 106/215 frames vs 103 upright, markers 409 vs 404), but YOLO was
trained on upright photos and collapses when the frame is flipped (bottle
30/215 vs 161, cup 0/215 vs 215). So "up" is taken from the markers fixed
to the payload (the table #3, else the container #0), whose top edge sits
within 3 degrees of horizontal on every upright take. Each frame, the
detector sees the frame turned so that edge is level (snapped to 90
degrees), and its boxes are mapped back onto the displayed frame. When
neither marker is visible the last known orientation is kept, since a
camera does not turn between two frames.

Loading (weights, CUDA context, first-inference autotuning) takes a few
seconds, so `load()` is meant to run off the GUI thread; `ready` flips once
everything is warm.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
YOLO_MODEL = ROOT / "yolo11m.pt"
HAND_MODEL = ROOT / "models" / "hands" / "hand_landmarker.task"

LAYERS = ("objects", "hands", "markers")

# COCO classes worth drawing for this task. Everything else (person, chair,
# the laptop on the desk) is clutter.
OBJECT_CLASSES = {
    "bottle": (0, 140, 255),      # BGR, orange
    "cup": (255, 180, 0),         # blue
    "wine glass": (255, 180, 0),
    "bowl": (255, 180, 0),
}
CONFIDENCE = 0.40

# Which marker id is stuck on what, as mounted on the recorded takes. This is
# NOT the Stage2.md plan (which had 0 on the bottle and 2 on the container):
# on the takes, id 0 is on the cup and id 2 is on the bottle.
MARKER_NAMES = {0: "container", 1: "cap", 2: "bottle", 3: "table"}
MARKER_COLOR = (80, 220, 80)
# Markers that do not move relative to the payload, best first. Their top
# edge is level when the payload is upright.
PAYLOAD_MARKERS = (3, 0)

# Payload roll (degrees, image y down) -> the cv2.rotate that levels the frame.
UNROTATE = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_CLOCKWISE}

# The 21 hand landmarks as bones: wrist, then thumb and four fingers, plus
# the palm edge.
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
HAND_COLORS = {"Left": (0, 200, 255), "Right": (255, 200, 0)}  # amber, cyan (as pose_test)


@dataclass
class Stats:
    objects: int = 0
    hands: int = 0
    markers: int = 0
    rotation: int = 0  # degrees the payload is turned in the image
    ms: dict = field(default_factory=dict)


class Perception:
    def __init__(self):
        self.ready = False
        self.error: str | None = None
        self.device_label = ""
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hands")
        self._last_ts = 0
        self.rotation = 0  # last known payload roll, a multiple of 90

    # --- setup -------------------------------------------------------------

    def load(self) -> None:
        """Load and warm every stage. Slow; call from a background thread."""
        try:
            import mediapipe as mp
            import torch
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                HandLandmarker,
                HandLandmarkerOptions,
                RunningMode,
            )
            from ultralytics import YOLO

            for path, what in ((YOLO_MODEL, "detector weights"), (HAND_MODEL, "hand model")):
                if not path.exists():
                    raise FileNotFoundError(f"missing {what} at {path}")

            if torch.cuda.is_available():
                self.device, gpu = 0, torch.cuda.get_device_name(0)
            elif torch.backends.mps.is_available():
                self.device, gpu = "mps", "Apple Silicon"
            else:
                self.device, gpu = "cpu", "CPU"

            self._mp = mp
            self.model = YOLO(str(YOLO_MODEL))
            self.class_ids = [i for i, n in self.model.names.items() if n in OBJECT_CLASSES]
            self.hand_landmarker = HandLandmarker.create_from_options(HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
                running_mode=RunningMode.VIDEO,
                num_hands=2,
            ))
            self.aruco = cv2.aruco.ArucoDetector(
                cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                cv2.aruco.DetectorParameters(),
            )

            # First inferences include CUDA setup and cuDNN autotuning: pay
            # for them here, not as a freeze on the first frame of a take.
            blank = np.zeros((720, 1280, 3), np.uint8)
            for _ in range(3):
                self._detect(blank)
                self._hands(blank)

            self.device_label = f"YOLO11m on {gpu} · hands on CPU"
            self.ready = True
        except Exception as exc:  # the GUI must still run without overlays
            self.error = f"{type(exc).__name__}: {exc}"

    # --- per frame ---------------------------------------------------------

    def annotate(self, frame: np.ndarray, layers) -> Stats:
        """Run the enabled layers on `frame` (BGR) and draw them onto it in place."""
        stats = Stats()
        if not self.ready or not layers:
            return stats

        with self._lock:
            t0 = time.perf_counter()
            hands_job = self._pool.submit(self._hands, frame) if "hands" in layers else None
            # Markers first: they say which way is up for the detector.
            markers = self._markers(frame) if layers & {"markers", "objects"} else []
            self.rotation = payload_rotation(markers, self.rotation)
            boxes = self._detect_levelled(frame, self.rotation) if "objects" in layers else []
            hands = hands_job.result() if hands_job else []
            elapsed = (time.perf_counter() - t0) * 1000
        if "markers" not in layers:
            markers = []

        for name, conf, box in boxes:
            _label_box(frame, box, f"{name} {conf:.2f}", OBJECT_CLASSES[name])
        for marker_id, corners in markers:
            pts = corners.astype(int)
            cv2.polylines(frame, [pts], True, MARKER_COLOR, 2, cv2.LINE_AA)
            x, y = pts[:, 0].min(), pts[:, 1].min()
            _tag(frame, (int(x), int(y) - 6), f"#{marker_id} {MARKER_NAMES.get(marker_id, '?')}",
                 MARKER_COLOR)
        for side, points in hands:
            _draw_hand(frame, points, HAND_COLORS.get(side, (255, 255, 255)))

        if self.rotation and "objects" in layers:
            _tag(frame, (10, frame.shape[0] - 14),
                 f"payload frame: turned {self.rotation} deg, detector levelled", (255, 255, 255))

        stats.objects, stats.hands, stats.markers = len(boxes), len(hands), len(markers)
        stats.rotation = self.rotation
        stats.ms = {"frame": elapsed}
        return stats

    def close(self) -> None:
        self._pool.shutdown(wait=False)
        if self.ready:
            self.hand_landmarker.close()

    # --- stages ------------------------------------------------------------

    def _detect(self, frame):
        result = self.model.predict(frame, device=self.device, conf=CONFIDENCE,
                                    classes=self.class_ids, verbose=False)[0]
        names = self.model.names
        return [
            (names[int(c)], float(p), tuple(int(v) for v in xyxy))
            for c, p, xyxy in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy)
        ]

    def _detect_levelled(self, frame, rotation: int):
        """Detect on the frame turned payload-up; boxes come back in `frame` coordinates."""
        if rotation == 0:
            return self._detect(frame)
        h, w = frame.shape[:2]
        boxes = self._detect(cv2.rotate(frame, UNROTATE[rotation]))
        return [(name, conf, unrotate_box(box, rotation, w, h)) for name, conf, box in boxes]

    def _hands(self, frame):
        """[(handedness, [(x, y) pixel per landmark])] for each hand found."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode needs strictly increasing timestamps, across takes too:
        # restarting a take resets the video clock but not the landmarker.
        ts = max(int(time.perf_counter() * 1000), self._last_ts + 1)
        self._last_ts = ts
        result = self.hand_landmarker.detect_for_video(image, ts)
        h, w = frame.shape[:2]
        return [
            (handed[0].category_name if handed else "",
             [(int(p.x * w), int(p.y * h)) for p in landmarks])
            for landmarks, handed in zip(result.hand_landmarks, result.handedness)
        ]

    def _markers(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco.detectMarkers(gray)
        if ids is None:
            return []
        return [(int(i), c.reshape(-1, 2)) for i, c in zip(ids.flatten(), corners)]


def payload_rotation(markers, previous: int) -> int:
    """Payload roll in the image from a fixed marker, snapped to 90 degrees."""
    found = dict(markers)
    for marker_id in PAYLOAD_MARKERS:
        if marker_id in found:
            top = found[marker_id][1] - found[marker_id][0]  # corner 0 -> 1: the top edge
            angle = float(np.degrees(np.arctan2(top[1], top[0])))
            return int(round(angle / 90.0)) % 4 * 90
    return previous


def unrotate_box(box, rotation: int, w: int, h: int):
    """Map a box found in the levelled frame back to the original w x h frame."""
    x1, y1, x2, y2 = box
    if rotation == 180:
        pts = [(w - 1 - x, h - 1 - y) for x, y in ((x1, y1), (x2, y2))]
    elif rotation == 90:  # levelled with ROTATE_90_COUNTERCLOCKWISE
        pts = [(w - 1 - y, x) for x, y in ((x1, y1), (x2, y2))]
    elif rotation == 270:  # levelled with ROTATE_90_CLOCKWISE
        pts = [(y, h - 1 - x) for x, y in ((x1, y1), (x2, y2))]
    else:
        return box
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _draw_hand(frame, points, colour) -> None:
    for a, b in HAND_EDGES:
        cv2.line(frame, points[a], points[b], colour, 2, cv2.LINE_AA)
    for p in points:
        cv2.circle(frame, p, 3, (0, 0, 255), -1, cv2.LINE_AA)


def _tag(frame, origin, text, colour) -> None:
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x, y = origin
    y = max(y, h + 4)
    cv2.rectangle(frame, (x, y - h - 4), (x + w + 6, y + base - 2), colour, -1)
    cv2.putText(frame, text, (x + 3, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def _label_box(frame, box, text, colour) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)
    _tag(frame, (x1, y1), text, colour)
