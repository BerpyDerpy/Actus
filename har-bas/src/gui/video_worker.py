"""Reads a video file or camera on its own thread and hands frames to the GUI.

Decoding, downscaling and the perception overlays all happen off the GUI
thread, so the window stays responsive. Every frame carries `t` on the
video's own timeline (frame / fps for a file), the same clock the step
scripts use.

For a file, decoding runs one thread further ahead: a reader thread decodes
and downscales into a short queue while this thread runs perception on the
previous frame. A 4K frame costs ~16 ms to decode and perception ~25 ms, so
overlapping them is what keeps a 4K take near real time.

Upside down: when `flipped()` says so, each frame is turned 180 degrees
BEFORE perception runs, so the overlays show how perception copes with an
inverted feed rather than a picture that was merely drawn upside down.

Backpressure: the worker sends one frame, then waits for the GUI to `ack()`
it before sending the next, so a busy GUI slows playback instead of piling
up a queue of decoded frames in memory.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

_END = object()


class VideoWorker(QThread):
    frame = pyqtSignal(QImage, float, int)  # image, t seconds, frame index
    opened = pyqtSignal(object)  # duration in seconds, or None for a live camera
    stats = pyqtSignal(object)  # perception Stats, about once a second
    ended = pyqtSignal(float, bool)  # last t, True if the file ran to its end
    failed = pyqtSignal(str)

    def __init__(self, source: Path | int, max_width: int = 1280, paced: bool = True,
                 perception=None, layers: Optional[Callable[[], frozenset]] = None,
                 flipped: Optional[Callable[[], bool]] = None):
        super().__init__()
        self.source = source
        self.max_width = max_width
        self.paced = paced
        self.perception = perception
        self.layers = layers or (lambda: frozenset())
        self.flipped = flipped or (lambda: False)
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._acked = threading.Event()
        self._acked.set()
        self._last_stats = 0.0

    @property
    def live(self) -> bool:
        return isinstance(self.source, int)

    # --- called from the GUI thread ----------------------------------------

    def ack(self) -> None:
        self._acked.set()

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._paused.set()
        else:
            self._paused.clear()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def stop(self) -> None:
        self._stop.set()
        self._acked.set()
        self.wait(3000)

    # --- worker thread -----------------------------------------------------

    def run(self) -> None:
        if self.live:
            self._run_live()
        else:
            self._run_file()

    def _run_file(self) -> None:
        cap = cv2.VideoCapture(str(self.source))
        if not cap.isOpened():
            self.failed.emit(f"cannot open video {self.source}")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.opened.emit(frames / fps if frames > 0 else None)
        decoded: queue.Queue = queue.Queue(maxsize=6)
        reader = threading.Thread(target=self._read_ahead, args=(cap, decoded),
                                  name="video-reader", daemon=True)
        reader.start()

        index, t = 0, 0.0
        anchor = time.perf_counter()
        reached_end = False
        try:
            while not self._stop.is_set():
                if self._paused.is_set():
                    self._stop.wait(0.05)
                    anchor = time.perf_counter() - t  # resume where we left off
                    continue
                try:
                    image = decoded.get(timeout=0.5)
                except queue.Empty:
                    continue
                if image is _END:
                    reached_end = True
                    break
                t = index / fps
                if self.paced:
                    delay = anchor + t - time.perf_counter()
                    if delay > 0 and self._stop.wait(delay):
                        break
                if not self._send(image, t, index):
                    break
                index += 1
        finally:
            self._stop.set()  # releases the reader if it is blocked on a full queue
            reader.join(timeout=2)
            cap.release()
        self.ended.emit(t, reached_end)

    def _read_ahead(self, cap, out: queue.Queue) -> None:
        while not self._stop.is_set():
            ok, image = cap.read()
            item = _downscale(image, self.max_width) if ok else _END
            while not self._stop.is_set():
                try:
                    out.put(item, timeout=0.2)
                    break
                except queue.Full:
                    continue
            if not ok:
                return

    def _run_live(self) -> None:
        from video_source import VideoSource

        t = 0.0
        try:
            with VideoSource(src=self.source) as source:
                self.opened.emit(None)
                sub = source.subscribe("gui")
                t0 = None
                while not self._stop.is_set():
                    frame = sub.read_latest(timeout=0.5)
                    if frame is None or self._paused.is_set():
                        continue
                    if t0 is None:
                        t0 = frame.timestamp
                    t = frame.timestamp - t0
                    image = _downscale(frame.image.copy(), self.max_width)
                    if not self._send(image, t, frame.index):
                        break
        except Exception as exc:  # camera missing, permissions, ...
            self.failed.emit(str(exc))
        self.ended.emit(t, False)

    def _send(self, image, t: float, index: int) -> bool:
        """Overlay, convert and hand over one downscaled BGR frame."""
        if self.flipped():
            image = cv2.rotate(image, cv2.ROTATE_180)
        layers = self.layers()
        if self.perception is not None and layers:
            stats = self.perception.annotate(image, layers)
            now = time.perf_counter()
            if now - self._last_stats >= 1.0:
                self._last_stats = now
                self.stats.emit(stats)

        while not self._acked.wait(0.5):
            if self._stop.is_set():
                return False
        if self._stop.is_set():
            return False
        self._acked.clear()
        self.frame.emit(_to_qimage(image), t, index)
        return True


def _downscale(image, max_width: int):
    if image.shape[1] > max_width:
        scale = max_width / image.shape[1]
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image


def _to_qimage(image) -> QImage:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
