"""Threaded webcam capture with independent, non-blocking consumers.

One background thread owns the camera and fans each frame out to every
registered subscriber. Each subscriber has its own bounded queue with a
drop-oldest policy, so a slow consumer (e.g. a disk writer flushing to
storage) never stalls a fast one (e.g. the inference loop).

Downstream consumers of this module are the local disk writer, the network
streamer and the inference loop -- none of which are implemented yet.
"""

from __future__ import annotations

import logging
import platform
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Frame:
    """A single captured frame plus the metadata consumers need to stay in sync.

    `image` is shared by reference across all subscribers -- treat it as
    read-only and call `.copy()` before drawing on it.
    """

    index: int
    timestamp: float
    image: np.ndarray

    @property
    def shape(self) -> tuple:
        return self.image.shape


class Subscription:
    """A single consumer's view of the stream.

    Backed by a bounded queue; when the consumer falls behind, the oldest
    frame is dropped so the consumer always sees recent data and the capture
    thread is never blocked.
    """

    def __init__(self, name: str, maxsize: int = 2):
        self.name = name
        self._q: queue.Queue[Frame] = queue.Queue(maxsize=maxsize)
        self.dropped = 0

    def _offer(self, frame: Frame) -> None:
        """Called by the capture thread only. Never blocks."""
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            try:
                self._q.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(frame)
            except queue.Full:
                self.dropped += 1

    def read(self, timeout: Optional[float] = 1.0) -> Optional[Frame]:
        """Pull the next frame for this consumer, or None if none arrived."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def read_latest(self, timeout: Optional[float] = 1.0) -> Optional[Frame]:
        """Pull the freshest queued frame, discarding any backlog."""
        frame = self.read(timeout=timeout)
        if frame is None:
            return None
        while True:
            try:
                frame = self._q.get_nowait()
            except queue.Empty:
                return frame

    def __len__(self) -> int:
        return self._q.qsize()


class VideoSource:
    """Opens a camera and serves its frames to any number of consumers.

    Reconnects on read failure with a capped backoff, so a camera that is
    unplugged and replugged recovers without restarting the process.
    """

    def __init__(
        self,
        src: int | str = 0,
        width: Optional[int] = 1280,
        height: Optional[int] = 720,
        fps: Optional[int] = 30,
        fourcc: Optional[str] = "MJPG",
        backend: Optional[int] = None,
        reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 10.0,
        max_read_failures: int = 30,
    ):
        self.src = src
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        # DirectShow opens far faster than MSMF on Windows and is less prone
        # to hanging on the first read.
        if backend is None:
            backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        self.backend = backend
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self.max_read_failures = max_read_failures

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._subs: list[Subscription] = []
        self._subs_lock = threading.Lock()
        self._frame_index = 0
        self._started = threading.Event()
        self.reconnects = 0

    # -- lifecycle ------------------------------------------------------

    def start(self, wait: float = 5.0) -> "VideoSource":
        if self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._capture_loop, name="VideoSource", daemon=True
        )
        self._thread.start()
        if wait:
            # Surfaces a dead camera at start() instead of at first read().
            if not self._started.wait(timeout=wait):
                log.warning("camera %s produced no frame within %.1fs", self.src, wait)
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._release()

    def __enter__(self) -> "VideoSource":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- consumers ------------------------------------------------------

    def subscribe(self, name: str, maxsize: int = 2) -> Subscription:
        """Register an independent consumer.

        Each subscriber gets its own queue, so consumers never contend with
        one another. Safe to call before or after start().
        """
        sub = Subscription(name, maxsize=maxsize)
        with self._subs_lock:
            self._subs.append(sub)
        log.debug("subscriber %r registered", name)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        with self._subs_lock:
            if sub in self._subs:
                self._subs.remove(sub)

    # -- internals ------------------------------------------------------

    def _open(self) -> bool:
        self._release()
        cap = cv2.VideoCapture(self.src, self.backend)
        if not cap.isOpened():
            cap.release()
            return False
        if self.fourcc:
            # Without this the camera negotiates uncompressed YUY2, which USB
            # bandwidth caps at ~7 FPS at 720p. MJPG gets us ~25 FPS.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps:
            cap.set(cv2.CAP_PROP_FPS, self.fps)
        # Keep the driver buffer shallow so we read live frames, not a backlog.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        return True

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _publish(self, frame: Frame) -> None:
        with self._subs_lock:
            subs = list(self._subs)
        for sub in subs:
            sub._offer(frame)

    def _capture_loop(self) -> None:
        delay = self.reconnect_delay
        failures = 0

        while not self._stop.is_set():
            if self._cap is None:
                if not self._open():
                    log.warning(
                        "cannot open camera %s, retrying in %.1fs", self.src, delay
                    )
                    if self._stop.wait(delay):
                        break
                    delay = min(delay * 2, self.max_reconnect_delay)
                    continue
                log.info("camera %s opened", self.src)
                delay = self.reconnect_delay
                failures = 0

            ok, image = self._cap.read()
            if not ok or image is None:
                failures += 1
                if failures >= self.max_read_failures:
                    log.warning(
                        "camera %s failed %d reads, reconnecting", self.src, failures
                    )
                    self.reconnects += 1
                    self._release()
                    failures = 0
                    if self._stop.wait(delay):
                        break
                    delay = min(delay * 2, self.max_reconnect_delay)
                else:
                    time.sleep(0.005)
                continue

            failures = 0
            self._frame_index += 1
            self._publish(
                Frame(
                    index=self._frame_index,
                    timestamp=time.time(),
                    image=image,
                )
            )
            self._started.set()

        self._release()
        log.info("capture loop stopped after %d frames", self._frame_index)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    DURATION = 5.0
    print(f"Opening default webcam, reading for {DURATION:.0f}s ...")

    with VideoSource(src=0) as source:
        consumer = source.subscribe("smoke-test")
        frames = 0
        first_shape = None
        start = time.perf_counter()

        while time.perf_counter() - start < DURATION:
            frame = consumer.read(timeout=1.0)
            if frame is None:
                continue
            if first_shape is None:
                first_shape = frame.shape
                print(f"first frame shape: {first_shape}")
            frames += 1

        elapsed = time.perf_counter() - start

    if frames == 0:
        print("NO FRAMES RECEIVED -- check that a camera is connected and free.")
    else:
        print(f"frame shape   : {first_shape}")
        print(f"frames read   : {frames}")
        print(f"elapsed       : {elapsed:.2f}s")
        print(f"capture FPS   : {frames / elapsed:.1f}")
        print(f"dropped       : {consumer.dropped}")
        print(f"reconnects    : {source.reconnects}")
