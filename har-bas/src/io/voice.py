"""Spoken prompts and alerts, off the caller's thread.

pyttsx3 blocks for as long as the sentence takes to say (several seconds for
an alert), so speech runs on its own worker thread fed by a queue; the video
loop never waits on it.

A fresh engine is created per utterance on purpose. Reusing one engine on
Windows (SAPI5, pyttsx3 2.99) speaks the first sentence and then returns
from every later runAndWait() in ~0.1 s without saying anything. Measured:
init costs nothing noticeable next to the speech itself.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

log = logging.getLogger(__name__)


class Voice:
    def __init__(self, enabled: bool = True, rate: int = 180):
        self.enabled = enabled
        self.rate = rate
        self._q: queue.Queue[Optional[str]] = queue.Queue()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "Voice":
        if self.enabled:
            self._thread = threading.Thread(target=self._run, name="voice", daemon=True)
            self._thread.start()
        return self

    def say(self, text: str) -> None:
        if self.enabled and text:
            self._q.put(text)

    def clear(self) -> None:
        """Drop everything queued but not yet spoken (the current sentence finishes)."""
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

    def stop(self, wait: float = 10.0) -> None:
        """Let queued speech finish (up to `wait` seconds), then shut down."""
        if self._thread is None:
            return
        self._q.put(None)
        self._thread.join(timeout=wait)
        self._thread = None

    def _run(self) -> None:
        import pyttsx3

        while True:
            text = self._q.get()
            if text is None:
                return
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", self.rate)
                engine.say(text)
                engine.runAndWait()
                del engine
            except Exception:  # a dead TTS engine must never take the demo down
                log.exception("speech failed for %r", text)
