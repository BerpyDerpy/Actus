"""The monitoring window: video, next step, procedure checklist, alerts, event log.

The window owns one Session at a time. Opening a take or pressing Restart
throws the old one away (its log is closed) and builds a new one, so every
run gets its own log file.

A take is a video plus, optionally, its step script. Opening `take.mp4`
picks up `take_steps.csv` beside it automatically. With no script, the
number keys drive the steps, and when the video reaches its end the presses
are saved as `take_steps.csv`, so pressing Restart replays them.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from event_log import EventLog
from session import Session
from sources import ScriptSource
from step_graph import StepGraph
from video_worker import VideoWorker
from voice import Voice
from widgets import (
    STYLESHEET,
    AlertList,
    EventTable,
    NextStepCard,
    StatusPill,
    SnapshotDialog,
    StepList,
    TimelineBar,
    VideoView,
    panel,
)


# overlay layer -> (button label, shortcut key)
OVERLAY_LAYERS = {"objects": ("Objects", "O"), "hands": ("Hands", "H"), "markers": ("Markers", "K")}


def script_for(video: Path) -> Path:
    return video.with_name(video.stem + "_steps.csv")


class MainWindow(QMainWindow):
    def __init__(self, graph: StepGraph, log_dir: Path, voice: Voice, paced: bool = True,
                 perception=None):
        super().__init__()
        self.graph = graph
        self.log_dir = log_dir
        self.voice = voice
        self.paced = paced
        self.perception = perception
        # Read by the video thread every frame; replaced whole, never mutated.
        self._layers: frozenset = frozenset(OVERLAY_LAYERS) if perception else frozenset()
        self._flipped = False  # also read by the video thread

        self.source: Optional[Path | int] = None  # what Restart reopens
        self.script_override: Optional[Path] = None
        self.session: Optional[Session] = None
        self.worker: Optional[VideoWorker] = None
        self.t = 0.0
        self._frame_index = 0
        self.state = "READY"

        self.setWindowTitle("Actus")
        self.setStyleSheet(STYLESHEET)
        self.resize(1440, 860)
        self._build()
        self._bind_keys()
        self._reset_panels()
        if perception is not None:
            self._watch_perception_load()

    # --- layout ------------------------------------------------------------

    def _build(self) -> None:
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(12, 10, 12, 8)
        outer.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("ACTUS")
        title.setStyleSheet("font-size: 15pt; font-weight: 700; letter-spacing: 3px;")
        self.status = StatusPill()
        self.clock = QLabel()
        self.clock.setStyleSheet("font-family: Consolas, monospace; font-size: 11pt;")
        self.mute_button = QPushButton("Mute")
        self.mute_button.setCheckable(True)
        self.mute_button.toggled.connect(self._set_muted)
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.status)
        header.addSpacing(12)
        header.addWidget(self.clock)
        header.addSpacing(12)
        header.addWidget(self.mute_button)
        outer.addLayout(header)

        self.video = VideoView()
        self.next_card = NextStepCard()
        self.next_card.answered.connect(self.answer)
        self.steps = StepList(self.graph.task_steps)
        alerts_panel, alerts_layout = panel("ALERTS")
        self.alerts = AlertList()
        self.alerts.opened.connect(self._show_snapshot)
        alerts_layout.addWidget(self.alerts)

        side = QVBoxLayout()
        side.setSpacing(10)
        side.addWidget(self.next_card)
        side.addWidget(self.steps)
        side.addWidget(alerts_panel, 1)
        side_widget = QWidget()
        side_widget.setLayout(side)
        side_widget.setFixedWidth(380)

        self.timeline = TimelineBar(self.graph.task_steps)
        video_column = QVBoxLayout()
        video_column.setSpacing(6)
        video_column.addWidget(self.video, 1)
        video_column.addWidget(self.timeline)

        top = QHBoxLayout()
        top.setSpacing(10)
        top.addLayout(video_column, 1)
        top.addWidget(side_widget)
        top_widget = QWidget()
        top_widget.setLayout(top)

        controls = QHBoxLayout()
        self.play_button = QPushButton("Pause")
        self.restart_button = QPushButton("Restart")
        self.play_button.clicked.connect(self.toggle_pause)
        self.restart_button.clicked.connect(self.restart)
        for button in (self.play_button, self.restart_button, self.mute_button):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        for button in (self.play_button, self.restart_button):
            controls.addWidget(button)
        self.hint = QLabel()
        self.hint.setStyleSheet("color: #8a93a0;")
        controls.addSpacing(16)
        controls.addWidget(self.hint)
        controls.addStretch()

        self.flip_button = QPushButton("Upside down")
        self.flip_button.setObjectName("layer")
        self.flip_button.setCheckable(True)
        self.flip_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.flip_button.toggled.connect(self._set_flipped)
        controls.addWidget(self.flip_button)
        controls.addSpacing(8)

        self.perception_label = QLabel()
        self.perception_label.setStyleSheet("color: #8a93a0; font-family: Consolas, monospace;")
        controls.addWidget(self.perception_label)
        controls.addSpacing(8)
        self.layer_buttons = {}
        for layer, (label, _key) in OVERLAY_LAYERS.items():
            button = QPushButton(label)
            button.setObjectName("layer")
            button.setCheckable(True)
            button.setChecked(layer in self._layers)
            button.setEnabled(False)  # until the models have loaded
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setVisible(self.perception is not None)
            button.toggled.connect(lambda on, name=layer: self._set_layer(name, on))
            controls.addWidget(button)
            self.layer_buttons[layer] = button

        log_panel, log_layout = panel("EVENT LOG")
        self.events = EventTable()
        log_layout.addWidget(self.events)
        bottom_widget = QWidget()
        bottom = QVBoxLayout(bottom_widget)
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.addLayout(controls)
        bottom.addWidget(log_panel)

        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(top_widget)
        split.addWidget(bottom_widget)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        split.setSizes([600, 200])
        outer.addWidget(split, 1)

        self.setCentralWidget(root)

    def _bind_keys(self) -> None:
        # Application-wide shortcuts, so a focused table or button never eats a step key.
        def shortcut(seq, slot):
            s = QShortcut(QKeySequence(seq), self)
            s.setContext(Qt.ShortcutContext.ApplicationShortcut)
            s.activated.connect(slot)

        for step in self.graph.steps:
            if step.hotkey is not None:
                shortcut(str(step.hotkey), lambda k=step.hotkey: self.step_key(k))
        shortcut("Space", self.toggle_pause)
        shortcut("Ctrl+O", self.open_dialog)
        shortcut("Ctrl+R", self.restart)
        shortcut("M", self.mute_button.toggle)
        shortcut("F", self.flip_button.toggle)
        shortcut("Y", lambda: self.answer(True))
        shortcut("N", lambda: self.answer(False))
        for layer, (_label, key) in OVERLAY_LAYERS.items():
            shortcut(key, self.layer_buttons[layer].toggle)

    # --- opening and running -----------------------------------------------

    def open_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open take", str(Path.cwd()), "Video (*.mp4 *.mov *.avi *.mkv)"
        )
        if path:
            self.open_take(Path(path))

    def open_take(self, video: Path, script: Optional[Path] = None) -> None:
        self.source = video
        self.script_override = script
        self.restart()

    def open_camera(self, index: int) -> None:
        self.source = index
        self.script_override = None
        self.restart()

    def restart(self) -> None:
        if self.source is None:
            return
        self._stop_run()
        self.voice.clear()
        self._reset_panels()

        live = isinstance(self.source, int)
        script_path = None
        if not live:
            script_path = self.script_override or script_for(self.source)
            if not script_path.exists():
                script_path = None
        script = ScriptSource(script_path, self.graph) if script_path else None

        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        log = EventLog(self.log_dir / f"{stamp}.jsonl", meta={
            "video": str(self.source),
            "source": f"script:{script_path}" if script else "hotkeys",
            "frontend": "gui",
        })
        self.session = Session(self.graph, log, self.voice, script)
        self.session.listeners.append(self._on_events)

        keys = "0 idle · 1–5 steps · " if not script else ""
        overlays = " · O H K overlays" if self.perception is not None else ""
        self.hint.setText(f"{keys}space pause · Ctrl+R restart · M mute · F flip{overlays}")

        self.worker = VideoWorker(self.source, paced=self.paced, perception=self.perception,
                                  layers=lambda: self._layers, flipped=lambda: self._flipped)
        self.worker.frame.connect(self._on_frame)
        self.worker.opened.connect(self.timeline.set_duration)
        self.worker.stats.connect(self._on_stats)
        self.worker.ended.connect(self._on_ended)
        self.worker.failed.connect(self._on_failed)
        self._set_state("LIVE" if live else "RUNNING")
        self.session.start()
        self.worker.start()

    def toggle_pause(self) -> None:
        if self.worker is None or self.worker.live or self.state not in ("RUNNING", "PAUSED"):
            return
        paused = not self.worker.paused
        self.worker.set_paused(paused)
        self._set_state("PAUSED" if paused else "RUNNING")

    def answer(self, yes: bool) -> None:
        """Y / N, or the card's buttons: the operator answers the pending question."""
        if self.session and self.state in ("RUNNING", "LIVE", "PAUSED"):
            self.session.answer(yes, self.t)

    def step_key(self, hotkey: int) -> None:
        if self.session and self.state in ("RUNNING", "LIVE", "PAUSED"):
            self.session.key(ord("0") + hotkey, self._frame_index, self.t)

    # --- worker signals ----------------------------------------------------

    def _on_frame(self, image, t: float, index: int) -> None:
        if self.sender() is not self.worker:
            return
        self.t, self._frame_index = t, index
        self.video.set_image(image)
        self.session.advance(t)
        self.timeline.set_time(t)
        self._refresh_live()
        self.worker.ack()

    def _on_ended(self, t: float, reached_end: bool) -> None:
        if self.sender() is not self.worker:
            return
        if reached_end:
            self.session.end_of_video(t)
            saved = None
            if isinstance(self.source, Path):
                saved = self.session.save_hotkeys(script_for(self.source))
            self._set_state("COMPLETE" if self.session.seq.complete else "INCOMPLETE")
            self._finish_session()
            self.statusBar().showMessage(f"log: {self.session.log.path}")
            if saved:
                print(f"saved step presses to {saved}")
        self.play_button.setEnabled(False)

    def _on_stats(self, stats) -> None:
        if self.sender() is not self.worker:
            return
        parts = []
        if "objects" in self._layers:
            parts.append(f"{stats.objects} obj")
        if "hands" in self._layers:
            parts.append(f"{stats.hands} hand{'s' if stats.hands != 1 else ''}")
        if "markers" in self._layers:
            parts.append(f"{stats.markers} mk")
        if stats.ms:
            parts.append(f"{stats.ms['frame']:.0f} ms")
        self.perception_label.setText("  ·  ".join(parts))

    def _show_snapshot(self, card) -> None:
        if card.image is not None:
            SnapshotDialog(card.image, f"{card.t:.1f}s   {card.message}", self).exec()

    def _on_failed(self, message: str) -> None:
        self.video.show_placeholder(message)
        self._set_state("STOPPED")

    def _on_events(self, events) -> None:
        snapshot = self.session.seq.snapshot()
        for e in events:
            self.events.add(e)
            if e.is_alert:
                image = self.video.current_image()
                self._save_snapshot(e, image)
                self.alerts.add(e.t, e.message, image)
                self.video.show_alert(e.message)
            elif e.kind == "ask":
                self.video.show_question(f"{e.message}    Y / N")
                self.next_card.set_question(e.detail["question"])
            elif e.kind in ("confirmed", "denied", "unanswered"):
                self.video.clear_question()
                self.next_card.clear_question()
        self.timeline.add_events(events)
        self.next_card.set_prompt(self.session.next_prompt, complete=snapshot["complete"])
        self.steps.update_from(snapshot, self._active_elapsed(), self._asking())

    # --- helpers -----------------------------------------------------------

    def _watch_perception_load(self) -> None:
        """Poll the background model load; enable the layer toggles once it is warm."""
        self.statusBar().showMessage("Loading perception models…")
        timer = QTimer(self)

        def check():
            if self.perception.ready:
                timer.stop()
                for button in self.layer_buttons.values():
                    button.setEnabled(True)
                self.statusBar().showMessage(f"Perception ready: {self.perception.device_label}", 8000)
            elif self.perception.error:
                timer.stop()
                self._layers = frozenset()
                self.statusBar().showMessage(f"Overlays off: {self.perception.error}")

        timer.timeout.connect(check)
        timer.start(250)

    def _set_layer(self, layer: str, on: bool) -> None:
        self._layers = self._layers | {layer} if on else self._layers - {layer}
        if not self._layers:
            self.perception_label.setText("")

    def _save_snapshot(self, event, image) -> None:
        """Keep the frame an alert fired on, next to the run log, and log where it went."""
        if image is None:
            return
        log = self.session.log
        folder = log.path.with_name(log.path.stem + "_alerts")
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{event.t:07.2f}s_{event.kind}_{event.step}.jpg"
        image.save(str(path), "JPG", 90)
        log.note({"t": round(event.t, 3), "kind": "snapshot", "alert": event.kind,
                  "step": event.step, "path": f"{folder.name}/{path.name}"})

    def _refresh_live(self) -> None:
        self.clock.setText(f"{self.t:.1f}s")
        if self.session.seq.active:
            self.steps.update_from(self.session.seq.snapshot(), self._active_elapsed(),
                                   self._asking())

    def _asking(self) -> Optional[str]:
        return self.session.pending[0] if self.session and self.session.pending else None

    def _set_flipped(self, flipped: bool) -> None:
        self._flipped = flipped

    def _active_elapsed(self) -> Optional[float]:
        since = self.session.seq.active_since if self.session else None
        return None if since is None else max(0.0, self.t - since)

    def _set_state(self, state: str) -> None:
        self.state = state
        self.status.set_state(state)
        self.play_button.setText("Resume" if state == "PAUSED" else "Pause")
        self.play_button.setEnabled(state in ("RUNNING", "PAUSED"))

    def _set_muted(self, muted: bool) -> None:
        self.voice.enabled = not muted
        self.mute_button.setText("Muted" if muted else "Mute")
        if muted:
            self.voice.clear()

    def _reset_panels(self) -> None:
        self.t = 0.0
        self.clock.setText("0.0s")
        self.events.setRowCount(0)
        self.alerts.clear()
        self.video.clear_alert()
        self.video.clear_question()
        self.next_card.clear_question()
        self.timeline.reset()
        self.next_card.set_prompt("")
        self.steps.update_from(
            {"steps": [{"id": s.id, "status": "pending", "too_fast": False, "duration": None}
                       for s in self.graph.task_steps]},
            None,
        )
        self.statusBar().clearMessage()
        if self.source is None:
            self.video.show_placeholder()
            self.hint.setText("Ctrl+O open a take")
            self._set_state("READY")

    def _stop_run(self) -> None:
        if self.worker is not None:
            worker, self.worker = self.worker, None
            worker.stop()
        self._finish_session()

    def _finish_session(self) -> None:
        if self.session is not None:
            self.session.close()

    def closeEvent(self, event) -> None:
        self._stop_run()
        self.voice.clear()
        super().closeEvent(event)
