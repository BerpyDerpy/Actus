"""The monitoring panels. Each one draws from the session; none holds protocol logic."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

TEXT = "#e6e6e6"
MUTED = "#8a93a0"
PANEL = "#1a1f26"
BG = "#111418"
GREEN = "#3fb950"
BLUE = "#58a6ff"
RED = "#f85149"
AMBER = "#d29922"

STYLESHEET = f"""
QWidget {{ background: {BG}; color: {TEXT}; font-family: "Segoe UI", sans-serif; font-size: 10pt; }}
QFrame#panel {{ background: {PANEL}; border-radius: 6px; }}
QLabel#caption {{ color: {MUTED}; font-size: 8pt; font-weight: 600; letter-spacing: 1px; background: transparent; }}
QLabel {{ background: transparent; }}
QPushButton {{ background: #262d36; border: 1px solid #333c47; border-radius: 4px; padding: 5px 12px; }}
QPushButton:hover {{ background: #2f3842; }}
QPushButton:checked {{ background: #3a2a2a; border-color: {RED}; }}
QPushButton#layer {{ color: {MUTED}; }}
QPushButton#layer:checked {{ background: #1d2b3a; border-color: {BLUE}; color: {TEXT}; }}
QTableWidget {{ background: {PANEL}; border: none; gridline-color: #262d36; }}
QHeaderView::section {{ background: {PANEL}; color: {MUTED}; border: none; padding: 4px; font-size: 8pt; }}
QStatusBar {{ color: {MUTED}; }}
QScrollArea {{ background: transparent; }}
QWidget#alertsInner {{ background: {PANEL}; }}
QFrame#alertCard {{ background: #221a1c; border-left: 3px solid {RED}; border-radius: 4px; }}
QFrame#alertCard:hover {{ background: #2a1f22; }}
QDialog {{ background: {BG}; }}
QScrollBar:vertical {{ background: {PANEL}; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #333c47; border-radius: 4px; min-height: 24px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
"""

QUESTION_KINDS = frozenset({"ask", "confirmed", "denied", "unanswered"})

# status -> (glyph, colour)
STEP_LOOK = {
    "pending": ("○", MUTED),
    "active": ("▶", BLUE),
    "done": ("✓", GREEN),
    "missed": ("✗", RED),
    "too_fast": ("!", AMBER),
    "asking": ("?", AMBER),
}


def panel(caption: Optional[str] = None) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("panel")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(12, 10, 12, 10)
    if caption:
        label = QLabel(caption)
        label.setObjectName("caption")
        layout.addWidget(label)
    return frame, layout


class VideoView(QLabel):
    """The video, scaled to fit, with an alert banner over its lower edge."""

    BANNER_MS = 5000

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(480, 270)
        self.setStyleSheet(f"background: black; color: {MUTED}; font-size: 12pt;")
        self._image: Optional[QImage] = None
        self.show_placeholder()

        self.banner = QLabel(self)
        self.banner.setWordWrap(True)
        self.banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.banner.setStyleSheet(
            f"background: rgba(200, 40, 40, 225); color: white; font-size: 14pt;"
            f"font-weight: 600; padding: 10px 16px; border-radius: 6px;"
        )
        self.banner.hide()
        self._banner_timer = QTimer(self, singleShot=True, timeout=self.banner.hide)

        # A question stays up until it is answered, so it gets its own banner.
        self.question = QLabel(self)
        self.question.setWordWrap(True)
        self.question.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.question.setStyleSheet(
            "background: rgba(210, 153, 34, 235); color: black; font-size: 14pt;"
            "font-weight: 600; padding: 10px 16px; border-radius: 6px;"
        )
        self.question.hide()

    def show_placeholder(self, text: str = "Open a take (Ctrl+O)") -> None:
        self._image = None
        self.setPixmap(QPixmap())
        self.setText(text)

    def current_image(self) -> Optional[QImage]:
        return self._image

    def set_image(self, image: QImage) -> None:
        self._image = image
        self._rescale()

    def show_alert(self, text: str) -> None:
        self.banner.setText(text)
        self._place_banner()
        self.banner.show()
        self.banner.raise_()
        self._banner_timer.start(self.BANNER_MS)

    def clear_alert(self) -> None:
        self._banner_timer.stop()
        self.banner.hide()

    def show_question(self, text: str) -> None:
        self.question.setText(text)
        self._place_banner()
        self.question.show()
        self.question.raise_()

    def clear_question(self) -> None:
        self.question.hide()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rescale()
        self._place_banner()

    def _rescale(self) -> None:
        if self._image is None:
            return
        self.setPixmap(QPixmap.fromImage(self._image).scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def _place_banner(self) -> None:
        width = int(self.width() * 0.8)
        self.banner.setFixedWidth(width)
        self.banner.adjustSize()
        self.banner.move((self.width() - width) // 2, self.height() - self.banner.height() - 24)
        self.question.setFixedWidth(width)
        self.question.adjustSize()
        self.question.move((self.width() - width) // 2, 24)


class NextStepCard(QFrame):
    """The next instruction, or, while the system is unsure, its yes/no question."""

    answered = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.setObjectName("panel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 14)
        self.caption = QLabel("NEXT STEP")
        self.caption.setObjectName("caption")
        self.text = QLabel("-")
        self.text.setWordWrap(True)
        self.text.setStyleSheet("font-size: 17pt; font-weight: 600;")
        layout.addWidget(self.caption)
        layout.addWidget(self.text)

        self.buttons = QWidget()
        row = QHBoxLayout(self.buttons)
        row.setContentsMargins(0, 6, 0, 0)
        for label, yes in (("Yes  (Y)", True), ("No  (N)", False)):
            button = QPushButton(label)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.clicked.connect(lambda _, v=yes: self.answered.emit(v))
            row.addWidget(button)
        layout.addWidget(self.buttons)
        self.buttons.hide()
        self.asking = False
        self._prompt = ("", False)

    def set_question(self, question: str) -> None:
        self.asking = True
        self.caption.setText("CONFIRM")
        self.caption.setStyleSheet(f"color: {AMBER};")
        self.text.setText(question)
        self.text.setStyleSheet(f"font-size: 17pt; font-weight: 600; color: {AMBER};")
        self.buttons.show()

    def clear_question(self) -> None:
        self.asking = False
        self.buttons.hide()
        self.caption.setStyleSheet("")
        self.set_prompt(*self._prompt)

    def set_prompt(self, text: str, complete: bool = False) -> None:
        self._prompt = (text, complete)
        if self.asking:
            return  # the question stays up until answered
        self.caption.setText("COMPLETE" if complete else "NEXT STEP")
        self.text.setText(text or "-")
        self.text.setStyleSheet(
            f"font-size: 17pt; font-weight: 600; color: {GREEN if complete else TEXT};"
        )


class StepList(QFrame):
    """One row per task step: glyph, number and name, duration."""

    def __init__(self, steps):
        super().__init__()
        self.setObjectName("panel")
        grid = QGridLayout(self)
        grid.setContentsMargins(14, 10, 14, 12)
        grid.setVerticalSpacing(8)
        caption = QLabel("PROCEDURE")
        caption.setObjectName("caption")
        grid.addWidget(caption, 0, 0, 1, 3)
        self.rows = {}
        for n, step in enumerate(steps, start=1):
            glyph, name, dur = QLabel(), QLabel(f"{n}  {step.name.capitalize()}"), QLabel()
            glyph.setFixedWidth(18)
            dur.setAlignment(Qt.AlignmentFlag.AlignRight)
            grid.addWidget(glyph, n, 0)
            grid.addWidget(name, n, 1)
            grid.addWidget(dur, n, 2)
            self.rows[step.id] = (glyph, name, dur)
        grid.setColumnStretch(1, 1)

    def update_from(self, snapshot: dict, active_elapsed: Optional[float],
                    asking: Optional[str] = None) -> None:
        for step in snapshot["steps"]:
            glyph, name, dur = self.rows[step["id"]]
            look = "too_fast" if step["too_fast"] else step["status"]
            if step["id"] == asking:
                look = "asking"
            char, colour = STEP_LOOK[look]
            active = step["status"] == "active"
            glyph.setText(char)
            glyph.setStyleSheet(f"color: {colour}; font-weight: 700;")
            name.setStyleSheet(
                f"color: {colour if look != 'pending' else TEXT};"
                f"font-weight: {'700' if active else '400'};"
            )
            seconds = active_elapsed if active else step["duration"]
            dur.setText("" if seconds is None else f"{seconds:.1f}s")
            dur.setStyleSheet(f"color: {MUTED};")


class AlertCard(QFrame):
    """One alert: the frame it fired on, when, and what was said. Click to enlarge."""

    THUMB = (128, 72)
    clicked = pyqtSignal()

    def __init__(self, t: float, message: str, image: Optional[QImage]):
        super().__init__()
        self.t, self.message, self.image = t, message, image
        self.setObjectName("alertCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor if image else Qt.CursorShape.ArrowCursor)
        row = QHBoxLayout(self)
        row.setContentsMargins(8, 8, 8, 8)
        row.setSpacing(10)

        thumb = QLabel()
        thumb.setFixedSize(*self.THUMB)
        thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb.setStyleSheet("background: black; border-radius: 3px;")
        if image is not None:
            thumb.setPixmap(QPixmap.fromImage(image).scaled(
                *self.THUMB, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))
        text = QVBoxLayout()
        text.setSpacing(2)
        when = QLabel(f"{t:.1f}s")
        when.setStyleSheet(f"color: {MUTED}; font-size: 8pt;")
        what = QLabel(message)
        what.setWordWrap(True)
        what.setStyleSheet(f"color: {RED};")
        text.addWidget(when)
        text.addWidget(what)
        text.addStretch()
        row.addWidget(thumb, 0, Qt.AlignmentFlag.AlignTop)
        row.addLayout(text, 1)

    def mousePressEvent(self, event) -> None:
        self.clicked.emit()


class AlertList(QScrollArea):
    """Alert history, newest at the bottom. Emits `opened(card)` when a card is clicked."""

    opened = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        inner.setObjectName("alertsInner")
        self._layout = QVBoxLayout(inner)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)
        self._layout.addStretch()
        self.setWidget(inner)
        self.cards: list[AlertCard] = []

    def add(self, t: float, text: str, image: Optional[QImage] = None) -> AlertCard:
        card = AlertCard(t, text, image)
        card.clicked.connect(lambda: self.opened.emit(card))
        self._layout.insertWidget(self._layout.count() - 1, card)
        self.cards.append(card)
        bar = self.verticalScrollBar()
        QTimer.singleShot(0, lambda: bar.setValue(bar.maximum()))
        return card

    def count(self) -> int:
        return len(self.cards)

    def clear(self) -> None:
        for card in self.cards:
            card.deleteLater()
        self.cards.clear()


class SnapshotDialog(QDialog):
    def __init__(self, image: QImage, caption: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Alert snapshot")
        layout = QVBoxLayout(self)
        picture = QLabel()
        picture.setPixmap(QPixmap.fromImage(image).scaled(
            960, 540, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))
        text = QLabel(caption)
        text.setWordWrap(True)
        text.setStyleSheet(f"color: {RED}; font-size: 12pt; font-weight: 600;")
        layout.addWidget(picture)
        layout.addWidget(text)


class EventTable(QTableWidget):
    COLUMNS = ("t", "event", "step", "message")

    def __init__(self):
        super().__init__(0, len(self.COLUMNS))
        self.setHorizontalHeaderLabels(self.COLUMNS)
        self.verticalHeader().hide()
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.setShowGrid(False)
        header = self.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft)
        for col in range(3):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.verticalHeader().setDefaultSectionSize(22)

    def add(self, event) -> None:
        row = self.rowCount()
        self.insertRow(row)
        values = (f"{event.t:.2f}", event.kind, event.step or "", event.message or "")
        colour = QColor(RED if event.is_alert else GREEN if event.kind == "complete"
                        else AMBER if event.kind in QUESTION_KINDS else TEXT)
        for col, value in enumerate(values):
            item = QTableWidgetItem(value)
            loud = event.is_alert or event.kind in QUESTION_KINDS
            item.setForeground(colour if loud or col == 1 else QColor(TEXT))
            self.setItem(row, col, item)
        self.scrollToBottom()


class StatusPill(QLabel):
    COLOURS = {
        "READY": MUTED, "RUNNING": BLUE, "LIVE": BLUE, "PAUSED": AMBER,
        "COMPLETE": GREEN, "INCOMPLETE": RED, "STOPPED": MUTED,
    }

    def set_state(self, state: str) -> None:
        colour = self.COLOURS.get(state, MUTED)
        self.setText(f"●  {state}")
        self.setStyleSheet(
            f"color: {colour}; border: 1px solid {colour}; border-radius: 10px;"
            f"padding: 2px 10px; font-weight: 600; font-size: 9pt;"
        )


class TimelineBar(QWidget):
    """The run so far, left to right: one segment per step, ticks for alerts
    (red) and questions (amber), and a playhead. Hover for details.

    A file's full length is known up front, so the bar fills as it plays; a
    live camera has no end, so the bar rescales as time runs on.
    """

    HEIGHT = 30

    def __init__(self, steps):
        super().__init__()
        self.setFixedHeight(self.HEIGHT)
        self.setMouseTracking(True)
        self.numbers = {s.id: n for n, s in enumerate(steps, start=1)}
        self.names = {s.id: s.name for s in steps}
        self.reset()

    def reset(self) -> None:
        self.duration: Optional[float] = None
        self.t = 0.0
        self.segments: list[list] = []  # [step id, start, end or None, status]
        self.ticks: list[tuple[float, str, str]] = []  # (t, colour, text)
        self.complete_at: Optional[float] = None
        self.update()

    def set_duration(self, duration: Optional[float]) -> None:
        self.duration = duration
        self.update()

    def set_time(self, t: float) -> None:
        self.t = t
        self.update()

    def add_events(self, events) -> None:
        for e in events:
            if e.kind == "started":
                self.segments.append([e.step, e.t, None, "active"])
            elif e.kind in ("done", "too_fast"):
                for seg in reversed(self.segments):
                    if seg[0] == e.step and seg[2] is None:
                        seg[2], seg[3] = e.t, "too_fast" if e.kind == "too_fast" else "done"
                        break
            if e.is_alert:
                self.ticks.append((e.t, RED, e.message))
            elif e.kind == "ask":
                self.ticks.append((e.t, AMBER, e.message))
            elif e.kind == "complete":
                self.complete_at = e.t
        self.update()

    def _span(self) -> float:
        return self.duration or max(60.0, self.t * 1.25)

    def _x(self, t: float) -> float:
        return 1 + (self.width() - 2) * min(t, self._span()) / self._span()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        h = self.height()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(PANEL))
        p.drawRoundedRect(QRectF(0, 0, self.width(), h), 4, 4)

        font = QFont(self.font())
        font.setPointSizeF(8)
        font.setBold(True)
        p.setFont(font)
        colours = {"active": BLUE, "done": GREEN, "too_fast": AMBER}
        for n, (step, start, end, status) in enumerate(self.segments):
            x1, x2 = self._x(start), self._x(end if end is not None else self.t)
            rect = QRectF(x1, 6, max(2.0, x2 - x1 - 1), h - 12)
            colour = QColor(colours[status])
            if status == "done" and n % 2:
                colour = colour.darker(125)  # neighbours stay distinguishable
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(colour)
            p.drawRoundedRect(rect, 3, 3)
            if rect.width() > 14:
                p.setPen(QColor(BG))
                p.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(self.numbers[step]))

        for t, colour, _ in self.ticks:
            x = self._x(t)
            p.setPen(QPen(QColor(colour), 2))
            p.drawLine(int(x), 2, int(x), h - 2)
        if self.complete_at is not None:
            x = self._x(self.complete_at)
            p.setPen(QPen(QColor(GREEN), 3))
            p.drawLine(int(x), 2, int(x), h - 2)

        x = self._x(self.t)
        p.setPen(QPen(QColor(TEXT), 1))
        p.drawLine(int(x), 0, int(x), h)
        p.end()

    def mouseMoveEvent(self, event) -> None:
        x = event.position().x()
        text = ""
        for t, _, message in self.ticks:
            if abs(self._x(t) - x) <= 4:
                text = f"{t:.1f}s  {message}"
                break
        else:
            for step, start, end, status in self.segments:
                stop = end if end is not None else self.t
                if self._x(start) <= x <= self._x(stop):
                    text = (f"{self.numbers[step]}  {self.names[step]}  "
                            f"{start:.1f}-{stop:.1f}s  ({status.replace('_', ' ')})")
                    break
        if text:
            QToolTip.showText(event.globalPosition().toPoint(), text, self)
        else:
            QToolTip.hideText()
