"""Actus monitoring GUI.

    python tools/gui.py                                   open a take from the window
    python tools/gui.py --video takes/take1.mp4           picks up take1_steps.csv
    python tools/gui.py --video takes/take2.mp4           no script yet: key it in
    python tools/gui.py --camera 0                        live webcam + hotkeys

Keys: 0-5 step hotkeys (when the take has no script), Space pause,
Ctrl+R restart, Ctrl+O open, M mute, O / H / K toggle the object, hand and
marker overlays. Every run writes logs/runs/<timestamp>.jsonl.

The overlay models (YOLO11m, MediaPipe hands, ArUco) load in the background
for a few seconds after launch; the video plays without overlays until they
are ready. --no-overlays skips loading them at all.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("fsm", "io", "gui", "perception"):
    sys.path.insert(0, str(ROOT / "src" / sub))

from PyQt6.QtWidgets import QApplication  # noqa: E402

from main_window import MainWindow  # noqa: E402
from perception import Perception  # noqa: E402
from step_graph import DEFAULT_PATH, load_step_graph  # noqa: E402
from voice import Voice  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group()
    source.add_argument("--video", type=Path, help="take to open on launch")
    source.add_argument("--camera", type=int, help="camera index, for live hotkey mode")
    ap.add_argument("--script", type=Path, help="steps CSV (default: <video>_steps.csv if present)")
    ap.add_argument("--graph", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--mute", action="store_true")
    ap.add_argument("--no-overlays", action="store_true", help="skip YOLO / hands / markers")
    args = ap.parse_args()

    if args.video is not None and not args.video.exists():
        ap.error(f"no such video: {args.video}")

    app = QApplication(sys.argv)
    voice = Voice().start()
    perception = None
    if not args.no_overlays:
        perception = Perception()
        threading.Thread(target=perception.load, name="perception-load", daemon=True).start()
    window = MainWindow(load_step_graph(args.graph), ROOT / "logs" / "runs", voice,
                        perception=perception)
    if args.mute:
        window.mute_button.setChecked(True)
    window.show()

    if args.video is not None:
        window.open_take(args.video.resolve(), args.script)
    elif args.camera is not None:
        window.open_camera(args.camera)

    code = app.exec()
    voice.stop(wait=0)
    if perception is not None:
        perception.close()
    sys.exit(code)


if __name__ == "__main__":
    main()
