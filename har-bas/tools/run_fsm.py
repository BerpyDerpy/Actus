"""Drive the step sequencer from a video and a script or hotkeys, speaking and logging.

This is the demo pipeline minus the model and minus the GUI. The only
simulated part is the step source; the sequencer, voice alerts and run log
are the real ones.

Recorded demo, the workflow:
    1. Film a take with any camera and save the mp4.
    2. Author its script by playing it back and tapping the step keys:
           python tools/run_fsm.py --video take.mp4
       Press 1-5 as each step begins and 0 once the hands are off at the end.
       On exit the presses are saved next to it as take_steps.csv.
       Replace a script by authoring again. The old one is kept as .bak.
    3. Replay, identical every time, nobody at the keys:
           python tools/run_fsm.py --video take.mp4 --script take_steps.csv

Other modes:
    python tools/run_fsm.py --video 0                  live webcam + hotkeys
    python tools/run_fsm.py --script demo/skip_s2.csv  no video: replay the
                                                       script instantly, e.g.
                                                       to check a script

Script times are the video's own timeline (frame / fps), so a script stays
in sync with its video no matter how fast the camera was really running.

Keys (video window): 0-5 step hotkeys (when no --script), y / n answer the
system's question when a `?s2` script row has it unsure, space pause (file
only), q / Esc quit. Every run writes logs/runs/<timestamp>.jsonl.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "fsm"))
sys.path.insert(0, str(ROOT / "src" / "io"))

from event_log import EventLog  # noqa: E402
from sequencer import ACTIVE, DONE, MISSED  # noqa: E402
from session import Session  # noqa: E402
from sources import ScriptSource  # noqa: E402
from step_graph import DEFAULT_PATH, load_step_graph  # noqa: E402
from voice import Voice  # noqa: E402

WINDOW = "Actus - step sequencer"
ALERT_SHOW_SECONDS = 5.0
DISPLAY_WIDTH = 1280  # takes shot at 1080p/4K are shrunk to fit a laptop screen


def print_events(events) -> None:
    for e in events:
        flag = "!!" if e.is_alert else "  "
        print(f"{flag} {e.t:7.2f}s  {e.kind:<12} {e.step or '':<3} {e.message or ''}")


def draw_hud(image, session: Session, t: float, footer: str) -> None:
    seq = session.seq
    active = session.graph.get(seq.active).name if seq.active else "-"
    marks = {DONE: "x", ACTIVE: ">", MISSED: "!"}
    checklist = "  ".join(
        f"[{'~' if r.too_fast else marks.get(r.status, ' ')}] {s.id}"
        for s, r in ((s, seq.records[s.id]) for s in session.graph.task_steps)
    )
    lines = [
        (f"t {t:6.1f}s   now: {active}", (255, 255, 255)),
        (f"next: {session.next_prompt}", (120, 255, 120)),
        (checklist, (255, 255, 255)),
    ]
    alert = session.last_alert
    if alert and t - alert.t < ALERT_SHOW_SECONDS:
        lines.append((alert.message, (60, 60, 255)))

    y = 30
    for text, colour in lines:
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(image, (8, y - h - 8), (18 + w, y + 8), (0, 0, 0), -1)
        cv2.putText(image, text, (13, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
        y += h + 20
    cv2.putText(image, footer, (13, image.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


def replay_headless(session: Session) -> None:
    last = 0.0
    for t, _ in session.script.rows:
        session.advance(t)
        last = t
    session.end_of_video(last)


def file_frames(path: Path, paced: bool, pause_flag: dict):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    anchor = time.perf_counter()
    index = 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                return
            t = index / fps
            if paced:
                if pause_flag.pop("resumed", False):
                    anchor = time.perf_counter() - t
                pause_flag["wait_ms"] = max(1, int((anchor + t - time.perf_counter()) * 1000))
            yield index, t, image
            index += 1
    finally:
        cap.release()


def live_frames(cam: int):
    from video_source import VideoSource

    with VideoSource(src=cam) as source:
        sub = source.subscribe("run_fsm")
        t0 = None
        while True:
            frame = sub.read_latest(timeout=2.0)
            if frame is None:
                continue
            if t0 is None:
                t0 = frame.timestamp
            yield frame.index, frame.timestamp - t0, frame.image


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="video file, or a camera index for live")
    ap.add_argument("--script", type=Path, help="steps CSV to replay; omit to use hotkeys")
    ap.add_argument("--save-script", type=Path,
                    help="where hotkey presses go (default: <video>_steps.csv for a file)")
    ap.add_argument("--graph", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--mute", action="store_true", help="no speech")
    ap.add_argument("--no-window", action="store_true",
                    help="with --video FILE --script: run unpaced with no display")
    args = ap.parse_args()

    if args.video is None and args.script is None:
        ap.error("give --video, --script, or both")

    graph = load_step_graph(args.graph)
    script = ScriptSource(args.script, graph) if args.script else None
    live = args.video is not None and args.video.isdigit()
    video = None if args.video is None or live else Path(args.video)
    if video is not None and not video.exists():
        ap.error(f"no such video: {video}")
    if args.no_window and script is None:
        ap.error("--no-window needs --script (hotkeys need the window)")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log = EventLog(
        ROOT / "logs" / "runs" / f"{stamp}.jsonl",
        meta={
            "video": args.video,
            "source": f"script:{args.script}" if script else "hotkeys",
            "graph": str(args.graph),
        },
    )
    voice = Voice(enabled=not args.mute).start()
    session = Session(graph, log, voice, script)
    session.listeners.append(print_events)
    session.start()

    try:
        if args.video is None:
            replay_headless(session)
            return

        pause = {}
        paced = not args.no_window
        frames = live_frames(int(args.video)) if live else file_frames(video, paced, pause)
        footer = ("SCRIPT " + script.path.name if script
                  else "keys: 0 idle  1-5 steps  space pause  q quit")

        t = 0.0
        reached_end = True
        for index, t, image in frames:
            session.advance(t)
            if args.no_window:
                continue

            if image.shape[1] > DISPLAY_WIDTH:
                scale = DISPLAY_WIDTH / image.shape[1]
                image = cv2.resize(image, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)
            draw_hud(image, session, t, footer)
            cv2.imshow(WINDOW, image)
            key = cv2.waitKey(pause.get("wait_ms", 1)) & 0xFF
            if key in (ord("q"), 27):
                reached_end = False
                break
            if key == ord(" ") and not live:
                while (k := cv2.waitKey(50) & 0xFF) not in (ord(" "), ord("q"), 27):
                    pass
                if k != ord(" "):
                    reached_end = False
                    break
                pause["resumed"] = True
            if key in (ord("y"), ord("n")):
                session.answer(key == ord("y"), t)
            session.key(key, index, t)

        if reached_end and not live:
            session.end_of_video(t)

        out = args.save_script or (video.with_name(video.stem + "_steps.csv") if video else None)
        if out is not None and session.save_hotkeys(out):
            print(f"saved {len(session.hotkeys.presses)} step presses to {out}")
    finally:
        cv2.destroyAllWindows()
        session.close()
        print(f"log: {log.path}")
        voice.stop()


if __name__ == "__main__":
    main()
