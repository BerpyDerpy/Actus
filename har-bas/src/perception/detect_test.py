"""Manual visual test: live YOLO object detection on the webcam feed.

The counterpart to `marker_test.py`, for detection rather than markers.
`sanity_checks/check_yolo.py` measures throughput but draws nothing; this
draws every box and label so the detector can be judged by eye -- does it
find the object, is the class right, does the box track it without flicker.

The weights here are stock YOLOv8n trained on COCO, so it recognises the 80
COCO classes (person, bottle, laptop, scissors ...) and nothing else. Lab
hardware will read as whatever COCO class is nearest, or not at all; that is
expected until the custom detector is trained.

Controls:
    q / Esc   quit
    s         save the annotated frame to logs/
    c         cycle the confidence threshold (0.25 / 0.40 / 0.60)
"""

from __future__ import annotations

import platform
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "yolov8n.pt"
SNAPSHOT_DIR = ROOT / "logs"
CONF_LEVELS = (0.25, 0.40, 0.60)
WARMUP_FRAMES = 5


def open_camera(index: int = 0) -> cv2.VideoCapture:
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    # MJPG is essential: this camera negotiates uncompressed YUY2 by default,
    # which USB bandwidth caps at ~7 FPS at 720p and would make the detector
    # look far slower than it is.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        sys.exit(f"ERROR: could not open camera {index}")
    return cap


def pick_device() -> tuple[str | int, str]:
    """Best compute device Ultralytics can target, plus a label to print.

    CUDA first (Windows/Linux dev boxes), then Apple Silicon's MPS backend,
    then CPU everywhere else.
    """
    if torch.cuda.is_available():
        return 0, torch.cuda.get_device_name(0)
    if torch.backends.mps.is_available():
        return "mps", "Apple Silicon (MPS)"
    return "cpu", "CPU only"


def main() -> None:
    if not MODEL_PATH.exists():
        sys.exit(f"ERROR: missing weights at {MODEL_PATH}")

    device, gpu = pick_device()
    print(f"device: {device}  ({gpu})")

    model = YOLO(str(MODEL_PATH))
    cap = open_camera()

    # First inferences include CUDA context setup and cuDNN autotuning, which
    # would otherwise show up as a multi-second freeze on the first frame.
    print("warming up ...")
    for _ in range(WARMUP_FRAMES):
        ok, frame = cap.read()
        if ok:
            model.predict(frame, device=device, verbose=False)

    print(f"{len(model.names)} COCO classes loaded")
    print("press q or Esc to quit, s to save a snapshot, c to change confidence")

    conf_index = 0
    frames = 0
    infer_time = 0.0
    start = time.perf_counter()
    last_report = start

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        conf = CONF_LEVELS[conf_index]
        t0 = time.perf_counter()
        results = model.predict(frame, device=device, conf=conf, verbose=False)
        infer_time += time.perf_counter() - t0
        frames += 1

        result = results[0]
        # plot() returns a fresh BGR copy with boxes, class names and scores
        # already drawn, in the same palette ultralytics uses everywhere else.
        annotated = result.plot()

        labels = [model.names[int(c)] for c in result.boxes.cls]
        now = time.perf_counter()
        elapsed = now - start
        fps = frames / elapsed if elapsed else 0.0
        ms = infer_time / frames * 1000 if frames else 0.0

        cv2.putText(
            annotated, f"{len(labels)} objects   {fps:.1f} FPS   {ms:.1f} ms   conf>{conf:.2f}",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )

        # A console echo makes it possible to confirm recognition without
        # staring at small on-screen text.
        if now - last_report >= 1.0:
            counts = Counter(labels)
            summary = ", ".join(f"{n}x {name}" for name, n in counts.most_common()) or "nothing"
            print(f"  [{elapsed:5.1f}s] {fps:5.1f} FPS  ->  {summary}")
            last_report = now

        cv2.imshow("YOLOv8n detection test (COCO)", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("c"):
            conf_index = (conf_index + 1) % len(CONF_LEVELS)
            print(f"confidence threshold -> {CONF_LEVELS[conf_index]:.2f}")
        if key == ord("s"):
            SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
            path = SNAPSHOT_DIR / f"detect_{int(time.time())}.png"
            cv2.imwrite(str(path), annotated)
            print(f"saved {path}")

    cap.release()
    cv2.destroyAllWindows()

    if frames:
        print(f"\n{frames} frames, {frames / (time.perf_counter() - start):.1f} FPS average, "
              f"{infer_time / frames * 1000:.1f} ms/frame inference")


if __name__ == "__main__":
    main()
