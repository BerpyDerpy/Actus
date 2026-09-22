"""Manual visual test: YOLO detection and MediaPipe pose on one feed.

`detect_test.py` and `pose_test.py` each answer "does this stage work". This
answers the question that actually matters for the demo -- do they work
*together*, on the same frame, fast enough to be worth watching. It is the
visual counterpart to `sanity_checks/check_combined.py`, which reports the
same numbers but draws nothing.

The two stages run serially on each frame, which is how the inference loop
is structured initially, so the FPS here is the honest end-to-end figure
rather than either stage's ceiling. Detection is on the GPU and pose is on
the CPU, so the per-stage breakdown in the HUD shows which one to attack
first when this needs to be faster.

The skeleton drawing and its landmark topology are imported from
`pose_test.py` rather than copied -- a 35-edge table is not worth
maintaining twice.

Controls:
    q / Esc   quit
    s         save the annotated frame to logs/
    c         cycle the detection confidence threshold (0.25 / 0.40 / 0.60)
    v         cycle the pose visibility threshold (0.00 / 0.50 / 0.75)
    n         toggle landmark index numbers
    d         toggle the detection layer
    p         toggle the pose layer
"""

from __future__ import annotations

import platform
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp
import torch
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)
from ultralytics import YOLO

# Run directly by path (the house pattern here) the sibling module is already
# importable; adding its directory explicitly means `python -m` works too.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pose_test import VISIBILITY_LEVELS, draw_pose  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
YOLO_MODEL = ROOT / "yolo11m.pt"
POSE_MODEL = ROOT / "models" / "pose" / "pose_landmarker_lite.task"
SNAPSHOT_DIR = ROOT / "logs"
CONF_LEVELS = (0.25, 0.40, 0.60)
WARMUP_FRAMES = 5


def open_camera(index: int = 0) -> cv2.VideoCapture:
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    # MJPG is essential: this camera negotiates uncompressed YUY2 by default,
    # which USB bandwidth caps at ~7 FPS at 720p and would hide the real cost
    # of running both stages.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        sys.exit(f"ERROR: could not open camera {index}")
    return cap


def main() -> None:
    for path, what in ((YOLO_MODEL, "detector weights"), (POSE_MODEL, "pose model")):
        if not path.exists():
            sys.exit(f"ERROR: missing {what} at {path}")

    device = 0 if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    print(f"detector: {gpu}   pose: CPU   mediapipe {mp.__version__}")

    model = YOLO(str(YOLO_MODEL))
    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    )

    cap = open_camera()

    # First inferences include CUDA context setup and cuDNN autotuning, which
    # would otherwise show up as a multi-second freeze on the first frame.
    print("warming up ...")
    for _ in range(WARMUP_FRAMES):
        ok, frame = cap.read()
        if ok:
            model.predict(frame, device=device, verbose=False)

    print("press q or Esc to quit, s to save, c/v thresholds, n numbers, d/p layers")

    conf_index = 0
    vis_index = 1
    show_numbers = False
    show_detection = True
    show_pose = True

    frames = 0
    cost = {"yolo": 0.0, "pose": 0.0}
    start = time.perf_counter()
    last_report = start

    with PoseLandmarker.create_from_options(pose_options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            now = time.perf_counter()
            conf = CONF_LEVELS[conf_index]
            min_visibility = VISIBILITY_LEVELS[vis_index]

            t0 = time.perf_counter()
            result = model.predict(frame, device=device, conf=conf, verbose=False)[0]
            t1 = time.perf_counter()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # VIDEO mode requires strictly increasing millisecond timestamps;
            # feeding it wall-clock time would break tracking across frames.
            pose_result = landmarker.detect_for_video(mp_image, int((now - start) * 1000))
            t2 = time.perf_counter()

            cost["yolo"] += t1 - t0
            cost["pose"] += t2 - t1
            frames += 1

            # plot() returns an annotated copy; with the layer off we draw the
            # skeleton straight onto the raw frame instead.
            annotated = result.plot() if show_detection else frame
            labels = [model.names[int(c)] for c in result.boxes.cls]

            drawn = 0
            if show_pose and pose_result.pose_landmarks:
                drawn = draw_pose(annotated, pose_result.pose_landmarks[0],
                                  min_visibility, show_numbers)

            elapsed = now - start
            fps = frames / elapsed if elapsed else 0.0
            yolo_ms = cost["yolo"] / frames * 1000
            pose_ms = cost["pose"] / frames * 1000

            cv2.putText(
                annotated,
                f"{len(labels)} objects   {drawn}/33 landmarks   {fps:.1f} FPS",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )
            cv2.putText(
                annotated,
                f"yolo {yolo_ms:.1f} ms + pose {pose_ms:.1f} ms = {yolo_ms + pose_ms:.1f} ms"
                f"   conf>{conf:.2f}  vis>{min_visibility:.2f}"
                f"{'' if show_detection else '  [det off]'}"
                f"{'' if show_pose else '  [pose off]'}",
                (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
            )

            if now - last_report >= 1.0:
                counts = Counter(labels)
                seen = ", ".join(f"{n}x {name}" for name, n in counts.most_common()) or "nothing"
                body = f"{drawn}/33 landmarks" if drawn else "no pose"
                print(f"  [{elapsed:5.1f}s] {fps:5.1f} FPS  ->  {seen}  |  {body}")
                last_report = now

            cv2.imshow("Combined detection + pose test", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                conf_index = (conf_index + 1) % len(CONF_LEVELS)
                print(f"confidence threshold -> {CONF_LEVELS[conf_index]:.2f}")
            if key == ord("v"):
                vis_index = (vis_index + 1) % len(VISIBILITY_LEVELS)
                print(f"visibility threshold -> {VISIBILITY_LEVELS[vis_index]:.2f}")
            if key == ord("n"):
                show_numbers = not show_numbers
            if key == ord("d"):
                show_detection = not show_detection
            if key == ord("p"):
                show_pose = not show_pose
            if key == ord("s"):
                SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
                path = SNAPSHOT_DIR / f"combined_{int(time.time())}.png"
                cv2.imwrite(str(path), annotated)
                print(f"saved {path}")

    cap.release()
    cv2.destroyAllWindows()

    if frames:
        total = sum(cost.values())
        elapsed = time.perf_counter() - start
        print(f"\n{frames} frames, {frames / elapsed:.1f} FPS end-to-end")
        print(f"  compute only: {total / frames * 1000:.1f} ms/frame "
              f"-> {frames / total:.1f} FPS ceiling")
        for stage, spent in cost.items():
            print(f"  {stage:<6} {spent / frames * 1000:6.1f} ms  "
                  f"({spent / elapsed * 100:4.1f}% of wall time)")


if __name__ == "__main__":
    main()
