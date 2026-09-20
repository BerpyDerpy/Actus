"""Manual visual test: live MediaPipe pose landmarks on the webcam feed.

The pose counterpart to `detect_test.py`. `sanity_checks/check_pose.py`
measures throughput but draws nothing; this draws the skeleton so tracking
can be judged by eye -- does it find the body, do the joints sit where the
joints are, does the skeleton hold together when limbs cross or leave frame.

MediaPipe 1.x ships no `mp.solutions`, so the usual `drawing_utils` /
POSE_CONNECTIONS helpers are unavailable and the skeleton is drawn here from
an explicit edge list over the 33 BlazePose landmarks.

Pose runs on CPU in this build (the Windows wheel has no GPU delegate), so
the FPS here is independent of the GPU and will contend with YOLO for CPU
time once both run in the same loop.

Controls:
    q / Esc   quit
    s         save the annotated frame to logs/
    v         cycle the visibility threshold (0.00 / 0.50 / 0.75)
    n         toggle landmark index numbers
"""

from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "pose" / "pose_landmarker_lite.task"
SNAPSHOT_DIR = ROOT / "logs"
VISIBILITY_LEVELS = (0.0, 0.5, 0.75)

# BlazePose's 33 landmarks, as the edges a human reads as a skeleton. Split
# left/right so the two sides can be coloured differently -- mirrored tracking
# is otherwise very hard to spot by eye.
FACE_EDGES = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10)]
LEFT_EDGES = [
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (11, 23), (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
]
RIGHT_EDGES = [
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (12, 24), (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]
TORSO_EDGES = [(11, 12), (23, 24)]

LEFT_COLOR = (0, 200, 255)    # amber
RIGHT_COLOR = (255, 200, 0)   # cyan
TORSO_COLOR = (255, 255, 255)
FACE_COLOR = (180, 180, 180)


def open_camera(index: int = 0) -> cv2.VideoCapture:
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    # MJPG is essential: this camera negotiates uncompressed YUY2 by default,
    # which USB bandwidth caps at ~7 FPS at 720p and would make tracking look
    # far choppier than it is.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        sys.exit(f"ERROR: could not open camera {index}")
    return cap


def draw_pose(frame, landmarks, min_visibility: float, show_numbers: bool) -> int:
    """Draw one person's skeleton. Returns the count of landmarks drawn.

    Landmarks below `min_visibility` are MediaPipe's own guesses at occluded
    joints; they are skipped so the drawing shows what is actually tracked
    rather than what has been inferred.
    """
    height, width = frame.shape[:2]
    points: list[tuple[int, int] | None] = []
    for lm in landmarks:
        # visibility is an optional field: None means the model reported no
        # estimate, which is not the same as reporting a low one, so it is
        # drawn rather than silently dropped.
        if lm.visibility is not None and lm.visibility < min_visibility:
            points.append(None)
            continue
        points.append((int(lm.x * width), int(lm.y * height)))

    for edges, color, thickness in (
        (FACE_EDGES, FACE_COLOR, 1),
        (TORSO_EDGES, TORSO_COLOR, 3),
        (LEFT_EDGES, LEFT_COLOR, 2),
        (RIGHT_EDGES, RIGHT_COLOR, 2),
    ):
        for a, b in edges:
            if points[a] is not None and points[b] is not None:
                cv2.line(frame, points[a], points[b], color, thickness, cv2.LINE_AA)

    drawn = 0
    for i, point in enumerate(points):
        if point is None:
            continue
        drawn += 1
        cv2.circle(frame, point, 3, (0, 0, 255), -1, cv2.LINE_AA)
        if show_numbers:
            cv2.putText(frame, str(i), (point[0] + 5, point[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    return drawn


def main() -> None:
    if not MODEL_PATH.exists():
        sys.exit(f"ERROR: missing pose model at {MODEL_PATH}\n{__doc__}")

    print(f"mediapipe: {mp.__version__}  model: {MODEL_PATH.name}  (CPU)")

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    )

    cap = open_camera()
    print("press q or Esc to quit, s to save, v for visibility, n for numbers")

    vis_index = 1
    show_numbers = False
    frames = 0
    detected = 0
    infer_time = 0.0
    start = time.perf_counter()
    last_report = start

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            now = time.perf_counter()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            t0 = time.perf_counter()
            # VIDEO mode requires strictly increasing millisecond timestamps;
            # feeding it wall-clock time would break tracking across frames.
            result = landmarker.detect_for_video(mp_image, int((now - start) * 1000))
            infer_time += time.perf_counter() - t0
            frames += 1

            min_visibility = VISIBILITY_LEVELS[vis_index]
            drawn = 0
            if result.pose_landmarks:
                detected += 1
                drawn = draw_pose(frame, result.pose_landmarks[0],
                                  min_visibility, show_numbers)

            elapsed = now - start
            fps = frames / elapsed if elapsed else 0.0
            ms = infer_time / frames * 1000 if frames else 0.0
            cv2.putText(
                frame,
                f"{drawn}/33 landmarks   {fps:.1f} FPS   {ms:.1f} ms   vis>{min_visibility:.2f}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )

            if now - last_report >= 1.0:
                state = f"{drawn}/33 landmarks" if drawn else "no pose"
                print(f"  [{elapsed:5.1f}s] {fps:5.1f} FPS  ->  {state}")
                last_report = now

            cv2.imshow("MediaPipe pose test (33 landmarks)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("v"):
                vis_index = (vis_index + 1) % len(VISIBILITY_LEVELS)
                print(f"visibility threshold -> {VISIBILITY_LEVELS[vis_index]:.2f}")
            if key == ord("n"):
                show_numbers = not show_numbers
            if key == ord("s"):
                SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
                path = SNAPSHOT_DIR / f"pose_{int(time.time())}.png"
                cv2.imwrite(str(path), frame)
                print(f"saved {path}")

    cap.release()
    cv2.destroyAllWindows()

    if frames:
        print(f"\n{frames} frames, {frames / (time.perf_counter() - start):.1f} FPS average, "
              f"{infer_time / frames * 1000:.1f} ms/frame inference, "
              f"pose found in {detected} ({detected / frames * 100:.0f}%)")


if __name__ == "__main__":
    main()
