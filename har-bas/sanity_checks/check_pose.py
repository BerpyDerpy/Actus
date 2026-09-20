"""Sanity check: MediaPipe Pose on the default webcam.

MediaPipe 1.x removed the legacy `mp.solutions.pose` API, so this uses the
Tasks API, which needs an explicit .task model bundle. Fetch it once with:

    curl -L -o models/pose/pose_landmarker_lite.task \\
      https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task

Pose runs on CPU in this build (the Windows MediaPipe wheel ships no GPU
delegate), so this number is independent of the GPU and will contend with
YOLO for CPU time in the combined check. Runs for 10 seconds.
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

DURATION = 10.0
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "pose" / "pose_landmarker_lite.task"


def open_camera(index: int = 0) -> cv2.VideoCapture:
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    # MJPG is essential: this camera negotiates uncompressed YUY2 by default,
    # which USB bandwidth caps at ~7 FPS at 720p and would make every number
    # below a measure of the webcam rather than of compute.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        sys.exit(f"ERROR: could not open camera {index}")
    return cap


def main() -> None:
    if not MODEL_PATH.exists():
        sys.exit(f"ERROR: missing pose model at {MODEL_PATH}\n{__doc__}")

    print(f"mediapipe: {mp.__version__}  model: {MODEL_PATH.name}")

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    )

    cap = open_camera()
    frames = 0
    detected = 0
    infer_time = 0.0
    start = time.perf_counter()
    last_report = start
    elapsed = 0.0

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            now = time.perf_counter()
            if now - start >= DURATION:
                elapsed = now - start
                break

            ok, frame = cap.read()
            if not ok:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # VIDEO mode requires strictly increasing millisecond timestamps.
            t0 = time.perf_counter()
            result = landmarker.detect_for_video(mp_image, int((now - start) * 1000))
            infer_time += time.perf_counter() - t0
            if result.pose_landmarks:
                detected += 1
            frames += 1

            if now - last_report >= 1.0:
                print(f"  [{now - start:4.1f}s] {frames / (now - start):5.1f} FPS")
                last_report = now

    cap.release()

    fps = frames / elapsed if elapsed else 0.0
    print()
    compute_fps = frames / infer_time if infer_time else 0.0
    print(f"RESULT check_pose: {fps:.1f} FPS end-to-end "
          f"({frames} frames in {elapsed:.1f}s, pose found in {detected})")
    print(f"  inference only  : {infer_time / frames * 1000:.1f} ms/frame "
          f"-> {compute_fps:.1f} FPS compute ceiling")


if __name__ == "__main__":
    main()
