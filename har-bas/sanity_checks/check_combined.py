"""Sanity check: YOLO11m + MediaPipe Pose + ArUco together on one feed.

This is the number that actually matters -- it is the real compute ceiling
for the full perception stack running serially on a single frame, which is
how the inference loop will be structured initially.

Also reports the per-stage cost breakdown so it is obvious which stage to
optimise first (threading, frame-skipping, or a smaller model).

Runs for 10 seconds.
"""

from __future__ import annotations

import platform
import sys
import time
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

DURATION = 10.0
YOLO_MODEL = "yolo11m.pt"
POSE_MODEL = Path(__file__).resolve().parents[1] / "models" / "pose" / "pose_landmarker_lite.task"
WARMUP_FRAMES = 5


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
    if not POSE_MODEL.exists():
        sys.exit(f"ERROR: missing pose model at {POSE_MODEL} (see check_pose.py)")

    device = 0 if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    print(f"device: {device}  ({gpu})")

    model = YOLO(YOLO_MODEL)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    )

    cap = open_camera()
    for _ in range(WARMUP_FRAMES):
        ok, frame = cap.read()
        if ok:
            model.predict(frame, device=device, verbose=False)

    frames = 0
    cost = {"yolo": 0.0, "pose": 0.0, "aruco": 0.0}
    start = time.perf_counter()
    last_report = start
    elapsed = 0.0

    with PoseLandmarker.create_from_options(pose_options) as landmarker:
        while True:
            now = time.perf_counter()
            if now - start >= DURATION:
                elapsed = now - start
                break

            ok, frame = cap.read()
            if not ok:
                continue

            t0 = time.perf_counter()
            model.predict(frame, device=device, verbose=False)
            t1 = time.perf_counter()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            landmarker.detect_for_video(mp_image, int((now - start) * 1000))
            t2 = time.perf_counter()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            aruco.detectMarkers(gray)
            t3 = time.perf_counter()

            cost["yolo"] += t1 - t0
            cost["pose"] += t2 - t1
            cost["aruco"] += t3 - t2
            frames += 1

            if now - last_report >= 1.0:
                print(f"  [{now - start:4.1f}s] {frames / (now - start):5.1f} FPS")
                last_report = now

    cap.release()

    fps = frames / elapsed if elapsed else 0.0
    print()
    total_compute = sum(cost.values())
    compute_fps = frames / total_compute if total_compute else 0.0
    print(f"RESULT check_combined: {fps:.1f} FPS end-to-end "
          f"({frames} frames in {elapsed:.1f}s)")
    print(f"  inference only  : {total_compute / frames * 1000:.1f} ms/frame "
          f"-> {compute_fps:.1f} FPS compute ceiling")
    if frames:
        print("per-frame cost breakdown:")
        for stage, total in cost.items():
            ms = total / frames * 1000
            print(f"  {stage:<6} {ms:6.1f} ms  ({total / elapsed * 100:4.1f}% of wall time)")


if __name__ == "__main__":
    main()
