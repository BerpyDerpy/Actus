"""Sanity check: stock YOLO11m inference on the default webcam.

Measures the sustained FPS of detection alone, to establish the compute
ceiling before any custom detector is trained. Runs for 10 seconds.
"""

from __future__ import annotations

import platform
import sys
import time

import cv2
import torch
from ultralytics import YOLO

DURATION = 10.0
MODEL = "yolo11m.pt"  # auto-downloaded by ultralytics on first use
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
    device = 0 if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    print(f"device: {device}  ({gpu})")

    model = YOLO(MODEL)
    cap = open_camera()

    # First inferences include CUDA context setup and cuDNN autotuning; they
    # are not representative of steady-state throughput.
    for _ in range(WARMUP_FRAMES):
        ok, frame = cap.read()
        if ok:
            model.predict(frame, device=device, verbose=False)

    frames = 0
    detections = 0
    infer_time = 0.0
    start = time.perf_counter()
    last_report = start

    while True:
        now = time.perf_counter()
        if now - start >= DURATION:
            break

        ok, frame = cap.read()
        if not ok:
            continue

        t0 = time.perf_counter()
        results = model.predict(frame, device=device, verbose=False)
        infer_time += time.perf_counter() - t0
        detections += len(results[0].boxes)
        frames += 1

        if now - last_report >= 1.0:
            print(f"  [{now - start:4.1f}s] {frames / (now - start):5.1f} FPS")
            last_report = now

    elapsed = time.perf_counter() - start
    cap.release()

    fps = frames / elapsed if elapsed else 0.0
    print()
    compute_fps = frames / infer_time if infer_time else 0.0
    print(f"RESULT check_yolo: {fps:.1f} FPS end-to-end "
          f"({frames} frames in {elapsed:.1f}s, {detections} total detections)")
    print(f"  inference only  : {infer_time / frames * 1000:.1f} ms/frame "
          f"-> {compute_fps:.1f} FPS compute ceiling")


if __name__ == "__main__":
    main()
