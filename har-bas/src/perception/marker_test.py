"""Manual visual test: live ArUco marker detection with pose axes.

Draws each detected marker's ID and its 3D pose axes on the webcam feed.
This is an eyeball test -- run it, wave a printed DICT_4X4_50 marker at the
camera, and confirm IDs are stable and axes do not jitter wildly.

Camera intrinsics here are a rough pinhole guess derived from the frame size,
NOT a real calibration. Axes will look approximately right but are not
metrically accurate; replace `approx_intrinsics()` with values from a proper
checkerboard calibration before any measurement depends on them.

Controls:  q / Esc  quit      s  save current frame to markers/
"""

from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

DICTIONARY = cv2.aruco.DICT_4X4_50
MARKER_LENGTH_M = 0.05  # placeholder: 5 cm printed marker
MARKERS_DIR = Path(__file__).resolve().parents[2] / "markers"


def approx_intrinsics(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Rough pinhole guess: focal length ~= image width, centre at midpoint."""
    focal = float(width)
    camera_matrix = np.array(
        [[focal, 0.0, width / 2.0],
         [0.0, focal, height / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    return camera_matrix, dist_coeffs


def marker_object_points(length: float) -> np.ndarray:
    """Corners of a marker in its own frame, matching detectMarkers order."""
    half = length / 2.0
    return np.array(
        [[-half, half, 0.0],
         [half, half, 0.0],
         [half, -half, 0.0],
         [-half, -half, 0.0]],
        dtype=np.float32,
    )


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
    cap = open_camera()
    ok, frame = cap.read()
    if not ok:
        sys.exit("ERROR: camera opened but returned no frame")
    height, width = frame.shape[:2]
    camera_matrix, dist_coeffs = approx_intrinsics(width, height)
    obj_points = marker_object_points(MARKER_LENGTH_M)

    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

    print(f"frame: {width}x{height}   dictionary: DICT_4X4_50")
    print("press q or Esc to quit, s to save a snapshot")

    frames = 0
    start = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frames += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _rejected = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for marker_corners, marker_id in zip(corners, ids.flatten()):
                found, rvec, tvec = cv2.solvePnP(
                    obj_points,
                    marker_corners.reshape(-1, 2),
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
                if not found:
                    continue
                cv2.drawFrameAxes(
                    frame, camera_matrix, dist_coeffs,
                    rvec, tvec, MARKER_LENGTH_M * 0.75, 2,
                )
                top_left = marker_corners.reshape(-1, 2)[0].astype(int)
                distance = float(np.linalg.norm(tvec))
                cv2.putText(
                    frame, f"id={marker_id}  {distance:.2f}m",
                    (top_left[0], max(top_left[1] - 12, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )

        count = 0 if ids is None else len(ids)
        elapsed = time.perf_counter() - start
        fps = frames / elapsed if elapsed else 0.0
        cv2.putText(
            frame, f"markers: {count}   {fps:.1f} FPS",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )

        cv2.imshow("ArUco marker test (DICT_4X4_50)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            MARKERS_DIR.mkdir(parents=True, exist_ok=True)
            path = MARKERS_DIR / f"snapshot_{int(time.time())}.png"
            cv2.imwrite(str(path), frame)
            print(f"saved {path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
