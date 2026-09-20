"""Manual visual test: live ArUco marker detection with pose axes.

Draws each detected marker's ID and its 3D pose axes on the webcam feed.
This is an eyeball test -- run it, wave a DICT_4X4_50 marker at the camera
(printed, or displayed on a phone), and confirm IDs are stable and axes do
not jitter wildly.

Markers shown on a phone screen need more from the detector than printed
ones, and stock `DetectorParameters()` is not enough:

  * A phone in dark mode, or with iOS Smart Invert on, renders a greyscale
    PNG white-on-black. An inverted marker never decodes with default
    settings -- the square is found, the bits are unreadable -- so
    `detectInvertedMarker` is on here. This was the measured cause of the
    markers/ images failing to detect from a phone.
  * Screens light unevenly and the lens softens cell edges, so the adaptive
    threshold sweeps a wider range of window sizes and the contour fit and
    bit-error tolerance are both loosened.
  * Subpixel corner refinement is on, without which the pose axes jitter
    visibly frame to frame.

When nothing decodes, press `r` to show rejected candidates. A red quad
around the marker means the square was found and only the bits failed
(inversion, blur, glare, or wrong dictionary); no quad at all means the
problem is upstream -- focus, distance, or the marker's white quiet zone
being cropped away by zooming in.

Camera intrinsics are a rough pinhole guess derived from the frame size, NOT
a real calibration, and `--marker-size` defaults to a printed 5 cm marker.
On a phone screen the marker is whatever size it displays at, so pass the
measured width or ignore the distance readout entirely. Replace
`approx_intrinsics()` with a checkerboard calibration before any measurement
depends on these numbers.

Controls:
    q / Esc   quit
    s         save the annotated frame to markers/
    r         toggle rejected-candidate outlines
    t         toggle tuned / stock detector parameters
    b         toggle the binarised view the detector works from
    [  ]      step exposure down / up (helps when a screen blows out)
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

DICTIONARY = cv2.aruco.DICT_4X4_50
MARKERS_DIR = Path(__file__).resolve().parents[2] / "markers"
DEFAULT_MARKER_SIZE_M = 0.05  # a printed 5 cm marker


def tuned_parameters() -> cv2.aruco.DetectorParameters:
    """Detector settings that cope with a marker on a phone screen."""
    p = cv2.aruco.DetectorParameters()
    # Dark mode renders the marker white-on-black; without this it is found
    # as a square and then discarded as undecodable.
    p.detectInvertedMarker = True
    # A screen lights unevenly, so no single threshold window suits the whole
    # marker; sweeping more of them finds the quad that one window misses.
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 53
    p.adaptiveThreshWinSizeStep = 4
    # Blur rounds the corners, so the polygon fit needs more slack.
    p.polygonalApproxAccuracyRate = 0.05
    # A phone at arm's length is a small marker.
    p.minMarkerPerimeterRate = 0.02
    # Tolerate more bad bits, and ignore more of each cell's edge, which is
    # where blur and JPEG ringing do their damage.
    p.errorCorrectionRate = 0.8
    p.perspectiveRemoveIgnoredMarginPerCell = 0.25
    # Without subpixel corners the pose axes visibly jitter.
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return p


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


def set_exposure(cap: cv2.VideoCapture, value: float) -> bool:
    """Ask for manual exposure. Not every camera or backend honours this."""
    # DirectShow wants 0.25 to mean "manual" on most UVC cameras.
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    return bool(cap.set(cv2.CAP_PROP_EXPOSURE, value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument(
        "--marker-size", type=float, default=DEFAULT_MARKER_SIZE_M,
        help="marker width in metres; measure it on screen if not printed "
             f"(default {DEFAULT_MARKER_SIZE_M})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cap = open_camera(args.camera)
    ok, frame = cap.read()
    if not ok:
        sys.exit("ERROR: camera opened but returned no frame")
    height, width = frame.shape[:2]
    camera_matrix, dist_coeffs = approx_intrinsics(width, height)
    obj_points = marker_object_points(args.marker_size)

    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    detectors = {
        True: cv2.aruco.ArucoDetector(dictionary, tuned_parameters()),
        False: cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters()),
    }

    print(f"frame: {width}x{height}   dictionary: DICT_4X4_50   "
          f"marker: {args.marker_size * 100:.1f} cm")
    print("press q or Esc to quit, s to save, r rejected, t params, b binary view")

    use_tuned = True
    show_rejected = False
    show_binary = False
    exposure = -6.0
    frames = 0
    decoded_frames = 0
    start = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frames += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detectors[use_tuned].detectMarkers(gray)

        if show_binary:
            # Indicative only: ArUco thresholds internally across a sweep of
            # window sizes, this shows one of them.
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 23, 7,
            )
            frame = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        if show_rejected and rejected:
            cv2.aruco.drawDetectedMarkers(frame, rejected, None, (0, 0, 255))

        if ids is not None and len(ids) > 0:
            decoded_frames += 1
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
                    rvec, tvec, args.marker_size * 0.75, 2,
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
            frame,
            f"markers: {count}   {fps:.1f} FPS   "
            f"{'tuned' if use_tuned else 'stock'} params",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )

        # Name the failure rather than leaving a blank screen: a rejected quad
        # means the square was found and only the bits failed.
        if count == 0:
            if rejected:
                hint = (f"{len(rejected)} square(s) found, none decoded -- "
                        "inverted? blurred? wrong dictionary?")
            else:
                hint = "no squares found -- check focus, distance, quiet zone"
            cv2.putText(frame, hint, (10, 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        cv2.imshow("ArUco marker test (DICT_4X4_50)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            show_rejected = not show_rejected
        if key == ord("t"):
            use_tuned = not use_tuned
            print(f"detector parameters -> {'tuned' if use_tuned else 'stock'}")
        if key == ord("b"):
            show_binary = not show_binary
        if key in (ord("["), ord("]")):
            exposure += -1.0 if key == ord("[") else 1.0
            applied = set_exposure(cap, exposure)
            print(f"exposure -> {exposure:.0f} ({'applied' if applied else 'refused'})")
        if key == ord("s"):
            MARKERS_DIR.mkdir(parents=True, exist_ok=True)
            path = MARKERS_DIR / f"snapshot_{int(time.time())}.png"
            cv2.imwrite(str(path), frame)
            print(f"saved {path}")

    cap.release()
    cv2.destroyAllWindows()

    if frames:
        print(f"\n{frames} frames, decoded a marker in {decoded_frames} "
              f"({decoded_frames / frames * 100:.0f}%)")


if __name__ == "__main__":
    main()
