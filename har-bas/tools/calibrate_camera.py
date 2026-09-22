"""ChArUco calibration for a recording camera, written to configs/.

Every marker-derived measurement (cap-to-bottle distance, bottle tilt, the
table anchor frame) needs real intrinsics. `marker_test.py` still uses a
pinhole guess; this tool replaces the guess with a measured camera matrix and
distortion coefficients, saved as `configs/camera_<name>.yaml`.

Calibration is valid for ONE camera at ONE resolution. Run it with the exact
capture settings the takes will be recorded with (1280x720, MJPG), before
filming. A calibration at another resolution does not carry over -- redo it.

Why ChArUco and not a plain checkerboard:
    Every chessboard corner is identified by the ArUco markers around it, so
    a view no longer needs the whole board in frame. A plain checkerboard is
    found all-or-nothing, which keeps it away from the frame edges -- exactly
    where lens distortion is strongest and most needs measuring. With ChArUco,
    the board can sit right in each frame corner, and those views are the
    valuable ones. Right at the edge, not past it: a corner whose neighbouring
    markers are cut off by the frame is not detected.

The board:
    10x7 squares (9x6 = 54 inner corners), 25 mm squares, 18 mm markers from
    DICT_5X5_100 by default. Deliberately NOT the project's DICT_4X4_50: the
    object markers are 4x4 ids 0-3, and a board built from the same
    dictionary would carry those same ids, so a bottle marker in view would
    read as a board square and marker_test.py would read the board as
    objects. A 5x5 marker never decodes as 4x4, so the two cannot collide.

    Generate the board with `--make-board`, print it at 100% scale (no "fit
    to page"), measure a square with a ruler and pass the real size with
    `--square-mm`; the marker size scales with it. Tape it to something rigid
    -- a board that bends calibrates a lens that does not exist.

    A board from elsewhere works if its dictionary matches, but its layout
    must be passed exactly (--squares-x/--squares-y, --legacy). Get it wrong
    and the markers decode but no corner ever does. When that happens the
    tool fits every layout to the markers it sees and prints the right flags.

Getting good views:
    * Reach the frame edges and corners, board edge touching the frame edge.
      In free mode the HUD shades red every part of the frame no captured
      corner has reached yet; aim for no red left.
    * Tilt the board 30-45 degrees, in both directions and about both axes,
      and vary the distance. Views facing the camera flat cannot separate
      focal length from distance: the focal length then comes out with a
      wide error bar, and every metric distance inherits it. Auto-capture
      stops taking flat views once it has MAX_FLAT of them, so the rest of
      the target has to be tilted. The HUD shows the current tilt.
    * Keep the board big in frame: the whole board should cover 10%+ of the
      frame in a typical view (the HUD shows "size"). From across a room a
      letter-size board is a few percent, and a small board hides the same
      perspective a flat one does. Come closer, or print a bigger board.
    * Hold still at each capture. Auto mode waits for the corners to stop
      moving, which is what keeps motion blur out.

A phone as the camera, previewed on the laptop:
    Stream from the phone (IP Webcam on Android: http://<phone-ip>:8080/video;
    DroidCam instead shows up as a camera index) and pass the URL as
    --camera. The live window, the HUD and the saved frames are all on the
    laptop; the phone only streams. Put the phone on a tripod and move the
    board, not the phone.

    A phone camera changes itself in ways a webcam does not, and any change
    between calibrating and recording silently invalidates the calibration.
    In the app, before calibrating, and identically when recording:
      * Lock focus (fixed / infinity, or tap-to-lock). Refocusing shifts the
        focal length -- "focus breathing" -- by a few percent.
      * Turn video stabilization OFF. It crops and warps each frame
        differently, so there is no single camera model to calibrate.
      * Fix zoom at 1x on the main lens. Many phones switch to the ultrawide
        or tele lens on their own at some distances or light levels.
      * Stream 1280x720 landscape, JPEG quality high (80+). A 1080p stream
        works with --resize, as long as the recorder downscales it the same
        way.
    Record what you set with --notes; it goes in the YAML.

    Keep both devices on the same 5 GHz wifi, or a hotspot off the laptop.
    Frames arrive 100-300 ms late, so the preview lags the board a little;
    auto-capture waits for the board to be still, so that does not matter.

Usage:
    python tools/calibrate_camera.py --make-board markers/charuco_10x7_25mm.png
    python tools/calibrate_camera.py --name webcam --square-mm 24.8
    python tools/calibrate_camera.py --name phone --camera http://192.168.1.7:8080/video \
        --notes "IP Webcam, 1280x720, focus locked, stabilization off, 1x"
    python tools/calibrate_camera.py --name webcam --force --from-dir
        logs/calibration/webcam/20260922_140633 logs/calibration/webcam/<later>

Each live session keeps its frames in `logs/calibration/<name>/<session>/`,
so a calibration can be redone offline with `--from-dir` (different flags,
bad frames deleted by hand, or several sessions combined -- say a good run
plus a short one that only fills the frame corners) without waving the board
around again.

Guided mode (the default):
    The live view shows a cyan box: the outline the board should have. Move
    and tilt the board until its outline (orange) fills the box -- arrows
    point each board corner to where it should go -- and hold still; it
    captures, and the next box appears. 20 boxes (--views, up to 30) cover
    near, far, tilted, rotated and frame-corner poses, then it calibrates. The box's size is
    the distance: a big box means hold the board close. --free goes back to
    capturing any new pose.

Controls (live):
    space     capture the current view
    a         toggle auto-capture (default on)
    n / p     guided: skip to the next box / back to the previous one
    g         toggle guided / free mode
    u         undo the last capture
    c         calibrate now (needs at least --min-views), or recalibrate
    d         after calibrating: toggle the undistorted preview
    q / Esc   quit (writes nothing unless a calibration has already run)
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import re
import struct
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "configs"
FRAMES_DIR = ROOT / "logs" / "calibration"

DICTIONARIES = {
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
}
# Marker side as a fraction of the square side on the generated board.
MARKER_RATIO = 0.72

# A view is only usable if its corners span a real area of the board: a
# single row or column of corners is a line, and a line cannot be calibrated.
# 9 is a 3x3 block: low enough that a board pushed into a frame corner, with
# much of it out of view, still counts.
MIN_CORNERS = 9
MIN_ROWS_COLS = 3
# Auto-capture: shared corners must move less than this (mean, px) between
# consecutive frames, for this many frames in a row.
STILL_PX = 1.5
STILL_FRAMES = 8
# Auto-capture: a new view must differ from every kept view by at least this
# much, as a fraction of the image diagonal (see `board_outline`).
MIN_NOVELTY = 0.06
# Tilt (board normal vs optical axis) that counts as a tilted view, how many
# flat views auto-capture will take, and how many tilted views a calibration
# needs before the focal length is trusted.
TILT_MIN_DEG = 20.0
MAX_FLAT = 10
MIN_TILTED = 6
# Median share of the frame the whole board should cover. A small board shows
# little perspective, so focal length trades off against distance just as it
# does with flat views.
MIN_BOARD_AREA = 0.10
# k3 (the 6th-power radial term) is only identifiable from corners near the
# frame corners, on a board big enough to show perspective; short of this
# many frame corners, or of MIN_BOARD_AREA, auto mode fixes it at zero
# instead of letting it trade off against fx.
K3_CORNERS = 3
# Focal-length 1-sigma above this fraction is flagged: every metric distance
# carries the same relative error.
FX_STD_WARN = 0.01
# Coverage is measured on a cells x cells grid over the image.
COVER_CELLS = 8
# Per-view outliers are dropped if their error exceeds both of these.
OUTLIER_FACTOR = 2.0
OUTLIER_FLOOR_PX = 0.5


@dataclass
class View:
    """One captured board observation: identified corners, board and image."""

    ids: np.ndarray        # (N,) ChArUco corner ids
    obj: np.ndarray        # (N, 1, 3) float32, board frame, metres
    img: np.ndarray        # (N, 1, 2) float32, pixels
    area: float = 0.0      # whole board's projected area, as a share of the frame


# -- board --------------------------------------------------------------


def make_charuco(args: argparse.Namespace) -> cv2.aruco.CharucoBoard:
    """The board, in metres to match marker_test.py's marker sizes.

    The unit only scales the extrinsics, never the intrinsics.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARIES[args.dict])
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        args.square_mm / 1000.0, args.marker_mm / 1000.0, dictionary,
    )
    board.setLegacyPattern(args.legacy)
    return board


def make_detector(board: cv2.aruco.CharucoBoard) -> cv2.aruco.CharucoDetector:
    params = cv2.aruco.DetectorParameters()
    # Markers on a partly visible, tilted board can be small.
    params.minMarkerPerimeterRate = 0.02
    return cv2.aruco.CharucoDetector(board, cv2.aruco.CharucoParameters(), params)


def write_board(path: Path, board: cv2.aruco.CharucoBoard,
                args: argparse.Namespace) -> None:
    """Write a printable board PNG that carries its print resolution."""
    px_per_mm = args.dpi / 25.4
    square = int(round(args.square_mm * px_per_mm))
    margin = int(round(10 * px_per_mm))
    # Sized so the board area is an exact multiple of the square, so the
    # generator never resamples and the printed squares are the stated size.
    size = (args.squares_x * square + 2 * margin, args.squares_y * square + 2 * margin)
    img = board.generateImage(size, marginSize=margin, borderBits=1)
    label = (f"ChArUco {args.squares_x}x{args.squares_y}, {args.dict}, "
             f"square {args.square_mm:g} mm, marker {args.marker_mm:g} mm @ {args.dpi} dpi"
             " -- print at 100%, then MEASURE a square")
    cv2.putText(img, label, (margin, img.shape[0] - margin // 3),
                cv2.FONT_HERSHEY_SIMPLEX, px_per_mm * 0.085, 0, max(1, args.dpi // 150))
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        sys.exit(f"ERROR: could not encode {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_png_with_dpi(buf.tobytes(), args.dpi))
    w_mm, h_mm = img.shape[1] / px_per_mm, img.shape[0] / px_per_mm
    print(f"wrote {path}  ({w_mm:.0f} x {h_mm:.0f} mm at {args.dpi} dpi)")
    print("print at 100% / actual size, measure one square, pass it as --square-mm")


def _png_with_dpi(png: bytes, dpi: int) -> bytes:
    """Insert a pHYs chunk after IHDR, so "actual size" printing means something."""
    ppm = int(round(dpi / 0.0254))
    body = b"pHYs" + struct.pack(">IIB", ppm, ppm, 1)
    chunk = struct.pack(">I", 9) + body + struct.pack(">I", zlib.crc32(body))
    ihdr_end = 8 + 4 + 4 + 13 + 4  # signature, then IHDR length/type/data/crc
    return png[:ihdr_end] + chunk + png[ihdr_end:]


# -- detection ----------------------------------------------------------


def detect_view(detector: cv2.aruco.CharucoDetector, board: cv2.aruco.CharucoBoard,
                gray: np.ndarray) -> tuple[View | None, tuple, str]:
    """Find the board's corners, returning a View only if it is calibratable.

    Also returns the raw detection for drawing, and a reason when rejected.
    """
    corners, ids, marker_corners, marker_ids = detector.detectBoard(gray)
    raw = (corners, ids, marker_corners, marker_ids)
    if ids is None or len(ids) == 0:
        n_markers = 0 if marker_ids is None else len(marker_ids)
        if n_markers >= 4:
            # Either the layout is wrong (--squares-x/y) or this one frame is
            # too steep or blurred to interpolate; the caller tells them apart.
            return None, raw, (f"{n_markers} markers, no corners -- too steep, "
                               "blurred, or wrong --squares-x/--squares-y")
        return None, raw, (f"{n_markers} markers, no corners -- closer? flat? well lit?"
                           if n_markers else "no board markers found")
    obj, img = board.matchImagePoints(corners, ids)
    view = View(ids.reshape(-1), obj.astype(np.float32), img.astype(np.float32))
    xy = view.obj.reshape(-1, 3)[:, :2]
    rows = len(np.unique(np.round(xy[:, 1], 6)))
    cols = len(np.unique(np.round(xy[:, 0], 6)))
    if len(view.ids) < MIN_CORNERS or rows < MIN_ROWS_COLS or cols < MIN_ROWS_COLS:
        return None, raw, (f"{len(view.ids)} corners in {rows} rows x {cols} cols -- "
                           f"need {MIN_CORNERS}+ across {MIN_ROWS_COLS}+ of each")
    h, w = gray.shape[:2]
    view.area = float(cv2.contourArea(board_outline(view, board)) / (w * h))
    return view, raw, ""


def guess_layout(gray: np.ndarray, marker_corners, marker_ids,
                 args: argparse.Namespace) -> tuple[int, int, bool, int] | None:
    """Find the board layout that the detected markers actually fit.

    Tries every plausible squares_x x squares_y (and the pre-4.6 legacy
    pattern where it differs) against the markers already found, and returns
    (squares_x, squares_y, legacy, corners) for the layout that interpolates
    the most corners. Ties go to the smallest board that holds every seen id,
    since a larger board would have shown its extra ids too. Only the layout
    is found: square and marker sizes still have to be measured.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARIES[args.dict])
    max_id = int(marker_ids.max())
    ratio = args.marker_mm / args.square_mm
    best = None
    for sx in range(3, 17):
        for sy in range(3, 17):
            n = sx * sy // 2
            if n <= max_id or n > dictionary.bytesList.shape[0]:
                continue
            # The legacy pattern only differs for an even number of rows.
            for legacy in (False, True) if sy % 2 == 0 else (False,):
                board = cv2.aruco.CharucoBoard((sx, sy), 1.0, ratio, dictionary)
                board.setLegacyPattern(legacy)
                ids = cv2.aruco.CharucoDetector(board).detectBoard(
                    gray, markerCorners=marker_corners, markerIds=marker_ids,
                )[1]
                found = 0 if ids is None else len(ids)
                key = (found, -n)
                if found and (best is None or key > best[0]):
                    best = (key, (sx, sy, legacy, found))
    return None if best is None else best[1]


def markers_without_corners(raw: tuple) -> bool:
    """Enough markers decoded that a correct layout would give corners."""
    _, ids, _, marker_ids = raw
    return (ids is None or len(ids) == 0) and marker_ids is not None \
        and len(marker_ids) >= 4


def layout_hint(guess: tuple[int, int, bool, int] | None,
                args: argparse.Namespace) -> str:
    """A rerun suggestion, only when another layout fits convincingly.

    A steep or partly visible board can let some wrong layout interpolate a
    corner or two by chance; that is noise, not evidence, so anything short
    of a calibratable view says nothing.
    """
    if guess is None or guess[3] < MIN_CORNERS:
        return ""
    sx, sy, legacy, found = guess
    if (sx, sy, legacy) == (args.squares_x, args.squares_y, args.legacy):
        return ""
    flags = f"--squares-x {sx} --squares-y {sy}" + (" --legacy" if legacy else "")
    return (f"board looks like {sx}x{sy} ({found} corners): rerun with {flags}, "
            "plus the measured --square-mm and --marker-mm")


def shared_displacement(a: View, b: View) -> float | None:
    """Mean movement of the corners two views share, or None if too few."""
    common, ia, ib = np.intersect1d(a.ids, b.ids, return_indices=True)
    if len(common) < 4:
        return None
    return float(np.linalg.norm(
        a.img.reshape(-1, 2)[ia] - b.img.reshape(-1, 2)[ib], axis=1).mean())


def board_outline(view: View, board: cv2.aruco.CharucoBoard) -> np.ndarray:
    """Where the whole board's four outer corners fall in the image.

    Fitted by homography from whichever corners are visible, so two views of
    the same pose compare as equal no matter which part of the board each saw.
    """
    H, _ = cv2.findHomography(view.obj.reshape(-1, 3)[:, :2], view.img.reshape(-1, 2))
    sx, sy = board.getChessboardSize()
    s = board.getSquareLength()
    outline = np.array([[[0, 0]], [[sx * s, 0]], [[sx * s, sy * s]], [[0, sy * s]]],
                       np.float32)
    return cv2.perspectiveTransform(outline, H).reshape(-1, 2)


def novelty(outline: np.ndarray, kept: list[np.ndarray], diagonal: float) -> float:
    """How far this pose is from the closest kept one, 0..1.

    Each corner's distance is capped at the diagonal, because a steeply
    tilted board projects its far corners wildly out of frame.
    """
    if not kept:
        return 1.0
    return min(
        float(np.minimum(np.linalg.norm(outline - k, axis=1), diagonal).mean())
        for k in kept
    ) / diagonal


def grid_hits(views: list[View], size: tuple[int, int], cells: int) -> np.ndarray:
    """Which cells of a cells x cells grid over the image any corner landed in."""
    w, h = size
    hit = np.zeros((cells, cells), bool)
    for v in views:
        pts = v.img.reshape(-1, 2)
        xs = np.clip((pts[:, 0] / w * cells).astype(int), 0, cells - 1)
        ys = np.clip((pts[:, 1] / h * cells).astype(int), 0, cells - 1)
        hit[ys, xs] = True
    return hit


def corners_reached(views: list[View], size: tuple[int, int]) -> int:
    """How many of the four frame-corner cells any corner landed in.

    Tracked apart from coverage because the frame corners are where a wide
    lens distorts most, and a board held in front of a person rarely goes
    there unprompted.
    """
    hit = grid_hits(views, size, COVER_CELLS)
    # int() each: numpy bools add as logical OR, which would cap this at 1.
    return sum(int(c) for c in (hit[0, 0], hit[0, -1], hit[-1, 0], hit[-1, -1]))


def tilt_deg(rvec: np.ndarray) -> float:
    """Angle between the board normal and the optical axis."""
    R, _ = cv2.Rodrigues(rvec)
    return float(np.degrees(np.arccos(min(1.0, abs(R[2, 2])))))


def view_tilt(view: View, K: np.ndarray, dist: np.ndarray) -> float | None:
    ok, rvec, _ = cv2.solvePnP(view.obj, view.img, K, dist)
    return tilt_deg(rvec) if ok else None


def provisional_intrinsics(views: list[View],
                           size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Rough intrinsics for the live tilt readout, before the real calibration.

    A quick fit once there are a few views; before that, a 70 degree HFOV
    guess, which is close enough to tell a flat board from a tilted one.
    """
    w, h = size
    if len(views) >= 6:
        _, K, dist, _, _ = cv2.calibrateCamera(
            [v.obj for v in views], [v.img for v in views], size, None, None,
            flags=cv2.CALIB_FIX_K3,
        )
        return K, dist
    f = (w / 2) / np.tan(np.radians(35))
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]]), np.zeros(5)


# -- calibration --------------------------------------------------------


def resolve_k3(mode: str, views: list[View], size: tuple[int, int]) -> tuple[int, str]:
    """calibrateCamera flags for k3, and why, for --k3 auto/free/fixed."""
    if mode == "free":
        return 0, "free (--k3 free)"
    if mode == "fixed":
        return cv2.CALIB_FIX_K3, "fixed at 0 (--k3 fixed)"
    reached = corners_reached(views, size)
    area = float(np.median([v.area for v in views]))
    if reached < K3_CORNERS:
        return cv2.CALIB_FIX_K3, (f"fixed at 0 (auto: only {reached} of 4 frame corners "
                                  "reached, too few to measure it)")
    # Corners alone are not enough: with a small board, k2 and k3 trade off
    # against each other and against fx. Two phone sessions of 3-4% boards
    # that both reached all 4 corners gave fx 959 and 855 with k3 free, and
    # predicted each other's frames no better than with it fixed.
    if area < MIN_BOARD_AREA:
        return cv2.CALIB_FIX_K3, (f"fixed at 0 (auto: board median {area * 100:.0f}% of "
                                  "the frame, too small to separate k3 from fx)")
    return 0, f"free (auto: {reached} of 4 frame corners, board median {area * 100:.0f}%)"


def calibrate(views: list[View], size: tuple[int, int], flags: int) -> dict:
    """Calibrate, drop per-view outliers once, recalibrate, and report."""

    def run(subset: list[View]):
        rms, K, dist, rvecs, tvecs, std, _, _ = cv2.calibrateCameraExtended(
            [v.obj for v in subset], [v.img for v in subset], size, None, None,
            flags=flags,
        )
        errors = []
        for v, r, t in zip(subset, rvecs, tvecs):
            proj, _ = cv2.projectPoints(v.obj, r, t, K, dist)
            residual = proj.reshape(-1, 2) - v.img.reshape(-1, 2)
            errors.append(float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))))
        # 1-sigma of fx, fy, cx, cy, and each view's tilt.
        extra = (std.reshape(-1)[:4], np.array([tilt_deg(r) for r in rvecs]))
        return rms, K, dist, np.array(errors), extra

    rms, K, dist, errors, extra = run(views)
    threshold = max(OUTLIER_FACTOR * float(np.median(errors)), OUTLIER_FLOOR_PX)
    keep = [i for i, e in enumerate(errors) if e <= threshold]
    rejected = [i for i in range(len(views)) if i not in keep]
    # Only prune if enough views survive to still constrain the model.
    if rejected and len(keep) >= max(8, len(views) // 2):
        rms, K, dist, errors, extra = run([views[i] for i in keep])
    else:
        keep, rejected = list(range(len(views))), []

    kept = [views[i] for i in keep]
    return {
        "rms": float(rms), "K": K, "dist": dist.reshape(-1),
        "errors": errors, "kept": keep, "rejected": rejected,
        "corners": int(sum(len(v.ids) for v in kept)),
        "coverage": float(grid_hits(kept, size, COVER_CELLS).mean()),
        "corners_reached": corners_reached(kept, size),
        "std": extra[0], "tilts": extra[1],
        "board_area": float(np.median([v.area for v in kept])),
    }


def verdict(rms: float) -> str:
    if rms < 0.5:
        return "good"
    if rms < 1.0:
        return "usable"
    return "poor -- recalibrate (blurred views? bent board? wrong --square-mm?)"


def problems(result: dict) -> list[str]:
    """Everything that makes this calibration less trustworthy than its RMS says.

    A low RMS only says the model fits the views it was given, not that those
    views pinned the model down.
    """
    out = []
    fx, fx_std = result["K"][0, 0], result["std"][0]
    tilted = int((result["tilts"] >= TILT_MIN_DEG).sum())
    if fx_std / fx > FX_STD_WARN:
        out.append(f"focal length uncertain: fx {fx:.0f} +/- {fx_std:.0f} px "
                   f"({fx_std / fx * 100:.1f}%, want < {FX_STD_WARN * 100:.0f}%), so "
                   "metric distances carry the same error")
    if tilted < MIN_TILTED:
        out.append(f"only {tilted} views tilted {TILT_MIN_DEG:.0f}+ deg (want "
                   f"{MIN_TILTED}+) -- tilt the board 30-45 deg, both ways")
    if result["board_area"] < MIN_BOARD_AREA:
        out.append(f"board small in frame: median {result['board_area'] * 100:.0f}% of "
                   f"the frame (want {MIN_BOARD_AREA * 100:.0f}%+) -- come closer or "
                   "print a bigger board; a small board hides the perspective that "
                   "fixes the focal length")
    if result["corners_reached"] < 4:
        out.append(f"board reached {result['corners_reached']} of 4 frame corners -- "
                   "put the board edge right at each frame corner (guided boxes 6-13)")
    elif result["coverage"] < 0.75:
        out.append("corners never reached large parts of the frame -- "
                   "cover the red areas")
    return out


def fov_deg(focal: float, extent: int) -> float:
    return float(np.degrees(2 * np.arctan(extent / (2 * focal))))


def report(result: dict, size: tuple[int, int], n_views: int) -> None:
    K, dist = result["K"], result["dist"]
    w, h = size
    dropped = f" (dropped outlier views {result['rejected']})" if result["rejected"] else ""
    print(f"\nviews        : {len(result['kept'])} used of {n_views}{dropped}, "
          f"{result['corners']} corners")
    print(f"RMS error    : {result['rms']:.3f} px  -> {verdict(result['rms'])}")
    print(f"per view     : mean {result['errors'].mean():.3f}  "
          f"max {result['errors'].max():.3f} px")
    sd = result["std"]
    print(f"fx fy        : {K[0, 0]:.1f} +/- {sd[0]:.1f}   {K[1, 1]:.1f} +/- {sd[1]:.1f}")
    print(f"cx cy        : {K[0, 2]:.1f} +/- {sd[2]:.1f}   {K[1, 2]:.1f} +/- {sd[3]:.1f}   "
          f"(image centre {w / 2:.0f} {h / 2:.0f})")
    print(f"dist         : {np.array2string(dist, precision=4, suppress_small=True)}")
    print(f"k3           : {result['k3_reason']}")
    print(f"board size   : median {result['board_area'] * 100:.0f}% of the frame")
    print(f"FOV          : {fov_deg(K[0, 0], w):.1f} x {fov_deg(K[1, 1], h):.1f} deg")
    tilts = result["tilts"]
    print(f"tilt         : {int((tilts >= TILT_MIN_DEG).sum())} of {len(tilts)} views "
          f"at {TILT_MIN_DEG:.0f}+ deg (max {tilts.max():.0f})")
    print(f"coverage     : {result['coverage'] * 100:.0f}% of the frame, "
          f"{result['corners_reached']} of 4 frame corners")
    for problem in problems(result):
        print(f"WARNING: {problem}")


# -- config i/o ---------------------------------------------------------


def write_config(path: Path, *, name: str, result: dict, size: tuple[int, int],
                 source: str, fourcc: str | None, args: argparse.Namespace,
                 n_views: int, measured_fps: float | None = None,
                 native_size: tuple[int, int] | None = None) -> None:
    K, dist = result["K"], result["dist"]
    w, h = size
    data = {
        "name": name,
        "calibrated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "image_width": w,
        "image_height": h,
        "source": source,
        # DirectShow reports the format it hands OpenCV, not what crossed the
        # USB cable, so YUY2 alongside ~30 fps at 720p still means MJPG.
        "fourcc_requested": args.fourcc if fourcc else None,
        "fourcc_reported": fourcc,
        "measured_fps": None if measured_fps is None else round(measured_fps, 1),
        # A downscaled stream: the recorder must deliver the same downscale.
        "source_size": None if native_size is None or native_size == size
        else list(native_size),
        "notes": args.notes,
        "model": "opencv_pinhole_k1k2p1p2k3",
        "camera_matrix": [[round(float(v), 4) for v in row] for row in K],
        "dist_coeffs": [round(float(v), 8) for v in dist],
        "rms_reprojection_px": round(result["rms"], 4),
        "intrinsics_std_px": {
            k: round(float(v), 3) for k, v in zip(("fx", "fy", "cx", "cy"), result["std"])
        },
        "per_view_error_px": {
            "mean": round(float(result["errors"].mean()), 4),
            "max": round(float(result["errors"].max()), 4),
        },
        "views_used": len(result["kept"]),
        "views_rejected": len(result["rejected"]),
        "views_captured": n_views,
        "corners_used": result["corners"],
        "frame_coverage": round(result["coverage"], 3),
        "frame_corners_reached": result["corners_reached"],
        "views_tilted": int((result["tilts"] >= TILT_MIN_DEG).sum()),
        "max_tilt_deg": round(float(result["tilts"].max()), 1),
        "board_area_median": round(result["board_area"], 3),
        "hfov_deg": round(fov_deg(K[0, 0], w), 2),
        "vfov_deg": round(fov_deg(K[1, 1], h), 2),
        "board": {
            "type": "charuco",
            "squares_x": args.squares_x,
            "squares_y": args.squares_y,
            "square_mm": args.square_mm,
            "marker_mm": round(args.marker_mm, 3),
            "dictionary": args.dict,
            "legacy_pattern": args.legacy,
        },
        "k3": args.k3_reason,
        "opencv_version": cv2.__version__,
    }
    header = (
        f"# Camera intrinsics for '{name}', written by tools/calibrate_camera.py.\n"
        f"# Valid ONLY at {w}x{h}. Recording at any other resolution needs a new\n"
        f"# calibration. Load with load_calibration() from the same tool.\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    # Block style for the warnings, one per line, so they read as sentences.
    warnings = yaml.safe_dump({"warnings": problems(result)}, sort_keys=False,
                              default_flow_style=False, width=1000)
    path.write_text(
        header + yaml.safe_dump(data, sort_keys=False, default_flow_style=None) + warnings,
        encoding="utf-8",
    )


def load_calibration(
    path: str | Path, image_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read (camera_matrix, dist_coeffs) from a camera YAML.

    Pass the (width, height) actually being captured: a mismatch raises
    rather than silently producing wrong geometry.
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if data.get("rms_reprojection_px") is None:
        raise ValueError(f"{path} holds no calibration (rms_reprojection_px is empty)")
    if image_size is not None:
        calibrated = (data["image_width"], data["image_height"])
        if tuple(image_size) != calibrated:
            raise ValueError(
                f"{path} was calibrated at {calibrated[0]}x{calibrated[1]}, "
                f"frames are {image_size[0]}x{image_size[1]} -- recalibrate"
            )
    K = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    return K, dist


# -- guided targets -----------------------------------------------------

# A view matches its target box when the board's outline lies within this
# share of the image diagonal of the box, corner for corner (~70 px at 720p).
MATCH_TOL = 0.05
# Where the corner and edge targets put the board's edge, as a share of the
# frame: negative is just inside it. Not past it -- a board pushed past the
# edge loses the markers around its outermost corners, and a ChArUco corner
# without its markers is not detected at all, so the corners nearest the
# frame corner would be exactly the ones missing.
OVERSHOOT = -0.01


@dataclass
class Target:
    label: str
    quad: np.ndarray       # (4, 2) image px


def _rotation(tx: float, ty: float, roll: float) -> np.ndarray:
    rx, ry, rz = np.radians([tx, ty, roll])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _describe(where: str, width: float, tx: float, ty: float, roll: float) -> str:
    parts = [where, "near" if width >= 0.4 else "mid" if width >= 0.3 else "farther"]
    if ty:
        parts.append(f"{'left' if ty > 0 else 'right'} edge {abs(ty):.0f} deg away")
    if tx:
        parts.append(f"{'bottom' if tx > 0 else 'top'} edge {abs(tx):.0f} deg away")
    if roll:
        parts.append(f"turned {abs(roll):.0f} deg {'clockwise' if roll > 0 else 'anticlockwise'}")
    if where in ("top-left", "top-right", "bottom-left", "bottom-right",
                 "left edge", "right edge", "top edge", "bottom edge"):
        parts.append("board edge right at the frame edge")
    return ", ".join(parts)


def make_targets(board: cv2.aruco.CharucoBoard, size: tuple[int, int],
                 scale: float = 1.0, count: int = 20) -> list[Target]:
    """The poses a good calibration needs, as boxes to line the board up with.

    Each box is the board's outline for one pose: its size sets the distance
    (big box = close), its shape sets the tilt, its place sets where in the
    frame. The set covers what the earlier runs lacked -- near, tilted views
    to pin the focal length, the four frame corners for distortion, and a few
    farther views for distance variety. Shapes come from a nominal 77 degree
    camera, so the tilts are approximate; that does not matter, as any
    tilted pose helps, and matching is done on the drawn box.
    """
    w, h = size
    f = 0.8 * w
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    sx, sy = board.getChessboardSize()
    s = board.getSquareLength()
    bw, bh = sx * s, sy * s
    outline = np.array([[0, 0, 0], [bw, 0, 0], [bw, bh, 0], [0, bh, 0]], np.float64)
    centre = np.array([bw / 2, bh / 2, 0.0])

    def quad_at_centre(width: float, tx: float, ty: float, roll: float) -> np.ndarray:
        R = _rotation(tx, ty, roll)
        d = f * bw / (width * scale * w)
        t = np.array([0.0, 0.0, d]) - R @ centre
        pts, _ = cv2.projectPoints(outline, cv2.Rodrigues(R)[0], t, K, None)
        return pts.reshape(4, 2)

    anchors = {  # (x, y) side of the frame each placement pushes against
        "top-left": (0, 0), "top-right": (1, 0), "bottom-left": (0, 1),
        "bottom-right": (1, 1), "left edge": (0, None), "right edge": (1, None),
        "top edge": (None, 0), "bottom edge": (None, 1),
    }
    # Priority order: --views takes the first N, so every prefix is a
    # balanced set. The first 20 hold everything a calibration needs -- near
    # tilted views for fx, all four frame corners flat and tilted for
    # distortion, and some distance variety; the rest add redundancy.
    specs = [
        # Centre, near: large views, the ones that pin the focal length.
        ("centre", 0.45, 0, 0, 0),
        ("centre", 0.45, 0, 35, 0), ("centre", 0.45, 0, -35, 0),
        ("centre", 0.45, 35, 0, 0), ("centre", 0.45, -35, 0, 0),
        # Frame corners, flat, then tilted: where distortion (and k3) lives.
        # Smaller boxes, so the first corner in from the board edge sits
        # close to the frame corner.
        ("top-left", 0.32, 0, 0, 0), ("top-right", 0.32, 0, 0, 0),
        ("bottom-left", 0.32, 0, 0, 0), ("bottom-right", 0.32, 0, 0, 0),
        ("top-left", 0.30, -25, 25, 0), ("top-right", 0.30, -25, -25, 0),
        ("bottom-left", 0.30, 25, 25, 0), ("bottom-right", 0.30, 25, -25, 0),
        # Centre, farther, tilted on the diagonals: distance variety.
        ("centre", 0.28, 25, 25, 0), ("centre", 0.28, -25, -25, 0),
        # In-plane rotation.
        ("centre", 0.40, 15, 0, 30),
        # Between centre and corners, steep.
        ((0.3, 0.35), 0.35, 0, 40, 0), ((0.7, 0.65), 0.35, -40, 0, 0),
        # Quadrants, farther.
        ((0.25, 0.3), 0.25, 20, 0, 0), ((0.75, 0.7), 0.25, -20, 0, 0),
        # ---- 20 above; the rest only when --views asks for more ----
        # Edge midpoints, tilted.
        ("left edge", 0.40, 0, 30, 0), ("right edge", 0.40, 0, -30, 0),
        ("top edge", 0.40, -30, 0, 0), ("bottom edge", 0.40, 30, 0, 0),
        ("centre", 0.28, 25, -25, 0),
        ("centre", 0.40, -15, 0, -30),
        ((0.7, 0.35), 0.35, 0, -40, 0), ((0.3, 0.65), 0.35, 40, 0, 0),
        ((0.75, 0.3), 0.25, 0, 20, 0), ((0.25, 0.7), 0.25, 0, -20, 0),
    ][:count]
    targets = []
    for where, width, tx, ty, roll in specs:
        quad = quad_at_centre(width, tx, ty, roll)
        lo, hi = quad.min(axis=0), quad.max(axis=0)
        # Place by moving the box in the image. A shifted perspective box is
        # still a pose a hand can reach, just slightly off-axis.
        if isinstance(where, tuple):
            shift = np.array([where[0] * w, where[1] * h]) - quad.mean(axis=0)
            label = "off-centre"
        elif where == "centre":
            shift = np.array([w / 2, h / 2]) - quad.mean(axis=0)
            label = "centre"
        else:
            ax, ay = anchors[where]
            label = where
            shift = np.array([w / 2, h / 2]) - quad.mean(axis=0)
            if ax is not None:
                shift[0] = (-OVERSHOOT * w - lo[0]) if ax == 0 else (w * (1 + OVERSHOOT) - hi[0])
            if ay is not None:
                shift[1] = (-OVERSHOOT * h - lo[1]) if ay == 0 else (h * (1 + OVERSHOOT) - hi[1])
        targets.append(Target(_describe(label, width, tx, ty, roll), quad + shift))
    return targets


def quad_error(outline: np.ndarray, quad: np.ndarray) -> float:
    """Mean corner distance between two quads, over every corner ordering.

    Matching the image quad is matching the pose, whichever way up the
    board is held, so all 8 orderings (4 rotations x 2 directions) count.
    """
    best = np.inf
    for seq in (outline, outline[::-1]):
        for k in range(4):
            best = min(best, float(np.linalg.norm(np.roll(seq, k, axis=0) - quad,
                                                  axis=1).mean()))
    return best


def draw_target(frame: np.ndarray, target: Target, outline: np.ndarray | None,
                matched: bool) -> None:
    """The box to fill, the board's current outline, and arrows between them."""
    colour = (0, 255, 0) if matched else (255, 255, 0)
    cv2.polylines(frame, [target.quad.astype(np.int32).reshape(-1, 1, 2)], True, colour, 3,
                  cv2.LINE_AA)
    if outline is None:
        return
    cv2.polylines(frame, [outline.astype(np.int32).reshape(-1, 1, 2)], True,
                  (0, 200, 255), 2, cv2.LINE_AA)
    if matched:
        return
    # Arrow from each board corner to the box corner it should move to.
    best, pairs = np.inf, None
    for seq in (outline, outline[::-1]):
        for k in range(4):
            cand = np.roll(seq, k, axis=0)
            err = float(np.linalg.norm(cand - target.quad, axis=1).mean())
            if err < best:
                best, pairs = err, cand
    for a, b in zip(pairs, target.quad):
        cv2.arrowedLine(frame, tuple(int(v) for v in a), tuple(int(v) for v in b),
                        (0, 200, 255), 2, cv2.LINE_AA, tipLength=0.08)


# -- live capture -------------------------------------------------------


def load_video_source():
    """Import src/io/video_source.py by path.

    `src/io` cannot be imported as a package: `io` is a stdlib module and
    Python finds that one first.
    """
    path = ROOT / "src" / "io" / "video_source.py"
    spec = importlib.util.spec_from_file_location("video_source", path)
    module = importlib.util.module_from_spec(spec)
    # @dataclass looks its module up in sys.modules while the class is built.
    sys.modules["video_source"] = module
    spec.loader.exec_module(module)
    return module.VideoSource


def open_source(args: argparse.Namespace):
    """Start capture and check the stream matches what will be recorded.

    Returns (source, subscription, first frame, negotiated info, measured fps).
    Capture goes through VideoSource -- the same class the recorder will use
    -- so a phone stream gets its latest-frame-only queue (a wifi backlog
    would otherwise make the preview lag seconds behind the board) and its
    reconnect on dropouts.
    """
    src: int | str = int(args.camera) if args.camera.isdigit() else args.camera
    VideoSource = load_video_source()
    source = VideoSource(src=src, width=args.width, height=args.height,
                         fps=args.fps, fourcc=args.fourcc)
    sub = source.subscribe("calibrate", maxsize=1)
    source.start(wait=10.0 if isinstance(src, str) else 5.0)
    first = sub.read_latest(timeout=5.0)
    if first is None:
        source.stop()
        if isinstance(src, str):
            sys.exit(f"ERROR: no frames from {src}. Is the phone app streaming, on "
                     "the same wifi, and does the URL open in the laptop's browser?")
        sys.exit(f"ERROR: camera {src} produced no frames")

    # Measured from frame timestamps: what arrives, not what the driver claims.
    stamps = [first.timestamp]
    t0 = time.perf_counter()
    while len(stamps) < 31 and time.perf_counter() - t0 < 3.0:
        f = sub.read(timeout=1.0)
        if f is not None:
            stamps.append(f.timestamp)
    measured_fps = (len(stamps) - 1) / max(stamps[-1] - stamps[0], 1e-6)
    return source, sub, first.image, dict(source.negotiated), measured_fps


def fit_frame(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Downscale a larger stream to the calibration size, if needed.

    Only a uniform scale is allowed: it scales fx, fy, cx, cy and leaves the
    distortion model valid. Cropping or stretching would change the camera.
    """
    if (image.shape[1], image.shape[0]) == size:
        return image
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def shade_uncovered(frame: np.ndarray, views: list[View], cells: int = 8) -> None:
    """Tint red every grid cell no captured corner has landed in yet."""
    if not views:
        return  # all red says nothing, and hides the board being lined up
    h, w = frame.shape[:2]
    hit = grid_hits(views, (w, h), cells)
    shade = frame.copy()
    for cy, cx in zip(*np.nonzero(~hit)):
        cv2.rectangle(shade, (cx * w // cells, cy * h // cells),
                      ((cx + 1) * w // cells, (cy + 1) * h // cells), (0, 0, 160), -1)
    cv2.addWeighted(shade, 0.25, frame, 0.75, 0, dst=frame)


def run_live(args: argparse.Namespace, board: cv2.aruco.CharucoBoard,
             out_path: Path) -> None:
    source, sub, frame, negotiated, measured_fps = open_source(args)
    native = (frame.shape[1], frame.shape[0])
    size = (args.width, args.height)
    fourcc = negotiated.get("fourcc")
    network = not args.camera.isdigit()
    print(f"camera {args.camera}: {native[0]}x{native[1]}  fourcc reported={fourcc}  "
          f"measured {measured_fps:.1f} fps")
    if measured_fps < 20:
        print("WARNING: under 20 fps -- " + (
            "weak wifi or a busy phone; a laptop hotspot on 5 GHz helps. "
            if network else "the camera may really be on uncompressed YUY2. ")
            + "Slow frames blur, so hold the board very still.")
    if native != size:
        # Worse than not calibrating: the YAML would look valid and every
        # measurement made at the real recording resolution would be off.
        same_shape = abs(native[0] / native[1] - size[0] / size[1]) < 0.01
        if not (args.resize and same_shape and native[0] > size[0]):
            source.stop()
            if not same_shape:
                why = ("a different aspect ratio -- portrait phone? Turn it to "
                       "landscape, and set 16:9 in the app")
            else:
                why = "pass --resize to downscale it, as the recorder will"
            sys.exit(f"ERROR: asked for {size[0]}x{size[1]}, source delivered "
                     f"{native[0]}x{native[1]}: {why}. Best is to set "
                     f"{size[0]}x{size[1]} in the phone app.")
        print(f"downscaling {native[0]}x{native[1]} -> {size[0]}x{size[1]}; "
              "the recorder must downscale the same way")
    w, h = size

    detector = make_detector(board)
    # One folder per session, so --from-dir can pick sessions rather than
    # silently mixing a bad earlier run into a good later one.
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    frames_dir = FRAMES_DIR / args.name / stamp
    frames_dir.mkdir(parents=True, exist_ok=True)
    diagonal = float(np.hypot(w, h))

    views: list[View] = []
    outlines: list[np.ndarray] = []
    saved: list[Path] = []
    auto = True
    prev: View | None = None
    still = 0
    flash_until = 0.0
    result: dict | None = None
    undistort = False
    maps = None
    hint, hint_at, hinted_layout = "", 0.0, None
    calibrated_from: list[Path] = []
    K_live, dist_live = provisional_intrinsics(views, size)
    tilts: list[float] = []
    guided = not args.free
    targets = make_targets(board, size, args.target_scale, args.views)
    if guided and len(targets) < args.views:
        print(f"guided mode has {len(targets)} boxes; --views {args.views} is capped "
              "there (--free has no cap)")
    ti = 0                            # index of the target being filled
    view_target: list[int | None] = []  # which target each view filled, if any

    print("space capture, a auto, u undo, c calibrate, d undistort, q quit")
    if guided:
        print(f"guided: {len(targets)} boxes -- move the board so its outline fills "
              "the box; it captures when matched and still. n skip box, p previous "
              "box, g free mode")
    window = f"calibrate '{args.name}' (ChArUco {args.squares_x}x{args.squares_y})"

    frame_ts = time.time()
    while True:
        latest = sub.read_latest(timeout=0.5)
        if latest is None:
            # Stream dropped: VideoSource is reconnecting. Keep the window
            # responsive and say so rather than freezing on the last frame.
            if time.time() - frame_ts > 2.0:
                print("waiting for camera ... (reconnecting)")
                frame_ts = time.time()
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            continue
        frame_ts = latest.timestamp
        # Frames are shared with the capture thread: copy before drawing.
        frame = fit_frame(latest.image, size).copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        view, raw, reason = detect_view(detector, board, gray)
        # Once a view has been captured the layout is proven right, so a
        # frame with markers but no corners is just a bad frame.
        if view is None and not views and markers_without_corners(raw):
            if time.perf_counter() - hint_at > 2.0:
                hint_at = time.perf_counter()
                guess = guess_layout(gray, raw[2], raw[3], args)
                new_hint = layout_hint(guess, args)
                # Say it once per suggested layout, not once per corner count.
                if new_hint and guess[:3] != hinted_layout:
                    print(new_hint)
                    hinted_layout = guess[:3]
                hint = new_hint
            reason = hint or reason

        moved = shared_displacement(view, prev) if view and prev else None
        still = still + 1 if moved is not None and moved < STILL_PX else 0
        prev = view
        outline = board_outline(view, board) if view else None
        nov = novelty(outline, outlines, diagonal) if view else 0.0
        tilt = view_tilt(view, K_live, dist_live) if view else None
        n_flat = sum(t < TILT_MIN_DEG for t in tilts)
        # Past MAX_FLAT flat views, another flat one adds nothing to the fit.
        flat_blocked = tilt is not None and tilt < TILT_MIN_DEG and n_flat >= MAX_FLAT
        target = targets[ti] if guided and ti < len(targets) and result is None else None
        err = quad_error(outline, target.quad) / diagonal \
            if target is not None and outline is not None else None
        matched = err is not None and err < MATCH_TOL

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

        if target is not None:
            # Guided: the box decides the pose, so novelty and the flat-view
            # cap are moot -- every box is a different, useful pose.
            auto_ok = auto and still >= STILL_FRAMES and matched
        else:
            auto_ok = (auto and still >= STILL_FRAMES and nov >= MIN_NOVELTY
                       and not flat_blocked and len(views) < args.views)
        capture = view is not None and (key == ord(" ") or auto_ok)
        if capture:
            path = frames_dir / f"{len(views) + 1:02d}.png"
            cv2.imwrite(str(path), frame)  # the clean frame, before any drawing
            views.append(view)
            outlines.append(outline)
            saved.append(path)
            K_live, dist_live = provisional_intrinsics(views, size)
            tilts = [view_tilt(v, K_live, dist_live) or 0.0 for v in views]
            still = 0
            flash_until = time.perf_counter() + 0.15
            filled = ti if target is not None and matched else None
            view_target.append(filled)
            label = f"box {ti + 1}/{len(targets)} done, " if filled is not None else ""
            print(f"captured {len(views)}  ({label}{len(view.ids)} corners, "
                  f"tilt {tilts[-1]:.0f} deg, size {view.area * 100:.0f}%)")
            if filled is not None:
                ti += 1

        if key == ord("a"):
            auto = not auto
            print(f"auto-capture {'on' if auto else 'off'}")
        if key == ord("n") and guided and ti < len(targets):
            print(f"skipped box {ti + 1}: {targets[ti].label}")
            ti += 1
        if key == ord("p") and guided and ti > 0:
            ti -= 1
            print(f"back to box {ti + 1}: {targets[ti].label}")
        if key == ord("g"):
            guided = not guided
            print("guided mode" if guided else "free mode: capture any new pose")
        if key == ord("u") and views:
            if view_target.pop() is not None:
                ti -= 1  # that view filled the box before this one
            views.pop()
            outlines.pop()
            saved.pop().unlink(missing_ok=True)
            K_live, dist_live = provisional_intrinsics(views, size)
            tilts = [view_tilt(v, K_live, dist_live) or 0.0 for v in views]
            print(f"undone, {len(views)} views")
        if key == ord("d") and result is not None:
            undistort = not undistort

        target_reached = result is None and (
            ti >= len(targets) if guided else len(views) >= args.views)
        if target_reached or key == ord("c"):
            if len(views) < args.min_views:
                print(f"need at least {args.min_views} views, have {len(views)}")
            elif result is not None and saved == calibrated_from:
                print("no new views since the last calibration -- space to add "
                      "some (frame corners, steeper tilts), then c")
            else:
                calibrated_from = list(saved)
                print("calibrating ...")
                flags, args.k3_reason = resolve_k3(args.k3, views, size)
                result = calibrate(views, size, flags)
                result["k3_reason"] = args.k3_reason
                report(result, size, len(views))
                write_config(out_path, name=args.name, result=result, size=size,
                             source=str(args.camera), fourcc=fourcc, args=args,
                             n_views=len(views), measured_fps=measured_fps,
                             native_size=native)
                print(f"wrote {out_path}")
                print("press d to check the undistorted view -- straight edges "
                      "should look straight, especially near the frame border")
                auto = False
                maps = cv2.initUndistortRectifyMap(
                    result["K"], result["dist"], None, result["K"], size, cv2.CV_16SC2,
                )

        # -- drawing: frame is already saved, so draw on it directly
        if undistort and maps is not None:
            frame = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)
        else:
            if target is None:
                shade_uncovered(frame, views)  # the boxes already cover the frame
            corners, ids, marker_corners, marker_ids = raw
            if marker_ids is not None and len(marker_ids):
                cv2.aruco.drawDetectedMarkers(frame, marker_corners)
            if ids is not None and len(ids):
                colour = (0, 255, 0) if view else (0, 0, 255)
                # OpenCV 5 returns corners as (N, 2); drawing wants (N, 1, 2).
                cv2.aruco.drawDetectedCornersCharuco(
                    frame, corners.reshape(-1, 1, 2), None, colour)
            if target is not None:
                draw_target(frame, target, outline, matched)
        if time.perf_counter() < flash_until:
            frame = 255 - frame

        if target is not None:
            if view is None:
                status, colour = reason, (0, 0, 255)
            elif not matched:
                status = (f"line the board up with the box -- {err * 100:.0f}% off, "
                          f"need under {MATCH_TOL * 100:.0f}%")
                colour = (0, 200, 255)
            elif still < STILL_FRAMES:
                status, colour = "matched -- hold still ...", (0, 255, 0)
            else:
                status = "matched -- " + ("capturing" if auto else "press space")
                colour = (0, 255, 0)
        elif result is not None:
            status = (f"RMS {result['rms']:.2f}px ({verdict(result['rms']).split()[0]}"
                      f"{', see warnings' if problems(result) else ''})"
                      f"   {'UNDISTORTED' if undistort else 'raw'}"
                      "   d toggle, space + c to add views and recalibrate")
            colour = (0, 255, 0) if result["rms"] < 1.0 else (0, 0, 255)
        elif view is None:
            status, colour = reason, (0, 0, 255)
        elif still < STILL_FRAMES:
            status, colour = "hold still ...", (0, 200, 255)
        elif flat_blocked:
            status, colour = (f"enough flat views -- tilt the board 30-45 deg "
                              f"(now {tilt:.0f})"), (0, 200, 255)
        elif nov < MIN_NOVELTY:
            status, colour = "move or tilt to a new pose -- try the red areas", (0, 200, 255)
        else:
            status = "ready -- " + ("capturing" if auto else "press space")
            colour = (0, 255, 0)
            if view.area < MIN_BOARD_AREA:
                status += f" (board small -- come closer, aim {MIN_BOARD_AREA * 100:.0f}%+)"

        corner_count = f"   corners {len(view.ids)}" if view else ""
        tilt_text = f"   tilt {tilt:.0f}" if tilt is not None else ""
        tilt_text += f"   size {view.area * 100:.0f}%" if view else ""
        n_tilted = len(tilts) - sum(t < TILT_MIN_DEG for t in tilts)
        cv2.putText(frame, f"views {len(views)}/{args.views}   "
                    f"tilted {n_tilted}/{MIN_TILTED}   "
                    f"auto {'on' if auto else 'off'}{corner_count}{tilt_text}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, status, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)
        if target is not None:
            cv2.putText(frame, f"box {ti + 1}/{len(targets)}: {target.label}",
                        (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow(window, frame)

    source.stop()
    cv2.destroyAllWindows()
    if result is None:
        print(f"quit without calibrating; {len(views)} frames kept in {frames_dir}")


# -- offline ------------------------------------------------------------


def run_offline(args: argparse.Namespace, board: cv2.aruco.CharucoBoard,
                out_path: Path) -> None:
    folders = [Path(f) for f in args.from_dir]
    paths = []
    for folder in folders:
        if not folder.is_dir():
            sys.exit(f"ERROR: {folder} is not a directory")
        paths += sorted(p for p in folder.iterdir()
                        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
    if not paths:
        sys.exit(f"ERROR: no images in {', '.join(map(str, folders))}")

    detector = make_detector(board)
    views: list[View] = []
    size: tuple[int, int] | None = None
    mismatch = None  # first frame with markers but no corners, for the layout hint
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"skip {p.name}: unreadable")
            continue
        this_size = (img.shape[1], img.shape[0])
        if size is None:
            size = this_size
        elif this_size != size:
            sys.exit(f"ERROR: {p.name} is {this_size[0]}x{this_size[1]}, others are "
                     f"{size[0]}x{size[1]} -- one calibration per resolution")
        view, raw, reason = detect_view(detector, board, img)
        if view is None:
            print(f"skip {p.parent.name}/{p.name}: {reason}")
            if mismatch is None and markers_without_corners(raw):
                mismatch = (img, raw)
            continue
        views.append(view)

    if size is None:
        sys.exit("ERROR: no readable images")
    # Any usable view proves the layout, so only guess when none worked.
    if not views and mismatch is not None:
        hint = layout_hint(guess_layout(mismatch[0], mismatch[1][2], mismatch[1][3], args),
                           args)
        if hint:
            print(hint)
    print(f"{len(views)} of {len(paths)} images usable at {size[0]}x{size[1]}")
    if len(views) < args.min_views:
        sys.exit(f"ERROR: need at least {args.min_views} usable views")

    flags, args.k3_reason = resolve_k3(args.k3, views, size)
    result = calibrate(views, size, flags)
    result["k3_reason"] = args.k3_reason
    report(result, size, len(views))
    write_config(out_path, name=args.name, result=result, size=size,
                 source="images:" + ",".join(f.as_posix() for f in folders),
                 fourcc=None, args=args,
                 n_views=len(views))
    print(f"wrote {out_path}")


# -- main ---------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--name", help="camera name; output is configs/camera_<name>.yaml")
    parser.add_argument("--camera", default="0",
                        help="camera index, or a stream URL for a phone (default 0)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--resize", action="store_true",
                        help="accept a larger stream of the same aspect ratio and "
                             "downscale it to --width x --height (a phone at 1080p)")
    parser.add_argument("--notes", default="",
                        help="free text saved in the YAML -- for a phone, the app and "
                             "the locked settings, e.g. 'IP Webcam, focus locked, "
                             "stabilization off, 1x main lens'")
    parser.add_argument("--squares-x", type=int, default=10, help="squares across (default 10)")
    parser.add_argument("--squares-y", type=int, default=7, help="squares down (default 7)")
    parser.add_argument("--square-mm", type=float, default=25.0,
                        help="MEASURED printed square size in mm (default 25)")
    parser.add_argument("--marker-mm", type=float,
                        help=f"marker size in mm (default {MARKER_RATIO} x --square-mm, "
                             "which stays right if the print came out scaled)")
    parser.add_argument("--dict", default="DICT_5X5_100", choices=sorted(DICTIONARIES),
                        help="board marker dictionary; never DICT_4X4_50, which the "
                             "object markers use (default DICT_5X5_100)")
    parser.add_argument("--legacy", action="store_true",
                        help="board uses the pre-4.6 OpenCV layout (only matters for "
                             "an even --squares-y; the live hint says when)")
    parser.add_argument("--views", type=int, default=20,
                        help="views per session: guided boxes, up to 30, or free "
                             "captures (default 20)")
    parser.add_argument("--min-views", type=int, default=10)
    parser.add_argument("--free", action="store_true",
                        help="no target boxes: capture any new pose (the old mode; "
                             "--views then sets the count)")
    parser.add_argument("--target-scale", type=float, default=1.0,
                        help="shrink (<1) or grow (>1) every box. Smaller boxes let "
                             "the board sit farther away -- use 0.7 if the phone "
                             "cannot focus on the board up close")
    parser.add_argument("--k3", choices=("auto", "free", "fixed"), default="auto",
                        help="the 6th-power distortion term. auto (default) frees it "
                             f"only when views reach {K3_CORNERS}+ frame corners; "
                             "without them it overfits and drags fx with it")
    parser.add_argument("--from-dir", nargs="+", metavar="DIR",
                        help="calibrate from saved images instead of live; several "
                             "session folders combine into one calibration")
    parser.add_argument("--out", help="override the output path")
    parser.add_argument("--force", action="store_true", help="overwrite an existing config")
    parser.add_argument("--make-board", metavar="PNG",
                        help="write a printable ChArUco board and exit")
    parser.add_argument("--dpi", type=int, default=300, help="board DPI (default 300)")
    args = parser.parse_args()

    if args.marker_mm is None:
        args.marker_mm = args.square_mm * MARKER_RATIO
    if not 0 < args.marker_mm < args.square_mm:
        parser.error("--marker-mm must be smaller than --square-mm")
    n_markers = (args.squares_x * args.squares_y) // 2
    if n_markers > cv2.aruco.getPredefinedDictionary(DICTIONARIES[args.dict]).bytesList.shape[0]:
        parser.error(f"{args.dict} has too few markers for a "
                     f"{args.squares_x}x{args.squares_y} board")
    if not args.make_board:
        if not args.name:
            parser.error("--name is required")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.name):
            parser.error("--name must be lowercase letters, digits, _ or -")
    return args


def main() -> None:
    args = parse_args()
    board = make_charuco(args)
    if args.make_board:
        write_board(Path(args.make_board), board, args)
        return

    out_path = Path(args.out) if args.out else CONFIGS_DIR / f"camera_{args.name}.yaml"
    if out_path.exists() and not args.force:
        sys.exit(f"ERROR: {out_path} exists; pass --force to replace it")

    print(f"board: ChArUco {args.squares_x}x{args.squares_y} squares, {args.dict}, "
          f"square {args.square_mm:g} mm, marker {args.marker_mm:.1f} mm")
    if args.from_dir:
        run_offline(args, board, out_path)
    else:
        run_live(args, board, out_path)


if __name__ == "__main__":
    main()
