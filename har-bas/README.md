# har-bas — inference/demo environment

Infrastructure for the on-board HAR demo target. This machine runs **inference
only**; training and labeling tooling lives on the training machine.

## Environment

The venv lives **outside** this tree, at `C:\Users\nandu\venvs\har-infer`, because
the project sits in a OneDrive folder and a CUDA torch install is several GB of
binaries that OneDrive would try to sync. Reproducibility comes from
`requirements-infer.txt`, not from the venv location.

```

SIH-2\Actus\venvs\har-infer\Scripts\activate
pip install -r requirements-infer.txt
```

Verified: torch 2.14.0+cu130, `torch.cuda.is_available() == True`,
RTX 4070 Laptop GPU (sm_89, 8 GB), driver CUDA 13.3.

## Layout

| Path | Purpose |
| --- | --- |
| `configs/` | `step_graph.yaml` (not yet written) |
| `markers/` | ArUco/AprilTag reference images — printable DICT_4X4_50 ids 0-3 |
| `models/detector/`, `models/pose/`, `models/interaction/` | weights, git-ignored |
| `src/perception/` | detection, pose, marker tracking |
| `src/fsm/` | step-sequencing state machine |
| `src/io/` | capture, disk writer, streamer, TTS, logging |
| `src/gui/` | monitoring dashboard |
| `sanity_checks/` | standalone benchmarks |

Model weights are git-ignored. Re-fetch the pose bundle with:

```
curl -L -o models/pose/pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

## Measured baseline

> **Stale:** the numbers below were measured against YOLOv8n. The detector is
> now YOLO11m, a much larger model, so these figures no longer apply — rerun
> the checks below and update this table.

Run from this directory. `end-to-end` is capped by the webcam (~24.5 FPS);
`compute` is the inference cost alone and is the real ceiling.

| Check | End-to-end | Compute ceiling | Per frame |
| --- | --- | --- | --- |
| `check_yolo.py` | 19.7 FPS | 67.7 FPS | 14.8 ms |
| `check_pose.py` | 24.2 FPS | 77.9 FPS | 12.8 ms |
| `check_combined.py` | 19.2 FPS | **28.2 FPS** | 35.4 ms |

Combined breakdown: YOLO 14.4 ms + Pose 14.2 ms + ArUco 6.8 ms.

> The webcam negotiates uncompressed YUY2 by default, which caps 720p at
> ~7 FPS. Every capture path here requests MJPG, which raises it to ~24.5 FPS.
> Keep that `CAP_PROP_FOURCC` call in any new capture code.

## Manual tests

Each opens a live window and is judged by eye, not by its numbers.

`src/perception/detect_test.py` draws YOLO11m's boxes and class labels on the
feed, and echoes what it recognises to the console once a second. The weights
are stock COCO, so it knows the 80 COCO classes and nothing else — lab
hardware reads as the nearest COCO class or not at all until the custom
detector is trained. `q`/`Esc` quits, `s` saves to `logs/`, `c` cycles the
confidence threshold.

`src/perception/pose_test.py` draws the 33 BlazePose landmarks as a skeleton,
left and right sides in different colours so mirrored tracking is visible at a
glance. Pose runs on CPU here, so its FPS is independent of the GPU. `q`/`Esc`
quits, `s` saves to `logs/`, `v` cycles the visibility threshold, `n` toggles
landmark index numbers.

`src/perception/combined_test.py` runs detection and pose on the same frame,
serially, the way the inference loop will. This is the one to watch for the
real end-to-end FPS; the HUD splits the per-frame cost between the two stages.
`d` and `p` toggle either layer when the overlays get busy, and `c`/`v`/`n`/`s`
behave as they do in the single-stage tests.

`src/perception/marker_test.py` draws each ArUco marker's ID and pose axes —
run it and hold up a marker from `markers/`, printed or on a phone. Stock
`DetectorParameters()` cannot read a marker a phone has inverted (dark mode,
or iOS Smart Invert), which is why this uses tuned parameters with
`detectInvertedMarker` on; `t` toggles back to stock to see the difference.
When nothing decodes, `r` outlines rejected candidates — a red quad means the
square was found and only the bits failed, no quad means focus, distance or a
cropped quiet zone. `b` shows the binarised view, `[`/`]` step exposure for a
screen that blows out.

Pass `--marker-size` with the marker's real width; the 5 cm default assumes a
printed marker, and a marker on a screen is whatever size it displays at.
Camera intrinsics are an uncalibrated pinhole guess, so pose axes and the
distance readout are indicative, not metric.
