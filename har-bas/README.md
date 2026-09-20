# har-bas — inference/demo environment

Infrastructure for the on-board HAR demo target. This machine runs **inference
only**; training and labeling tooling lives on the training machine.

## Environment

The venv lives **outside** this tree, at `C:\Users\nandu\venvs\har-infer`, because
the project sits in a OneDrive folder and a CUDA torch install is several GB of
binaries that OneDrive would try to sync. Reproducibility comes from
`requirements-infer.txt`, not from the venv location.

```
C:\Users\nandu\venvs\har-infer\Scripts\activate
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

## Manual test

`src/perception/marker_test.py` is a visual test — run it and hold up a marker
from `markers/`. Camera intrinsics are an uncalibrated pinhole guess, so pose
axes are indicative, not metric.
