# har-bas — inference/demo environment

Infrastructure for the on-board HAR demo target. This machine runs **inference
only**; training and labeling tooling lives on the training machine.

## Environment

### Windows / CUDA (training/demo target)

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

### macOS / Apple Silicon (dev port)

No OneDrive constraint here, so the venv lives at `har-bas/.venv` (git-ignored).
Use `requirements-infer-macos.txt`, not `requirements-infer.txt` -- several
pins differ from the Windows/CUDA set (see the comments in that file for why):
torch has no CUDA build to fetch, mediapipe is pinned to the 0.10.x line
because 1.x crashes PoseLandmarker on macOS, and opencv/numpy follow suit.

```
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements-infer-macos.txt
pip install --no-deps pyttsx3==2.99
```

Detection and pose both run happily on CPU, and detection additionally gets
Apple Silicon's Metal backend for free -- code picks `device="mps"` via
`torch.backends.mps.is_available()` wherever it previously checked
`torch.cuda.is_available()`, falling back to CPU otherwise. No code changes
are needed to run the exact same scripts as the Windows/CUDA target.

Verified: torch 2.14.0, `torch.backends.mps.is_available() == True`, Apple M2.

Camera access needs a one-time macOS permission grant: the first run of any
script that opens `cv2.VideoCapture` triggers a system prompt for whichever
app owns the terminal (Terminal.app, iTerm, VS Code, ...); if it's denied or
missed, re-enable it under System Settings -> Privacy & Security -> Camera.

## Layout

| Path | Purpose |
| --- | --- |
| `configs/` | `step_graph.yaml`: the step list every other part reads |
| `markers/` | ArUco/AprilTag reference images — printable DICT_4X4_50 ids 0-3 |
| `models/detector/`, `models/pose/`, `models/interaction/` | weights, git-ignored |
| `src/perception/` | detection, pose, marker tracking |
| `src/fsm/` | step-sequencing state machine |
| `src/io/` | capture, disk writer, streamer, TTS, logging |
| `src/gui/` | monitoring dashboard |
| `sanity_checks/` | standalone benchmarks |
| `tests/` | unit tests: `python -m unittest discover -s tests -v` |
| `demo/` | example step scripts for `tools/run_fsm.py` |

Model weights are git-ignored. Re-fetch the pose bundle with:

```
curl -L -o models/pose/pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

and the hand model (used by the GUI overlays) with:

```
curl -L -o models/hands/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
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

## Step sequencer (demo pipeline)

`src/fsm/sequencer.py` is the real state machine. It takes "step X started"
observations and turns them into next-step prompts and skipped /
out-of-order / too-fast / stuck alerts. Until a model exists, the step
observations come from a script or from hotkeys (`src/fsm/sources.py`).

`tools/run_fsm.py` connects that to voice (`src/io/voice.py`) and the JSONL
run log (`src/io/event_log.py`, under `logs/runs/`):

```
python tools/run_fsm.py --video take.mp4                            # tap 0-5, saves take_steps.csv
python tools/run_fsm.py --video take.mp4 --script take_steps.csv    # deterministic replay
python tools/run_fsm.py --script demo/skip_s2.csv --mute            # no video, check a script
```

`tools/gui.py` is the monitoring window over the same session (`src/gui/`):
video with alert banner, next step, procedure checklist, alert history and
event log. Opening `take.mp4` picks up `take_steps.csv` beside it; with no
script, keys 0-5 drive it and are saved as the script when the video ends.
Every alert keeps the frame it fired on: a thumbnail on the alert card (click
to enlarge), a JPG in `logs/runs/<run>_alerts/`, and a `snapshot` line in the
run log pointing at it.

The video carries three overlay layers from `src/perception/perception.py`,
toggled with the Objects / Hands / Markers buttons (keys O, H, K): YOLO11m
boxes filtered to bottle / cup / glass / bowl, MediaPipe hand landmarks, and
ArUco markers labelled by what they are stuck on. They show what the
perception stack sees; they do not drive the steps yet. Hands rather than
body pose because the takes are framed on the table: BlazePose needs the
upper body in view and drew wrong skeletons on these takes. Models load in
the background for ~6 s after launch; `--no-overlays` skips them.

Under the video, the timeline bar shows one segment per step as the run
unfolds, red ticks for alerts, amber ticks for questions; hover for details.

**Upside down** (F) turns the feed 180 degrees before perception runs, so it
tests how perception copes rather than just drawing upside down. Hands and
markers do not care; YOLO does (flipped take1: cup found on 0/215 frames
instead of 215/215). So detection is relative to the payload, not the
camera: the fixed table / container markers say which way is up, and YOLO
sees the frame levelled to them. With that, flipped and 90-degree-rotated
take1 detect exactly as upright does.

**Knowing when it is unsure** is scripted like the steps. A script row `?s2`
makes it say "I can't see clearly. Is the cap off?" (each step's `question`
in `step_graph.yaml`) and wait. Y or the card's Yes button counts the step
from when it asked; N repeats the instruction. A `yes` / `no` row answers
from the script instead, and a question still open when the next step
arrives lapses unanswered. `takes/take1_unsure_steps.csv` is take1 with the
unscrew step asked rather than seen:

```
python tools/gui.py --video takes/take1.mp4 --script takes/take1_unsure_steps.csv
```

```
python tools/gui.py --video takes/take1.mp4    # or launch bare and Ctrl+O
python tools/gui.py --camera 0                 # live webcam + hotkeys
```

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
