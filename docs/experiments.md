# Experiment Log

Every detection / tracking experiment is recorded here as a new entry,
newest at the bottom. Numbers are measured on this project's data only.

---

## 2026-10-10 — 118575_yolo11m_botsort

**Goal:** first full-frame tracking baseline on the SoccerTrack v2 MOT clip,
in the same pixel coordinate space as the ground truth (no resizing of the
video, no cropping).

### Data

- Dataset: SoccerTrack v2, match 118575, panoramic full-pitch camera
- Clip: first 2 minutes of the MOT clip
  - video: `data/SoccerTrackv2/mot/118575.mp4`
  - ground truth: `data/SoccerTrackv2/mot/118575_gt.txt`
- Video properties: 4096 x 1080, 25 fps, 3000 frames, 120.0 s
- GT format (MOT, comma separated, 10 columns):
  `frame, id, x_left, y_top, width, height, -1, -1, -1, -1`
  (pixel units in the original 4096 x 1080 frame; columns 7-10 are always -1)
- GT frame numbers are **0-based** (0 ... 2999), covering all 3000 video frames

### Command

```
yolo track model=yolo11m.pt source=data/SoccerTrackv2/mot/118575.mp4
  tracker=botsort.yaml classes=0 imgsz=2560 half=True device=0
  save=True save_txt=True save_conf=True
  project=outputs/tracking name=118575_yolo11m_botsort exist_ok=True
```

Settings: COCO-pretrained YOLO11m, class 0 (person) only, input size 2560
(letterboxed to 704 x 2560 by Ultralytics), FP16, default BoT-SORT
configuration (`botsort.yaml` shipped with Ultralytics), default detection
confidence threshold (lowest saved confidence: 0.10).

Note: with Ultralytics 8.4 a relative `project=` path is placed under
`runs/detect/`. The run was written to
`runs/detect/outputs/tracking/118575_yolo11m_botsort` and moved afterwards
to `outputs/tracking/118575_yolo11m_botsort`. Future runs should pass an
absolute `project=` path.

### Environment

- GPU: NVIDIA GeForce RTX 4080 Laptop GPU (CUDA available)
- PyTorch 2.6.0+cu124, Ultralytics 8.4.171, Python 3.12

### Runtime

- Wall-clock time: 598 s for 3000 frames → **5.0 fps** end to end
  (includes model loading, tracking and writing the annotated video)
- Ultralytics per-frame timing: 14.5 ms preprocess, 63.6 ms inference,
  2.9 ms postprocess (~81 ms → ~12 fps for the model alone)

### Outputs (not versioned)

- Annotated video: `outputs/tracking/118575_yolo11m_botsort/118575.avi`
  (4096 x 1080, 25 fps, 3000 frames)
- Labels: `outputs/tracking/118575_yolo11m_botsort/labels/`, 3000 files
  (`118575_<frame>.txt`, no empty files)
- Label row format (space separated, 7 columns):
  `class x_center y_center width height confidence track_id`
  — box coordinates **normalised** to [0, 1] by image width / height
- Label frame numbers are **1-based** (1 ... 3000):
  label file `118575_N.txt` corresponds to GT frame `N - 1`

These outputs, the evaluation folder below and `tracks.csv` were deleted on
2026-10-16, when the ROI run replaced this run; the numbers in this entry
are kept as the before-ROI baseline (`tracks.csv` remains in git history).

Versioned evidence in `docs/`:

- `tracking_118575_frame.jpg` — annotated frame at t = 60 s from the output video
- `tracking_118575_log.txt` — full terminal output of the run

### Summary numbers

| Measure | Tracker output | Ground truth |
|---|---|---|
| Rows (boxes) | 74,437 | 66,000 |
| Unique IDs | 575 | 22 |
| Boxes per frame (mean) | 24.81 | 22.00 |

Observations:

- The tracker finds ~2.8 more boxes per frame than the GT, which labels only
  the 22 players (likely extra: referees, substitutes, staff, spectators).
- 575 track IDs for 22 players means heavy ID fragmentation (largest raw
  ID: 4774). No matching against the GT has been done yet; these are raw
  counts, not MOT metrics.

### Performance evaluation (with GT)

Run on 2026-10-11 with `src/evaluate_tracking.py` v1 (reads the label files
into pixel boxes, shifts frame numbers to the GT's 0-based numbering, own
implementation of CLEAR MOT and ID metrics; only the Hungarian algorithm
from scipy).
Outputs: `outputs/evaluation/118575_yolo11m_botsort/`
(`performance_metrics.json`, `performance_report.md`, `per_frame.csv`,
`alignment_check.jpg`).

- Matching: per frame, Hungarian assignment on cost 1 - IoU; a pair counts
  only if IoU >= 0.5.
- GT filter (MOT rule): rows with conf == 0 or a non-person class are
  excluded. In this GT, conf, class and visibility are all -1 (not
  annotated), so all 66,000 rows are evaluated.
- Self-test (GT evaluated against itself): Precision = Recall = MOTA =
  IDF1 = 1, IDSW = FP = FN = 0 — passed.
- `alignment_check.jpg` (frames 0, 1499, 2999): GT and predicted boxes lie
  on top of each other; no coordinate or frame offset.

| Measure | Value |
|---|---|
| GT boxes / predicted boxes | 66,000 / 74,437 |
| GT IDs / predicted IDs | 22 / 575 |
| TP / FP / FN | 57,263 / 17,174 / 8,737 |
| ID switches | 770 |
| Precision / Recall | 0.769 / 0.868 |
| MOTA | 0.596 |
| MOTP (mean IoU of matches) | 0.810 |
| IDF1 (IDTP = 24,339) | 0.347 |
| Mostly tracked / partially tracked / mostly lost | 18 / 4 / 0 |

Notes:

- Error budget of MOTA: FP 17,174 (64 %), FN 8,737 (33 %), IDSW 770 (3 %).
- FP split: 15,221 unmatched predictions have IoU <= 0.1 with every GT box
  (people not in the GT: referees, staff, people behind the goal); only
  1,953 overlap a GT box (box too loose / too tight, or a duplicate).
- 4,324 of the 8,737 FN have a prediction nearby (IoU > 0.1) that missed the
  0.5 threshold — mostly small, distant players where a few pixels of box
  error change the IoU a lot.
- IDF1 is far below MOTA: detection is reasonable, but identities fragment
  (575 IDs for 22 players), so one player rarely keeps one ID for long.

---

## 2026-10-16 — 118575_yolo11m_botsort_roi

**Goal:** remove people outside the pitch (substitutes, staff, spectators)
from tracking by restricting the model to a region of interest (ROI).
Same video, model and tracker as `118575_yolo11m_botsort`, so the only
change is the ROI.

### Method

`src/track.py` v2 reads the video frame by frame with OpenCV:

1. Pixels outside the ROI polygon are painted black (`cv2.fillPoly` mask).
   The video is **not cropped**, the frame stays 4096 x 1080, so every
   coordinate remains in the original pixel space of the ground truth.
2. `model.track(masked_frame, persist=True, ...)` runs detection and
   BoT-SORT on the masked frame.
3. A tracked box is dropped if its foot point (bottom centre,
   `x + w/2, y + h`) lies outside the polygon (`cv2.pointPolygonTest`).
   The mask alone can leave half-visible people on the polygon edge.
   The filter acts on the tracker output, so dropped boxes may still have
   used a track ID inside BoT-SORT.

### Settings

| Setting | Value |
|---|---|
| Video | `data/SoccerTrackv2/mot/118575.mp4` (4096 x 1080, 25 fps, 3000 frames) |
| ROI | `data/roi/118575_roi.json` (21 corners, original pixels) |
| Model | `yolo11m.pt` (COCO-pretrained), class 0 (person) |
| Tracker | `botsort.yaml` (Ultralytics default) |
| Input size | `imgsz=2560` |
| Precision | FP16 (`half=True`) |
| Confidence threshold | Ultralytics default |
| Environment | RTX 4080 Laptop GPU, PyTorch 2.6.0+cu124, Ultralytics 8.4.171 |

Ultralytics 8.4.171 prints `'half' is deprecated ... Use 'quantize'`;
the setting still takes effect in this version.

### Outputs

In `outputs/tracking/118575_yolo11m_botsort_roi/`:

- `labels/118575_<N>.txt`: 3000 files, N = 1 ... 3000, same format as the
  run without ROI (`class xc yc w h conf id`, normalised). The first rows
  of frame 1 match the earlier run to 6 digits; `evaluate_tracking.py`
  reads the folder unchanged.
- `tracks.csv`: `frame` (0-based), `track_id`, `x, y, w, h` (pixels,
  top-left), `conf` (versioned).
- `118575.mp4`: boxes drawn on the original (unmasked) frame, ROI outline
  semi-transparent (575 MB, not versioned; deleted on 2026-10-16 in favour
  of `118575_tracks.mp4`).
- `118575_tracks.mp4`: tracking video regenerated from this run's `tracks.csv` by `src/draw_boxes_video.py` (yellow ROI outline).
- `sample_frame.jpg`: frame 1500 of the output video.

The output of the run without ROI is kept for comparison.

### Summary numbers

| Measure | Without ROI | With ROI | Change | GT |
|---|---|---|---|---|
| Boxes | 74,437 | 62,565 | -11,872 (-15.9 %) | 66,000 |
| Unique IDs | 575 | 448 | -127 (-22.1 %) | 22 |
| Boxes per frame (mean) | 24.81 | 20.86 | -3.95 | 22.00 |
| Removed by foot-point filter | - | 2,895 | | |
| Runtime | 598 s (5.0 fps) | 601.5 s (5.0 fps) | | |

Observations:

- Boxes per frame dropped below the 22 GT players. The ROI removes people
  outside the pitch, but some players may also be missed (occlusion,
  distant players). Only an evaluation against the GT can separate these
  two effects; it has not been run yet.
- In frame 1500, the person standing outside the touchline at the right
  corner (not in the GT, ID 2022 in the run without ROI) is no longer
  tracked.
- The ROI edge runs very close to the near touchline. The white outline
  lies on the white line there and is hard to see in the video.

---

## 2026-10-29 — Foot point

**Goal:** give every box one ground point. This is the point that later goes
to the pitch (homography), so heatmaps, distance and speed are all built on
it.

### Definition

`foot_point(x, y, w, h) = (x + w/2, y + h)`, the bottom centre of the box
(x, y = top-left corner, original 4096 x 1080 pixels).

- **Why the bottom centre:** it is the box point closest to where the
  player touches the ground. The box centre lies about half a body height
  above the ground; projected onto the pitch with perspective, it lands
  behind the player, further from the camera.
- **Limits:** a player who falls, slides or jumps does not touch the ground
  at this point, so the position is off for those frames. A box that is
  cut by occlusion or by the frame edge also moves the point.

### Where it is defined

Only in `src/track.py` (`foot_point()`). Every other use goes through it:

- the ROI filter in `track.py` (a box is kept if its foot point is inside
  the polygon),
- `tracks.csv`, which now has the columns `foot_x, foot_y`
  (`add_foot_columns()`),
- `src/draw_boxes_video.py`, which draws the point as a small filled dot,
  for the tracker output and for the ground truth.

`tracks.csv` of `118575_yolo11m_botsort_roi` was rebuilt from the saved
label files with `track.rebuild_tracks_csv()`, without running the tracker
again: 62,565 rows, 448 IDs (unchanged). Box values differ from the earlier
file by at most 0.1 px (rounding of the normalised label values).

### Current output video

**Decision:** from this stage on, the project continues on the **ground
truth** boxes and IDs (22 players, 22 IDs), not on the tracker output
(448 IDs, IDF1 far below MOTA). The later stages (homography, distance,
speed, heatmaps) can then be measured without tracking errors mixed in.
The tracker is kept for comparison.

`outputs/118575_latest.mp4` is the project's **current output video**:
the ground truth boxes, GT IDs, foot points (computed from the GT boxes with
`track.foot_point()`) and the ROI outline on the original frame
(`SOURCE = "gt"` in `draw_boxes_video.py`). Every new stage overwrites it.
It replaces `outputs/tracking/118575_gt.mp4`.
The tracker output drawn the same way (`SOURCE = "tracking"`) stays in its
run folder: `outputs/tracking/118575_yolo11m_botsort_roi/118575_tracks.mp4`.

Both videos were checked against the source: 4096 x 1080, 25 fps, 3000
frames, H.264 (CRF 23), about 140 MB each (not versioned). The middle frame
of the current video (frame 1500, 22 GT boxes) is
`outputs/118575_latest_frame.jpg` (versioned).
