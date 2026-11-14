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

---

## 2026-10-29 — Movement (pixel space)

**Goal:** per-frame foot point movement for every GT player, in pixels and
px/s only. Metres, m/s and km/h come after the homography stage, which
sets the pixel-to-metre scale.

### Input

- `outputs/tracking/118575_gt/tracks.csv`: the ground truth in tracks.csv
  format (`frame, track_id, x, y, w, h, conf, foot_x, foot_y`), written by
  `track.export_gt_tracks_csv()`. 66,000 rows, 22 IDs, every ID in all
  3000 frames, no frame gaps. GT boxes are whole pixels, so foot_x moves
  in steps of 0.5 px.
- fps (25.00) and duration (120.00 s) are read from
  `data/SoccerTrackv2/mot/118575.mp4`.

### Method (`src/movement.py` v1)

Per track, sorted by frame: `timestamp = frame / fps`; `dx, dy, dt` to the
previous row; `pixel_speed = sqrt(dx² + dy²) / dt`. The first row of each
track has these fields empty. Movement uses the foot point, not the box
centre (the box centre also moves when only the box height changes).

Derived columns are added next to the untouched raw columns:
`smooth_x, smooth_y` (centred moving average inside the track),
`smooth_speed`, `gap_flag`, `outlier_flag`. Project rule: rows are never
deleted and never interpolated, only flagged.

| Constant | Value | Reason |
|---|---|---|
| `GAP_FRAMES` | 3 | a step longer than 3 frames (0.12 s) is a gap |
| `SMOOTH_WINDOW` | 5 frames (0.2 s) | removes 1-2 px box jitter, keeps changes of direction |
| `OUTLIER_SPEED_PX_S` | 290 px/s | ≈ p99.7 of the raw speed (288.7 px/s) |

Raw speed distribution (65,978 steps, px/s): p50 35.4, p90 112.5,
p99 226.4, p99.5 265.8, p99.7 288.7, p99.9 340.2, max 575.1.
290 px/s (11.6 px in one frame) flags about the fastest 0.3 % of steps for
a closer look. It is a pixel threshold: a player near the camera covers
more pixels for the same run, so flags gather near the camera. A metric
threshold replaces it after the homography.

### Summary numbers

| Measure | Value |
|---|---|
| Rows / tracks | 66,000 / 22 |
| Consistency checks passed | 4 / 4 (timestamps, monotonic frames, no duplicates, 120.00 s covered) |
| Gap rows | 0 |
| Outlier rows (> 290 px/s) | 198 (0.30 %) |
| Short-lived tracks (< 2 s) | 0 |
| Raw speed mean / median / max | 49.1 / 35.4 / 575.1 px/s |
| Smoothed speed mean / median / max | 43.3 / 28.3 / 342.5 px/s |

Observation: the smallest non-zero raw step is 0.5 px per frame
(12.5 px/s), so box jitter alone creates speed. Smoothing lowers the mean
by 12 % and the maximum by 40 %.

Smoothing check (measurement only, in `consistency_report.md`): raw vs
smoothed speed differs by 20.0 % at the median (35.4 vs 28.3 px/s) and by
13.8 % at p99 (226.4 vs 195.3 px/s). A standing player (net movement
< 2 px in 1 s, 4,115 rows) still has a raw speed of 6.3 px/s on average
(p90 25.0 px/s) from 0.5-1 px box steps. The difference is above the 5 %
"negligible" limit, so the smoothed columns are kept.

### Outputs

- `outputs/movement/118575_gt/movement.csv`: all columns, 3 decimals
  (8.4 MB, not versioned).
- `outputs/movement/118575_gt/consistency_report.md` (versioned).
- `outputs/118575_latest.mp4` regenerated with fading 3 s foot point
  trails (`draw_boxes_video.py` v4, `USE_SMOOTH = True`): newest point
  large and opaque, older points smaller and fainter, in the track colour.
  Rendering takes about 7.5 min for 3000 frames.

---

## 2026-11-14 — Pitch landmark selection

**Goal:** collect pairs of (pixel position, pitch position in metres) for the
pitch calibration. No homography is computed in this step.

### Command

```
python src/calibrate.py select --video data/SoccerTrackv2/mot/118575.mp4 --frame 0
```

Re-click one landmark: `python src/calibrate.py select --redo NAME`.
A second run without `--redo` asks only for the landmarks that have no row
yet.

### Method (`src/calibrate.py` v1, `select`)

- The landmark names and metre values come from `config/field_config.yaml`
  (31 landmarks, pitch model 105 x 68 m; see "Pitch coordinate system" in
  `docs/data.md`). The tool asks for them in config order.
- The frame is shown scaled down to fit the screen. A magnifier in a top
  corner shows the original pixels around the cursor (4x) with a crosshair.
  The saved value is the **original 4096 x 1080 pixel** under the crosshair.
- The mouse moves in window pixels, so a click can only land on every
  `1 / display scale` original pixel ("click step", printed by the tool).
- Controls: left click = save, `s` = skip, `u` = undo last click,
  `q` = finish.

### Outputs

- `config/calibration_points.csv`: `name, x_m, y_m, px_x, px_y`, one row per
  clicked landmark (skipped landmarks have no row).
- `outputs/calibration/118575_points.jpg`: the full-resolution frame with
  every clicked point drawn and labelled (name and metres; not versioned).

### Clicked points

To be filled in after the selection run.

| Measure | Value |
|---|---|
| Frame used | |
| Click step (original px per window px) | |
| Landmarks clicked / in config | / 31 |
| Landmarks skipped | |

| Landmark | x_m | y_m | px_x | px_y | Note |
|---|---|---|---|---|---|
| corner_left_far | 0 | 0 | | | |
| corner_left_near | 0 | 68 | | | |
| corner_right_far | 105 | 0 | | | |
| corner_right_near | 105 | 68 | | | |
| halfway_far | 52.5 | 0 | | | |
| halfway_near | 52.5 | 68 | | | |
| centre_spot | 52.5 | 34 | | | |
| centre_circle_far | 52.5 | 24.85 | | | |
| centre_circle_near | 52.5 | 43.15 | | | |
| centre_circle_left | 43.35 | 34 | | | |
| centre_circle_right | 61.65 | 34 | | | |
| left_pen_far_goal | 0 | 13.84 | | | |
| left_pen_near_goal | 0 | 54.16 | | | |
| left_pen_far_in | 16.5 | 13.84 | | | |
| left_pen_near_in | 16.5 | 54.16 | | | |
| left_goalarea_far_goal | 0 | 24.84 | | | |
| left_goalarea_near_goal | 0 | 43.16 | | | |
| left_goalarea_far_in | 5.5 | 24.84 | | | |
| left_goalarea_near_in | 5.5 | 43.16 | | | |
| left_post_far | 0 | 30.34 | | | |
| left_post_near | 0 | 37.66 | | | |
| right_pen_far_goal | 105 | 13.84 | | | |
| right_pen_near_goal | 105 | 54.16 | | | |
| right_pen_far_in | 88.5 | 13.84 | | | |
| right_pen_near_in | 88.5 | 54.16 | | | |
| right_goalarea_far_goal | 105 | 24.84 | | | |
| right_goalarea_near_goal | 105 | 43.16 | | | |
| right_goalarea_far_in | 99.5 | 24.84 | | | |
| right_goalarea_near_in | 99.5 | 43.16 | | | |
| right_post_far | 105 | 30.34 | | | |
| right_post_near | 105 | 37.66 | | | |

The Note column records why a landmark was skipped or is uncertain (hidden
by a player, outside the frame, line not visible).

---

## 2026-11-14 — Pixel-to-metre mapping: three variants

**Goal:** turn the 31 clicked landmarks into a mapping from video pixels to
pitch metres, and measure how far its positions are from the true ones.

### Command

```
python src/calibrate.py fit
```

Inputs: `config/calibration_points.csv` (31 landmarks, raw 4096 x 1080
pixels and metres), `config/118575_camera_intrinsics.npz` (fisheye lens
model shipped with the dataset: `K`, `D`, `Knew`), `overlap_m = 10` from
`config/field_config.yaml`.

### Method (`src/calibrate.py` v2, `fit`)

The panoramic image is curved, so one homography does not fit the whole
pitch. The pitch is cut into pieces of equal length along x and every piece
gets its own homography (`cv2.findHomography`, method 0 = least squares,
pixel -> metres).

| Variant | Pixels | Pieces | Split at x_m |
|---|---|---|---|
| `raw_2piece` | raw | 2 | 52.5 |
| `raw_3piece` | raw | 3 | 35, 70 |
| `undist_2piece` | undistorted with the lens model | 2 | 52.5 |

- **Piece rule:** a piece is fitted on the landmarks whose `x_m` lies in its
  range extended by `overlap_m` on both sides (fewer than 5 landmarks: the
  range is widened and a warning is printed; this did not happen).
  Landmarks per piece: 19 + 19 (two pieces), 13 + 7 + 13 (three pieces).
- **Mapping a pixel:** a global homography fitted on all landmarks gives a
  rough `x_m`, the rough `x_m` selects the piece, the homography of that
  piece gives the position.
- **Undistortion** is applied to points only
  (`cv2.fisheye.undistortPoints(..., P=Knew)`); the video is never
  undistorted.
- Later stages call one function: `pixel_to_pitch(points_px, method)`.

### Evaluation: leave-one-out

Each landmark is left out in turn, the variant is fitted on the other 30
and the left-out landmark is predicted. The error is the distance between
the predicted and the true position in metres, so it is measured at a point
the mapping has not seen.

| Variant | Mean (m) | Median (m) | Max (m) | Worst landmark |
|---|---|---|---|---|
| `raw_2piece` | 3.24 | 2.21 | 12.78 | `halfway_far` |
| `raw_3piece` | 2.53 | 1.45 | 11.49 | `centre_circle_right` |
| `undist_2piece` | 1.52 | 1.21 | 6.46 | `corner_left_near` |

### Selected variant

`undist_2piece` is selected (`mapping_method` in `field_config.yaml`). It
has the lowest mean, median and maximum leave-one-out error, and in the grid
image its lines bend with the touchlines, while the raw variants draw
straight lines over a curved pitch (`raw_2piece` misses the near touchline,
`raw_3piece` shows visible jumps at its two splits). Its largest errors are
at the two near corners (6.46 m and 4.80 m), where a left-out landmark lies
outside the remaining ones. All positions from this mapping are approximate.

### Outputs

- `config/homography.npz`: the fitted matrices of all three variants
  (global and per piece, split positions) and the `video` they belong to.
- `outputs/calibration/118575_fit_report.md`: variant table and error per
  landmark; `118575_fit_errors.csv`: predicted position and error per
  landmark and variant (not versioned).
- `outputs/calibration/118575_grid_<method>.jpg`: frame 0 with a 5 m grid
  (yellow) and the pitch markings (red) projected back into the image, one
  per variant (not versioned).

---

## 2026-11-14 — Touchline points added (67 landmarks)

**Goal:** the 31 clicked landmarks give only three points on each touchline
(two corners and the end of the halfway line), and the largest mapping
errors were at the near touchline. 36 touchline points were added and the
three variants were fitted again.

### What changed

- `config/field_config.yaml`: 36 new landmarks `touchline_far_<x>m` and
  `touchline_near_<x>m`, every 5.5 m along both touchlines.
- `config/calibration_points.csv`: 36 rows appended. Their pixel positions
  are **not clicked**; they come from the dataset keypoint file
  (`118575_keypoints.json`). The 31 clicked rows are unchanged (checked byte
  by byte against a copy made before the change).
- On frame 0 the 36 points lie on the touchlines, at the edges of the
  mowing stripes.
- Same command (`python src/calibrate.py fit`), same code, `overlap_m = 10`.
  Landmarks per piece: 41 + 41 (two pieces), 29 + 27 + 29 (three pieces).

### Leave-one-out error, 31 vs 67 landmarks

| Variant | Landmarks | Mean (m) | Median (m) | Max (m) | Worst landmark |
|---|---|---|---|---|---|
| `raw_2piece` | 31 | 3.24 | 2.21 | 12.78 | `halfway_far` |
| `raw_2piece` | 67 | 2.90 | 3.11 | 6.23 | `corner_right_near` |
| `raw_3piece` | 31 | 2.53 | 1.45 | 11.49 | `centre_circle_right` |
| `raw_3piece` | 67 | 1.83 | 1.74 | 4.88 | `corner_right_near` |
| `undist_2piece` | 31 | 1.52 | 1.21 | 6.46 | `corner_left_near` |
| `undist_2piece` | 67 | 1.49 | 1.24 | 5.32 | `corner_left_near` |

The two rows of a variant are not measured on the same points. The 67-point
run split by group:

| Variant | 31 clicked: mean / max (m) | 36 touchline: mean / max (m) |
|---|---|---|
| `raw_2piece` | 3.36 / 6.23 | 2.51 / 4.30 |
| `raw_3piece` | 2.15 / 4.88 | 1.56 / 2.94 |
| `undist_2piece` | 1.60 / 5.32 | 1.40 / 3.58 |

### Result

`undist_2piece` stays selected: it is still the best variant on every
measure except the maximum, where `raw_3piece` is lower (4.88 m vs 5.32 m).
For `undist_2piece` the extra points changed little: the mean went from
1.52 m to 1.49 m, and on the 31 clicked landmarks alone it is 1.60 m. The
maximum dropped from 6.46 m to 5.32 m. The raw variants gained more
(`raw_3piece` 2.53 m -> 1.83 m, its maximum 11.49 m -> 4.88 m), because its
middle piece no longer rests on seven points. The remaining error of
`undist_2piece` therefore does not come from too few points: two
homographies do not describe this image exactly, and the largest errors
stay at the two near corners. All positions remain approximate.

Outputs were regenerated in place: `config/homography.npz`,
`outputs/calibration/118575_fit_report.md`, `118575_fit_errors.csv`, the
three `118575_grid_<method>.jpg` and `118575_points.jpg`.

---

## 2026-11-14 — Calibration from reference points

**Goal:** choose the pixel-to-metre mapping that gives the most accurate
player positions and speeds, and make it the default.

### Reference points

The calibration is built from reference points: named points of the pitch
with a known position in metres and a pixel position on frame 0. The pitch
grid in the check images is drawn from them.

- **31 own points** were selected by hand on the frame with
  `python src/calibrate.py select`: crossings of marked lines, the centre
  spot, the two side points of the centre circle and the feet of the goal
  posts (`source = own`).
- **36 touchline marks** every 5.5 m come from the dataset keypoint file
  (`source = dataset`). A mark is used only if the 31 own points alone
  predict it within 2 m of its nominal position (blended 3 pieces on
  undistorted pixels). 21 marks pass; 15 do not: 14 on the far touchline and
  `touchline_near_5.5m`.
- `config/calibration_points.csv` keeps all 67 rows with the columns
  `source` and `use`; the calibration uses the 52 rows with `use = 1`.
  `fit` repeats the 2 m check and reports it.

### Method

`reference_points`: each position is mapped with a homography fitted to all
reference points, where the points near that position count more. The
weight of a reference point for a position is
`max(exp(-(d / radius_m)^2), min_weight)`, with `d` the distance in metres
between the two. The position of a pixel is not known before it is mapped,
so a single homography with equal weights gives a rough position first and
the weights are set around that. The weighting idea follows moving DLT
(Zaragoza et al., "As-Projective-As-Possible Image Stitching with Moving
DLT", CVPR 2013).

Settings (`calibration` in `config/field_config.yaml`):
`method: reference_points`, `radius_m: 30`, `min_weight: 0.05`,
`pixel_space: undistorted`. The blended 3-piece mapping stays available as
method `pieces`, only as the baseline for comparison.
`pixel_to_pitch(points_px, method=None)` is the single entry point; every
point is mapped directly (66,000 foot points take about 1 s).

### What was compared

190 configurations, all scored in the same way:

| Factor | Values |
|---|---|
| Method | blended 3 pieces (baseline); reference points |
| Radius | 10, 15, 20, 30, 40, 60 m |
| Minimum weight | 0, 0.05, 0.15 |
| Point set | own 31; all 67; own + marks with 16.5 <= x_m <= 88.5; all 67 with the marks weighted 0.3; own + marks within 2 m of the prediction from the own points |
| Pixel space | undistorted; raw |

Scores:

- leave-one-out error at the reference points;
- positions and speeds of the 22 GT tracks over the whole clip (66,000
  positions; speeds in 2640 one-second bins) against the reference
  positions and speeds of the dataset (see "Reference positions of the
  players" in `docs/data.md`);
- smoothness of the pitch grid that a configuration draws.

Selection rule, fixed before the run: discard every configuration whose
median position error is more than 10 % above the best one, then take the
lowest speed error.

### Baseline vs selected

| | Baseline: blended 3 pieces, all 67 points | Selected: reference points, 30 m, 0.05, 52 points |
|---|---|---|
| Position error, median (m) | 1.80 | 0.93 |
| Position error, p95 (m) | 4.02 | 2.65 |
| Speed, mean absolute error (m/s) | 0.276 | 0.284 |
| Speed, bias (ours minus reference, m/s) | +0.104 | +0.182 |
| Distance covered, difference (%) | 5.5 | 8.3 |

Both on undistorted pixels. The selected mapping halves the position error.
Speed and distance are not improved; they are slightly worse than with the
baseline, within the spread between neighbouring configurations.

Leave-one-out error at the 52 reference points (fit report), mean / median
/ max: `reference_points` 1.14 / 0.75 / 3.85 m, `pieces` on the same points
1.28 / 0.95 / 5.79 m.

### Findings

1. **The waviness of the grid has no measurable effect on speed.** Over the
   configurations with reference points on undistorted pixels, the rank
   correlation between speed error and the two waviness scores is +0.04 and
   +0.06. The speed error stays between 0.26 and 0.29 m/s for radii of 10 to
   40 m without minimum weight, although the grid is much wavier at 10 m.
2. **A radius of 20-30 m is best.** Median position error by radius (minimum
   weight 0.05, selected point set): 1.54 m at 10 m, 1.03 m at 15 m, 0.95 m
   at 20 m, 0.93 m at 30 m, 1.04 m at 40 m, 1.53 m at 60 m. 20 m and 30 m
   cannot be told apart. A minimum weight of 0.05 lowers the position error
   and raises the speed error; 0.15 is worse on both.
3. **The far-touchline dataset marks deviate systematically along the
   line.** Predicted from the own points, they are off by -3.8 m on the left
   to +3.2 m on the right, near zero at the halfway line. It is a smooth
   shift of the whole row, not single bad marks. Leaving these marks out
   lowers the median position error from 1.20 m (all 67 points) to 0.93 m.
   Leave-one-out cannot see such a shift, because a mark is always predicted
   well by its neighbours 5.5 m away.

### Limits

- The reference positions lie on a grid of 1.05 x 0.68 m (two decimals of
  the pitch size). Rounding alone gives 0.34 m mean error even for a perfect
  mapping, and the reference tracking has an unknown error of its own.
- One 2-minute clip and one calibration frame; 4 of the 24 pitch regions
  have too few samples to be scored.
- In the far strip the selected mapping is worse than the baseline in 3 of
  20 regions (for example 0.68 m -> 1.29 m), and its far touchline runs up
  to 4.7 m beside the dataset marks that are not used.
- All positions and speeds remain approximate.

### Outputs

- `config/homography.npz`: the 52 reference points in the pixel space of
  the calibration, their metre positions, the settings and the `video` they
  belong to (written by `fit`, read by `pixel_to_pitch`).
- `outputs/calibration/118575_fit_report.md`, `118575_fit_errors.csv`: both
  methods, the 2 m check of the dataset marks, error per point (not
  versioned).
- `outputs/calibration/118575_grid.jpg` (default) and
  `118575_grid_pieces.jpg` (baseline): frame 0 with a 5 m grid (yellow), the
  pitch markings (red) and the points of the points file (not versioned).
- The comparison of the 190 configurations was run with a scratch script;
  its results are in `outputs/calibration/experiments/` (not versioned).

---

## 2026-11-14 — The calibration passes through the reference points

**Goal:** the pitch lines of the grid image must run through the reference
points that were selected on the frame, and positions and speeds must be
computed from that same mapping.

### Problem

With the setting selected in the section above (52 reference points, radius
30 m, minimum weight 0.05) the mapping only came close to its own reference
points. In the grid image the drawn lines missed them by 7.6 px on average
and by up to 22 px (0.94 m mean, 3.06 m max on the pitch), most visibly at
the corners, and the far touchline ran beside the 15 dataset marks that
were not used.

### Change (`src/calibrate.py`, method `reference_points`)

- **Weight.** The weight of a reference point for a position is now
  `max(exp(-(d / radius_m)^2), min_weight) / (d^2 + 0.05^2)`, with `d` the
  distance in metres. The division by `d^2` lets the weight grow without
  limit as the position reaches a reference point, so the mapping passes
  through every reference point instead of only close to it.
- **Position in steps.** A pixel is mapped from a rough position first and
  then three more times, each time with the weights set around the position
  found before. After that a reference pixel lands on its own pitch
  position.
- **All 67 points are used** (`use = 1` for every row of
  `config/calibration_points.csv`), including the 15 far-touchline marks
  that were left out before. `radius_m: 30` and `min_weight: 0.05` are
  unchanged.
- `fit` now measures and reports whether the mapping passes through its
  reference points.

### Result

| | Baseline `pieces`, 67 points | Previous default, 52 points | Now, 67 points |
|---|---|---|---|
| Error at the reference points, mean / max (m) | 1.18 / 4.61 | 0.94 / 3.06 | 0.00 / 0.01 |
| Reference point to drawn line, mean / max (px) | 11.3 / 38 | 7.6 / 22 | 0.0 / 0 |
| Leave-one-out, own points, mean / max (m) | 1.57 / 5.63 | 1.39 / 3.05 | 0.76 / 2.63 |
| Player position error, median (m) | 1.80 | 0.93 | 1.36 |
| Player position error, p95 (m) | 4.02 | 2.65 | 3.19 |
| Speed, mean absolute error (m/s) | 0.276 | 0.284 | 0.260 |
| Speed, bias (ours minus reference, m/s) | +0.105 | +0.182 | +0.113 |
| Distance covered, difference (%) | 5.5 | 8.3 | 5.5 |

Player rows: the 22 GT tracks over the whole clip against the reference
positions and speeds of the dataset, as in the section above.

- The mapping now passes through all 67 reference points, and the pitch
  lines of `118575_grid.jpg` run through them.
- A reference point that is left out is predicted better than before
  (0.76 m against 1.39 m on the own points).
- Speed and distance agree better with the reference than with the
  previous default, and slightly better than with the baseline.
- **The median player position error against the dataset reference is
  higher than with the previous default: 1.36 m against 0.93 m.** By the
  selection rule of the section above this setting would not have been
  chosen. It is used because the calibration has to honour the reference
  points; it is still lower than the baseline (1.80 m).
- The far right of the pitch is the weakest area: mean position error
  1.96 m and 3.59 m in the two far-right regions.
- Radius and minimum weight now matter little: for radii of 20 to 60 m and
  minimum weights of 0 to 0.15 the median position error stays between 1.30
  and 1.45 m and the speed error between 0.26 and 0.29 m/s.
- 15 dataset marks are still more than 2 m from where the own points alone
  put them, 14 of them along the far touchline; the fit report lists them.
  They are used anyway: they lie on the visible touchline, and only their
  place along the line is uncertain.

The limits of the section above still hold: the reference positions lie on
a 1.05 x 0.68 m grid, and all of this is measured on one 2-minute clip.
All positions and speeds remain approximate.

Mapping the 66,000 foot points of the clip takes about 3 s.
