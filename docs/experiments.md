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

### Quality evaluation (no GT)

Run on 2026-10-10 with `src/tracking_io.py` v1 (labels → `tracks.csv`,
pixel boxes, frames shifted to 0-based to match the GT numbering) and
`src/evaluate_quality.py` v1. The ground truth was **not** used.
Outputs: `outputs/evaluation/118575_yolo11m_botsort/`
(`quality_metrics.json`, `quality_report.md`, `detections_per_frame.png`,
`track_lifetimes.png`).

| Measure | Value |
|---|---|
| Boxes per frame: mean / std / min / max | 24.81 / 1.83 / 18 / 31 |
| Frames without boxes | 0 |
| Unique IDs / ID ratio (IDs ÷ mean boxes per frame) | 575 / 23.2 |
| Track lifetime: mean / median | 5.43 s / 1.04 s |
| Lifetime buckets: < 1 s / 1-5 s / 5-30 s / > 30 s | 285 / 159 / 105 / 26 |
| Short tracks (< 2 s) | 347 (60.3 %) |
| Tracks with gaps / missing frames inside tracks | 250 / 3577 |
| Jumps (> 50 px between consecutive frames, same ID) | 0 |
| Confidence: mean / median / share < 0.4 | 0.612 / 0.686 / 17.7 % |

Notes:

- Jump threshold 50 px/frame: rough image scale ~38 px/m on the near
  touchline (no calibration yet); a 10 m/s sprint is ~15 px/frame at 25 fps.
  Measured consecutive-frame centre shifts: 99.9th percentile 12.5 px,
  maximum 19.6 px. The tracker does not produce jumps; its identity errors
  show up as fragmentation (new IDs) instead.
- Jumps are only checked between consecutive frames; a move across a gap
  inside a track is not counted.
