# oyuncu_analiz

Single-camera player tracking in amateur football: an analytical
feasibility study. Players are detected and tracked in one fixed panoramic
video, and each player's foot point will be mapped to approximate metric
pitch coordinates.

## Project output

- **Working video (the only one):** `data/SoccerTrackv2/mot/118575.mp4`,
  SoccerTrack v2 match 118575, 2 minutes, 4096 x 1080, 25 fps, with
  ground truth `data/SoccerTrackv2/mot/118575_gt.txt`. Every script runs
  on this video.
- **Positions come from the ground truth:** the project continues on the
  GT boxes and IDs (22 players); the own tracker is kept for comparison.
- **Current output video:** `outputs/118575_latest.mp4`: GT boxes, GT IDs,
  foot points and the ROI outline on the original frame. Every new stage
  overwrites it. Its middle frame: `outputs/118575_latest_frame.jpg`.
- **Tracker comparison video:**
  `outputs/tracking/118575_yolo11m_botsort_roi/118575_tracks.mp4`.

Videos and data are not in the repository.

### Run order

Run from the project root:

1. `python src/select_roi.py`: click the pitch polygon,
   saves `data/roi/118575_roi.json`.
2. `python src/track.py`: YOLO11m + BoT-SORT inside the ROI (CUDA GPU),
   writes `outputs/tracking/118575_yolo11m_botsort_roi/` (labels,
   `tracks.csv` with foot points).
3. `python src/evaluate_tracking.py`: compares the tracking with the
   ground truth, writes the `evaluation/` folder of the run.
4. `python -c "import sys; sys.path.insert(0, 'src'); import track; track.export_gt_tracks_csv()"`:
   writes the ground truth as `outputs/tracking/118575_gt/tracks.csv`
   (with foot points).
5. `python src/movement.py`: foot point movement in pixels and px/s
   (`dx, dy, dt, pixel_speed`, smoothing, gap / outlier flags), writes
   `outputs/movement/118575_gt/movement.csv` and `consistency_report.md`.
6. `python src/draw_boxes_video.py`: draws the table chosen by `SOURCE`
   (`"gt"`, default → `outputs/118575_latest.mp4`, with foot point trails
   from step 5; `"tracking"` → the tracker video in the run folder).

Steps 2–3 only matter for the tracker comparison; the current output needs
steps 1 (ROI outline), 4, 5 and 6.

Experiment notes and measured numbers: `docs/experiments.md`.
