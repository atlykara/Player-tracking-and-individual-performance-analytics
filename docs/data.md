# Data

## Source

- **Dataset:** SoccerTrack v2, match **118575**, MOT challenge clip
  (SoccerTrack Challenge 2025, training set).
- **Files used:** `118575.mp4` (video) and `118575.txt` (MOT bounding-box
  ground truth), stored in `data/SoccerTrackv2/mot/`.
- **License:** CC BY 4.0.
- Raw and derived data are not committed (`data/` is git-ignored).

The ground truth indexes frames of this clip, not the full-match panorama video.

## Original clip (`118575.mp4`)

| Property | Value |
|---|---|
| Resolution | 4096 x 1080 (panoramic, full pitch) |
| Frame rate | 25 fps |
| Frames | 6000 |
| Duration | 240 s (4 min) |
| Streams | H.264 video + AAC audio |

## Ground truth format

MOT format, one row per box, comma-separated:

```
frame,id,x,y,w,h,-1,-1,-1,-1
0,1,2327.0,266.00,42.00,63.00,-1,-1,-1,-1
```

- `frame`: **0-based** (first frame of the video is `0`).
- `id`: player identity, 22 ids (outfield players and goalkeepers;
  no ball, no referees).
- `x, y, w, h`: box in pixels, top-left origin, original video resolution.
- The last four columns are unused (`-1`).

The original file has 132000 rows (frames 0-5999, 22 boxes per frame).

## Working clip: first 2 minutes

All following experiments use this clip and its matching ground truth.

| File | Content |
|---|---|
| `data/SoccerTrackv2/mot/118575_120s.mp4` | frames 0-2999 of `118575.mp4` |
| `data/SoccerTrackv2/mot/118575_120s_gt.txt` | GT rows for frames 0-2999 |

### Video

```
ffmpeg -i data/SoccerTrackv2/mot/118575.mp4 -frames:v 3000 \
  -c:v libx264 -crf 18 -preset medium -an data/SoccerTrackv2/mot/118575_120s.mp4
```

The first 3000 frames are re-encoded with H.264 at CRF 18 (visually lossless).
The resolution is unchanged and the audio is removed.

Checked after cutting (ffprobe, frames counted by decoding):
4096 x 1080, 25 fps, 3000 frames, 120.0 s.

Frame alignment was checked by decoding both videos sequentially. For every
clip frame `j`, the closest source frame (mean absolute difference over
neighbours `j-1`, `j`, `j+1`) was frame `j`. The only exceptions were 4 frames
where two consecutive frames were nearly identical. Clip frame `j` therefore
corresponds to GT frame `j`.

### Ground truth

```
awk -F, '$1>=0 && $1<=2999' data/SoccerTrackv2/mot/118575.txt \
  > data/SoccerTrackv2/mot/118575_120s_gt.txt
```

The row format is unchanged and frame numbers stay 0-based.

| | Value |
|---|---|
| Rows | 66000 |
| Frames | 0-2999 (3000 frames) |
| Unique ids | 22 |
| Boxes per frame | 22 |

### Box viewing videos

`src/draw_boxes_video.py` draws a box table on the working clip so it can be
watched like a tracking result. The same function draws the ground truth and
the tracker output, so both videos follow the same rules and can be compared
directly (v1 of the script, `draw_gt_video.py`, drew only the GT).

| Input | Output (not versioned) |
|---|---|
| `118575_gt.txt` via `evaluate_tracking.load_ground_truth()` | `outputs/tracking/118575_gt.mp4` |
| `outputs/tracking/118575_yolo11m_botsort_roi/tracks.csv` | `outputs/tracking/118575_yolo11m_botsort_roi/118575_tracks.mp4` |

Both videos keep the clip's resolution and fps (3000 frames) and are
re-encoded with ffmpeg (H.264, CRF 23).

Drawing rules:

- the input is any table with the columns `frame, track_id, x, y, w, h`
  (pixels, top-left origin, 0-based frames);
- every ID keeps one colour for the whole clip (hue derived from the ID)
  and shows its number above the box;
- rows with `is_valid = False` (GT reader, MOT rule: `conf == 0` or
  non-person class) are drawn thin and grey; this GT has none, and tables
  without the column (tracks.csv) are treated as all valid;
- top-left corner: table name, frame number and the number of boxes in it.

## Region of interest

The panoramic camera also sees the area around the pitch: substitutes,
coaches and officials standing outside the lines are detected as persons and
would enter the tracking as false players. A polygon around the pitch (the
region of interest, ROI) marks where players can be; detections outside it
can be discarded.

**How it was selected:** `src/select_roi.py` shows the first frame of
`118575.mp4` scaled down to fit the screen. The pitch corners were clicked
by hand, slightly **outside** the touchlines and goal lines, so that players
standing on a line are not cut off. A magnifier at the bottom centre
shows the original pixels around the cursor (4x) for precise clicks on the
far corners. Controls: left click adds a corner,
`u` removes the last one, Enter saves (at least 3 corners), Esc cancels.

| File | Content |
|---|---|
| `data/roi/118575_roi.json` | polygon corners (versioned) |
| `data/roi/118575_roi_check.jpg` | polygon drawn on the first frame, for visual checking (versioned) |

JSON format:

```
{"video": "data/SoccerTrackv2/mot/118575.mp4", "width": 4096, "height": 1080,
 "points": [[x, y], ...]}
```

**Coordinate space rule:** the points are stored in **original video
pixels** (4096 x 1080), rounded to 0.1 px. Clicks on the scaled-down window
are divided by the display scale before saving. The ground truth and the
tracker output are in this same pixel space, so inside/outside tests compare
like with like. The `video` field records which video the polygon belongs to.

`load_roi()` in the same module returns the corners as an `(N, 2)` int32
array for later steps.

## Pitch coordinate system

Player positions are reported on a flat model of the pitch, in metres.
The model and all named points are defined in `config/field_config.yaml`;
no pitch measurement is written in Python code.

| | Definition |
|---|---|
| Model size | 105 x 68 m, from the dataset metadata of match 118575 (`118575_tracker_box_metadata.xml`: `<pitch width="105" height="68"/>`); also the FIFA standard size |
| x axis | along the pitch length: `0` = left goal line, `105` = right goal line |
| y axis | across the pitch: `0` = far touchline, `68` = near touchline (camera side) |
| Origin | `(0, 0)` = left far corner |
| Units | metres (`x_m`, `y_m`) |

"Left / right" and "far / near" are as seen from the camera. Both axes run
in the same direction as the pixel axes of the video (x to the right, y
towards the camera, which is down in the image), so image and pitch need no
axis flip.

The pixel positions of the landmarks are read off the image, so every
metric position derived from them is approximate.

### Landmarks

A landmark is a named point that can be found both in the image and on the
model. `field_config.yaml` lists 67 of them, each with `x_m` and `y_m`; they
are the 67 reference points of the calibration:

- **31 line landmarks**, where a marked line meets or crosses another one.
- **36 touchline points**, every 5.5 m along both touchlines, where the
  edges of the mowing stripes meet the touchline. No marked line crosses
  the touchline there. They were added because the line landmarks give only
  three points on each touchline.

Naming: `<object>_<side>`.

| Part of the name | Meaning |
|---|---|
| `far` / `near` | on the side of the far / near touchline |
| `left` / `right` | in the half of the left / right goal |
| `_goal` | the point lies on the goal line |
| `_in` | the point lies on the inner line of the box, inside the pitch |

| Group | Landmarks |
|---|---|
| Pitch outline | `corner_left_far`, `corner_left_near`, `corner_right_far`, `corner_right_near` |
| Halfway line | `halfway_far`, `halfway_near` (ends on the touchlines) |
| Centre | `centre_spot`; `centre_circle_far`, `centre_circle_near` (circle meets the halfway line); `centre_circle_left`, `centre_circle_right` (circle points closest to each goal) |
| Penalty areas | `left_pen_*`, `right_pen_*`: `far_goal`, `near_goal`, `far_in`, `near_in` |
| Goal areas | `left_goalarea_*`, `right_goalarea_*`: the same four corners |
| Goal posts | `left_post_far`, `left_post_near`, `right_post_far`, `right_post_near` (foot of the post on the goal line; optional) |
| Touchline points | `touchline_far_<x>m`, `touchline_near_<x>m` with `<x>` = 5.5, 11, ... 49.5 and 55.5, 61, ... 99.5 (18 per touchline) |

The right side mirrors the left side (`x -> 105 - x`). Line dimensions used:
penalty area 16.5 m deep and 40.32 m wide, goal area 5.5 m deep and 18.32 m
wide, goal 7.32 m wide, centre circle radius 9.15 m.

`centre_circle_left` and `centre_circle_right` are not line crossings: no
marked line passes through them, so they are harder to click than the other
landmarks.

The pixel position of each landmark is stored in
`config/calibration_points.csv` (pixels of the original 4096 x 1080 frame).
A landmark without a pixel position has no row.

| Column | Content |
|---|---|
| `name`, `x_m`, `y_m` | landmark name and its position in metres |
| `px_x`, `px_y` | pixel position on frame 0 |
| `source` | where the pixel position comes from: `own` = selected on the frame with `calibrate.py select`; `dataset` = taken from the keypoint file of the dataset |
| `use` | `1` = reference point of the calibration; `0` = kept in the file but not used |

All 67 rows have `use = 1`: the calibration is built from all 67 reference
points and passes through every one of them. 15 touchline points (14 on the
far touchline) are more than 2 m from where the line landmarks alone put
them; the fit report lists them. See "The calibration passes through the
reference points" in `docs/experiments.md`.

## Reference positions of the players

The dataset also gives the pitch position and the speed of every player for
the whole match. They are used only to check our positions and speeds,
never to produce them.

- **File:** `data/referans/raw/118575/118575_tracker_box_data.xml` (227 MB,
  not versioned). Player ids, teams and the pitch size
  (`<pitch width="105" height="68"/>`) are in
  `118575_tracker_box_metadata.xml` in the same folder.
- **Structure:** one `<frame>` per video frame, one `<player>` per player
  on the pitch, plus the ball:

```
<frame matchTime="1439967" frameNumber="36251" eventPeriod="FIRST_HALF" ballStatus="NEUTRAL">
  <player playerId="476352" loc="[0.06, 0.48]" speed="0.57"/>
  ...
  <ball playerId="ball" loc="[0.42, 0.17]" speed="NA"/>
</frame>
```

| Field | Meaning | Unit |
|---|---|---|
| `frameNumber` | frame counter inside the period, 25 fps; the first half runs from 251 to 72351 | frames |
| `eventPeriod` | `FIRST_HALF` or `SECOND_HALF`; the frame counter starts again in the second half | - |
| `matchTime` | time since kick-off | ms |
| `playerId` | player id from the metadata file | - |
| `loc` | `[x, y]`, position as a fraction of the pitch length and width | `x_m = 105 * x`, `y_m = 68 * y`; same axes as ours, no flip |
| `speed` | speed of the player | m/s |

`loc` has two decimals, so the reference positions lie on a grid of
1.05 m x 0.68 m.

**Clip offset:** clip frame `j` = reference `frameNumber` `6252 + j`
(first half). The working clip starts 240.0 s after the first reference
frame. The offset was found by matching our 22 player positions to the
reference positions of the same instant (Hungarian matching) for 30 clip
frames and taking the median of the best offsets; the median player
position error is lowest at exactly this offset. Each GT track was assigned
to the reference player it was matched with most often (lowest share
93.3 %).
