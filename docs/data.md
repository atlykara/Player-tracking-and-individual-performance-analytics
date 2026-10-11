# Data

## Source

- **Dataset:** SoccerTrack v2, match **118575**, MOT challenge clip
  (SoccerTrack Challenge 2025, training set).
- **Files used:** `118575.mp4` (video) and `118575.txt` (MOT bounding-box
  ground truth), stored locally in `data/SoccerTrackv2/mot/`.
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
| `118575_gt.txt` via `evaluate_tracking.load_ground_truth()` | `outputs/gt_video/118575_gt.mp4` |
| `outputs/tracking/118575_yolo11m_botsort/tracks.csv` | `outputs/gt_video/118575_tracks.mp4` |

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
