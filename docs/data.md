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
