# Movement consistency report (pixel space)

Units: pixels and px/s only. Speeds are approximate and depend on the
distance to the camera; no metric value is given before the homography.

## Checks

- [x] timestamp == frame / fps
- [x] frames increase inside every track
- [x] no duplicate (track_id, frame)
- [x] covered time 120.00 s ~ video 120.00 s

## Counts

- Rows: 66000, tracks: 22, steps with a speed: 65978
- Gap rows (dt > 3 frames): 0 (0.00 %)
- Outlier rows (pixel_speed > 290 px/s): 198 (0.30 %)
- Short-lived tracks (< 2 s): 0

## Speed (px/s)

| | mean | median | max |
|---|---|---|---|
| raw | 49.1 | 35.4 | 575.1 |
| smoothed (5 frames) | 43.3 | 28.3 | 342.5 |

Raw percentiles: p50 = 35.4, p90 = 112.5, p99 = 226.4, p99.5 = 265.8, p99.7 = 288.7, p99.9 = 340.2, max = 575.1

## Smoothing effect (measurement only)

| | raw px/s | smoothed px/s | difference |
|---|---|---|---|
| median | 35.4 | 28.3 | 20.0 % |
| p99 | 226.4 | 195.3 | 13.8 % |

Standing player (net movement < 2 px in 1 s, 4115 rows): raw speed mean 6.3, p90 25.0 px/s.

Smoothing changes the median by 20.0 % (>= 5 %): the smoothed columns are kept.

## Interpretation

All 4 checks pass. There are 0 gaps and 0 short-lived tracks, so every track covers 120.0-120.0 s of the 120 s clip. The smallest non-zero raw step is 0.5 px per frame (12.5 px/s), so box jitter alone produces speed: smoothing lowers the mean from 49.1 to 43.3 px/s. 198 rows (0.30 %) exceed 290 px/s; they are flagged, not removed.
