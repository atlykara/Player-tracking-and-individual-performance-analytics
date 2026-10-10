# Tracking quality report (no ground truth)

Labels: `outputs/tracking/118575_yolo11m_botsort/labels`  
Video: `data/SoccerTrackv2/mot/118575.mp4`

| Metric | Value | Meaning |
|---|---|---|
| Frames | 3000 | Number of frames in the video. |
| Tracked boxes | 74437 | Total boxes with a track ID over all frames. |
| Boxes per frame (mean) | 24.812 | Average number of people tracked in one frame. |
| Boxes per frame (std) | 1.825 | How much that number changes from frame to frame. |
| Boxes per frame (min) | 18 | Fewest boxes seen in a single frame. |
| Boxes per frame (max) | 31 | Most boxes seen in a single frame. |
| Frames without boxes | 0 | Frames where the tracker reported nobody. |
| Unique track IDs | 575 | Number of different IDs handed out in the clip. |
| ID ratio | 23.174 | Unique IDs divided by mean boxes per frame; 1.0 would mean every person kept one ID. |
| Track lifetime mean (s) | 5.427 | Average time from first to last appearance of an ID. |
| Track lifetime median (s) | 1.04 | Typical lifetime, less affected by a few long tracks. |
| Tracks < 1 s | 285 | IDs that live less than one second. |
| Tracks 1-5 s | 159 | IDs that live between 1 and 5 seconds. |
| Tracks 5-30 s | 105 | IDs that live between 5 and 30 seconds. |
| Tracks > 30 s | 26 | IDs that live longer than 30 seconds. |
| Short tracks (< 2.0 s) | 347 | IDs too short to belong to a real, continuously visible player. |
| Short tracks (%) | 60.348 | Share of all IDs that are short. |
| Tracks with gaps | 250 | IDs that disappear for some frames and come back with the same ID. |
| Missing frames in tracks | 3577 | Total frames missing inside those gaps. |
| Jumps (> 50 px/frame) | 0 | Consecutive-frame moves of one ID that are physically impossible, likely ID switches. |
| Confidence mean | 0.612 | Average detector confidence of tracked boxes. |
| Confidence median | 0.686 | Typical detector confidence. |
| Confidence < 0.4 (%) | 17.722 | Share of boxes the detector was unsure about. |
