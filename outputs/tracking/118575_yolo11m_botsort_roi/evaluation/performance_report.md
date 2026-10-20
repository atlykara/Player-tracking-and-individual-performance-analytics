# Tracking evaluation report (ROI run vs. ground truth)

Labels: `outputs/tracking/118575_yolo11m_botsort_roi/labels`  
Ground truth: `data/SoccerTrackv2/mot/118575_gt.txt`  
IoU threshold: 0.5

| Metric | Value | Meaning |
|---|---|---|
| MOTA | 0.7685 | Overall score: 1 minus all errors (FN + FP + ID switches) per GT box; can be negative. |
| IDF1 | 0.3548 | Identity score: how well one player keeps one track ID over the whole clip. |
| Precision | 0.9112 | Share of predicted boxes that match a real player. |
| Recall | 0.8637 | Share of real players (GT boxes) that were found. |
| ID switches | 729 | Times a GT player became matched to a different track ID than before. |
| GT IDs / Predicted IDs | 22 / 448 | Real players vs. track IDs handed out; equal numbers would mean no fragmentation. |
| FP / FN | 5558 / 8993 | Extra or misplaced boxes / missed GT boxes. |

Values refer to this 3000-frame clip only; a box counts as correct when it overlaps a GT box with IoU >= 0.5.
