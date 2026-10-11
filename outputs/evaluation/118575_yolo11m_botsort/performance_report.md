# Tracking performance report (with ground truth)

Labels: `outputs/tracking/118575_yolo11m_botsort/labels`  
Ground truth: `data/SoccerTrackv2/mot/118575_gt.txt`  
IoU threshold: 0.5

| Metric | Value | Meaning |
|---|---|---|
| Frames | 3000 | Frames compared. |
| GT boxes | 66000 | Valid player boxes in the ground truth. |
| Predicted boxes | 74437 | Boxes reported by the tracker. |
| GT IDs | 22 | Different players in the ground truth. |
| Predicted IDs | 575 | Different track IDs handed out by the tracker. |
| TP | 57263 | Predicted boxes matched to a GT box with IoU >= threshold. |
| FP | 17174 | Predicted boxes with no matching GT box (extra or misplaced boxes). |
| FN | 8737 | GT boxes with no matching prediction (missed players). |
| ID switches | 770 | Times a GT player became matched to a different track ID than before. |
| Precision | 0.7693 | Share of predicted boxes that are correct. |
| Recall | 0.8676 | Share of GT boxes that were found. |
| MOTA | 0.5957 | 1 minus all errors (FN + FP + IDSW) per GT box; can be negative. |
| MOTP (mean IoU) | 0.8097 | How tightly matched boxes overlap the GT, on average. |
| IDTP | 24339 | Matched boxes that also carry the one track ID assigned to that player for the whole clip. |
| IDF1 | 0.3466 | F1 score of identity: how well one player keeps one ID over the whole clip. |
| Mostly tracked | 18 | GT players matched in more than 80% of their frames. |
| Partially tracked | 4 | GT players between the two limits. |
| Mostly lost | 0 | GT players matched in less than 20% of their frames. |
