# Speed parameters against the reference

All values are approximate. Our speed and the reference speed of the dataset are averaged in
1 s bins per track (2640 bins). Bias = ours minus reference. Distance = per-track
sum of dist_m against the reference speed summed over the same frames. Peak = fastest moment
of every track against the 5-frame smoothing with the same speed window, mean over the tracks.

| smooth_frames | speed_window_s (frames used) | MAE (m/s) | Bias (m/s) | Correlation | Ours p95 / p99 / max (m/s) | Distance difference (%) | Peak vs first (%) | Within the tie band |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.5 (12) | 0.261 | +0.113 | 0.964 | 5.37 / 6.72 / 8.56 | 38.3 | +0.0 | no |
| 9 | 0.5 (12) | 0.238 | +0.066 | 0.969 | 5.32 / 6.64 / 8.46 | 15.9 | -10.4 | no |
| 13 | 0.5 (12) | 0.218 | +0.026 | 0.974 | 5.32 / 6.60 / 8.38 | 7.9 | -15.6 | no |
| 25 | 0.5 (12) | 0.195 | -0.066 | 0.982 | 5.24 / 6.48 / 8.22 | 1.6 | -24.5 | yes |
| **5** | **1** (24) | 0.191 | -0.030 | 0.982 | 5.27 / 6.52 / 8.21 | 38.3 | +0.0 | yes |
| 9 | 1 (24) | 0.191 | -0.047 | 0.982 | 5.26 / 6.49 / 8.23 | 15.9 | -3.4 | yes |
| 13 | 1 (24) | 0.193 | -0.061 | 0.982 | 5.24 / 6.48 / 8.23 | 7.9 | -5.4 | yes |
| 25 | 1 (24) | 0.199 | -0.103 | 0.983 | 5.18 / 6.42 / 8.10 | 1.6 | -8.6 | yes |

Reference speed in the same bins: p95 5.30, p99 6.60, max 7.71 m/s.

Rule: among the pairs within 0.02 m/s of the lowest mean absolute error (0.191 m/s), the bias closest to zero; if equal, the shorter smoothing window.

**Chosen: smooth_frames = 5, speed_window_s = 1** (mean absolute error 0.191 m/s, bias -0.030 m/s, peak +0.0 % against 5 frames). Written to `config/field_config.yaml`.

The rule looks at speed only. With the chosen smoothing the summed dist_m is 38.3 % away from the reference distance; the smallest difference in the table is 1.6 % (smooth_frames = 25). A step from frame to frame adds up the remaining jitter of the positions, so short smoothing makes the summed distance too long.
