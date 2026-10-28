"""Foot point movement in pixel space (v1).

Reads a tracks.csv (here: the ground truth boxes with foot points) and adds,
per track, the time stamp, the step to the previous row (dx, dy, dt) and the
pixel speed. Units are pixels and pixels per second ONLY: the pixel-to-metre
scale is not known before the homography stage, and one pixel covers a
different distance near and far from the camera.

Movement uses foot_x, foot_y and not the box centre: the foot point is the
point that will be mapped to the pitch, and the box centre also moves when
only the box height changes (a player bending, a leg stretched out).

Project rule: rows are never deleted and never interpolated. Suspicious rows
only get flags (gap_flag, outlier_flag) next to the untouched raw columns;
the analytics stage decides what to use. Deleting or filling rows here would
hide measurement problems instead of showing them.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# --- Settings -------------------------------------------------------------
TRACKS_CSV = "outputs/tracking/118575_gt/tracks.csv"
VIDEO_PATH = "data/SoccerTrackv2/mot/118575.mp4"   # fps and duration are read from here
OUT_DIR = "outputs/movement/118575_gt"
GAP_FRAMES = 3          # a step longer than 3 frames (0.12 s at 25 fps) is a gap
SMOOTH_WINDOW = 5       # centred moving average over 5 frames (0.2 s): removes
                        # 1-2 px box jitter but keeps real changes of direction
# Measured raw speed distribution on the GT (66,000 rows): p99 = 226 px/s,
# p99.5 = 266, p99.7 = 289, p99.9 = 340, max = 575 px/s. 290 px/s flags about
# the fastest 0.3 % of steps (11.6 px in one frame) for a closer look. It is a
# pixel threshold: near the camera a player covers more pixels, so flags
# gather there; a metric threshold follows after the homography.
OUTLIER_SPEED_PX_S = 290
SHORT_TRACK_SECONDS = 2.0   # tracks shorter than this are reported as short-lived
SMOOTH_NEGLIGIBLE_PCT = 5   # median speed change below 5 %: smoothing is not needed
STANDING_PX = 2             # net foot movement below 2 px in 1 s counts as standing
PERCENTILES = [50, 90, 99, 99.5, 99.7, 99.9]


def video_fps_and_duration(path):
    """Return (fps, duration in seconds) of the video."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps, n_frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return fps, n_frames / fps


def compute_movement(df, fps):
    """Add timestamp, dx, dy, dt, pixel_speed, smoothed values and flags.

    Every value is computed inside one track (groupby track_id), so the first
    row of a track has empty dx, dy, dt and speeds, and smoothing never mixes
    two tracks. Raw columns are not changed.
    """
    df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)
    df["timestamp"] = df["frame"] / fps
    by_track = df.groupby("track_id")
    df["dx"] = by_track["foot_x"].diff()
    df["dy"] = by_track["foot_y"].diff()
    df["dt"] = by_track["timestamp"].diff()
    df["pixel_speed"] = np.hypot(df["dx"], df["dy"]) / df["dt"]

    for col in ["x", "y"]:
        df[f"smooth_{col}"] = by_track[f"foot_{col}"].transform(
            lambda s: s.rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean())
    by_track = df.groupby("track_id")
    step = np.hypot(by_track["smooth_x"].diff(), by_track["smooth_y"].diff())
    df["smooth_speed"] = step / df["dt"]

    df["gap_flag"] = df["dt"] > GAP_FRAMES / fps
    df["outlier_flag"] = df["pixel_speed"] > OUTLIER_SPEED_PX_S
    return df


def speed_percentiles(speed):
    """Percentiles and max of a speed column, NaN rows (first rows) ignored."""
    speed = speed.dropna()
    stats = {f"p{p}": float(np.percentile(speed, p)) for p in PERCENTILES}
    stats["max"] = float(speed.max())
    return stats


def smoothing_effect(df, fps):
    """Report lines comparing raw and smoothed speed (measurement only).

    If smoothing changed the median by less than SMOOTH_NEGLIGIBLE_PCT, the
    input has no jitter worth removing. A "standing" row is one whose foot
    point moved less than STANDING_PX between 0.5 s before and 0.5 s after it;
    its raw speed is pure box jitter.
    """
    speed, smooth = df["pixel_speed"].dropna(), df["smooth_speed"].dropna()
    rows = ["## Smoothing effect (measurement only)", "",
            "| | raw px/s | smoothed px/s | difference |", "|---|---|---|---|"]
    for name, q in [("median", 50), ("p99", 99)]:
        raw, smo = np.percentile(speed, q), np.percentile(smooth, q)
        rows.append(f"| {name} | {raw:.1f} | {smo:.1f} | {100 * (raw - smo) / raw:.1f} % |")
    half = round(fps / 2)
    by_track = df.groupby("track_id")
    net = np.hypot(by_track["foot_x"].shift(-half) - by_track["foot_x"].shift(half),
                   by_track["foot_y"].shift(-half) - by_track["foot_y"].shift(half))
    standing = df.loc[net < STANDING_PX, "pixel_speed"].dropna()
    median_diff = 100 * (speed.median() - smooth.median()) / speed.median()
    rows += ["", f"Standing player (net movement < {STANDING_PX} px in 1 s, {len(standing)} rows): "
             f"raw speed mean {standing.mean():.1f}, p90 {standing.quantile(0.9):.1f} px/s.", ""]
    if median_diff < SMOOTH_NEGLIGIBLE_PCT:
        rows.append("GT input has no jitter; smoothing is not needed.")
    else:
        rows.append(f"Smoothing changes the median by {median_diff:.1f} % "
                    f"(>= {SMOOTH_NEGLIGIBLE_PCT} %): the smoothed columns are kept.")
    return rows + [""]


def write_consistency_report(df, out_md, fps, video_seconds):
    """Check the movement table and write a short Markdown report."""
    n, n_steps = len(df), int(df["dt"].notna().sum())
    life = df.groupby("track_id")["frame"].agg(lambda f: (f.max() - f.min() + 1) / fps)
    covered = (df["frame"].max() - df["frame"].min() + 1) / fps
    checks = {
        "timestamp == frame / fps": bool(np.allclose(df["timestamp"], df["frame"] / fps)),
        "frames increase inside every track": bool((df["dt"].dropna() > 0).all()),
        "no duplicate (track_id, frame)": not df.duplicated(["track_id", "frame"]).any(),
        f"covered time {covered:.2f} s ~ video {video_seconds:.2f} s": abs(covered - video_seconds) <= 1 / fps,
    }
    speed, smooth = df["pixel_speed"].dropna(), df["smooth_speed"].dropna()
    gaps, outliers = int(df["gap_flag"].sum()), int(df["outlier_flag"].sum())
    short = int((life < SHORT_TRACK_SECONDS).sum())
    pct = speed_percentiles(df["pixel_speed"])
    step = np.hypot(df["dx"], df["dy"])
    min_step = float(step[step > 0].min())   # box jitter cannot be smaller than this

    lines = ["# Movement consistency report (pixel space)", "",
             "Units: pixels and px/s only. Speeds are approximate and depend on the",
             "distance to the camera; no metric value is given before the homography.", "",
             "## Checks", ""]
    lines += [f"- [{'x' if ok else ' '}] {name}" for name, ok in checks.items()]
    lines += ["", "## Counts", "",
              f"- Rows: {n}, tracks: {df['track_id'].nunique()}, steps with a speed: {n_steps}",
              f"- Gap rows (dt > {GAP_FRAMES} frames): {gaps} ({100 * gaps / n_steps:.2f} %)",
              f"- Outlier rows (pixel_speed > {OUTLIER_SPEED_PX_S} px/s): {outliers} "
              f"({100 * outliers / n_steps:.2f} %)",
              f"- Short-lived tracks (< {SHORT_TRACK_SECONDS:.0f} s): {short}", "",
              "## Speed (px/s)", "",
              "| | mean | median | max |", "|---|---|---|---|",
              f"| raw | {speed.mean():.1f} | {speed.median():.1f} | {speed.max():.1f} |",
              f"| smoothed ({SMOOTH_WINDOW} frames) | {smooth.mean():.1f} | {smooth.median():.1f} "
              f"| {smooth.max():.1f} |", "",
              "Raw percentiles: " + ", ".join(f"{k} = {v:.1f}" for k, v in pct.items()), ""]
    lines += smoothing_effect(df, fps)
    lines += ["## Interpretation", "",
              f"All {len(checks)} checks {'pass' if all(checks.values()) else 'do NOT all pass'}. "
              f"There are {gaps} gaps and {short} short-lived tracks, so every track covers "
              f"{life.min():.1f}-{life.max():.1f} s of the {video_seconds:.0f} s clip. "
              f"The smallest non-zero raw step is {min_step:g} px per frame ({min_step * fps:g} px/s), "
              f"so box jitter alone produces speed: smoothing lowers the mean from {speed.mean():.1f} "
              f"to {smooth.mean():.1f} px/s. "
              f"{outliers} rows ({100 * outliers / n_steps:.2f} %) exceed {OUTLIER_SPEED_PX_S} px/s; "
              "they are flagged, not removed."]
    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checks


def main():
    """Compute movement for TRACKS_CSV, write movement.csv and the report."""
    fps, video_seconds = video_fps_and_duration(VIDEO_PATH)
    df = compute_movement(pd.read_csv(TRACKS_CSV), fps)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "movement.csv", index=False, float_format="%.3f")
    checks = write_consistency_report(df, out_dir / "consistency_report.md", fps, video_seconds)

    print(f"Input            : {TRACKS_CSV} ({len(df)} rows, {df['track_id'].nunique()} tracks)")
    print(f"fps / duration   : {fps:.2f} / {video_seconds:.2f} s")
    print("Raw speed (px/s) : " + ", ".join(
        f"{k} {v:.1f}" for k, v in speed_percentiles(df["pixel_speed"]).items()))
    print(f"Flags            : gap {int(df['gap_flag'].sum())}, "
          f"outlier {int(df['outlier_flag'].sum())} (> {OUTLIER_SPEED_PX_S} px/s)")
    print(f"Checks passed    : {sum(checks.values())} / {len(checks)}")
    print(f"Outputs in       : {OUT_DIR}")


if __name__ == "__main__":
    main()
