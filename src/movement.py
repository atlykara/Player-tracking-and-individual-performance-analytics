"""Foot point movement in pixel space, and the position on the pitch (v2).

Reads a tracks.csv (here: the ground truth boxes with foot points) and adds,
per track, the time stamp, the step to the previous row (dx, dy, dt) and the
pixel speed. The movement columns are in pixels and pixels per second ONLY:
one pixel covers a different distance near and far from the camera.

Since the calibration exists, every foot point is also mapped to the pitch:
x_m, y_m in metres and the flag on_pitch. These are positions only; metric
speed and metric smoothing are not computed here yet. The positions are
approximate (see the calibration fit report).

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

from calibrate import load_field_config, pitch_lines, pixel_to_pitch
from draw_boxes_video import id_color

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
# A position up to 2 m outside the lines still counts as on the pitch: players
# do step over the lines, and the mapping itself is off by 1-2 m in places.
PITCH_MARGIN_M = 2.0
FIGURE_SURFACE, FIGURE_LINE, FIGURE_TEXT = "#1a1a19", "#c3c2b7", "#f0efec"   # dark: the track colours are bright


def video_fps_and_duration(path):
    """Return (fps, duration in seconds) of the video."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps, n_frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return fps, n_frames / fps


def compute_movement(df, fps, window=SMOOTH_WINDOW):
    """Add timestamp, dx, dy, dt, pixel_speed, smoothed values and flags.

    window is the smoothing window in frames (default SMOOTH_WINDOW); it is a
    parameter so that other window lengths can be compared on the same input.

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
            lambda s: s.rolling(window, center=True, min_periods=1).mean())
    by_track = df.groupby("track_id")
    step = np.hypot(by_track["smooth_x"].diff(), by_track["smooth_y"].diff())
    df["smooth_speed"] = step / df["dt"]

    df["gap_flag"] = df["dt"] > GAP_FRAMES / fps
    df["outlier_flag"] = df["pixel_speed"] > OUTLIER_SPEED_PX_S
    return df


def add_pitch_position(df):
    """Add x_m, y_m (metres on the pitch) and on_pitch to the movement table.

    x_m runs along the pitch from the left goal line (0) to the right one;
    y_m runs across it from the far touchline (0) to the touchline on the
    camera side. The RAW foot point is mapped: smoothing in metres is a later
    step, and the pixel columns stay as they are.

    on_pitch = 1 if the position lies on the pitch or at most PITCH_MARGIN_M
    outside it. Project rule: a row with on_pitch = 0 is kept and only flagged.
    """
    pitch = load_field_config()["pitch"]
    metres = pixel_to_pitch(df[["foot_x", "foot_y"]].to_numpy())
    df["x_m"], df["y_m"] = metres[:, 0], metres[:, 1]
    inside = (df["x_m"].between(-PITCH_MARGIN_M, pitch["length_m"] + PITCH_MARGIN_M)
              & df["y_m"].between(-PITCH_MARGIN_M, pitch["width_m"] + PITCH_MARGIN_M))
    df["on_pitch"] = inside.astype(int)
    return df


def save_positions_figure(df, path):
    """Top view of the pitch with every metric position, one colour per track.

    The camera side is at the bottom, as in the video. A track has the same
    colour as in the box videos (id_color). The dashed frame is the on_pitch
    limit; positions outside it are marked with a white x.
    """
    import matplotlib
    matplotlib.use("Agg")                   # writes the file, opens no window
    import matplotlib.pyplot as plt

    config = load_field_config()
    length, width = float(config["pitch"]["length_m"]), float(config["pitch"]["width_m"])
    _, markings = pitch_lines(config["landmarks"], length, width)
    fig, ax = plt.subplots(figsize=(13, 9.4), facecolor=FIGURE_SURFACE)
    ax.set_facecolor(FIGURE_SURFACE)
    for line in markings:
        ax.plot(line[:, 0], line[:, 1], color=FIGURE_LINE, linewidth=1.2)
    m = PITCH_MARGIN_M
    ax.plot([-m, length + m, length + m, -m, -m], [-m, -m, width + m, width + m, -m],
            color=FIGURE_LINE, linewidth=0.8, linestyle="--")
    for track_id, track in df.groupby("track_id"):
        b, g, r = id_color(track_id)
        ax.plot(track["x_m"], track["y_m"], ".", markersize=1.5, color=(r / 255, g / 255, b / 255),
                label=str(track_id))
    off = df[df["on_pitch"] == 0]
    if len(off):
        ax.plot(off["x_m"], off["y_m"], "x", markersize=5, color="white", label="on_pitch = 0")
    ax.set_xlim(-6, length + 6)
    ax.set_ylim(width + 6, -6)              # y downwards: far touchline at the top, camera at the bottom
    ax.set_aspect("equal")
    ax.set_xlabel("x_m (0 = left goal line)", color=FIGURE_TEXT)
    ax.set_ylabel(f"y_m (0 = far touchline, {width:g} = camera side)", color=FIGURE_TEXT)
    ax.tick_params(colors=FIGURE_TEXT)
    for side in ax.spines.values():
        side.set_color(FIGURE_SURFACE)
    ax.set_title(f"Foot point positions on the pitch: {len(df)} positions, {df['track_id'].nunique()} tracks "
                 f"(approximate; dashed = on_pitch limit, {PITCH_MARGIN_M:g} m outside the lines)",
                 color=FIGURE_TEXT, fontsize=11, loc="left")
    legend = ax.legend(title="track id", ncol=12, markerscale=8, loc="upper center", bbox_to_anchor=(0.5, -0.09),
                       frameon=False, labelcolor=FIGURE_TEXT, fontsize=9)
    legend.get_title().set_color(FIGURE_TEXT)
    if len(off):
        legend.legend_handles[-1].set_markersize(6)     # the x is already large enough
    fig.savefig(path, dpi=130, facecolor=FIGURE_SURFACE, bbox_inches="tight")
    plt.close(fig)


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
    """Compute movement for TRACKS_CSV, write movement.csv, the report and the pitch figure."""
    fps, video_seconds = video_fps_and_duration(VIDEO_PATH)
    df = compute_movement(pd.read_csv(TRACKS_CSV), fps)
    df = add_pitch_position(df)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "movement.csv", index=False, float_format="%.3f")
    save_positions_figure(df, out_dir / "positions_pitch.png")
    checks = write_consistency_report(df, out_dir / "consistency_report.md", fps, video_seconds)

    print(f"Input            : {TRACKS_CSV} ({len(df)} rows, {df['track_id'].nunique()} tracks)")
    print(f"fps / duration   : {fps:.2f} / {video_seconds:.2f} s")
    print("Raw speed (px/s) : " + ", ".join(
        f"{k} {v:.1f}" for k, v in speed_percentiles(df["pixel_speed"]).items()))
    print(f"Flags            : gap {int(df['gap_flag'].sum())}, "
          f"outlier {int(df['outlier_flag'].sum())} (> {OUTLIER_SPEED_PX_S} px/s)")
    print(f"Checks passed    : {sum(checks.values())} / {len(checks)}")
    off = int((df["on_pitch"] == 0).sum())
    print(f"Pitch position   : x_m {df['x_m'].min():.2f} .. {df['x_m'].max():.2f} m, "
          f"y_m {df['y_m'].min():.2f} .. {df['y_m'].max():.2f} m (approximate)")
    print(f"on_pitch = 0     : {off} rows ({100 * off / len(df):.2f} %), more than {PITCH_MARGIN_M:g} m outside the lines")
    print(f"Outputs in       : {OUT_DIR}")


if __name__ == "__main__":
    main()
