"""Foot point movement in pixel space, position and speed on the pitch (v4).

Reads a tracks.csv (here: the ground truth boxes with foot points) and adds,
per track, the time stamp, the step to the previous row (dx, dy, dt) and the
pixel speed. The movement columns are in pixels and pixels per second ONLY:
one pixel covers a different distance near and far from the camera.

Since the calibration exists, every foot point is also mapped to the pitch:
x_m, y_m in metres and the flag on_pitch. From these come the smoothed
position, the speed in m/s and km/h and the distance per frame. All metric
values are approximate (see the calibration fit report).

    python src/movement.py           movement table, summaries and figures
    python src/movement.py params    first choose smooth_frames and speed_window_s
                                     against the reference of the dataset and
                                     write them to config/field_config.yaml

The reference positions and speeds of the dataset are used only to check our
values and to choose the two parameters, never to produce a position or speed.

Movement uses foot_x, foot_y and not the box centre: the foot point is the
point that will be mapped to the pitch, and the box centre also moves when
only the box height changes (a player bending, a leg stretched out).

Project rule: rows are never deleted and never interpolated. Suspicious rows
only get flags (gap_flag, outlier_flag) next to the untouched raw columns;
the analytics stage decides what to use. Deleting or filling rows here would
hide measurement problems instead of showing them.
"""

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from calibrate import CONFIG_PATH, load_field_config, pitch_lines, pixel_to_pitch
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
# Reference of the dataset: position and speed of every player in every frame.
REFERENCE_XML = "data/referans/raw/118575/118575_tracker_box_data.xml"
VALIDATION_DIR = "outputs/validation/118575"    # parsed reference (.npy) and id_map.csv
VOTE_STEP = 10              # every 10th frame votes for the player of a track
MIN_VOTE_SHARE = 0.8        # below this a track is not clearly one player: warn
# Speed above 10 m/s (36 km/h) is flagged: the fastest reference speed in this
# clip is 8.2 m/s and top players reach about 10 m/s in a sprint, so a higher
# value is treated as an artefact of the boxes or the mapping. Flagged rows stay.
SPEED_FLAG_MPS = 10.0
MIN_HALF_WINDOW = 2         # speed needs positions at least 2 frames before and after
SMOOTH_GRID = [5, 9, 13, 25]        # smoothing windows tried (frames)
WINDOW_GRID = [0.5, 1.0]            # speed windows tried (seconds)
MAE_TIE_MPS = 0.02          # settings this close to the best speed error count as equal
CHART_SURFACE, CHART_INK, CHART_MUTED, CHART_GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
OURS_COLOR, REFERENCE_COLOR, RAW_COLOR = "#2a78d6", "#eb6834", "#b9b8b2"


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


def load_reference():
    """Reference positions and speeds of the dataset for the first half.

    Returns (frame numbers, player ids, data) with data[frame, player] =
    (x_m, y_m, speed in m/s). The XML gives a position as fractions of the
    pitch length and width, on the same axes as ours. Reading the 227 MB file
    takes a while, so the arrays are cached as .npy.
    """
    cache = Path(VALIDATION_DIR)
    files = [cache / f"reference_{part}.npy" for part in ("frames", "ids", "data")]
    if not all(f.exists() for f in files):
        pitch = load_field_config()["pitch"]
        frames, records = [], []
        for _, elem in ET.iterparse(REFERENCE_XML, events=("end",)):
            if elem.tag != "frame":
                continue
            if elem.get("eventPeriod") == "FIRST_HALF":
                frames.append(int(float(elem.get("frameNumber"))))
                records.append({int(p.get("playerId")): json.loads(p.get("loc"))
                                + [pd.to_numeric(p.get("speed"), errors="coerce")] for p in elem.iter("player")})
            elif frames:
                break                           # the second half begins: done
            elem.clear()                        # free the players of this frame
        ids = np.array(sorted({pid for record in records for pid in record}))
        data = np.full((len(frames), len(ids), 3), np.nan, np.float32)
        for i, record in enumerate(records):
            for pid, (x, y, speed) in record.items():
                data[i, np.searchsorted(ids, pid)] = (x * pitch["length_m"], y * pitch["width_m"], speed)
        cache.mkdir(parents=True, exist_ok=True)
        for f, array in zip(files, (np.array(frames), ids, data)):
            np.save(f, array)
    return [np.load(f) for f in files]


def reference_for_tracks(df):
    """Reference (x_m, y_m, speed) for every row of df, and the track -> player table.

    Clip frame j is reference frameNumber reference_offset + j. A GT track is
    the reference player it is matched with most often: in every VOTE_STEP-th
    frame our positions are matched to the reference positions so that the
    total distance is smallest (Hungarian matching), and each match is a vote.
    """
    frames, ids, data = load_reference()
    first_row = load_field_config()["reference_offset"] - int(frames[0])
    tracks = np.sort(df["track_id"].unique())
    votes = np.zeros((len(tracks), len(ids)), int)
    for frame, group in df[df["frame"] % VOTE_STEP == 0].groupby("frame"):
        ref = data[first_row + frame]
        present = np.flatnonzero(~np.isnan(ref[:, 0]))
        cost = np.hypot(group["x_m"].to_numpy()[:, None] - ref[present, 0],
                        group["y_m"].to_numpy()[:, None] - ref[present, 1])
        rows, cols = linear_sum_assignment(cost)
        votes[np.searchsorted(tracks, group["track_id"].to_numpy()[rows]), present[cols]] += 1
    player = votes.argmax(axis=1)
    id_map = pd.DataFrame({"track_id": tracks, "player_id": ids[player],
                           "vote_share": (votes.max(axis=1) / votes.sum(axis=1)).round(3)})
    id_map.to_csv(Path(VALIDATION_DIR) / "id_map.csv", index=False)
    if (id_map["vote_share"] < MIN_VOTE_SHARE).any() or id_map["player_id"].duplicated().any():
        print("Warning: the track -> player table is not clear, see id_map.csv")
    rows = data[first_row + df["frame"].to_numpy(), player[np.searchsorted(tracks, df["track_id"].to_numpy())]]
    return rows.astype(float), id_map


def add_metric_speed(df, fps, smooth_frames, speed_window_s):
    """Add x_m_smooth, y_m_smooth, speed_mps, speed_kmh, dist_m, speed_mps_raw and speed_flag.

    Everything is computed inside one piece of a track; a gap (gap_flag) starts
    a new piece, so neither the smoothing nor a speed window reaches across it.

    - x_m_smooth, y_m_smooth: centred moving average over smooth_frames frames.
      Far from the camera one pixel is about half a metre, so the raw
      positions jump; the average removes most of that.
    - speed_mps at frame t: distance between the smoothed positions k frames
      before and after t, divided by the time between them. 2k / fps is
      speed_window_s (k is rounded down to whole frames). Near the ends of a
      piece the widest symmetric window that fits is used; with fewer than
      MIN_HALF_WINDOW frames on a side the speed is left empty.
    - dist_m: distance covered in the frame, speed_mps times the time since the
      previous row. It is NOT the step between two smoothed positions: such
      steps add up the jitter that is left in the positions, and their sum was
      38 % above the reference distance, while speed times time is within 2 %.
    - speed_mps_raw: frame-to-frame speed of the unsmoothed positions, kept
      only to show what the smoothing does.
    - speed_flag: 1 if speed_mps > SPEED_FLAG_MPS. Flagged rows are kept.
    """
    pieces = [df["track_id"], df.groupby("track_id")["gap_flag"].cumsum().rename("piece")]
    for col in ("x_m", "y_m"):
        df[f"{col}_smooth"] = df.groupby(pieces)[col].transform(
            lambda s: s.rolling(smooth_frames, center=True, min_periods=1).mean())
    half = max(int(speed_window_s * fps / 2), MIN_HALF_WINDOW)
    position, time = df[["x_m_smooth", "y_m_smooth"]].to_numpy(), df["timestamp"].to_numpy()
    speed = np.full(len(df), np.nan)
    for rows in df.groupby(pieces).indices.values():        # the rows of one piece, in frame order
        i = np.arange(len(rows))
        k = np.minimum(half, np.minimum(i, len(rows) - 1 - i))
        ok = k >= MIN_HALF_WINDOW
        before, after = rows[i[ok] - k[ok]], rows[i[ok] + k[ok]]
        speed[rows[ok]] = np.hypot(*(position[after] - position[before]).T) / (time[after] - time[before])
    df["speed_mps"], df["speed_kmh"] = speed, speed * 3.6
    by_piece = df.groupby(pieces)
    df["dist_m"] = df["speed_mps"] * by_piece["timestamp"].diff()
    df["speed_mps_raw"] = np.hypot(by_piece["x_m"].diff(), by_piece["y_m"].diff()) / by_piece["timestamp"].diff()
    df["speed_flag"] = (df["speed_mps"] > SPEED_FLAG_MPS).astype(int)
    return df


def speed_scores(df, reference, fps):
    """Compare speed_mps and dist_m of df with the reference rows; return a dict of scores.

    Both speed series are averaged in 1 s bins per track (a bin needs more than
    half of its frames), so a small time shift between the two does not count.
    """
    table = pd.DataFrame({"track": df["track_id"], "second": (df["frame"] // round(fps)).to_numpy(),
                          "ours": df["speed_mps"].to_numpy(), "ref": reference[:, 2]})
    table = table[table["ours"].notna() & table["ref"].notna()]
    bins = table.groupby(["track", "second"]).agg(ours=("ours", "mean"), ref=("ref", "mean"), n=("ours", "size"))
    bins = bins[bins["n"] > fps / 2]
    steps = df["dist_m"].notna().to_numpy() & ~np.isnan(reference[:, 2])
    ours_m = df["dist_m"].where(steps, 0).groupby(df["track_id"]).sum()
    ref_m = pd.Series(np.where(steps, reference[:, 2], 0) / fps).groupby(df["track_id"].to_numpy()).sum()
    scores = {"mae": float((bins["ours"] - bins["ref"]).abs().mean()), "bias": float((bins["ours"] - bins["ref"]).mean()),
              "corr": float(np.corrcoef(bins["ours"], bins["ref"])[0, 1]), "bins": len(bins),
              "dist_diff_pct": float(((ours_m - ref_m).abs() / ref_m).mean() * 100)}
    for name in ("ours", "ref"):
        scores.update({f"p95_{name}": float(bins[name].quantile(0.95)), f"p99_{name}": float(bins[name].quantile(0.99)),
                       f"max_{name}": float(bins[name].max())})
    return scores


def choose_speed_parameters(df, fps, reference, out_md):
    """Try every pair of SMOOTH_GRID x WINDOW_GRID against the reference and pick one.

    Rule, fixed before looking at the numbers: among the pairs whose mean
    absolute error is within MAE_TIE_MPS of the best, take the one with the
    bias closest to zero; if that is still equal, the shorter smoothing window.
    Returns (smooth_frames, speed_window_s) and writes the table to out_md.
    """
    rows, peaks = [], {}
    for window in WINDOW_GRID:
        for smooth in SMOOTH_GRID:
            trial = add_metric_speed(df.copy(), fps, smooth, window)
            peaks[smooth, window] = trial.groupby("track_id")["speed_mps"].max()
            # Long windows flatten sprints: the fastest moment of every track against the 5-frame result.
            loss = float((peaks[smooth, window] / peaks[SMOOTH_GRID[0], window] - 1).mean() * 100)
            rows.append({"smooth_frames": smooth, "speed_window_s": window,
                         "window_frames": 2 * max(int(window * fps / 2), MIN_HALF_WINDOW),
                         **speed_scores(trial, reference, fps), "peak_vs_first_pct": loss})
    table = pd.DataFrame(rows)
    close = table[table["mae"] <= table["mae"].min() + MAE_TIE_MPS]
    best = close.assign(abs_bias=close["bias"].abs().round(3)).sort_values(["abs_bias", "smooth_frames"]).iloc[0]
    smooth, window = int(best["smooth_frames"]), float(best["speed_window_s"])

    lines = ["# Speed parameters against the reference", "",
             "All values are approximate. Our speed and the reference speed of the dataset are averaged in",
             f"1 s bins per track ({int(table['bins'].iloc[0])} bins). Bias = ours minus reference. Distance = per-track",
             "sum of dist_m (speed x time) against the reference speed summed over the same frames. Peak = fastest moment",
             f"of every track against the {SMOOTH_GRID[0]}-frame smoothing with the same speed window, mean over the tracks.", "",
             "| smooth_frames | speed_window_s (frames used) | MAE (m/s) | Bias (m/s) | Correlation | Ours p95 / p99 / max (m/s) "
             "| Distance difference (%) | Peak vs first (%) | Within the tie band |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        mark = "**" if (r["smooth_frames"], r["speed_window_s"]) == (smooth, window) else ""
        lines.append(f"| {mark}{int(r['smooth_frames'])}{mark} | {mark}{r['speed_window_s']:g}{mark} ({int(r['window_frames'])}) | "
                     f"{r['mae']:.3f} | {r['bias']:+.3f} | {r['corr']:.3f} | {r['p95_ours']:.2f} / {r['p99_ours']:.2f} / "
                     f"{r['max_ours']:.2f} | {r['dist_diff_pct']:.1f} | {r['peak_vs_first_pct']:+.1f} | "
                     f"{'yes' if r['mae'] <= table['mae'].min() + MAE_TIE_MPS else 'no'} |")
    lines += ["", f"Reference speed in the same bins: p95 {table['p95_ref'].iloc[0]:.2f}, p99 {table['p99_ref'].iloc[0]:.2f}, "
              f"max {table['max_ref'].iloc[0]:.2f} m/s.", "",
              f"Rule: among the pairs within {MAE_TIE_MPS:g} m/s of the lowest mean absolute error ({table['mae'].min():.3f} m/s), "
              "the bias closest to zero; if equal, the shorter smoothing window.", "",
              f"**Chosen: smooth_frames = {smooth}, speed_window_s = {window:g}** (mean absolute error {best['mae']:.3f} m/s, "
              f"bias {best['bias']:+.3f} m/s, peak {best['peak_vs_first_pct']:+.1f} % against {SMOOTH_GRID[0]} frames). "
              f"Written to `{CONFIG_PATH}`.", "",
              f"The rule looks at speed only. With the chosen pair the summed dist_m is {best['dist_diff_pct']:.1f} % away from "
              f"the reference distance; the smallest difference in the table is {table['dist_diff_pct'].min():.1f} %."]
    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return smooth, window, table


def write_config_value(key, value):
    """Replace the value of one top-level key of the config file; comments stay as they are."""
    with open(CONFIG_PATH, encoding="utf-8", newline="") as f:
        text = f.read()
    text, found = re.subn(rf"(?m)^({key}:[ ]*)[^ #\r\n]+", rf"\g<1>{value}", text)
    if found != 1:
        raise KeyError(f"'{key}' is not a top-level key of {CONFIG_PATH}")
    with open(CONFIG_PATH, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def write_speed_summary(df, out_csv):
    """One row per track: distance, speed statistics in km/h and the share of flagged rows."""
    by_track = df.groupby("track_id")
    summary = pd.DataFrame({
        "distance_m": by_track["dist_m"].sum(), "mean_kmh": by_track["speed_kmh"].mean(),
        "median_kmh": by_track["speed_kmh"].median(), "p95_kmh": by_track["speed_kmh"].quantile(0.95),
        "max_kmh": by_track["speed_kmh"].max(), "flagged_share_pct": 100 * by_track["speed_flag"].mean()})
    summary.round(2).to_csv(out_csv)
    return summary


def save_speed_figure(df, reference, settings_text, path):
    """(a) speed distribution, ours and reference; (b) the track with the fastest reference moment over time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5), facecolor=CHART_SURFACE, width_ratios=[1, 1.5])
    for ax in (left, right):
        ax.set_facecolor(CHART_SURFACE)
        ax.grid(True, color=CHART_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=CHART_MUTED, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(CHART_GRID)
    edges = np.arange(0, 38, 1.0)
    left.hist(reference[:, 2] * 3.6, bins=edges, histtype="step", linewidth=2, color=REFERENCE_COLOR, label="reference")
    left.hist(df["speed_kmh"].dropna(), bins=edges, histtype="step", linewidth=2, color=OURS_COLOR, label="ours")
    left.set_xlabel("speed (km/h), every frame of every track", color=CHART_MUTED)
    left.set_ylabel("rows", color=CHART_MUTED)
    left.set_title("(a) Speed distribution", color=CHART_INK, fontsize=11, loc="left", pad=30)
    left.legend(frameon=False, labelcolor=CHART_MUTED)

    track_id = df.loc[int(np.nanargmax(reference[:, 2])), "track_id"]      # the track with the fastest reference moment
    rows = (df["track_id"] == track_id).to_numpy()
    time = df.loc[rows, "timestamp"]
    top = 1.25 * max(df.loc[rows, "speed_kmh"].max(), np.nanmax(reference[rows, 2]) * 3.6)
    right.plot(time, df.loc[rows, "speed_mps_raw"] * 3.6, color=RAW_COLOR, linewidth=0.4, alpha=0.6,
               label="ours, raw (frame to frame)")
    right.plot(time, reference[rows, 2] * 3.6, color=REFERENCE_COLOR, linewidth=2, label="reference")
    right.plot(time, df.loc[rows, "speed_kmh"], color=OURS_COLOR, linewidth=2, label=f"ours ({settings_text})")
    right.set_ylim(0, top)
    right.set_xlabel("time (s)", color=CHART_MUTED)
    right.set_ylabel("speed (km/h)", color=CHART_MUTED)
    right.set_title(f"(b) Track {track_id} over time (raw values above {top:.0f} km/h are cut off)", color=CHART_INK,
                    fontsize=11, loc="left", pad=30)
    legend = right.legend(frameon=False, labelcolor=CHART_MUTED, ncol=3, loc="lower left", bbox_to_anchor=(0, 0.995),
                          borderaxespad=0)
    legend.legend_handles[0].set_linewidth(2)
    legend.legend_handles[0].set_alpha(1)
    fig.suptitle("Metric speed of the foot points against the reference of the dataset (approximate)", color=CHART_INK,
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130, facecolor=CHART_SURFACE)
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
    """Compute movement for TRACKS_CSV; write movement.csv, the reports and the figures.

    With the argument "params" the two speed parameters are first chosen
    against the reference and written to the config file.
    """
    fps, video_seconds = video_fps_and_duration(VIDEO_PATH)
    df = compute_movement(pd.read_csv(TRACKS_CSV), fps)
    df = add_pitch_position(df)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    reference, id_map = reference_for_tracks(df)
    if "params" in sys.argv[1:]:
        smooth, window, _ = choose_speed_parameters(df, fps, reference, out_dir / "speed_param_test.md")
        write_config_value("smooth_frames", smooth)
        write_config_value("speed_window_s", f"{window:g}")
        print(f"Parameters       : smooth_frames = {smooth}, speed_window_s = {window:g} chosen against the reference "
              f"and written to {CONFIG_PATH} (table: {out_dir / 'speed_param_test.md'})")
    config = load_field_config()
    smooth, window = int(config["smooth_frames"]), float(config["speed_window_s"])
    df = add_metric_speed(df, fps, smooth, window)
    df.to_csv(out_dir / "movement.csv", index=False, float_format="%.3f")
    summary = write_speed_summary(df, out_dir / "speed_summary.csv")
    save_speed_figure(df, reference, f"{smooth} frames, {window:g} s", out_dir / "speed_metric.png")
    scores = speed_scores(df, reference, fps)
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
    flagged = int(df["speed_flag"].sum())
    print(f"Metric speed     : smooth_frames {smooth}, speed_window_s {window:g}; median {df['speed_kmh'].median():.1f}, "
          f"p95 {df['speed_kmh'].quantile(0.95):.1f}, max {df['speed_kmh'].max():.1f} km/h (approximate)")
    print(f"speed_flag = 1   : {flagged} rows ({100 * flagged / len(df):.2f} %), above {SPEED_FLAG_MPS:g} m/s; "
          f"rows without a speed: {int(df['speed_mps'].isna().sum())}")
    print(f"Distance         : {summary['distance_m'].min():.0f} .. {summary['distance_m'].max():.0f} m per track, "
          f"mean {summary['distance_m'].mean():.0f} m (sum of dist_m = speed x time)")
    print(f"Against reference: speed error {scores['mae']:.3f} m/s, bias {scores['bias']:+.3f} m/s, correlation "
          f"{scores['corr']:.3f}, distance difference {scores['dist_diff_pct']:.1f} % "
          f"(track -> player votes: lowest share {100 * id_map['vote_share'].min():.1f} %)")
    print(f"Outputs in       : {OUT_DIR}")


if __name__ == "__main__":
    main()
