"""Tracking quality without ground truth (v1).

Looks only at the tracker output and measures signs of good or bad tracking:
how many boxes per frame, how many IDs, how long IDs live, holes inside a
track, impossible jumps and detection confidence. The ground truth is NOT
used here; it is kept for a later, independent evaluation.
"""

import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # write PNG files, no window
import matplotlib.pyplot as plt
import numpy as np

from tracking_io import load_ultralytics_tracks

# --- Settings -------------------------------------------------------------
LABELS_DIR = "outputs/tracking/118575_yolo11m_botsort/labels"
VIDEO_PATH = "data/SoccerTrackv2/mot/118575.mp4"
OUTPUT_DIR = "outputs/evaluation/118575_yolo11m_botsort"

SHORT_TRACK_SEC = 2.0     # a real player stays visible far longer than 2 s
LOW_CONF = 0.4            # confidence below this counts as a weak detection
LIFETIME_BINS_SEC = [0, 1, 5, 30, np.inf]
# Rough image scale (no calibration yet): the 105 m near touchline spans about
# 3900 px, i.e. ~38 px/m at most. A sprinting player (~10 m/s) moves 0.4 m per
# frame at 25 fps, ~15 px. 50 px per frame is ~1.3 m/frame (~33 m/s), which
# no human can do even with box jitter, so it marks a tracking error.
JUMP_THRESHOLD_PX = 50


def video_info(video_path):
    """Return (fps, frame_count) read from the video file."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, frame_count


def compute_metrics(tracks, fps, frame_count):
    """Compute all quality numbers; also return series needed for the plots."""
    m = {}
    # Detections per frame; frames without any box count as 0.
    per_frame = tracks.groupby("frame").size().reindex(range(frame_count), fill_value=0)
    m["frames"] = frame_count
    m["detections"] = len(tracks)
    m["det_per_frame_mean"] = per_frame.mean()
    m["det_per_frame_std"] = per_frame.std()
    m["det_per_frame_min"] = int(per_frame.min())
    m["det_per_frame_max"] = int(per_frame.max())
    m["frames_without_detection"] = int((per_frame == 0).sum())

    m["unique_ids"] = int(tracks["track_id"].nunique())
    m["id_ratio"] = m["unique_ids"] / m["det_per_frame_mean"]

    # Lifetime = first to last appearance, inclusive, in seconds.
    span = tracks.groupby("track_id")["frame"].agg(["min", "max", "nunique"])
    span_frames = span["max"] - span["min"] + 1
    lifetime = span_frames / fps
    m["lifetime_mean_sec"] = lifetime.mean()
    m["lifetime_median_sec"] = lifetime.median()
    counts = np.histogram(lifetime, bins=LIFETIME_BINS_SEC)[0]
    for name, n in zip(["lt_1s", "1_5s", "5_30s", "gt_30s"], counts):
        m[f"lifetime_{name}"] = int(n)
    m["short_tracks"] = int((lifetime < SHORT_TRACK_SEC).sum())
    m["short_tracks_pct"] = 100 * m["short_tracks"] / m["unique_ids"]

    # Gaps: frames between first and last appearance where the ID is missing.
    missing = span_frames - span["nunique"]
    m["tracks_with_gaps"] = int((missing > 0).sum())
    m["missing_frames_total"] = int(missing.sum())

    # Jumps: box centre moves too far between two consecutive frames of one ID.
    t = tracks.sort_values(["track_id", "frame"])
    cx, cy = t["x"] + t["w"] / 2, t["y"] + t["h"] / 2
    same_id = t["track_id"].diff() == 0
    next_frame = t["frame"].diff() == 1
    dist = np.hypot(cx.diff(), cy.diff())
    m["jumps"] = int((same_id & next_frame & (dist > JUMP_THRESHOLD_PX)).sum())

    m["conf_mean"] = tracks["conf"].mean()
    m["conf_median"] = tracks["conf"].median()
    m["conf_below_low_pct"] = 100 * (tracks["conf"] < LOW_CONF).mean()

    m = {k: round(float(v), 3) if isinstance(v, float) else v for k, v in m.items()}
    return m, per_frame, lifetime


# Metric key -> (label, one-sentence meaning) for the Markdown report.
DESCRIPTIONS = {
    "frames": ("Frames", "Number of frames in the video."),
    "detections": ("Tracked boxes", "Total boxes with a track ID over all frames."),
    "det_per_frame_mean": ("Boxes per frame (mean)", "Average number of people tracked in one frame."),
    "det_per_frame_std": ("Boxes per frame (std)", "How much that number changes from frame to frame."),
    "det_per_frame_min": ("Boxes per frame (min)", "Fewest boxes seen in a single frame."),
    "det_per_frame_max": ("Boxes per frame (max)", "Most boxes seen in a single frame."),
    "frames_without_detection": ("Frames without boxes", "Frames where the tracker reported nobody."),
    "unique_ids": ("Unique track IDs", "Number of different IDs handed out in the clip."),
    "id_ratio": ("ID ratio", "Unique IDs divided by mean boxes per frame; 1.0 would mean every person kept one ID."),
    "lifetime_mean_sec": ("Track lifetime mean (s)", "Average time from first to last appearance of an ID."),
    "lifetime_median_sec": ("Track lifetime median (s)", "Typical lifetime, less affected by a few long tracks."),
    "lifetime_lt_1s": ("Tracks < 1 s", "IDs that live less than one second."),
    "lifetime_1_5s": ("Tracks 1-5 s", "IDs that live between 1 and 5 seconds."),
    "lifetime_5_30s": ("Tracks 5-30 s", "IDs that live between 5 and 30 seconds."),
    "lifetime_gt_30s": ("Tracks > 30 s", "IDs that live longer than 30 seconds."),
    "short_tracks": (f"Short tracks (< {SHORT_TRACK_SEC} s)", "IDs too short to belong to a real, continuously visible player."),
    "short_tracks_pct": ("Short tracks (%)", "Share of all IDs that are short."),
    "tracks_with_gaps": ("Tracks with gaps", "IDs that disappear for some frames and come back with the same ID."),
    "missing_frames_total": ("Missing frames in tracks", "Total frames missing inside those gaps."),
    "jumps": (f"Jumps (> {JUMP_THRESHOLD_PX} px/frame)", "Consecutive-frame moves of one ID that are physically impossible, likely ID switches."),
    "conf_mean": ("Confidence mean", "Average detector confidence of tracked boxes."),
    "conf_median": ("Confidence median", "Typical detector confidence."),
    "conf_below_low_pct": (f"Confidence < {LOW_CONF} (%)", "Share of boxes the detector was unsure about."),
}


def write_report(m, path):
    """Write the metrics as an English Markdown table."""
    lines = ["# Tracking quality report (no ground truth)", "",
             f"Labels: `{LABELS_DIR}`  ", f"Video: `{VIDEO_PATH}`", "",
             "| Metric | Value | Meaning |", "|---|---|---|"]
    lines += [f"| {label} | {m[key]} | {text} |" for key, (label, text) in DESCRIPTIONS.items()]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_plots(per_frame, lifetime, fps, out_dir):
    """Boxes per frame over time, and a histogram of track lifetimes."""
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(per_frame.index / fps, per_frame.values, linewidth=0.8)
    ax.set(xlabel="Time (s)", ylabel="Tracked boxes", title="Tracked boxes per frame")
    fig.tight_layout()
    fig.savefig(out_dir / "detections_per_frame.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lifetime, bins=60)
    ax.set_yscale("log")  # many very short tracks would hide the long ones
    ax.axvline(SHORT_TRACK_SEC, color="red", linestyle="--", label=f"{SHORT_TRACK_SEC} s")
    ax.set(xlabel="Track lifetime (s)", ylabel="Number of tracks (log)", title="Track lifetimes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "track_lifetimes.png", dpi=120)
    plt.close(fig)


def evaluate():
    """Load tracks, compute metrics, write JSON, Markdown report and plots."""
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    fps, frame_count = video_info(VIDEO_PATH)
    tracks = load_ultralytics_tracks(LABELS_DIR, VIDEO_PATH)

    m, per_frame, lifetime = compute_metrics(tracks, fps, frame_count)
    (out_dir / "quality_metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    write_report(m, out_dir / "quality_report.md")
    save_plots(per_frame, lifetime, fps, out_dir)

    for key, (label, _) in DESCRIPTIONS.items():
        print(f"{label:<32}: {m[key]}")
    print(f"Outputs saved to {out_dir}")


if __name__ == "__main__":
    evaluate()
