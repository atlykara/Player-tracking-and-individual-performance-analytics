"""Read Ultralytics tracking labels into one table (v1).

`yolo track ... save_txt=True save_conf=True` writes one text file per frame,
`<video>_<N>.txt`, with one row per tracked box:

    class x_center y_center width height confidence track_id

Box values are normalised to [0, 1]. This module turns all files into one
pandas DataFrame with pixel boxes in the original video resolution:

    frame, track_id, x, y, w, h, conf      (x, y = top-left corner, px)
"""

import re
from pathlib import Path

import cv2
import pandas as pd

# --- Settings -------------------------------------------------------------
LABELS_DIR = "outputs/tracking/118575_yolo11m_botsort/labels"
VIDEO_PATH = "data/SoccerTrackv2/mot/118575.mp4"
CSV_PATH = "outputs/tracking/118575_yolo11m_botsort/tracks.csv"

# Ultralytics numbers label files from 1 (frame 1 = first frame), but the
# SoccerTrack v2 ground truth for this clip numbers frames from 0 (0 ... 2999,
# measured). Subtracting 1 puts both in the same 0-based frame system.
FRAME_OFFSET = -1

LABEL_COLUMNS = ["cls", "xc", "yc", "wn", "hn", "conf", "track_id"]


def video_size(video_path):
    """Return (width, height) of the video in pixels."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


def load_ultralytics_tracks(labels_dir, video_path):
    """Read every label file and return one DataFrame in pixel coordinates.

    Frames with no detection have no label file (or an empty one); they are
    simply absent from the table, which is not an error.
    """
    width, height = video_size(video_path)

    tables = []
    for path in Path(labels_dir).glob("*.txt"):
        if path.stat().st_size == 0:
            continue
        frame = int(re.search(r"_(\d+)\.txt$", path.name).group(1)) + FRAME_OFFSET
        table = pd.read_csv(path, sep=" ", header=None, names=LABEL_COLUMNS)
        table["frame"] = frame
        tables.append(table)
    if not tables:
        raise ValueError(f"No label rows found in {labels_dir}")
    raw = pd.concat(tables, ignore_index=True)

    # A box without a track ID cannot be followed over time; report and drop it.
    missing_id = raw["track_id"].isna().sum()
    raw = raw.dropna(subset=["track_id"])

    # Normalised centre/size -> top-left corner and size in pixels.
    tracks = pd.DataFrame({
        "frame": raw["frame"],
        "track_id": raw["track_id"].astype(int),
        "x": (raw["xc"] - raw["wn"] / 2) * width,
        "y": (raw["yc"] - raw["hn"] / 2) * height,
        "w": raw["wn"] * width,
        "h": raw["hn"] * height,
        "conf": raw["conf"],
    })
    tracks = tracks.sort_values(["frame", "track_id"]).reset_index(drop=True)

    print(f"Video size       : {width} x {height}")
    print(f"Label files read : {len(tables)}")
    print(f"Rows             : {len(tracks)} (dropped {missing_id} without track ID)")
    print(f"Frame range      : {tracks['frame'].min()} ... {tracks['frame'].max()} (0-based)")
    return tracks


if __name__ == "__main__":
    tracks = load_ultralytics_tracks(LABELS_DIR, VIDEO_PATH)
    tracks.round(2).to_csv(CSV_PATH, index=False)
    print(f"Saved to         : {CSV_PATH}")
