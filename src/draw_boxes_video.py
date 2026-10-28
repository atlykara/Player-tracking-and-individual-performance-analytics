"""Draw tracked boxes, foot points and trails on the clip (v4).

v1 drew only the ground truth. v2 made the drawing generic: any table with
the columns `frame, track_id, x, y, w, h` (pixels, top-left origin, frame
0-based) can be drawn, so the ground truth and the tracker output
(`tracks.csv`) go through the same code and look the same. v3 adds the foot
point (a small filled dot, the point that later goes to the pitch) and
chooses the table with one constant, SOURCE. v4 draws the last few seconds
of every foot point as a fading trail, read from movement.py's output, so
movement and its jitter can be seen in the video.

Every ID keeps one colour for the whole video and shows its number above
the box, so a viewer can follow one player. The GT is read with
evaluate_tracking.load_ground_truth() and its foot points come from
track.foot_point(), so the GT reader and the foot point formula each exist
once.
"""

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from evaluate_tracking import load_ground_truth
from select_roi import load_roi
from track import add_foot_columns

# --- Settings -------------------------------------------------------------
# The project continues on the ground truth boxes and IDs, so "gt" is the
# default; "tracking" (ROI run's tracks.csv) is kept for comparison.
SOURCE = "gt"                    # "gt" or "tracking"
VIDEO_PATH = "data/SoccerTrackv2/mot/118575.mp4"
GT_PATH = "data/SoccerTrackv2/mot/118575_gt.txt"
RUN_NAME = "118575_yolo11m_botsort_roi"   # tracking run whose tracks.csv is drawn
TRACKS_PATH = f"outputs/tracking/{RUN_NAME}/tracks.csv"
ROI_PATH = "data/roi/118575_roi.json"
# The GT video is the project's current result video; every new stage
# overwrites it. The tracker video stays in its run folder for comparison.
OUTPUT_PATHS = {"gt": "outputs/118575_latest.mp4",
                "tracking": f"outputs/tracking/{RUN_NAME}/118575_tracks.mp4"}
FRAME_IMAGE_PATH = "outputs/118575_latest_frame.jpg"   # middle frame of the current video
DRAW_ROI_OUTLINE = True          # thin yellow ROI polygon
DRAW_FOOT_POINT = True           # small filled dot at the foot point
ROI_COLOR = (0, 255, 255)        # BGR yellow
FOOT_RADIUS = 4                  # px; visible at 4096 px width, smaller than a player
LINE_WIDTH = 2
FONT_SCALE = 0.7
INVALID_COLOR = (128, 128, 128)   # rows with is_valid = False: grey and thin
H264_CRF = 23                     # ffmpeg quality; 23 is the usual default
# Foot point trails from movement.py; drawn only if the table exists.
DRAW_TRAILS = True
MOVEMENT_PATHS = {"gt": "outputs/movement/118575_gt/movement.csv"}
TRAIL_SECONDS = 3.0              # length of the trail behind each player
TRAIL_FADE_STEPS = 4             # opacity levels; each costs one blend of the frame


def id_color(track_id):
    """Fixed, well-separated BGR colour for a track ID.

    The hue walks around OpenCV's 0-179 hue wheel in steps of 37 (180 and 37
    share no factor), so neighbouring IDs get clearly different colours.
    """
    hsv = np.uint8([[[(track_id * 37) % 180, 220, 255]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def draw_frame(img, rows, frame, title):
    """Draw one frame's rows and a counter '<title> frame N boxes K' top-left."""
    for row in rows.itertuples():
        x, y, w, h = int(row.x), int(row.y), int(row.w), int(row.h)
        if row.is_valid:
            color, width = id_color(row.track_id), LINE_WIDTH
        else:
            color, width = INVALID_COLOR, 1
        cv2.rectangle(img, (x, y), (x + w, y + h), color, width)
        cv2.putText(img, str(row.track_id), (x, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, 2)
        if DRAW_FOOT_POINT:
            cv2.circle(img, (round(row.foot_x), round(row.foot_y)), FOOT_RADIUS, color, -1, cv2.LINE_AA)
    n_valid = int(rows["is_valid"].sum())
    cv2.putText(img, f"{title}  frame {frame}  boxes {n_valid}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)


def load_trails(source):
    """Return {track_id: (frames, xy points)} from movement.csv, or None."""
    path = MOVEMENT_PATHS.get(source)
    if not DRAW_TRAILS or path is None or not Path(path).exists():
        print(f"Trails           : off (no movement table for {source!r})")
        return None
    cols = ["foot_x", "foot_y"]
    table = pd.read_csv(path, usecols=["frame", "track_id", *cols]).sort_values("frame")
    print(f"Trails           : {path} ({cols[0]}, {cols[1]}), {TRAIL_SECONDS:g} s")
    return {tid: (g["frame"].to_numpy(), g[cols].to_numpy())
            for tid, g in table.groupby("track_id")}


def draw_trails(img, trails, frame, fps):
    """Draw the last TRAIL_SECONDS of every track: newest point large and opaque.

    Points are split into TRAIL_FADE_STEPS age bands. Each band is drawn on a
    copy of the frame and blended in with its own opacity, oldest band first,
    so a few blends give a smooth fade instead of one blend per point.
    """
    n = round(TRAIL_SECONDS * fps)
    for band in range(TRAIL_FADE_STEPS):          # 0 = oldest ... last = newest
        overlay = img.copy()
        for tid, (frames, xy) in trails.items():
            lo, hi = np.searchsorted(frames, [frame - n + 1, frame + 1])
            for f, (x, y) in zip(frames[lo:hi], xy[lo:hi]):
                fresh = 1 - (frame - f) / n        # 1 = current frame, -> 0 = oldest
                if min(int(fresh * TRAIL_FADE_STEPS), TRAIL_FADE_STEPS - 1) == band:
                    radius = max(1, round(FOOT_RADIUS * fresh))
                    cv2.circle(overlay, (round(x), round(y)), radius, id_color(tid), -1, cv2.LINE_AA)
        alpha = (band + 1) / TRAIL_FADE_STEPS
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)


def video_info(path):
    """Return (width, height, fps, frame count) of a video."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    info = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            round(cap.get(cv2.CAP_PROP_FPS), 2), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return info


def write_video(boxes, video_path, output_path, title, roi=None, frame_image_path=None,
                trails=None):
    """Read the clip frame by frame, draw the boxes and write an mp4 of the same size/fps.

    `boxes` needs the columns frame, track_id, x, y, w, h, foot_x, foot_y. An
    optional `is_valid` column (from the GT reader) marks rows drawn grey;
    tables without it, like tracks.csv, are treated as all valid. `roi` (N x 2
    pixel polygon) is drawn as a thin outline when given. If
    `frame_image_path` is given, the middle frame is also saved as an image.
    `trails` (from load_trails) are drawn under the boxes.
    """
    if "is_valid" not in boxes.columns:
        boxes = boxes.assign(is_valid=True)
    width, height, fps, n_frames = video_info(video_path)
    size = (width, height)
    cap = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    by_frame = dict(list(boxes.groupby("frame")))
    empty = boxes.iloc[:0]
    frame = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if trails:
            draw_trails(img, trails, frame, fps)
        draw_frame(img, by_frame.get(frame, empty), frame, title)
        if roi is not None:
            cv2.polylines(img, [roi], True, ROI_COLOR, 1, cv2.LINE_AA)
        if frame_image_path and frame == n_frames // 2:
            cv2.imwrite(str(frame_image_path), img)
            print(f"Middle frame     : {frame_image_path}")
        out.write(img)
        frame += 1
    cap.release()
    out.release()
    print(f"Frames written   : {frame} ({size[0]} x {size[1]}, {fps:.2f} fps)")


def compress_with_ffmpeg(path):
    """Re-encode with H.264 if ffmpeg exists; OpenCV's mp4v files are very large."""
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found  : video left as written by OpenCV (mp4v)")
        return
    tmp = path.with_name(path.stem + "_h264.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
                    "-c:v", "libx264", "-crf", str(H264_CRF), "-an", str(tmp)], check=True)
    tmp.replace(path)
    print(f"Compressed (H.264): {path.stat().st_size / 1e6:.1f} MB")


def check_same_format(source_path, output_path):
    """Fail if the output's resolution, fps or frame count differ from the source."""
    src, dst = video_info(source_path), video_info(output_path)
    print(f"Source / output  : {src} / {dst}  (width, height, fps, frames)")
    if src != dst:
        raise RuntimeError(f"Output video differs from source: {src} vs {dst}")


def draw_boxes(boxes, output_path, title, video_path=VIDEO_PATH, roi=None, frame_image_path=None,
               trails=None):
    """Draw one box table on the clip, compress and check the result."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"{title:<17}: {len(boxes)} boxes, {boxes['track_id'].nunique()} IDs")
    write_video(boxes, video_path, output, title, roi, frame_image_path, trails)
    compress_with_ffmpeg(output)
    check_same_format(video_path, output)
    print(f"Saved to         : {output}")


def main(source=SOURCE):
    """Draw the table chosen by `source` ("tracking" or "gt") with the same function."""
    if source == "gt":
        boxes, frame_image = load_ground_truth(GT_PATH), FRAME_IMAGE_PATH
    elif source == "tracking":
        boxes, frame_image = pd.read_csv(TRACKS_PATH), None
    else:
        raise ValueError(f"SOURCE must be 'tracking' or 'gt', not {source!r}")
    if "foot_x" not in boxes.columns:   # the GT has no foot point; same formula as tracking
        boxes = add_foot_columns(boxes)
    roi = load_roi(ROI_PATH) if DRAW_ROI_OUTLINE else None
    draw_boxes(boxes, OUTPUT_PATHS[source], source, roi=roi, frame_image_path=frame_image,
               trails=load_trails(source))


if __name__ == "__main__":
    main()
