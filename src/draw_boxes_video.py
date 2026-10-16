"""Draw tracked boxes on the clip to make a box table watchable (v2).

v1 drew only the ground truth. v2 makes the drawing generic: any table with
the columns `frame, track_id, x, y, w, h` (pixels, top-left origin, frame
0-based) can be drawn, so the ground truth and the tracker output
(`tracks.csv`) go through the same code and look the same. Keeping one
drawing function avoids two copies that drift apart (the same mistake the
earlier project made with its windowing logic).

Every ID keeps one colour for the whole video and shows its number above
the box, so a viewer can follow one player. The GT is read with
evaluate_tracking.load_ground_truth(), so there is still a single GT reader.
"""

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from evaluate_tracking import load_ground_truth
from select_roi import load_roi

# --- Settings -------------------------------------------------------------
VIDEO_PATH = "data/SoccerTrackv2/mot/118575.mp4"
GT_PATH = "data/SoccerTrackv2/mot/118575_gt.txt"
RUN_NAME = "118575_yolo11m_botsort_roi"   # tracking run whose tracks.csv is drawn
TRACKS_PATH = f"outputs/tracking/{RUN_NAME}/tracks.csv"
ROI_PATH = "data/roi/118575_roi.json"
DRAW_ROI_OUTLINE = True          # thin yellow ROI polygon on the tracks video
ROI_COLOR = (0, 255, 255)        # BGR yellow
GT_OUTPUT_PATH = "outputs/tracking/118575_gt.mp4"
TRACKS_OUTPUT_PATH = f"outputs/tracking/{RUN_NAME}/118575_tracks.mp4"
LINE_WIDTH = 2
FONT_SCALE = 0.7
INVALID_COLOR = (128, 128, 128)   # rows with is_valid = False: grey and thin
H264_CRF = 23                     # ffmpeg quality; 23 is the usual default


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
    n_valid = int(rows["is_valid"].sum())
    cv2.putText(img, f"{title}  frame {frame}  boxes {n_valid}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)


def write_video(boxes, video_path, output_path, title, roi=None):
    """Read the clip frame by frame, draw the boxes and write an mp4 of the same size/fps.

    `boxes` needs the columns frame, track_id, x, y, w, h. An optional
    `is_valid` column (from the GT reader) marks rows drawn grey; tables
    without it, like tracks.csv, are treated as all valid. `roi` (N x 2
    pixel polygon) is drawn as a thin outline when given.
    """
    if "is_valid" not in boxes.columns:
        boxes = boxes.assign(is_valid=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    by_frame = dict(list(boxes.groupby("frame")))
    empty = boxes.iloc[:0]
    frame = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        draw_frame(img, by_frame.get(frame, empty), frame, title)
        if roi is not None:
            cv2.polylines(img, [roi], True, ROI_COLOR, 1, cv2.LINE_AA)
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


def draw_boxes(boxes, output_path, title, video_path=VIDEO_PATH, roi=None):
    """Draw one box table on the clip, compress the result and report the path."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"{title:<17}: {len(boxes)} boxes, {boxes['track_id'].nunique()} IDs")
    write_video(boxes, video_path, output, title, roi)
    compress_with_ffmpeg(output)
    print(f"Saved to         : {output}")


def main():
    """Draw the ground truth and the tracker output with the same function."""
    draw_boxes(load_ground_truth(GT_PATH), GT_OUTPUT_PATH, "GT")
    roi = load_roi(ROI_PATH) if DRAW_ROI_OUTLINE else None
    draw_boxes(pd.read_csv(TRACKS_PATH), TRACKS_OUTPUT_PATH, "tracks", roi=roi)


if __name__ == "__main__":
    main()
