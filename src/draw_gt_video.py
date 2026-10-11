"""Draw the ground-truth boxes on the clip to make the GT watchable (v1).

Every GT ID keeps one colour for the whole video and shows its number above
the box, so a viewer can follow one player and judge what "perfect" tracking
looks like before comparing it with the tracker output. The GT is read with
evaluate_tracking.load_ground_truth(), so there is a single GT reader.
"""

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from evaluate_tracking import load_ground_truth

# --- Settings -------------------------------------------------------------
VIDEO_PATH = "data/SoccerTrackv2/mot/118575.mp4"
GT_PATH = "data/SoccerTrackv2/mot/118575_gt.txt"
OUTPUT_PATH = "outputs/gt_video/118575_gt.mp4"
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


def draw_frame(img, rows, frame):
    """Draw one frame's GT rows and the frame counter in the top-left corner."""
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
    cv2.putText(img, f"frame {frame}  GT boxes {n_valid}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)


def write_video(gt, video_path, output_path):
    """Read the clip frame by frame, draw the GT and write an mp4 of the same size/fps."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    by_frame = dict(list(gt.groupby("frame")))
    empty = gt.iloc[:0]
    frame = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        draw_frame(img, by_frame.get(frame, empty), frame)
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


def main():
    """Load GT, draw it on the clip, compress the result."""
    output = Path(OUTPUT_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    gt = load_ground_truth(GT_PATH)
    write_video(gt, VIDEO_PATH, output)
    compress_with_ffmpeg(output)
    print(f"Saved to         : {output}")


if __name__ == "__main__":
    main()
