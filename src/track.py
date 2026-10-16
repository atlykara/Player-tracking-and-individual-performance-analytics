"""Player tracking with a pitch region of interest (v2): YOLO11m + BoT-SORT.

v1 ran Ultralytics on the whole frame. People outside the pitch
(substitutes, staff, spectators) became tracks and false positives. v2 reads
the video frame by frame and hides everything outside the ROI polygon from
the model.

Design rule: the video is NOT cropped and the frame size never changes.
Pixels outside the polygon are painted black and the model sees this masked
frame. All coordinates therefore stay in the original pixel space, so the
label format, the ground truth and evaluate_tracking.py work unchanged.

Outputs in OUTPUT_DIR:
- labels/<video>_<N>.txt  same format and naming as `yolo track save_txt`
  (N is 1-based, rows: class xc yc w h conf id, normalised)
- tracks.csv              frame (0-based), track_id, x, y, w, h, conf in pixels
- <video>.mp4             boxes drawn on the ORIGINAL (unmasked) frame
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import colors

from select_roi import load_roi

# --- Settings -------------------------------------------------------------
VIDEO_PATH = "data/SoccerTrackv2/mot/118575.mp4"
ROI_PATH = "data/roi/118575_roi.json"   # None: no ROI, v1 behaviour
MODEL = "yolo11m.pt"
TRACKER = "botsort.yaml"   # Ultralytics' built-in BoT-SORT config
IMGSZ = 2560               # video is 4096 px wide; default 640 misses far players
CLASSES = [0]              # COCO class 0 = person
HALF = True                # FP16 inference, as in the run without ROI
OUTPUT_DIR = "outputs/tracking/118575_yolo11m_botsort_roi"

DRAW_ROI_OUTLINE = True
ROI_COLOR = (255, 255, 255)
ROI_ALPHA = 0.4            # outline opacity: visible but does not hide players
BOX_THICKNESS = 2
FONT_SCALE = 0.7
FONT_THICKNESS = 2


def load_checked_roi(path, video_path):
    """Load the ROI and warn if it was selected on a different video."""
    with open(path, encoding="utf-8") as f:
        roi_video = json.load(f)["video"]
    if roi_video != video_path:
        print(f"WARNING: ROI was selected on {roi_video}, not on {video_path}")
    return load_roi(path)


def roi_mask(shape, polygon):
    """Single-channel mask: 255 inside the polygon, 0 outside."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def feet_inside(boxes, polygon):
    """True for boxes whose foot point (bottom centre) lies inside the polygon.

    The mask alone is not enough: a person standing on the polygon edge is
    half visible and can still be detected as a cut box. The foot point is
    the point that later goes to the pitch, so it decides in or out.
    """
    keep = []
    for x1, y1, x2, y2 in boxes.xyxy.tolist():
        foot = ((x1 + x2) / 2, y2)
        keep.append(cv2.pointPolygonTest(polygon, foot, False) >= 0)
    return keep


def draw_tracks(frame, boxes):
    """Draw a thin box and a compact 'id conf' label for every track."""
    for (x1, y1, x2, y2), tid, conf in zip(
        boxes.xyxy.int().tolist(), boxes.id.int().tolist(), boxes.conf.tolist()
    ):
        color = colors(tid, True)  # same colour for the same ID in every frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        cv2.putText(frame, f"{tid} {conf:.2f}", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SCALE, color, FONT_THICKNESS, cv2.LINE_AA)


def draw_roi(frame, polygon):
    """Thin, semi-transparent polygon outline."""
    overlay = frame.copy()
    cv2.polylines(overlay, [polygon], True, ROI_COLOR, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, ROI_ALPHA, frame, 1 - ROI_ALPHA, 0, dst=frame)


def run_tracking():
    """Track players inside the ROI and write labels, tracks.csv and a video."""
    if not torch.cuda.is_available():   # stop instead of silently using the CPU
        raise RuntimeError("CUDA GPU not available; refusing to run on CPU.")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    out_dir = Path(OUTPUT_DIR)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    stem = Path(VIDEO_PATH).stem
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(str(out_dir / f"{stem}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    polygon = load_checked_roi(ROI_PATH, VIDEO_PATH) if ROI_PATH else None
    mask = roi_mask((size[1], size[0]), polygon) if ROI_PATH else None
    model = YOLO(MODEL)
    rows, frame_idx, removed = [], 0, 0
    start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        masked = cv2.bitwise_and(frame, frame, mask=mask) if ROI_PATH else frame
        result = model.track(masked, persist=True, tracker=TRACKER, classes=CLASSES,
                             imgsz=IMGSZ, half=HALF, device=0, verbose=False)[0]
        has_tracks = result.boxes.id is not None   # None: no confirmed track yet
        if has_tracks and ROI_PATH:
            keep = feet_inside(result.boxes, polygon)
            removed += keep.count(False)
            result = result[torch.tensor(keep, device=result.boxes.data.device)]

        if has_tracks and len(result.boxes):
            # Ultralytics numbers video label files from 1, as `yolo track` did.
            result.save_txt(out_dir / "labels" / f"{stem}_{frame_idx + 1}.txt", save_conf=True)
            for (x, y, w, h), tid, conf in zip(result.boxes.xywh.tolist(),
                                               result.boxes.id.int().tolist(),
                                               result.boxes.conf.tolist()):
                rows.append((frame_idx, tid, x - w / 2, y - h / 2, w, h, conf))
            draw_tracks(frame, result.boxes)
        if DRAW_ROI_OUTLINE and ROI_PATH:
            draw_roi(frame, polygon)
        writer.write(frame)   # the original frame, not the masked one
        frame_idx += 1
    cap.release()
    writer.release()

    tracks = pd.DataFrame(rows, columns=["frame", "track_id", "x", "y", "w", "h", "conf"])
    tracks.round({"x": 1, "y": 1, "w": 1, "h": 1, "conf": 2}).to_csv(out_dir / "tracks.csv", index=False)

    elapsed = time.time() - start
    print(f"Frames processed   : {frame_idx}")
    print(f"Elapsed time       : {elapsed:.1f} s ({frame_idx / elapsed:.1f} fps)")
    print(f"Unique track IDs   : {tracks['track_id'].nunique()}")
    print(f"Boxes per frame    : {len(tracks) / frame_idx:.2f} (mean)")
    print(f"Removed by ROI feet: {removed}")
    print(f"Outputs in         : {OUTPUT_DIR}")


if __name__ == "__main__":
    run_tracking()
