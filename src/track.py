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
- tracks.csv              frame (0-based), track_id, x, y, w, h, conf,
                          foot_x, foot_y in pixels
- <video>.mp4             boxes drawn on the ORIGINAL (unmasked) frame

torch and ultralytics are imported inside the functions that need them, so
other scripts can import foot_point() without loading the model libraries.
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

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
GT_TRACKS_PATH = "outputs/tracking/118575_gt/tracks.csv"   # GT in tracks.csv format

DRAW_ROI_OUTLINE = True
ROI_COLOR = (255, 255, 255)
ROI_ALPHA = 0.4            # outline opacity: visible but does not hide players
BOX_THICKNESS = 2
FONT_SCALE = 0.7
FONT_THICKNESS = 2


def foot_point(x, y, w, h):
    """Bottom centre of a box (x, y = top-left corner, pixels): (x + w/2, y + h).

    This is the box point closest to where the player touches the ground.
    The box centre lies above the ground, so after perspective projection it
    lands on the pitch behind the player. Homography, heatmaps, distance and
    speed will all use this point. Limits: a player who falls, slides or is
    in the air does not touch the ground at this point, so the position is
    off for those frames.
    """
    return x + w / 2, y + h


def add_foot_columns(tracks):
    """Add foot_x, foot_y to a table with the columns x, y, w, h."""
    tracks["foot_x"], tracks["foot_y"] = foot_point(tracks["x"], tracks["y"], tracks["w"], tracks["h"])
    return tracks


def save_tracks_csv(tracks, path):
    """Write tracks.csv with foot point columns, rounded to 0.1 px."""
    tracks = add_foot_columns(tracks[["frame", "track_id", "x", "y", "w", "h", "conf"]].copy())
    tracks.round({"x": 1, "y": 1, "w": 1, "h": 1, "conf": 2, "foot_x": 1, "foot_y": 1}).to_csv(
        path, index=False)


def rebuild_tracks_csv(labels_dir=f"{OUTPUT_DIR}/labels", output_path=f"{OUTPUT_DIR}/tracks.csv"):
    """Rebuild tracks.csv from the saved label files, without running the tracker.

    The labels are read with evaluate_tracking.load_predictions(), so there
    is one label reader in the project.
    """
    from evaluate_tracking import load_predictions
    tracks = load_predictions(labels_dir, VIDEO_PATH)
    save_tracks_csv(tracks, output_path)
    print(f"Rebuilt {output_path}: {len(tracks)} rows, {tracks['track_id'].nunique()} IDs")
    return tracks


def export_gt_tracks_csv(output_path=GT_TRACKS_PATH):
    """Write the ground truth as a tracks.csv (same columns, with foot points).

    The project continues on the GT boxes and IDs, so later stages read the
    GT in the same format as a tracker run. GT conf is -1 (not annotated).
    """
    from evaluate_tracking import GT_PATH, load_ground_truth
    gt = load_ground_truth(GT_PATH)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_tracks_csv(gt, output_path)
    print(f"Exported {output_path}: {len(gt)} rows, {gt['track_id'].nunique()} IDs")
    return gt


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
        foot = foot_point(x1, y1, x2 - x1, y2 - y1)
        keep.append(cv2.pointPolygonTest(polygon, foot, False) >= 0)
    return keep


def draw_tracks(frame, boxes):
    """Draw a thin box and a compact 'id conf' label for every track."""
    from ultralytics.utils.plotting import colors
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
    import torch
    from ultralytics import YOLO

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
    save_tracks_csv(tracks, out_dir / "tracks.csv")

    elapsed = time.time() - start
    print(f"Frames processed   : {frame_idx}")
    print(f"Elapsed time       : {elapsed:.1f} s ({frame_idx / elapsed:.1f} fps)")
    print(f"Unique track IDs   : {tracks['track_id'].nunique()}")
    print(f"Boxes per frame    : {len(tracks) / frame_idx:.2f} (mean)")
    print(f"Removed by ROI feet: {removed}")
    print(f"Outputs in         : {OUTPUT_DIR}")


if __name__ == "__main__":
    run_tracking()
