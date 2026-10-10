"""First tracking test (v1): YOLO11m + BoT-SORT on a 60 s clip.

Runs player detection and tracking on the full frame and saves only the
annotated video. No CSV yet; the goal is to look at the tracking result.
"""

import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import colors

# --- Settings -------------------------------------------------------------
VIDEO_PATH = "data/processed/test_60s.mp4"
MODEL_PATH = "yolo11m.pt"
TRACKER = "botsort.yaml"   # Ultralytics' built-in BoT-SORT config
CLASSES = [0]              # COCO class 0 = person
IMGSZ = 2560               # video is 4096 px wide; default 640 misses far players
OUTPUT_PATH = "outputs/tracking/test_60s.mp4"  # overwritten on every run

# Own drawing instead of Ultralytics' save=True: its label "id:2 person 0.87"
# is several times wider than a far player and hides neighbours.
BOX_THICKNESS = 2
FONT_SCALE = 0.7
FONT_THICKNESS = 2


def draw_tracks(frame, boxes):
    """Draw a thin box and a compact 'id conf' label for every track."""
    if boxes.id is None:  # tracker has no confirmed track on this frame
        return
    for (x1, y1, x2, y2), tid, conf in zip(
        boxes.xyxy.int().tolist(), boxes.id.int().tolist(), boxes.conf.tolist()
    ):
        color = colors(tid, True)  # same colour for the same ID in every frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        label = f"{tid} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)
        cv2.rectangle(frame, (x1, y1 - h - 6), (x1 + w + 4, y1), color, -1)
        # Black text on light backgrounds (white, yellow), white text otherwise.
        b, g, r = color
        text_color = (0, 0, 0) if 0.299 * r + 0.587 * g + 0.114 * b > 150 else (255, 255, 255)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SCALE, text_color, FONT_THICKNESS, cv2.LINE_AA)


def run_tracking():
    """Track people in the video, write the annotated video, print a summary."""
    # Stop early instead of silently running on the CPU.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available; refusing to run on CPU.")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    model = YOLO(MODEL_PATH)
    start = time.time()

    # stream=True yields one result per frame, so memory stays low.
    results = model.track(
        source=VIDEO_PATH,
        tracker=TRACKER,
        classes=CLASSES,
        imgsz=IMGSZ,
        device=0,
        stream=True,
        verbose=False,
    )

    frame_count = 0
    track_ids = set()
    for result in results:
        frame_count += 1
        if result.boxes.id is not None:
            track_ids.update(result.boxes.id.int().tolist())
        frame = result.orig_img.copy()
        draw_tracks(frame, result.boxes)
        writer.write(frame)
    writer.release()

    elapsed = time.time() - start
    print(f"Frames processed : {frame_count}")
    print(f"Elapsed time     : {elapsed:.1f} s ({frame_count / elapsed:.1f} fps)")
    print(f"Unique track IDs : {len(track_ids)}")
    print(f"Video saved to   : {OUTPUT_PATH}")


if __name__ == "__main__":
    run_tracking()
