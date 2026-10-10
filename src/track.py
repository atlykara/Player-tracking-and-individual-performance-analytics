"""First tracking test (v1): YOLO11m + BoT-SORT on a 60 s clip.

Runs player detection and tracking on the full frame and saves only the
annotated video. No CSV yet; the goal is to look at the tracking result.
"""

import time
from pathlib import Path

import torch
from ultralytics import YOLO

# --- Settings -------------------------------------------------------------
VIDEO_PATH = "data/processed/test_60s.mp4"
MODEL_PATH = "yolo11m.pt"
TRACKER = "botsort.yaml"   # Ultralytics' built-in BoT-SORT config
CLASSES = [0]              # COCO class 0 = person
IMGSZ = 2560               # video is 4096 px wide; default 640 misses far players
OUTPUT_DIR = "outputs"
OUTPUT_NAME = "tracking"   # result goes to outputs/tracking/


def run_tracking():
    """Track people in the video and print a short summary."""
    # Stop early instead of silently running on the CPU.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available; refusing to run on CPU.")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    model = YOLO(MODEL_PATH)
    start = time.time()

    # stream=True yields one result per frame, so memory stays low.
    # exist_ok=True overwrites outputs/tracking/ instead of tracking2, 3 ...
    # project must be absolute: Ultralytics puts a relative one under runs/detect/.
    results = model.track(
        source=VIDEO_PATH,
        tracker=TRACKER,
        classes=CLASSES,
        imgsz=IMGSZ,
        device=0,
        save=True,
        project=str(Path(OUTPUT_DIR).resolve()),
        name=OUTPUT_NAME,
        exist_ok=True,
        stream=True,
        verbose=False,
    )

    frame_count = 0
    track_ids = set()
    for result in results:
        frame_count += 1
        # boxes.id is None on frames where the tracker has no confirmed track.
        if result.boxes.id is not None:
            track_ids.update(result.boxes.id.int().tolist())

    elapsed = time.time() - start
    print(f"Frames processed : {frame_count}")
    print(f"Elapsed time     : {elapsed:.1f} s ({frame_count / elapsed:.1f} fps)")
    print(f"Unique track IDs : {len(track_ids)}")
    print(f"Video saved in   : {OUTPUT_DIR}/{OUTPUT_NAME}/")


if __name__ == "__main__":
    run_tracking()
