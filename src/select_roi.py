"""Selection of the pitch region of interest (ROI) (v1).
This tool lets the user click the corners of a polygon around the pitch on the first video frame and saves it as JSON.
Later steps keep only detections whose foot point lies inside the polygon.

Controls: left click = add corner, u = undo last corner,
Enter = save (at least 3 corners), Esc = cancel.
"""

import ctypes
import json
from pathlib import Path

import cv2
import numpy as np

# --- Settings -------------------------------------------------------------
VIDEO_PATH = "data/SoccerTrackv2/mot/118575.mp4"
ROI_PATH = "data/roi/118575_roi.json"
CHECK_PATH = "data/roi/118575_roi_check.jpg"
FALLBACK_SCREEN = (1600, 900)      # used if the screen size cannot be read
SCREEN_MARGIN = (40, 120)          # room for window border and taskbar (px)
ZOOM_HALF = 25                     # magnifier shows +-25 original px around the cursor
ZOOM_FACTOR = 4                    # ... enlarged 4x -> 200 x 200 inset
WINDOW = "Select ROI"
HELP_TEXT = "left click: add corner | u: undo | Enter: save (>=3) | Esc: cancel"
POINT_COLOR = (0, 0, 255)          # BGR red
EDGE_COLOR = (0, 255, 255)         # BGR yellow
CLOSING_COLOR = (255, 0, 255)      # BGR magenta: edge from last back to first corner


def read_first_frame(video_path):
    """Return the first frame of the video in original resolution."""
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise FileNotFoundError(f"Cannot read first frame: {video_path}")
    return frame


def display_scale(w, h):
    """Scale that fits a w x h frame on the screen (never above 1)."""
    try:
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        screen = (user32.GetSystemMetrics(0), user32.GetSystemMetrics(1))
    except (AttributeError, OSError):         # not Windows
        screen = FALLBACK_SCREEN
    max_w, max_h = screen[0] - SCREEN_MARGIN[0], screen[1] - SCREEN_MARGIN[1]
    return min(max_w / w, max_h / h, 1.0)


def shrink(img, scale):
    """Resize for display"""
    size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def draw_magnifier(view, padded, cursor, scale):
    """Paste an enlarged crop of the original frame around the cursor."""
    cx, cy = int(cursor[0] / scale), int(cursor[1] / scale)   # original pixels
    # `padded` has a ZOOM_HALF border, so the crop never leaves the image.
    crop = padded[cy:cy + 2 * ZOOM_HALF, cx:cx + 2 * ZOOM_HALF]
    side = 2 * ZOOM_HALF * ZOOM_FACTOR
    zoom = cv2.resize(crop, (side, side), interpolation=cv2.INTER_NEAREST)
    mid = side // 2
    cv2.line(zoom, (mid, 0), (mid, side), (0, 255, 0), 1)
    cv2.line(zoom, (0, mid), (side, mid), (0, 255, 0), 1)
    # Bottom centre: the pitch corners and side lines are near the image edges.
    x0, y0 = (view.shape[1] - side) // 2, view.shape[0] - side - 10
    if x0 > 0 and y0 > 0:
        view[y0:y0 + side, x0:x0 + side] = zoom


def draw_polygon(img, points, scale=1.0, closed=False):
    """Draw corners and edges; points are original pixels, drawn at `scale`."""
    pts = [(int(round(x * scale)), int(round(y * scale))) for x, y in points]
    for a, b in zip(pts, pts[1:]):
        cv2.line(img, a, b, EDGE_COLOR, 2)
    if len(pts) >= 3:
        cv2.line(img, pts[-1], pts[0], EDGE_COLOR if closed else CLOSING_COLOR, 2)
    for p in pts:
        cv2.circle(img, p, 5, POINT_COLOR, -1)
    return img


def select_points(frame):
    """Let the user click polygon corners; return original-pixel points or None."""
    h, w = frame.shape[:2]
    scale = display_scale(w, h)
    small = shrink(frame, scale)
    padded = cv2.copyMakeBorder(frame, ZOOM_HALF, ZOOM_HALF, ZOOM_HALF, ZOOM_HALF,
                                cv2.BORDER_CONSTANT)
    points, cursor = [], [0, 0]

    def on_mouse(event, x, y, flags, param):
        cursor[:] = [x, y]
        if event == cv2.EVENT_LBUTTONDOWN:
            # Back to original resolution: display pixel / scale.
            points.append((x / scale, y / scale))

    # AUTOSIZE: the window shows `small` 1:1, so a mouse position is a pixel
    # of `small` and dividing by `scale` gives the original pixel.
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    while True:
        view = draw_polygon(small.copy(), points, scale)
        draw_magnifier(view, padded, cursor, scale)
        cv2.putText(view, f"{HELP_TEXT} | corners: {len(points)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("u") and points:
            points.pop()
        elif key == 13 and len(points) >= 3:      # Enter
            break
        elif key == 27:                           # Esc
            points = None
            break
    cv2.destroyWindow(WINDOW)
    return points


def load_roi(path=ROI_PATH):
    """Load a saved ROI as an (N, 2) int32 array in original video pixels."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return np.array(data["points"], dtype=np.float64).round().astype(np.int32)


def main():
    frame = read_first_frame(VIDEO_PATH)
    h, w = frame.shape[:2]
    print(f"Video: {VIDEO_PATH} ({w} x {h})")
    points = select_points(frame)
    if points is None:
        print("Cancelled, nothing saved.")
        return

    roi = {"video": VIDEO_PATH, "width": w, "height": h,
           "points": [[round(x, 1), round(y, 1)] for x, y in points]}
    Path(ROI_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(ROI_PATH, "w", encoding="utf-8") as f:
        json.dump(roi, f, indent=2)
    print(f"Saved {len(points)} corners to {ROI_PATH}")

    # Check image: polygon drawn on the full-resolution frame from the saved file.
    check = draw_polygon(frame.copy(), load_roi(ROI_PATH), closed=True)
    cv2.imwrite(CHECK_PATH, check)
    print(f"Check image: {CHECK_PATH} (any key closes the window)")
    cv2.imshow("ROI check", shrink(check, display_scale(w, h)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
