"""Pitch calibration, step 1 (v1): click named pitch landmarks on a video frame.

A homography needs pairs of (pixel position, pitch position in metres). The
metre side comes from config/field_config.yaml; this tool collects the pixel
side by hand and stores the pairs in a CSV. No homography is computed here.

Usage (from the repository root):
    python src/calibrate.py select --video data/SoccerTrackv2/mot/118575.mp4 --frame 0
    python src/calibrate.py select --redo centre_spot

Controls: left click = save the current landmark, s = skip it, u = undo the
last click, q = finish. Points already in the CSV are kept: only missing
landmarks are asked, or the one named with --redo.

Coordinate rule: the window shows a scaled-down frame, but every saved value
is a pixel of the ORIGINAL frame, the one under the crosshair of the magnifier.
"""

import argparse
import csv
from pathlib import Path

import cv2
import yaml

from select_roi import display_scale, shrink

# --- Settings -------------------------------------------------------------
CONFIG_PATH = "config/field_config.yaml"
DEFAULT_VIDEO = "data/SoccerTrackv2/mot/118575.mp4"
OUT_DIR = "outputs/calibration"
ZOOM_HALF = 25          # magnifier shows +-25 original px around the cursor
ZOOM_FACTOR = 4         # ... and draws every original pixel as 4 x 4
BAR_HEIGHT = 60         # black status bar under the frame: text never hides the pitch
WINDOW = "Select pitch landmarks"
HELP_TEXT = "left click: save | s: skip | u: undo last | q: finish"
FONT = cv2.FONT_HERSHEY_SIMPLEX
POINT_COLOR = (0, 0, 255)       # BGR red
CURRENT_COLOR = (0, 255, 255)   # BGR yellow: the landmark being asked
CSV_COLUMNS = ["name", "x_m", "y_m", "px_x", "px_y"]


def load_field_config(path=CONFIG_PATH):
    """Return ({name: (x_m, y_m)} in config order, path of the points file)."""
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    landmarks = {name: (float(p["x_m"]), float(p["y_m"]))
                 for name, p in config["landmarks"].items()}
    return landmarks, config["points_file"]


def read_frame(video_path, index):
    """Return frame `index` (0-based) of the video in original resolution."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise FileNotFoundError(f"Cannot read frame {index} of {video_path}")
    return frame


def load_points(csv_path, landmarks):
    """Return {name: (px_x, px_y)} from an earlier run (empty if no file).

    Only the pixel columns are read: the metre values always come from the
    config, so the two files cannot disagree.
    """
    points = {}
    if Path(csv_path).exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["name"] in landmarks:
                    points[row["name"]] = (float(row["px_x"]), float(row["px_y"]))
                else:
                    print(f"Warning: '{row['name']}' is not in {CONFIG_PATH}, ignored")
    return points


def save_points(csv_path, landmarks, points):
    """Write the clicked landmarks in config order (skipped ones have no row)."""
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for name, (x_m, y_m) in landmarks.items():
            if name in points:
                writer.writerow([name, x_m, y_m] + [f"{v:g}" for v in points[name]])


def draw_text(img, text, org, size, color):
    """Text with a black outline, readable on grass and on white lines."""
    cv2.putText(img, text, org, FONT, size, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, size, color, 1, cv2.LINE_AA)


def draw_points(img, landmarks, points, scale=1.0, with_metres=True):
    """Draw every clicked point (original pixels, drawn at `scale`) with its name."""
    for name, (px, py) in points.items():
        p = (int(round(px * scale)), int(round(py * scale)))
        # A thin cross, not a filled dot: the clicked pixel stays visible.
        cv2.drawMarker(img, p, POINT_COLOR, cv2.MARKER_CROSS, 14, 1)
        label = name + (" ({:g}, {:g}) m".format(*landmarks[name]) if with_metres else "")
        draw_text(img, label, (p[0] + 6, p[1] - 6), 0.5 if with_metres else 0.4, (255, 255, 255))
    return img


def draw_inset(view, padded, cx, cy, cursor_on_left):
    """Paste a magnified view of the original pixels around (cx, cy)."""
    size = 2 * ZOOM_HALF + 1                      # odd: (cx, cy) is the middle pixel
    crop = padded[cy:cy + size, cx:cx + size]     # `padded` has a ZOOM_HALF border
    side = size * ZOOM_FACTOR
    zoom = cv2.resize(crop, (side, side), interpolation=cv2.INTER_NEAREST)
    mid, gap = side // 2, 2 * ZOOM_FACTOR
    # Crosshair with a hole in the middle, so it does not cover the target pixel.
    for a, b in [(0, mid - gap), (mid + gap, side)]:
        cv2.line(zoom, (mid, a), (mid, b), (0, 255, 0), 1)
        cv2.line(zoom, (a, mid), (b, mid), (0, 255, 0), 1)
    cv2.rectangle(zoom, (0, 0), (side - 1, side - 1), (255, 255, 255), 1)
    # Top corner on the other side of the cursor: always visible, never on
    # top of the place that is being clicked.
    x0 = view.shape[1] - side if cursor_on_left else 0
    view[0:side, x0:x0 + side] = zoom


def select_points(frame, landmarks, points, todo):
    """Ask for every landmark in `todo`; `points` (original pixels) is updated in place."""
    h, w = frame.shape[:2]
    scale = display_scale(w, h)
    small = shrink(frame, scale)
    sh, sw = small.shape[:2]
    padded = cv2.copyMakeBorder(frame, ZOOM_HALF, ZOOM_HALF, ZOOM_HALF, ZOOM_HALF,
                                cv2.BORDER_CONSTANT)
    before = dict(points)                   # values to restore on undo
    cursor, clicks, clicked = [0, 0], [], []   # clicked: positions in `todo`, this run

    def on_mouse(event, x, y, flags, param):
        # Back to original resolution: display pixel / scale (kept inside the frame).
        cursor[:] = [min(max(int(x / scale), 0), w - 1), min(max(int(y / scale), 0), h - 1)]
        if event == cv2.EVENT_LBUTTONDOWN and y < sh:
            clicks.append(tuple(cursor))

    # AUTOSIZE: the window shows the view 1:1, so a mouse position is a pixel of `small`.
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    i = 0
    while True:
        # After the last landmark the window stays open (name = None), so the
        # last click can still be undone; only q ends the selection.
        name = todo[i] if i < len(todo) else None
        title = ("[{}/{}] {}  ({:g}, {:g}) m".format(i + 1, len(todo), name, *landmarks[name])
                 if name else "All landmarks asked: u = undo last, q = finish")
        view = cv2.copyMakeBorder(small, 0, BAR_HEIGHT, 0, 0, cv2.BORDER_CONSTANT)
        draw_points(view, landmarks, points, scale, with_metres=False)
        draw_inset(view, padded, cursor[0], cursor[1], cursor[0] < w / 2)
        draw_text(view, title, (10, sh + 24), 0.7, CURRENT_COLOR)
        draw_text(view, f"{HELP_TEXT} | saved: {len(points)}/{len(landmarks)} | "
                  f"cursor: {cursor[0]}, {cursor[1]} px", (10, sh + 50), 0.5, (255, 255, 255))
        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(20) & 0xFF
        click = clicks.pop() if clicks else None
        clicks.clear()
        if click and name:
            points[name] = click
            clicked.append(i)
            i += 1
        elif key == ord("s") and name:
            i += 1
        elif key == ord("u") and clicked:
            i = clicked.pop()               # back to the landmark of the last click
            if todo[i] in before:
                points[todo[i]] = before[todo[i]]
            else:
                del points[todo[i]]
        elif key == ord("q") or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()
    return 1 / scale                        # original pixels per window pixel


def select(video, frame_index, redo=None):
    """Collect the missing landmarks (or `redo`), write the CSV and the check image."""
    landmarks, points_file = load_field_config()
    if redo and redo not in landmarks:
        raise SystemExit(f"Unknown landmark '{redo}'. Names: {', '.join(landmarks)}")
    points = load_points(points_file, landmarks)
    todo = [redo] if redo else [name for name in landmarks if name not in points]
    frame = read_frame(video, frame_index)
    print(f"Video        : {video}, frame {frame_index} ({frame.shape[1]} x {frame.shape[0]})")
    print(f"Landmarks    : {len(landmarks)} in config, {len(points)} already clicked, {len(todo)} to ask")
    if todo:
        step = select_points(frame, landmarks, points, todo)
        # The mouse moves in window pixels, so clicks land on every `step`-th original pixel.
        print(f"Click step   : {step:.2f} original px per window px")

    save_points(points_file, landmarks, points)
    check_path = Path(OUT_DIR) / f"{Path(video).stem}_points.jpg"
    check_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(check_path), draw_points(frame.copy(), landmarks, points))
    for name, (px, py) in points.items():
        print("  {:<26} ({:>6g}, {:>5g}) m -> ({:g}, {:g}) px".format(name, *landmarks[name], px, py))
    missing = [name for name in landmarks if name not in points]
    print(f"Saved        : {len(points)} points to {points_file}")
    print(f"Not clicked  : {', '.join(missing) if missing else 'none'}")
    print(f"Check image  : {check_path}")


def main():
    parser = argparse.ArgumentParser(description="Pitch calibration tools.")
    commands = parser.add_subparsers(dest="command", required=True)
    sel = commands.add_parser("select", help="click pitch landmarks on a video frame")
    sel.add_argument("--video", default=DEFAULT_VIDEO)
    sel.add_argument("--frame", type=int, default=0, help="0-based frame number")
    sel.add_argument("--redo", metavar="NAME", help="click this one landmark again")
    args = parser.parse_args()
    if args.command == "select":
        select(args.video, args.frame, args.redo)


if __name__ == "__main__":
    main()
