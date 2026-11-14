"""Pitch calibration (v3): select reference points, then map video pixels to pitch metres.

The calibration is built from reference points: named points of the pitch whose
position in metres is known (config/field_config.yaml) and whose pixel position
was selected by hand on a video frame ("select"). "fit" turns them into the
mapping from pixels to metres, measures its error and draws the pitch grid
that follows from the reference points back into the frame.

Usage (from the repository root):
    python src/calibrate.py select --video data/SoccerTrackv2/mot/118575.mp4 --frame 0
    python src/calibrate.py select --redo centre_spot
    python src/calibrate.py fit

select controls: left click = save the current landmark, s = skip it, u = undo
the last click, q = finish. Points already in the CSV are kept: only missing
landmarks are asked, or the one named with --redo.

Coordinate rule: the window shows a scaled-down frame, but every saved value
is a pixel of the ORIGINAL frame, the one under the crosshair of the magnifier.

fit: the panoramic image is curved, so one homography cannot describe the
whole pitch. Method "reference_points" (the default): each position is mapped
with a homography fitted to all reference points, where the points near that
position count more; the mapping passes through the reference points
themselves. Method "pieces" (only the baseline for comparison): the
pitch is cut into equal pieces along its length, one homography per piece,
neighbouring pieces blended. Both work on pixels with the lens distortion
removed. Each method is scored leave-one-out: every reference point is
predicted by a fit that did not see it. Later stages only call
pixel_to_pitch(). Positions are approximate.
"""

import argparse
import csv
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import yaml

from select_roi import display_scale, shrink

# --- Settings -------------------------------------------------------------
CONFIG_PATH = "config/field_config.yaml"
DEFAULT_VIDEO = "data/SoccerTrackv2/mot/118575.mp4"
OUT_DIR = "outputs/calibration"
# Fisheye lens model shipped with the dataset for this match (K, D, Knew).
INTRINSICS_PATH = "config/118575_camera_intrinsics.npz"
# Written by "fit" and read by pixel_to_pitch(): the reference points in use
# (in the pixel space of the calibration, and in metres) and the settings.
# The homographies themselves are computed from these when points are mapped.
HOMOGRAPHY_PATH = "config/homography.npz"
METHODS = ("reference_points", "pieces")   # the default is calibration.method in the config
BASELINE_METHOD = "pieces"                 # fitted and reported only for comparison
SAVED_SETTINGS = ("radius_m", "min_weight", "pieces", "overlap_m", "pixel_space")
# The mapping has to pass through the reference points themselves. Closer than this
# to a reference point, that point outweighs all the others and the fit goes through it.
EXACT_DISTANCE_M = 0.05
# A pixel is mapped in steps: a rough position first, then again with the weights
# set around the result. After 3 steps a reference pixel lands on its own position.
POSITION_PASSES = 3
# The fit report flags a dataset mark that the own points alone predict further
# away than this from its nominal position (information only).
MARK_LIMIT_M = 2.0
TINY_WEIGHT = 1e-6      # lower bound of a weight; keeps the fit defined if min_weight is 0
MIN_PIECE_POINTS = 5    # a homography is fixed by 4 points; 5 leaves one point to average over
CHUNK = 20000           # points mapped at once (limits memory, not accuracy)
GRID_STEP_M = 5         # spacing of the grid drawn back into the image
SAMPLE_STEP_M = 0.25    # pitch lines are sampled this densely, so curved lines are drawn curved
GRID_COLOR = (0, 255, 255)      # BGR yellow
MARKING_COLOR = (0, 0, 255)     # BGR red
USED_COLOR = (255, 255, 255)    # BGR white: reference point in use
UNUSED_COLOR = (255, 0, 255)    # BGR magenta: dataset mark that is not used
ZOOM_HALF = 25          # magnifier shows +-25 original px around the cursor
ZOOM_FACTOR = 4         # ... and draws every original pixel as 4 x 4
BAR_HEIGHT = 60         # black status bar under the frame: text never hides the pitch
WINDOW = "Select pitch landmarks"
HELP_TEXT = "left click: save | s: skip | u: undo last | q: finish"
FONT = cv2.FONT_HERSHEY_SIMPLEX
POINT_COLOR = (0, 0, 255)       # BGR red
CURRENT_COLOR = (0, 255, 255)   # BGR yellow: the landmark being asked
CSV_COLUMNS = ["name", "x_m", "y_m", "px_x", "px_y", "source", "use"]


def load_field_config(path=CONFIG_PATH):
    """Return the config; "landmarks" becomes {name: (x_m, y_m)} in config order."""
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["landmarks"] = {name: (float(p["x_m"]), float(p["y_m"]))
                           for name, p in config["landmarks"].items()}
    return config


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
    """Return ({name: (px_x, px_y)}, {name: (source, use)}) from the points file.

    source: "own" = selected by hand on the frame, "dataset" = pixel position
    taken from the dataset. use: 1 = the point is a reference point of the
    calibration, 0 = it stays in the file but is not used. The metre values are
    not read: they always come from the config, so the two files cannot disagree.
    """
    points, info = {}, {}
    if Path(csv_path).exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["name"] in landmarks:
                    points[row["name"]] = (float(row["px_x"]), float(row["px_y"]))
                    info[row["name"]] = (row.get("source") or "own", int(row.get("use") or 1))
                else:
                    print(f"Warning: '{row['name']}' is not in {CONFIG_PATH}, ignored")
    return points, info


def save_points(csv_path, landmarks, points, info):
    """Write the points in config order (skipped ones have no row).

    source and use are kept as they are; a point without them was selected by
    hand in this run, so it is written as an own point that is used.
    """
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for name, (x_m, y_m) in landmarks.items():
            if name in points:
                writer.writerow([name, x_m, y_m] + [f"{v:g}" for v in points[name]]
                                + list(info.get(name, ("own", 1))))


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
    config = load_field_config()
    landmarks, points_file = config["landmarks"], config["points_file"]
    if redo and redo not in landmarks:
        raise SystemExit(f"Unknown landmark '{redo}'. Names: {', '.join(landmarks)}")
    points, info = load_points(points_file, landmarks)
    todo = [redo] if redo else [name for name in landmarks if name not in points]
    frame = read_frame(video, frame_index)
    print(f"Video        : {video}, frame {frame_index} ({frame.shape[1]} x {frame.shape[0]})")
    print(f"Landmarks    : {len(landmarks)} in config, {len(points)} already clicked, {len(todo)} to ask")
    if todo:
        before = dict(points)
        step = select_points(frame, landmarks, points, todo)
        # The mouse moves in window pixels, so clicks land on every `step`-th original pixel.
        print(f"Click step   : {step:.2f} original px per window px")
        for name, pixel in points.items():
            if before.get(name) != pixel:           # selected by hand in this run
                info[name] = ("own", 1)

    save_points(points_file, landmarks, points, info)
    check_path = Path(OUT_DIR) / f"{Path(video).stem}_points.jpg"
    check_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(check_path), draw_points(frame.copy(), landmarks, points))
    for name, (px, py) in points.items():
        print("  {:<26} ({:>6g}, {:>5g}) m -> ({:g}, {:g}) px  {} {}".format(
            name, *landmarks[name], px, py, *info.get(name, ("own", 1))))
    missing = [name for name in landmarks if name not in points]
    print(f"Saved        : {len(points)} points to {points_file}")
    print(f"Not clicked  : {', '.join(missing) if missing else 'none'}")
    print(f"Check image  : {check_path}")


# --- fit: pixel -> metre mapping -------------------------------------------

@lru_cache(maxsize=1)
def load_lens(path=INTRINSICS_PATH):
    """Return (K, D, Knew) of the fisheye lens model."""
    lens = np.load(path, allow_pickle=True)
    return lens["K"], lens["D"], lens["Knew"]


def undistort_points(points_px):
    """Raw pixels -> undistorted pixels. Only points are undistorted, never the video."""
    K, D, Knew = load_lens()
    pts = np.asarray(points_px, np.float64).reshape(-1, 1, 2)
    return cv2.fisheye.undistortPoints(pts, K, D, P=Knew).reshape(-1, 2)


def distort_points(points_und):
    """Undistorted pixels -> raw pixels (the inverse of undistort_points)."""
    K, D, Knew = load_lens()
    pts = np.asarray(points_und, np.float64).reshape(-1, 2)
    # distortPoints expects normalised coordinates, so Knew is taken off first.
    norm = np.stack([(pts[:, 0] - Knew[0, 2]) / Knew[0, 0],
                     (pts[:, 1] - Knew[1, 2]) / Knew[1, 1]], axis=1)
    return cv2.fisheye.distortPoints(norm.reshape(-1, 1, 2), K, D).reshape(-1, 2)


def to_pixel_space(points_px, pixel_space):
    """Raw video pixels -> the pixels the calibration works in."""
    if pixel_space == "undistorted":
        return undistort_points(points_px)
    return np.asarray(points_px, np.float64).reshape(-1, 2)


def to_raw_pixels(points, pixel_space):
    """Pixels of the calibration -> raw video pixels (the inverse of to_pixel_space)."""
    return distort_points(points) if pixel_space == "undistorted" else points


def transform(H, points):
    """Apply a 3x3 homography to (N, 2) points."""
    pts = np.asarray(points, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


# Method "reference_points" ---------------------------------------------------

def centre_and_scale(points):
    """Centre of the points and the scale that gives them a mean distance of sqrt(2).

    Pixels are in the thousands and metres in the tens; with this both enter
    the equations at the same size, which keeps the solution stable.
    """
    centre = points.mean(axis=0)
    return centre, np.sqrt(2) / np.sqrt(((points - centre) ** 2).sum(axis=1).mean())


def fit_reference_points(px, metres, radius_m, min_weight):
    """Prepare the reference points mapping; the homographies are solved when points are mapped.

    A homography has nine entries. Every reference point (pixel p, pitch
    position q) gives two linear equations in them. They are kept per point as
    a 9 x 9 block, so the weighted sum over all points can be formed for many
    positions at once.
    """
    (c_px, s_px), (c_m, s_m) = centre_and_scale(px), centre_and_scale(metres)
    a, b = (px - c_px) * s_px, (metres - c_m) * s_m
    zero, one = np.zeros(len(a)), np.ones(len(a))
    row_x = np.stack([a[:, 0], a[:, 1], one, zero, zero, zero,
                      -b[:, 0] * a[:, 0], -b[:, 0] * a[:, 1], -b[:, 0]], axis=1)
    row_y = np.stack([zero, zero, zero, a[:, 0], a[:, 1], one,
                      -b[:, 1] * a[:, 0], -b[:, 1] * a[:, 1], -b[:, 1]], axis=1)
    blocks = row_x[:, :, None] * row_x[:, None, :] + row_y[:, :, None] * row_y[:, None, :]
    return {"metres": metres, "blocks": blocks.reshape(len(a), 81), "scaling": (c_px, s_px, c_m, s_m),
            "radius_m": float(radius_m), "min_weight": float(min_weight)}


def weighted_homographies(model, positions_m=None):
    """One homography per position (in the scaled coordinates), by weighted least squares.

    Weight of a reference point for a position, with d the distance in metres
    between the two:
        max(exp(-(d / radius_m)^2), min_weight) / (d^2 + EXACT_DISTANCE_M^2)
    The first factor makes near points count more and keeps a small weight for
    far ones. The division by d^2 lets the weight grow without limit as the
    position reaches a reference point, so the mapping passes through every
    reference point instead of only close to it.
    positions_m = None gives all points the same weight: the ordinary single
    homography. The least-squares solution is the eigenvector with the smallest
    eigenvalue of the weighted sum of the blocks.
    """
    if positions_m is None:
        weights = np.ones((1, len(model["metres"])))
    else:
        d2 = ((positions_m[:, None, :] - model["metres"][None, :, :]) ** 2).sum(axis=2)
        weights = np.maximum(np.exp(-d2 / model["radius_m"] ** 2), max(model["min_weight"], TINY_WEIGHT))
        weights = weights / (d2 + EXACT_DISTANCE_M ** 2)
    return np.linalg.eigh((weights @ model["blocks"]).reshape(-1, 9, 9))[1][:, :, 0].reshape(-1, 3, 3)


def reference_points_mapping(model, points, to_pitch=True):
    """Map with the reference points: each position is mapped with a homography
    fitted to all reference points, where the points near that position count
    more. A reference point itself is mapped exactly onto its pitch position.

    to_pitch=True: pixels -> metres. The position of a pixel is not known before
    it is mapped, so the single homography with equal weights gives a rough
    position first; then the pixel is mapped POSITION_PASSES times, each time
    with the weights set around the position found before.
    to_pitch=False: metres -> pixels (for drawing); here the position is given.
    """
    points = np.asarray(points, np.float64).reshape(-1, 2)
    c_px, s_px, c_m, s_m = model["scaling"]
    out = np.empty_like(points)
    for lo in range(0, len(points), CHUNK):
        part = points[lo:lo + CHUNK]
        if to_pitch:
            p = np.column_stack([(part - c_px) * s_px, np.ones(len(part))])
            q = p @ weighted_homographies(model)[0].T
            for _ in range(POSITION_PASSES):
                position = q[:, :2] / q[:, 2:3] / s_m + c_m
                q = np.einsum("kij,kj->ki", weighted_homographies(model, position), p)
            out[lo:lo + CHUNK] = q[:, :2] / q[:, 2:3] / s_m + c_m
        else:
            q = np.column_stack([(part - c_m) * s_m, np.ones(len(part))])
            p = np.linalg.solve(weighted_homographies(model, part), q[:, :, None])[:, :, 0]
            out[lo:lo + CHUNK] = p[:, :2] / p[:, 2:3] / s_px + c_px
    return out


# Method "pieces" (baseline) ---------------------------------------------------

def piece_landmarks(x_m, lo, hi, overlap_m, warn=True):
    """Piece rule: indices of the reference points that fit the piece lo..hi (metres).

    The range is extended by overlap_m on both sides, so neighbouring pieces
    share points and agree near the split. A piece with fewer than
    MIN_PIECE_POINTS points is widened further in 1 m steps.
    """
    margin = overlap_m
    while True:
        idx = np.flatnonzero((x_m >= lo - margin) & (x_m <= hi + margin))
        if len(idx) >= MIN_PIECE_POINTS or len(idx) == len(x_m):
            break
        margin += 1.0
    if warn and margin > overlap_m:
        print(f"Warning: piece {lo:g}-{hi:g} m widened to +-{margin:g} m "
              f"to reach {len(idx)} reference points")
    return idx


def fit_pieces(px, metres, n_pieces, length_m, overlap_m, warn=True):
    """One homography per piece of equal length along x, plus one for the whole pitch.

    Method 0 = plain least squares over all given points, no outlier rejection:
    every reference point was checked on the frame, so every point should count.
    """
    edges = np.linspace(0.0, length_m, n_pieces + 1)
    pieces = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = piece_landmarks(metres[:, 0], lo, hi, overlap_m, warn)
        pieces.append(cv2.findHomography(px[idx], metres[idx], 0)[0])
    return {"global": cv2.findHomography(px, metres, 0)[0], "pieces": np.array(pieces),
            "centres": (edges[:-1] + edges[1:]) / 2}


def pieces_mapping(model, points, to_pitch=True):
    """Map with blended pieces: a piece counts fully at its own centre and falls
    linearly to nothing at the centres of its two neighbours, so there is no jump
    where the piece changes. The whole-pitch homography gives the rough x_m of a
    pixel, which sets the weights. to_pitch=False: metres -> pixels (for drawing).
    """
    points = np.asarray(points, np.float64).reshape(-1, 2)
    n = len(model["pieces"])
    x_m = transform(model["global"], points)[:, 0] if to_pitch else points[:, 0]
    # Position in "piece units": 0 at the first centre, n - 1 at the last one.
    pos = np.interp(np.nan_to_num(x_m), model["centres"], np.arange(n)) if n > 1 else np.zeros(len(points))
    left = np.clip(np.floor(pos).astype(int), 0, max(n - 2, 0))
    weights = np.zeros((len(points), n))
    weights[np.arange(len(points)), left] = 1 - (pos - left)
    if n > 1:
        weights[np.arange(len(points)), left + 1] = pos - left
    out = np.zeros_like(points)
    for i, H in enumerate(model["pieces"]):
        used = weights[:, i] > 0
        if used.any():
            out[used] += weights[used, i:i + 1] * transform(H if to_pitch else np.linalg.inv(H), points[used])
    return out


# Both methods -----------------------------------------------------------------

def fit_mapping(method, px, metres, calibration, length_m, warn=True):
    """Fit `method` on reference points given in the pixel space of the calibration."""
    if method == "pieces":
        return fit_pieces(px, metres, int(calibration["pieces"]), length_m, float(calibration["overlap_m"]), warn)
    return fit_reference_points(px, metres, calibration["radius_m"], calibration["min_weight"])


def apply_mapping(method, model, points, to_pitch=True):
    """Pixels of the calibration -> metres (to_pitch=True), or metres -> pixels."""
    mapping = pieces_mapping if method == "pieces" else reference_points_mapping
    return mapping(model, points, to_pitch)


def pixel_to_pitch(points_px, method=None):
    """Raw video pixels (N, 2) -> approximate pitch positions (N, 2) in metres.

    The one function later stages use. method=None takes calibration.method
    from the config; "pieces" gives the baseline. The reference points and the
    settings are the ones the last "fit" run saved. Every point is mapped
    directly, without a lookup table. The positions are approximate (see the
    fit report).
    """
    calibration = load_field_config()["calibration"]
    method = method or calibration["method"]
    saved = np.load(HOMOGRAPHY_PATH)
    settings = {key: saved[key].item() for key in SAVED_SETTINGS}
    changed = [key for key in SAVED_SETTINGS if settings[key] != calibration[key]]
    if changed:
        print(f"Warning: {', '.join(changed)} changed in {CONFIG_PATH} since the last fit; "
              "run 'python src/calibrate.py fit' again")
    model = fit_mapping(method, saved["points_px"], saved["points_m"], settings, saved["length_m"].item(), warn=False)
    raw = np.asarray(points_px, np.float64).reshape(-1, 2)
    ok = np.isfinite(raw).all(axis=1)
    metres = np.full_like(raw, np.nan)              # no pixel in, no position out
    metres[ok] = apply_mapping(method, model, to_pixel_space(raw[ok], settings["pixel_space"]))
    return metres


def leave_one_out(method, px, metres, calibration, length_m):
    """Predict every reference point with a fit on all the others."""
    predicted = np.empty_like(metres)
    for i in range(len(px)):
        rest = np.arange(len(px)) != i
        model = fit_mapping(method, px[rest], metres[rest], calibration, length_m, warn=False)
        predicted[i] = apply_mapping(method, model, px[i:i + 1])[0]
    return predicted


def check_dataset_marks(px, metres, own, calibration, length_m):
    """Error vectors (dx, dy) of the dataset marks when they are predicted from the own points alone.

    A check of the dataset marks against the points selected by hand: the fit
    report lists every mark and flags the ones further than MARK_LIMIT_M from
    their nominal position. It does not decide what is used; that is the "use"
    column of the points file. The prediction is made with the method "pieces":
    with own points only, the method "reference_points" is not stable along the
    touchline between corner and halfway line, where no own point lies.
    """
    model = fit_mapping("pieces", px[own], metres[own], calibration, length_m, warn=False)
    return apply_mapping("pieces", model, px[~own]) - metres[~own]


def pitch_lines(landmarks, length_m, width_m):
    """Return (grid lines, pitch markings); every line is (N, 2) points in metres.

    Lines are sampled every SAMPLE_STEP_M: a straight pitch line is a curve in
    the panoramic image, and only a dense polyline shows that curve.
    """
    def segment(a, b):
        n = int(np.hypot(b[0] - a[0], b[1] - a[1]) / SAMPLE_STEP_M) + 2
        return np.linspace(a, b, n)

    grid = [segment((x, 0), (x, width_m)) for x in np.arange(0, length_m + 0.01, GRID_STEP_M)]
    grid += [segment((0, y), (length_m, y)) for y in np.arange(0, width_m + 0.01, GRID_STEP_M)]
    # Markings are built from the landmarks, so no pitch measurement is written here.
    pairs = [("corner_left_far", "corner_right_far"), ("corner_left_near", "corner_right_near"),
             ("corner_left_far", "corner_left_near"), ("corner_right_far", "corner_right_near"),
             ("halfway_far", "halfway_near")]
    for box in ("left_pen", "left_goalarea", "right_pen", "right_goalarea"):
        pairs += [(f"{box}_far_goal", f"{box}_far_in"), (f"{box}_far_in", f"{box}_near_in"),
                  (f"{box}_near_in", f"{box}_near_goal")]
    marks = [segment(landmarks[a], landmarks[b]) for a, b in pairs]
    radius = landmarks["centre_spot"][1] - landmarks["centre_circle_far"][1]
    angle = np.linspace(0, 2 * np.pi, 361)
    marks.append(np.array(landmarks["centre_spot"])
                 + radius * np.stack([np.cos(angle), np.sin(angle)], axis=1))
    return grid, marks


def save_grid_image(frame, to_pixels, grid, marks, raw_points, used, path):
    """Draw the metre grid, the pitch markings and the points of the points file into the frame.

    `to_pixels` maps metres to raw pixels with the fitted mapping, so the image
    shows the pitch as the reference points define it.
    """
    img = frame.copy()
    for lines, color, thickness in [(grid, GRID_COLOR, 1), (marks, MARKING_COLOR, 2)]:
        for line in lines:
            px = to_pixels(line)
            ok = np.isfinite(px).all(axis=1) & (np.abs(px) < 3 * img.shape[1]).all(axis=1)
            for a, b, both in zip(px[:-1], px[1:], ok[:-1] & ok[1:]):
                if both:
                    cv2.line(img, (int(round(a[0])), int(round(a[1]))),
                             (int(round(b[0])), int(round(b[1]))), color, thickness, cv2.LINE_AA)
    for (x, y), in_use in zip(raw_points, used):
        cv2.drawMarker(img, (int(round(x)), int(round(y))), USED_COLOR if in_use else UNUSED_COLOR,
                       cv2.MARKER_TILTED_CROSS, 11, 1 if in_use else 2, cv2.LINE_AA)
    draw_text(img, f"white x: reference point in use ({int(used.sum())})", (20, 30), 0.6, USED_COLOR)
    draw_text(img, f"magenta x: dataset mark, not used ({int((~used).sum())})", (20, 56), 0.6, UNUSED_COLOR)
    cv2.imwrite(str(path), img)


def error_cells(values):
    """mean / median / max of an error array, as text for the report."""
    if len(values) == 0:
        return "-"
    return f"{values.mean():.2f} / {np.median(values):.2f} / {values.max():.2f}"


def write_fit_report(path, names, metres, own, used, errors, passing, mark_errors, calibration, default):
    """Markdown report: the two methods, the check of the dataset marks, one row per point.

    errors[method]: for a reference point in use the leave-one-out error, for a
    point that is not used the error of the plain prediction (it was never part
    of the fit either way). The point table is sorted by the error of the
    default method, largest first, so its weakest places are at the top.
    """
    kind = np.where(own, "line intersection", "interpolated touchline mark")
    source = np.where(own, "own", "dataset")
    lines = ["# Calibration fit report", "",
             "All positions are approximate. Errors are distances in metres between the predicted",
             "and the nominal pitch position of a point. A reference point in use is predicted",
             "leave-one-out, by a fit that did not see it; a point that is not used is predicted",
             "by the full fit.", "",
             f"Reference points in use: {int(used.sum())} ({int((used & own).sum())} own, selected by hand on the frame; "
             f"{int((used & ~own).sum())} dataset touchline marks). Not used: {int((~used).sum())} dataset marks.",
             f"Settings: radius_m = {calibration['radius_m']:g}, min_weight = {calibration['min_weight']:g}, "
             f"pixel_space = {calibration['pixel_space']}; pieces = {calibration['pieces']}, "
             f"overlap_m = {calibration['overlap_m']:g}.", "",
             "## Methods", "",
             "| Method | Leave-one-out, all in use: mean / median / max (m) | Own points | Dataset marks in use "
             "| Dataset marks not used |",
             "|---|---|---|---|---|"]
    for method, err in errors.items():
        role = " (default)" if method == default else " (baseline)" if method == BASELINE_METHOD else ""
        lines.append(f"| {method}{role} | {error_cells(err[used])} | {error_cells(err[used & own])} | "
                     f"{error_cells(err[used & ~own])} | {error_cells(err[~used])} |")
    lines += ["", "Does the fitted mapping pass through its own reference points?", "",
              "| Method | Error at the reference points: mean / median / max (m) "
              "| Distance between a reference point and where the grid image draws it: mean / median / max (px) |",
              "|---|---|---|"]
    for method, (at_points, drawn) in passing.items():
        lines.append(f"| {method} | {error_cells(at_points)} | {error_cells(drawn)} |")
    lines += ["", "`reference_points`: each position is mapped with a homography fitted to all reference points,",
              "where the points near that position count more.",
              "`pieces`: blended pieces along the pitch length; kept only as the baseline for comparison.", "",
              "## Dataset marks checked against the own points", "",
              f"Every dataset mark is predicted from the {int(own.sum())} own points alone (method `pieces`). "
              f"{int((np.hypot(*mark_errors.T) > MARK_LIMIT_M).sum())} of {len(mark_errors)} marks are further than "
              f"{MARK_LIMIT_M:g} m from their nominal position. This is information only: which points are used is "
              "set in the `use` column of the points file.", "",
              f"| Mark | dx (m) | dy (m) | Error (m) | Within {MARK_LIMIT_M:g} m | Use |", "|---|---|---|---|---|---|"]
    mark_dist = np.hypot(*mark_errors.T)
    for k, i in enumerate(np.flatnonzero(~own)):
        lines.append(f"| {names[i]} | {mark_errors[k, 0]:+.2f} | {mark_errors[k, 1]:+.2f} | {mark_dist[k]:.2f} | "
                     f"{'yes' if mark_dist[k] <= MARK_LIMIT_M else 'no'} | {int(used[i])} |")
    order = [default] + [method for method in errors if method != default]
    lines += ["", "dx = predicted minus nominal position along the touchline, dy = across it.", "",
              f"## Error per point (m), sorted by `{default}`", "",
              "| Point | Type | Source | Use | x_m | y_m | " + " | ".join(order) + " |",
              "|---|---|---|---|---|---|" + "---|" * len(order)]
    for i in np.argsort(-errors[default]):
        lines.append(f"| {names[i]} | {kind[i]} | {source[i]} | {int(used[i])} | {metres[i, 0]:g} | {metres[i, 1]:g} | "
                     + " | ".join(f"{errors[m][i]:.2f}" for m in order) + " |")
    lines += ["", "Type: \"line intersection\" = a marked point of the pitch, selected by hand on the frame",
              "(crossings of marked lines; also the centre spot, the two side points of the centre",
              "circle and the feet of the goal posts). \"interpolated touchline mark\" = a point every",
              "5.5 m along a touchline, with no marked line crossing it; its pixel position comes from",
              "the dataset."]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def fit(video, frame_index):
    """Fit both methods, score them, write report, CSV and grid images, save the calibration."""
    config = load_field_config()
    landmarks, pitch, calibration = config["landmarks"], config["pitch"], config["calibration"]
    default = calibration["method"]
    points, info = load_points(config["points_file"], landmarks)
    names = list(points)
    raw = np.array([points[name] for name in names], np.float64)
    metres = np.array([landmarks[name] for name in names], np.float64)
    own = np.array([info[name][0] == "own" for name in names])
    used = np.array([info[name][1] == 1 for name in names])
    px = to_pixel_space(raw, calibration["pixel_space"])
    length_m = float(pitch["length_m"])
    frame = read_frame(video, frame_index)
    grid, marks = pitch_lines(landmarks, length_m, float(pitch["width_m"]))
    out_dir, stem = Path(OUT_DIR), Path(video).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Points       : {len(names)} in {config['points_file']}; {int(used.sum())} reference points in use "
          f"({int((used & own).sum())} own, {int((used & ~own).sum())} dataset marks)")
    print(f"Settings     : method {default}, radius {calibration['radius_m']:g} m, min weight "
          f"{calibration['min_weight']:g}, {calibration['pixel_space']} pixels")

    errors, passing, rows = {}, {}, []
    for method in METHODS:
        model = fit_mapping(method, px[used], metres[used], calibration, length_m)
        predicted = np.empty_like(metres)
        predicted[used] = leave_one_out(method, px[used], metres[used], calibration, length_m)
        if (~used).any():
            predicted[~used] = apply_mapping(method, model, px[~used])
        err = errors[method] = np.hypot(*(predicted - metres).T)
        # Does the mapping pass through its own reference points? In metres, and in
        # pixels between a reference point and where the grid image draws it.
        drawn = to_raw_pixels(apply_mapping(method, model, metres[used], to_pitch=False), calibration["pixel_space"])
        passing[method] = (np.hypot(*(apply_mapping(method, model, px[used]) - metres[used]).T),
                           np.hypot(*(drawn - raw[used]).T))
        rows += [[name, *metres[i], "own" if own[i] else "dataset", int(used[i]), method,
                  *predicted[i].round(3), round(float(err[i]), 3)] for i, name in enumerate(names)]
        image = out_dir / (f"{stem}_grid.jpg" if method == default else f"{stem}_grid_{method}.jpg")
        save_grid_image(frame, lambda m: to_raw_pixels(apply_mapping(method, model, m, to_pitch=False),
                                                       calibration["pixel_space"]), grid, marks, raw, used, image)
        print(f"  {method:<16} leave-one-out error: mean {err[used].mean():.2f} m, median "
              f"{np.median(err[used]):.2f} m, max {err[used].max():.2f} m "
              f"({names[int(np.flatnonzero(used)[err[used].argmax()])]}) -> {image.name}")
        print(f"  {'':<16} at its reference points: {passing[method][0].mean():.2f} m mean, {passing[method][0].max():.2f} m max; "
              f"drawn {passing[method][1].mean():.1f} px mean, {passing[method][1].max():.0f} px max from the point")

    mark_errors = check_dataset_marks(px, metres, own, calibration, length_m)
    far_marks = int((np.hypot(*mark_errors.T) > MARK_LIMIT_M).sum())

    np.savez(HOMOGRAPHY_PATH, video=video, frame=frame_index, points_px=px[used], points_m=metres[used],
             length_m=length_m, **{key: calibration[key] for key in SAVED_SETTINGS})
    with open(out_dir / f"{stem}_fit_errors.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "x_m", "y_m", "source", "use", "method", "pred_x_m", "pred_y_m", "error_m"])
        writer.writerows(rows)
    write_fit_report(out_dir / f"{stem}_fit_report.md", names, metres, own, used, errors, passing, mark_errors,
                     calibration, default)
    print(f"Dataset marks: {int((used & ~own).sum())} in use, {int((~used).sum())} not used; {far_marks} are more than "
          f"{MARK_LIMIT_M:g} m from the prediction of the own points (see the report)")
    print(f"Calibration  : {HOMOGRAPHY_PATH}")
    print(f"Report       : {out_dir / (stem + '_fit_report.md')}, {stem}_fit_errors.csv")


def main():
    parser = argparse.ArgumentParser(description="Pitch calibration tools.")
    commands = parser.add_subparsers(dest="command", required=True)
    sel = commands.add_parser("select", help="select pitch landmarks on a video frame")
    sel.add_argument("--redo", metavar="NAME", help="select this one landmark again")
    fit_cmd = commands.add_parser("fit", help="fit and score the pixel -> metre mapping")
    for cmd in (sel, fit_cmd):
        cmd.add_argument("--video", default=DEFAULT_VIDEO)
        cmd.add_argument("--frame", type=int, default=0, help="0-based frame number")
    args = parser.parse_args()
    if args.command == "select":
        select(args.video, args.frame, args.redo)
    else:
        fit(args.video, args.frame)


if __name__ == "__main__":
    main()
