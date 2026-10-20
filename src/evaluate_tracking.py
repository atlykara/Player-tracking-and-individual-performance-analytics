"""Tracking performance against the ground truth (v1).

Compares the tracker output with the SoccerTrack v2 ground truth (GT) frame
by frame and computes the key tracking metrics: MOTA, IDF1, precision,
recall, ID switches, FP / FN and the number of IDs. All counting is written
here on purpose, without a ready-made MOT evaluation library, so every number
can be explained step by step. Only scipy's Hungarian algorithm is reused.

Ultralytics `yolo track ... save_txt=True save_conf=True` writes one label
file per frame, `<video>_<N>.txt`, one row per tracked box:

    class x_center y_center width height confidence track_id

with box values normalised to [0, 1]. The GT is a MOT text file:

    frame, id, x, y, w, h, conf, class, visibility, unused
"""

import json
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# --- Settings -------------------------------------------------------------
LABELS_DIR = "outputs/tracking/118575_yolo11m_botsort_roi/labels"
VIDEO_PATH = "data/SoccerTrackv2/mot/118575.mp4"
GT_PATH = "data/SoccerTrackv2/mot/118575_gt.txt"
OUTPUT_DIR = "outputs/tracking/118575_yolo11m_botsort_roi/evaluation"

IOU_THRESHOLD = 0.5       # standard CLEAR MOT value: boxes must overlap at least half
PREVIEW_WIDTH = 2048      # width of alignment_check.jpg

# Ultralytics numbers label files from 1, the GT numbers frames from 0
# (0 ... 2999, measured). Subtracting 1 puts both in the same 0-based system.
FRAME_OFFSET = -1
LABEL_COLUMNS = ["cls", "xc", "yc", "wn", "hn", "conf", "track_id"]
GT_COLUMNS = ["frame", "track_id", "x", "y", "w", "h", "conf", "cls", "visibility", "unused"]
PERSON_CLASS = 1          # MOT17 class id "pedestrian"
NOT_ANNOTATED = -1        # value used in a column the annotators did not fill
BOX = ["x", "y", "w", "h"]


# --- Loading --------------------------------------------------------------
def video_size(video_path):
    """Return (width, height) of the video in pixels."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return size


def load_predictions(labels_dir=LABELS_DIR, video_path=VIDEO_PATH):
    """Read all Ultralytics label files into one table in pixel coordinates.

    Columns: frame, track_id, x, y, w, h, conf (x, y = top-left corner).
    Frames with no detection have no label file; they are simply absent.
    """
    width, height = video_size(video_path)
    tables = []
    for path in Path(labels_dir).glob("*.txt"):
        if path.stat().st_size == 0:
            continue
        table = pd.read_csv(path, sep=" ", header=None, names=LABEL_COLUMNS)
        table["frame"] = int(re.search(r"_(\d+)\.txt$", path.name).group(1)) + FRAME_OFFSET
        tables.append(table)
    if not tables:
        raise ValueError(f"No label rows found in {labels_dir}")
    raw = pd.concat(tables, ignore_index=True)

    # A box without a track ID cannot be followed over time; report and drop it.
    missing_id = int(raw["track_id"].isna().sum())
    raw = raw.dropna(subset=["track_id"])

    # Normalised centre/size -> top-left corner and size in pixels.
    pred = pd.DataFrame({
        "frame": raw["frame"],
        "track_id": raw["track_id"].astype(int),
        "x": (raw["xc"] - raw["wn"] / 2) * width,
        "y": (raw["yc"] - raw["hn"] / 2) * height,
        "w": raw["wn"] * width,
        "h": raw["hn"] * height,
        "conf": raw["conf"],
    }).sort_values(["frame", "track_id"]).reset_index(drop=True)
    print(f"Predictions      : {len(pred)} boxes, {pred['track_id'].nunique()} IDs, "
          f"frames {pred['frame'].min()}..{pred['frame'].max()} "
          f"(dropped {missing_id} rows without track ID)")
    return pred


def load_ground_truth(path=GT_PATH):
    """Read every GT row and add an `is_valid` column.

    MOT rule: a GT row with conf == 0 is marked "ignore" by the annotators,
    and rows of non-person classes (vehicles, static objects, ...) are not
    tracking targets; both get is_valid = False and are not evaluated. In
    this dataset conf, class and visibility are all -1 ("not annotated"),
    which is neither 0 nor a foreign class, so every row stays valid.
    """
    gt = pd.read_csv(path, header=None, names=GT_COLUMNS)
    for col in ["conf", "cls", "visibility"]:
        print(f"GT {col:<10} values: {gt[col].value_counts().to_dict()}")
    gt["is_valid"] = (gt["conf"] != 0) & gt["cls"].isin([PERSON_CLASS, NOT_ANNOTATED])
    print(f"GT rows          : {len(gt)} total, {int(gt['is_valid'].sum())} valid, "
          f"{gt['track_id'].nunique()} IDs")
    return gt


# --- Matching and metrics -------------------------------------------------
def iou_matrix(a, b):
    """IoU between every box in a (n x 4) and b (m x 4); boxes are x, y, w, h."""
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 0:1] + a[:, 2:3], a[:, 1:2] + a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    inter_w = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    inter_h = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = inter_w * inter_h
    union = (a[:, 2:3] * a[:, 3:4]) + (b[:, 2] * b[:, 3]) - inter
    return inter / union


def match_frame(gt_boxes, pred_boxes):
    """Hungarian matching on cost 1 - IoU; pairs below IOU_THRESHOLD are dropped.

    Returns a list of (gt_index, pred_index).
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return []
    iou = iou_matrix(gt_boxes, pred_boxes)
    rows, cols = linear_sum_assignment(1 - iou)
    return [(r, c) for r, c in zip(rows, cols) if iou[r, c] >= IOU_THRESHOLD]


def evaluate(gt, pred):
    """Match every frame, count errors and return the metrics as a dict.

    Only GT rows with is_valid = True take part.
    """
    gt = gt[gt["is_valid"]]
    gt_by_frame = dict(list(gt.groupby("frame")))
    pred_by_frame = dict(list(pred.groupby("frame")))
    empty = gt.iloc[:0]
    frames = sorted(set(gt_by_frame) | set(pred_by_frame))

    last_pred_of = {}         # GT ID -> prediction ID it was last matched to
    pair_frames = Counter()   # (GT ID, pred ID) -> frames they were matched
    tp = idsw = 0
    for frame in frames:
        g = gt_by_frame.get(frame, empty)
        p = pred_by_frame.get(frame, empty)
        for gi, pi in match_frame(g[BOX].to_numpy(float), p[BOX].to_numpy(float)):
            gid, pid = g["track_id"].iat[gi], p["track_id"].iat[pi]
            # ID switch: this GT player is now followed by a different track ID.
            if gid in last_pred_of and last_pred_of[gid] != pid:
                idsw += 1
            last_pred_of[gid] = pid
            pair_frames[(gid, pid)] += 1
            tp += 1

    # Every box that was not matched is an error of its own side.
    n_gt, n_pred = len(gt), len(pred)
    fp, fn = n_pred - tp, n_gt - tp

    # IDF1: one fixed pred ID per GT ID for the whole video (one-to-one),
    # chosen to maximise the frames where the pair is matched (IDTP).
    gt_ids, pred_ids = sorted(gt["track_id"].unique()), sorted(pred["track_id"].unique())
    counts = np.zeros((len(gt_ids), len(pred_ids)))
    g_index = {g: i for i, g in enumerate(gt_ids)}
    p_index = {p: i for i, p in enumerate(pred_ids)}
    for (gid, pid), n in pair_frames.items():
        counts[g_index[gid], p_index[pid]] = n
    r, c = linear_sum_assignment(counts, maximize=True)
    idtp = counts[r, c].sum()

    m = {
        "iou_threshold": IOU_THRESHOLD, "frames": len(frames),
        "gt_ids": len(gt_ids), "pred_ids": len(pred_ids),
        "tp": tp, "fp": fp, "fn": fn, "idsw": idsw,
        "precision": tp / (tp + fp), "recall": tp / (tp + fn),
        "mota": 1 - (fn + fp + idsw) / n_gt,
        "idf1": 2 * idtp / (n_gt + n_pred),
    }
    return {k: round(float(v), 4) if isinstance(v, (float, np.floating)) else v
            for k, v in m.items()}


def self_test(gt):
    """Evaluate the GT against itself; everything must be perfect.

    This checks the counting code, not the tracker: if a perfect prediction
    does not give 1.0 / 0 everywhere, the evaluation itself is wrong.
    """
    m = evaluate(gt, gt[gt["is_valid"]].copy())
    ok = (m["precision"] == m["recall"] == m["mota"] == m["idf1"] == 1
          and m["idsw"] == m["fp"] == m["fn"] == 0)
    print(f"Self-test        : {'PASSED' if ok else 'FAILED'} "
          f"(P={m['precision']} R={m['recall']} MOTA={m['mota']} IDF1={m['idf1']} "
          f"IDSW={m['idsw']} FP={m['fp']} FN={m['fn']})")
    if not ok:
        raise RuntimeError("Self-test failed: the evaluation code is wrong.")


# --- Outputs --------------------------------------------------------------
# Report rows: (label, metric keys shown as "a / b", one-sentence meaning).
REPORT_ROWS = [
    ("MOTA", ["mota"], "Overall score: 1 minus all errors (FN + FP + ID switches) per GT box; can be negative."),
    ("IDF1", ["idf1"], "Identity score: how well one player keeps one track ID over the whole clip."),
    ("Precision", ["precision"], "Share of predicted boxes that match a real player."),
    ("Recall", ["recall"], "Share of real players (GT boxes) that were found."),
    ("ID switches", ["idsw"], "Times a GT player became matched to a different track ID than before."),
    ("GT IDs / Predicted IDs", ["gt_ids", "pred_ids"], "Real players vs. track IDs handed out; equal numbers would mean no fragmentation."),
    ("FP / FN", ["fp", "fn"], "Extra or misplaced boxes / missed GT boxes."),
]


def write_report(m, path):
    """Write the key metrics as a short English Markdown report."""
    lines = ["# Tracking evaluation report (ROI run vs. ground truth)", "",
             f"Labels: `{LABELS_DIR}`  ", f"Ground truth: `{GT_PATH}`  ",
             f"IoU threshold: {IOU_THRESHOLD}", "",
             "| Metric | Value | Meaning |", "|---|---|---|"]
    for label, keys, text in REPORT_ROWS:
        lines.append(f"| {label} | {' / '.join(str(m[k]) for k in keys)} | {text} |")
    lines += ["", f"Values refer to this {m['frames']}-frame clip only; a box counts as "
                  f"correct when it overlaps a GT box with IoU >= {IOU_THRESHOLD}."]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_alignment_check(gt, pred, path):
    """First, middle and last frame with GT (green) and predictions (red), stacked."""
    cap = cv2.VideoCapture(VIDEO_PATH)
    last = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    images = []
    for frame in [0, last // 2, last]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {frame} from {VIDEO_PATH}")
        for table, color in [(gt[gt["is_valid"]], (0, 255, 0)), (pred, (0, 0, 255))]:
            for x, y, w, h in table.loc[table["frame"] == frame, BOX].to_numpy(int):
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f"frame {frame}  GT=green  pred=red", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)
        images.append(img)
    cap.release()
    stacked = np.vstack(images)
    scale = PREVIEW_WIDTH / stacked.shape[1]
    stacked = cv2.resize(stacked, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), stacked)


def main():
    """Self-test first; then evaluate the tracker and write all outputs."""
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = load_ground_truth()
    self_test(gt)

    pred = load_predictions()
    m = evaluate(gt, pred)
    (out_dir / "performance_metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    write_report(m, out_dir / "performance_report.md")
    save_alignment_check(gt, pred, out_dir / "alignment_check.jpg")

    for label, keys, _ in REPORT_ROWS:
        print(f"{label:<23}: {' / '.join(str(m[k]) for k in keys)}")
    print(f"Outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
