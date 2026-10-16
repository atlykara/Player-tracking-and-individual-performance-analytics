"""Tracking performance against the ground truth (v1).

Compares the tracker output with the SoccerTrack v2 ground truth (GT) frame
by frame and computes the CLEAR MOT metrics (MOTA, MOTP) and the ID metrics
(IDF1, ID switches, mostly tracked / lost). All counting is written here on
purpose, without a ready-made MOT evaluation library, so every number can be
explained step by step. Only scipy's Hungarian algorithm is reused.

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
OUTPUT_DIR = "outputs/evaluation/118575_yolo11m_botsort_roi"

IOU_THRESHOLD = 0.5       # standard CLEAR MOT value: boxes must overlap at least half
MOSTLY_TRACKED = 0.8      # GT track matched in more than 80 % of its frames
MOSTLY_LOST = 0.2         # GT track matched in less than 20 % of its frames
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

    Returns a list of (gt_index, pred_index, iou).
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return []
    iou = iou_matrix(gt_boxes, pred_boxes)
    rows, cols = linear_sum_assignment(1 - iou)
    return [(r, c, iou[r, c]) for r, c in zip(rows, cols) if iou[r, c] >= IOU_THRESHOLD]


def evaluate(gt, pred):
    """Match every frame, count errors and return (metrics, per-frame table).

    Only GT rows with is_valid = True take part.
    """
    gt = gt[gt["is_valid"]]
    gt_by_frame = dict(list(gt.groupby("frame")))
    pred_by_frame = dict(list(pred.groupby("frame")))
    empty = gt.iloc[:0]

    last_pred_of = {}         # GT ID -> prediction ID it was last matched to
    pair_frames = Counter()   # (GT ID, pred ID) -> frames they were matched
    iou_sum, rows = 0.0, []
    for frame in sorted(set(gt_by_frame) | set(pred_by_frame)):
        g = gt_by_frame.get(frame, empty)
        p = pred_by_frame.get(frame, empty)
        matches = match_frame(g[BOX].to_numpy(float), p[BOX].to_numpy(float))

        idsw = 0
        for gi, pi, iou in matches:
            gid, pid = g["track_id"].iat[gi], p["track_id"].iat[pi]
            # ID switch: this GT player is now followed by a different track ID.
            if gid in last_pred_of and last_pred_of[gid] != pid:
                idsw += 1
            last_pred_of[gid] = pid
            pair_frames[(gid, pid)] += 1
            iou_sum += iou
        tp = len(matches)
        rows.append({"frame": frame, "n_gt": len(g), "n_pred": len(p), "tp": tp,
                     "fp": len(p) - tp, "fn": len(g) - tp, "idsw": idsw})
    per_frame = pd.DataFrame(rows)

    tp, fp, fn, idsw = (int(per_frame[c].sum()) for c in ["tp", "fp", "fn", "idsw"])
    n_gt, n_pred = len(gt), len(pred)

    # IDF1: one fixed pred ID per GT ID for the whole video (one-to-one),
    # chosen to maximise the frames where the pair is matched (IDTP).
    gt_ids, pred_ids = sorted(gt["track_id"].unique()), sorted(pred["track_id"].unique())
    counts = np.zeros((len(gt_ids), len(pred_ids)))
    g_index = {g: i for i, g in enumerate(gt_ids)}
    p_index = {p: i for i, p in enumerate(pred_ids)}
    for (gid, pid), n in pair_frames.items():
        counts[g_index[gid], p_index[pid]] = n
    r, c = linear_sum_assignment(counts, maximize=True)
    idtp = int(counts[r, c].sum())

    # Mostly tracked / lost: share of each GT track's frames that were matched.
    matched_per_gt = Counter()
    for (gid, _), n in pair_frames.items():
        matched_per_gt[gid] += n
    ratio = pd.Series({g: matched_per_gt[g] / n for g, n in gt["track_id"].value_counts().items()})

    m = {
        "iou_threshold": IOU_THRESHOLD,
        "frames": len(per_frame), "gt_boxes": n_gt, "pred_boxes": n_pred,
        "gt_ids": len(gt_ids), "pred_ids": len(pred_ids),
        "tp": tp, "fp": fp, "fn": fn, "idsw": idsw,
        "precision": tp / (tp + fp), "recall": tp / (tp + fn),
        "mota": 1 - (fn + fp + idsw) / n_gt,
        "motp": iou_sum / tp if tp else 0.0,
        "idtp": idtp, "idf1": 2 * idtp / (n_gt + n_pred),
        "mostly_tracked": int((ratio > MOSTLY_TRACKED).sum()),
        "mostly_lost": int((ratio < MOSTLY_LOST).sum()),
    }
    m["partially_tracked"] = m["gt_ids"] - m["mostly_tracked"] - m["mostly_lost"]
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}, per_frame


def self_test(gt):
    """Evaluate the GT against itself; everything must be perfect.

    This checks the counting code, not the tracker: if a perfect prediction
    does not give 1.0 / 0 everywhere, the evaluation itself is wrong.
    """
    m, _ = evaluate(gt, gt[gt["is_valid"]].copy())
    ok = (m["precision"] == m["recall"] == m["mota"] == m["idf1"] == 1
          and m["idsw"] == m["fp"] == m["fn"] == 0)
    print(f"Self-test        : {'PASSED' if ok else 'FAILED'} "
          f"(P={m['precision']} R={m['recall']} MOTA={m['mota']} IDF1={m['idf1']} "
          f"IDSW={m['idsw']} FP={m['fp']} FN={m['fn']})")
    if not ok:
        raise RuntimeError("Self-test failed: the evaluation code is wrong.")


# --- Outputs --------------------------------------------------------------
# Metric key -> (label, one-sentence meaning) for the Markdown report.
DESCRIPTIONS = {
    "frames": ("Frames", "Frames compared."),
    "gt_boxes": ("GT boxes", "Valid player boxes in the ground truth."),
    "pred_boxes": ("Predicted boxes", "Boxes reported by the tracker."),
    "gt_ids": ("GT IDs", "Different players in the ground truth."),
    "pred_ids": ("Predicted IDs", "Different track IDs handed out by the tracker."),
    "tp": ("TP", "Predicted boxes matched to a GT box with IoU >= threshold."),
    "fp": ("FP", "Predicted boxes with no matching GT box (extra or misplaced boxes)."),
    "fn": ("FN", "GT boxes with no matching prediction (missed players)."),
    "idsw": ("ID switches", "Times a GT player became matched to a different track ID than before."),
    "precision": ("Precision", "Share of predicted boxes that are correct."),
    "recall": ("Recall", "Share of GT boxes that were found."),
    "mota": ("MOTA", "1 minus all errors (FN + FP + IDSW) per GT box; can be negative."),
    "motp": ("MOTP (mean IoU)", "How tightly matched boxes overlap the GT, on average."),
    "idtp": ("IDTP", "Matched boxes that also carry the one track ID assigned to that player for the whole clip."),
    "idf1": ("IDF1", "F1 score of identity: how well one player keeps one ID over the whole clip."),
    "mostly_tracked": ("Mostly tracked", f"GT players matched in more than {MOSTLY_TRACKED:.0%} of their frames."),
    "partially_tracked": ("Partially tracked", "GT players between the two limits."),
    "mostly_lost": ("Mostly lost", f"GT players matched in less than {MOSTLY_LOST:.0%} of their frames."),
}


def write_report(m, path):
    """Write the metrics as an English Markdown table."""
    lines = ["# Tracking performance report (with ground truth)", "",
             f"Labels: `{LABELS_DIR}`  ", f"Ground truth: `{GT_PATH}`  ",
             f"IoU threshold: {IOU_THRESHOLD}", "",
             "| Metric | Value | Meaning |", "|---|---|---|"]
    lines += [f"| {label} | {m[key]} | {text} |" for key, (label, text) in DESCRIPTIONS.items()]
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
    m, per_frame = evaluate(gt, pred)
    (out_dir / "performance_metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    write_report(m, out_dir / "performance_report.md")
    per_frame.to_csv(out_dir / "per_frame.csv", index=False)
    save_alignment_check(gt, pred, out_dir / "alignment_check.jpg")

    for key, (label, _) in DESCRIPTIONS.items():
        print(f"{label:<20}: {m[key]}")
    print(f"Outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
