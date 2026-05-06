"""
Classical detectors used only in the detection pilot (Section 5.3 of the paper).

These exist to satisfy the course's detection-module requirement and to provide
a fair baseline against which YOLOv8 is justified as the main detector. They
are not used in the main tracking experiments.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from detection.yolo_cache import FrameDetections


@dataclass
class HSVRange:
    h_lo: int
    h_hi: int
    s_lo: int = 60
    s_hi: int = 255
    v_lo: int = 50
    v_hi: int = 255

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        if self.h_lo <= self.h_hi:
            return cv2.inRange(hsv, (self.h_lo, self.s_lo, self.v_lo),
                                    (self.h_hi, self.s_hi, self.v_hi))
        # Wraps around 180 (red): two ranges.
        a = cv2.inRange(hsv, (self.h_lo, self.s_lo, self.v_lo),
                              (180, self.s_hi, self.v_hi))
        b = cv2.inRange(hsv, (0, self.s_lo, self.v_lo),
                              (self.h_hi, self.s_hi, self.v_hi))
        return cv2.bitwise_or(a, b)


class MOG2Detector:
    """Background-subtraction detector. Best on stationary-camera clips."""
    def __init__(self, history: int = 200, var_threshold: float = 16.0,
                 min_area: int = 800, max_area: int = 80_000,
                 min_aspect: float = 1.2, max_aspect: float = 6.0):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=history,
                                                      varThreshold=var_threshold,
                                                      detectShadows=True)
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect = min_aspect  # height/width; players are taller than wide
        self.max_aspect = max_aspect
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 11))

    def detect(self, frame_bgr: np.ndarray) -> FrameDetections:
        fg = self.bg.apply(frame_bgr)
        # Treat shadows (value 127) as background.
        fg = (fg == 255).astype(np.uint8) * 255
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.kernel)
        n, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        boxes, scores = [], []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < self.min_area or area > self.max_area:
                continue
            ar = h / max(w, 1)
            if ar < self.min_aspect or ar > self.max_aspect:
                continue
            boxes.append([x, y, x + w, y + h])
            scores.append(min(1.0, area / 5000.0))
        return FrameDetections(
            boxes_xyxy=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            scores=np.asarray(scores, dtype=np.float32),
        )


class HSVDetector:
    """Detect players by team-color HSV range. Useful as a baseline,
    sensitive to lighting."""
    def __init__(self, ranges: list[HSVRange], min_area: int = 600,
                 min_aspect: float = 1.0):
        self.ranges = ranges
        self.min_area = min_area
        self.min_aspect = min_aspect
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 9))

    def detect(self, frame_bgr: np.ndarray) -> FrameDetections:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        full_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for r in self.ranges:
            full_mask = cv2.bitwise_or(full_mask, r.mask(hsv))
        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, self.kernel)
        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_DILATE, self.kernel)
        n, _, stats, _ = cv2.connectedComponentsWithStats(full_mask, connectivity=8)
        boxes, scores = [], []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < self.min_area:
                continue
            if h / max(w, 1) < self.min_aspect:
                continue
            boxes.append([x, y, x + w, y + h])
            scores.append(min(1.0, area / 4000.0))
        return FrameDetections(
            boxes_xyxy=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            scores=np.asarray(scores, dtype=np.float32),
        )


# Convenience: a small pilot evaluator that compares detectors to GT on a clip.

def boxes_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU. a: (N,4), b: (M,4) in xyxy. Returns (N,M)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    xa1, ya1, xa2, ya2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    xb1, yb1, xb2, yb2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(xa1, xb1)
    inter_y1 = np.maximum(ya1, yb1)
    inter_x2 = np.minimum(xa2, xb2)
    inter_y2 = np.minimum(ya2, yb2)
    iw = np.clip(inter_x2 - inter_x1, 0, None)
    ih = np.clip(inter_y2 - inter_y1, 0, None)
    inter = iw * ih
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter + 1e-9
    return inter / union


def evaluate_detector_on_sequence(detector, sequence, iou_thresh: float = 0.5):
    """Run a detector over every frame and compute precision/recall/F1
    against GT. Detector must implement .detect(frame_bgr) -> FrameDetections.
    """
    tp = fp = fn = 0
    for frame_idx, img_path in sequence.iter_frames():
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        det = detector.detect(img)
        gt_rows = sequence.gt_at_frame(frame_idx)
        # GT is xywh; convert to xyxy.
        if gt_rows.size > 0:
            gt_xyxy = gt_rows[:, 2:6].copy()
            gt_xyxy[:, 2] += gt_xyxy[:, 0]
            gt_xyxy[:, 3] += gt_xyxy[:, 1]
        else:
            gt_xyxy = np.empty((0, 4), dtype=np.float32)

        if det.n == 0 and gt_xyxy.shape[0] == 0:
            continue
        if det.n == 0:
            fn += gt_xyxy.shape[0]
            continue
        if gt_xyxy.shape[0] == 0:
            fp += det.n
            continue

        ious = boxes_iou(det.boxes_xyxy, gt_xyxy)
        # Greedy matching: each GT matched at most once, each det matched at most once.
        det_matched = np.zeros(det.n, dtype=bool)
        gt_matched = np.zeros(gt_xyxy.shape[0], dtype=bool)
        # Sort all (det, gt) pairs by IoU desc.
        flat = [(ious[i, j], i, j) for i in range(ious.shape[0]) for j in range(ious.shape[1])]
        flat.sort(reverse=True)
        for iou, i, j in flat:
            if iou < iou_thresh:
                break
            if det_matched[i] or gt_matched[j]:
                continue
            det_matched[i] = True
            gt_matched[j] = True
        tp += int(det_matched.sum())
        fp += int((~det_matched).sum())
        fn += int((~gt_matched).sum())

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
