"""
Shared geometry utilities for SORT, EIoU-SORT, ambiguity gating, and
InteractionPrior experiments.
"""

from __future__ import annotations

import numpy as np


def area_xyxy(box: np.ndarray) -> float:
    w = max(0.0, float(box[2] - box[0]))
    h = max(0.0, float(box[3] - box[1]))
    return w * h


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0

    union = area_xyxy(a) + area_xyxy(b) - inter + 1e-9
    return inter / union


def expand_box_xyxy(box: np.ndarray, alpha: float) -> np.ndarray:
    x1, y1, x2, y2 = map(float, box)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)

    new_w = w * (1.0 + alpha)
    new_h = h * (1.0 + alpha)

    return np.array(
        [
            cx - 0.5 * new_w,
            cy - 0.5 * new_h,
            cx + 0.5 * new_w,
            cy + 0.5 * new_h,
        ],
        dtype=np.float32,
    )


def expansion_iou(a: np.ndarray, b: np.ndarray, alpha: float = 0.5) -> float:
    return iou_xyxy(expand_box_xyxy(a, alpha), expand_box_xyxy(b, alpha))


def centers_xyxy(boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty((0, 2), dtype=np.float32)

    cx = 0.5 * (boxes[:, 0] + boxes[:, 2])
    cy = 0.5 * (boxes[:, 1] + boxes[:, 3])
    return np.stack([cx, cy], axis=1).astype(np.float32)


def pairwise_center_distance(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    ca = centers_xyxy(boxes_a)
    cb = centers_xyxy(boxes_b)
    diff = ca[:, None, :] - cb[None, :, :]
    return np.linalg.norm(diff, axis=-1).astype(np.float32)


def pairwise_overlap(
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
    use_eiou: bool,
    eiou_alpha: float,
) -> np.ndarray:
    n, m = len(boxes_a), len(boxes_b)
    out = np.zeros((n, m), dtype=np.float32)

    if n == 0 or m == 0:
        return out

    for i in range(n):
        for j in range(m):
            if use_eiou:
                out[i, j] = expansion_iou(boxes_a[i], boxes_b[j], alpha=eiou_alpha)
            else:
                out[i, j] = iou_xyxy(boxes_a[i], boxes_b[j])

    return out
