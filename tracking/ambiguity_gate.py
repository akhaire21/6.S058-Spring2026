"""
Experiment 1 component: ambiguity gate.

A row is ambiguous if:
1) the best match cost is weak,
2) the best-vs-second-best margin is small, or
3) the track is in a crowded detection neighborhood.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from tracking.geometry import centers_xyxy


def compute_ambiguity_flags(
    base_cost: np.ndarray,
    pred_boxes: np.ndarray,
    det_boxes: np.ndarray,
    weak_cost: float,
    margin_thresh: float,
    crowd_radius: float,
    min_neighbors: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    n, m = base_cost.shape

    ambiguous = np.zeros((n,), dtype=bool)
    weak = np.zeros((n,), dtype=bool)
    margin = np.zeros((n,), dtype=bool)
    crowded = np.zeros((n,), dtype=bool)

    if n == 0 or m == 0:
        return ambiguous, {"weak": weak, "margin": margin, "crowded": crowded}

    det_centers = centers_xyxy(det_boxes)
    pred_centers = centers_xyxy(pred_boxes)

    for i in range(n):
        row = np.asarray(base_cost[i], dtype=np.float32)
        finite = row[np.isfinite(row)]

        if len(finite) == 0:
            continue

        sorted_costs = np.sort(finite)
        best = float(sorted_costs[0])
        second = float(sorted_costs[1]) if len(sorted_costs) >= 2 else float("inf")

        weak[i] = best > float(weak_cost)
        margin[i] = (second - best) < float(margin_thresh)

        dists = np.linalg.norm(det_centers - pred_centers[i], axis=1)
        crowded[i] = int(np.sum(dists <= float(crowd_radius))) >= int(min_neighbors)

        ambiguous[i] = weak[i] or margin[i] or crowded[i]

    return ambiguous, {"weak": weak, "margin": margin, "crowded": crowded}
