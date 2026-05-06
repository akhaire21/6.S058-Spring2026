"""
Experiment 1 / 2.2 component: InteractionPrior.

For a candidate assignment, compute the implied velocity from the current track
to the candidate detection. Penalize it if it disagrees with the mean velocity
of nearby tracks.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from tracking.geometry import centers_xyxy


def neighbor_mean_velocity(
    tracks,
    track_idx: int,
    max_age: int,
    rel_k: int,
    rel_radius: float,
) -> Tuple[np.ndarray, bool]:
    if len(tracks) <= 1:
        return np.zeros(2, dtype=np.float32), False

    ci = tracks[track_idx].center()
    candidates = []

    for q, t in enumerate(tracks):
        if q == track_idx:
            continue

        if t.time_since_update > max_age:
            continue

        d = float(np.linalg.norm(t.center() - ci))
        if d <= float(rel_radius):
            candidates.append((d, q))

    if not candidates:
        return np.zeros(2, dtype=np.float32), False

    candidates.sort(key=lambda x: x[0])
    keep = [q for _, q in candidates[: int(rel_k)]]

    if len(keep) == 0:
        return np.zeros(2, dtype=np.float32), False

    mean_vel = np.mean([tracks[q].vel for q in keep], axis=0).astype(np.float32)
    return mean_vel, True


def relation_cost_row(
    tracks,
    track_idx: int,
    det_boxes: np.ndarray,
    max_age: int,
    rel_k: int,
    rel_radius: float,
    rel_norm_px: float,
):
    mean_vel, has_neighbors = neighbor_mean_velocity(
        tracks=tracks,
        track_idx=track_idx,
        max_age=max_age,
        rel_k=rel_k,
        rel_radius=rel_radius,
    )

    if not has_neighbors or len(det_boxes) == 0:
        return np.zeros((len(det_boxes),), dtype=np.float32), False

    det_centers = centers_xyxy(det_boxes)
    last = tracks[track_idx].last_obs_center.astype(np.float32)

    implied_vel = det_centers - last[None, :]
    rel = np.linalg.norm(implied_vel - mean_vel[None, :], axis=1)
    rel = rel / max(1.0, float(rel_norm_px))

    return rel.astype(np.float32), True
