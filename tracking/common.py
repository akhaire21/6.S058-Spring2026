"""
Shared tracking utilities used across all experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class TrackRow:
    frame_idx: int
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0


class TrackState:
    """Constant-velocity track state used by all SORT-style variants."""

    def __init__(self, track_id: int, init_box: np.ndarray, score: float, cfg):
        self.id = int(track_id)
        self.box = init_box.astype(np.float32).copy()
        self.score = float(score)
        self.cfg = cfg

        self.vel = np.zeros(2, dtype=np.float32)
        self.last_obs_center = self.center().copy()

        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.confirmed = cfg.min_hits <= 1

    def center(self) -> np.ndarray:
        return np.array(
            [
                (self.box[0] + self.box[2]) * 0.5,
                (self.box[1] + self.box[3]) * 0.5,
            ],
            dtype=np.float32,
        )

    def predict(self):
        self.box[[0, 2]] += self.vel[0]
        self.box[[1, 3]] += self.vel[1]
        self.age += 1
        self.time_since_update += 1

    def update(self, det_box: np.ndarray, det_score: float):
        det_center = np.array(
            [
                (det_box[0] + det_box[2]) * 0.5,
                (det_box[1] + det_box[3]) * 0.5,
            ],
            dtype=np.float32,
        )

        inst_vel = det_center - self.last_obs_center
        self.vel = self.cfg.vel_beta * self.vel + (1.0 - self.cfg.vel_beta) * inst_vel

        self.box = det_box.astype(np.float32).copy()
        self.score = float(det_score)
        self.last_obs_center = det_center
        self.time_since_update = 0
        self.hits += 1

        if self.hits >= self.cfg.min_hits:
            self.confirmed = True


def extract_det_arrays(det_obj, frame_idx: int):
    frames = np.asarray(det_obj.frames)
    boxes = np.asarray(det_obj.boxes)
    mask = frames == frame_idx

    frame_boxes = boxes[mask].astype(np.float32)
    scores = getattr(det_obj, "scores", None)

    if scores is None:
        frame_scores = np.ones((len(frame_boxes),), dtype=np.float32)
    else:
        frame_scores = np.asarray(scores)[mask].astype(np.float32)

    return frame_boxes, frame_scores


def run_tracker_on_sequence(tracker, sequence, detections):
    rows: List[TrackRow] = []
    seq_len = int(sequence.info.seq_length)

    for frame_idx in range(1, seq_len + 1):
        det_boxes, det_scores = extract_det_arrays(detections, frame_idx)
        rows.extend(tracker.step(frame_idx, det_boxes, det_scores))

    return rows


def save_mot_txt(rows: List[TrackRow], out_path: str | Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for r in rows:
            w = r.x2 - r.x1
            h = r.y2 - r.y1
            f.write(
                f"{r.frame_idx},{r.track_id},{r.x1:.2f},{r.y1:.2f},"
                f"{w:.2f},{h:.2f},{r.score:.4f},-1,-1,-1\n"
            )


def summarize_rows(rows: List[TrackRow], short_len: int = 20):
    from collections import Counter

    counts = Counter(r.track_id for r in rows)

    return {
        "rows": len(rows),
        "unique_ids": len(counts),
        "short_tracks": sum(1 for v in counts.values() if v < short_len),
        "mean_track_len": (sum(counts.values()) / max(len(counts), 1)),
    }
