"""
Experiment 2.1: Standalone EIoU-SORT.

Tracker without ambiguity gating. When use_eiou=True, it applies
expanded IoU uniformly to all track-detection associations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracking.common import (
    TrackRow,
    TrackState,
    run_tracker_on_sequence,
    save_mot_txt,
    summarize_rows,
)
from tracking.geometry import pairwise_center_distance, pairwise_overlap


@dataclass
class EIOUSORTConfig:
    alpha: float = 0.7
    beta: float = 0.3
    max_age: int = 30
    min_hits: int = 3
    max_match_cost: float = 0.90
    eiou_alpha: float = 0.5
    use_eiou: bool = False
    vel_beta: float = 0.7
    conf_threshold: float = 0.0
    tentative_output: bool = False


class StandaloneEIOUSORT:
    def __init__(self, cfg: EIOUSORTConfig, img_w: int, img_h: int):
        self.cfg = cfg
        self.img_w = int(img_w)
        self.img_h = int(img_h)
        self.img_diag = max(1.0, float((img_w ** 2 + img_h ** 2) ** 0.5))

        self.tracks: List[TrackState] = []
        self.next_id = 1

    def _spawn(self, det_box: np.ndarray, det_score: float):
        self.tracks.append(TrackState(self.next_id, det_box, det_score, self.cfg))
        self.next_id += 1

    def _build_cost(self, pred_boxes: np.ndarray, det_boxes: np.ndarray) -> np.ndarray:
        overlap = pairwise_overlap(
            pred_boxes,
            det_boxes,
            use_eiou=self.cfg.use_eiou,
            eiou_alpha=self.cfg.eiou_alpha,
        )

        center_dist = pairwise_center_distance(pred_boxes, det_boxes) / self.img_diag
        cost = self.cfg.alpha * (1.0 - overlap) + self.cfg.beta * center_dist
        return cost.astype(np.float32)

    def step(self, frame_idx: int, det_xyxy: np.ndarray, det_scores: np.ndarray):
        if det_xyxy is None:
            det_xyxy = np.empty((0, 4), dtype=np.float32)
        if det_scores is None:
            det_scores = np.ones((len(det_xyxy),), dtype=np.float32)

        det_xyxy = np.asarray(det_xyxy, dtype=np.float32)
        det_scores = np.asarray(det_scores, dtype=np.float32)

        keep = det_scores >= float(self.cfg.conf_threshold)
        det_xyxy = det_xyxy[keep]
        det_scores = det_scores[keep]

        for t in self.tracks:
            t.predict()

        pred_boxes = (
            np.stack([t.box for t in self.tracks], axis=0).astype(np.float32)
            if len(self.tracks) > 0
            else np.empty((0, 4), dtype=np.float32)
        )

        matches: List[Tuple[int, int]] = []
        unmatched_dets = set(range(len(det_xyxy)))

        if len(self.tracks) > 0 and len(det_xyxy) > 0:
            cost = self._build_cost(pred_boxes, det_xyxy)
            row_ind, col_ind = linear_sum_assignment(cost)

            for r, c in zip(row_ind.tolist(), col_ind.tolist()):
                if cost[r, c] <= float(self.cfg.max_match_cost):
                    matches.append((r, c))
                    unmatched_dets.discard(c)

        for ti, di in matches:
            self.tracks[ti].update(det_xyxy[di], det_scores[di])

        for di in sorted(unmatched_dets):
            self._spawn(det_xyxy[di], det_scores[di])

        self.tracks = [t for t in self.tracks if t.time_since_update <= int(self.cfg.max_age)]

        rows: List[TrackRow] = []
        for t in self.tracks:
            should_emit = (t.time_since_update == 0) and (t.confirmed or self.cfg.tentative_output)
            if not should_emit:
                continue

            rows.append(
                TrackRow(
                    frame_idx=int(frame_idx),
                    track_id=int(t.id),
                    x1=float(t.box[0]),
                    y1=float(t.box[1]),
                    x2=float(t.box[2]),
                    y2=float(t.box[3]),
                    score=float(t.score),
                )
            )

        return rows
