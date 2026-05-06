"""
Experiment 1: Ambiguity-Aware SORT.

This tracker builds a normal SORT-style cost matrix. Rows that cross an
ambiguity threshold receive richer cues:
1) Expanded IoU
2) InteractionPrior based on nearby-track velocity consistency

Experiment 2.2 uses this tracker with use_ambiguity_gate=False and
use_relation_prior=True to test InteractionPrior as a standalone cue.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracking.ambiguity_gate import compute_ambiguity_flags
from tracking.common import (
    TrackRow,
    TrackState,
    run_tracker_on_sequence,
    save_mot_txt,
    summarize_rows,
)
from tracking.geometry import pairwise_center_distance, pairwise_overlap
from tracking.interaction_prior import relation_cost_row


@dataclass
class AAEIOUSORTConfig:
    # Baseline SORT-style association
    alpha: float = 0.7
    beta: float = 0.3
    max_age: int = 30
    min_hits: int = 3
    max_match_cost: float = 0.90
    vel_beta: float = 0.7
    conf_threshold: float = 0.0
    tentative_output: bool = False

    # Ablation switches
    always_eiou: bool = False
    use_ambiguity_gate: bool = True
    use_eiou_on_ambiguous: bool = True
    use_relation_prior: bool = True

    # Ambiguity thresholds
    ambiguity_weak_cost: float = 0.55
    ambiguity_margin: float = 0.08
    ambiguity_radius: float = 70.0
    ambiguity_min_neighbors: int = 2

    # Richer cues
    eiou_alpha: float = 0.50
    lambda_eiou: float = 0.35
    lambda_rel: float = 0.25

    # InteractionPrior
    rel_k: int = 3
    rel_radius: float = 120.0
    rel_norm_px: float = 80.0


class AmbiguityAwareEIOUSORT:
    def __init__(self, cfg: AAEIOUSORTConfig, img_w: int, img_h: int):
        self.cfg = cfg
        self.img_w = int(img_w)
        self.img_h = int(img_h)
        self.img_diag = max(1.0, float((img_w ** 2 + img_h ** 2) ** 0.5))

        self.tracks: List[TrackState] = []
        self.next_id = 1

        self.debug = {
            "frames": 0,
            "total_rows": 0,
            "ambiguous_rows": 0,
            "weak_cost_rows": 0,
            "small_margin_rows": 0,
            "crowded_rows": 0,
            "matches": 0,
            "ambiguous_matches": 0,
            "unambiguous_matches": 0,
            "relation_rows": 0,
        }

    def config_dict(self) -> Dict[str, Any]:
        return asdict(self.cfg)

    def get_debug_summary(self) -> Dict[str, Any]:
        out = dict(self.debug)
        out["ambiguity_rate"] = out["ambiguous_rows"] / max(out["total_rows"], 1)
        out["ambiguous_match_rate"] = out["ambiguous_matches"] / max(out["matches"], 1)
        return out

    def _spawn(self, det_box: np.ndarray, det_score: float):
        self.tracks.append(TrackState(self.next_id, det_box, det_score, self.cfg))
        self.next_id += 1

    def _build_base_cost(self, pred_boxes: np.ndarray, det_boxes: np.ndarray) -> np.ndarray:
        overlap = pairwise_overlap(
            pred_boxes,
            det_boxes,
            use_eiou=self.cfg.always_eiou,
            eiou_alpha=self.cfg.eiou_alpha,
        )

        center_dist = pairwise_center_distance(pred_boxes, det_boxes) / self.img_diag
        cost = self.cfg.alpha * (1.0 - overlap) + self.cfg.beta * center_dist
        return cost.astype(np.float32)

    def _build_final_cost(
        self,
        pred_boxes: np.ndarray,
        det_boxes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        base_cost = self._build_base_cost(pred_boxes, det_boxes)
        n, m = base_cost.shape

        if n == 0 or m == 0:
            return base_cost, np.zeros((n,), dtype=bool)

        ambiguous, reasons = compute_ambiguity_flags(
            base_cost=base_cost,
            pred_boxes=pred_boxes,
            det_boxes=det_boxes,
            weak_cost=self.cfg.ambiguity_weak_cost,
            margin_thresh=self.cfg.ambiguity_margin,
            crowd_radius=self.cfg.ambiguity_radius,
            min_neighbors=self.cfg.ambiguity_min_neighbors,
        )

        self.debug["total_rows"] += int(n)
        self.debug["ambiguous_rows"] += int(np.sum(ambiguous))
        self.debug["weak_cost_rows"] += int(np.sum(reasons["weak"]))
        self.debug["small_margin_rows"] += int(np.sum(reasons["margin"]))
        self.debug["crowded_rows"] += int(np.sum(reasons["crowded"]))

        final_cost = base_cost.copy()

        if self.cfg.use_ambiguity_gate:
            active_rows = ambiguous
        else:
            active_rows = np.ones((n,), dtype=bool)

        if self.cfg.use_eiou_on_ambiguous and not self.cfg.always_eiou:
            eiou_overlap = pairwise_overlap(
                pred_boxes,
                det_boxes,
                use_eiou=True,
                eiou_alpha=self.cfg.eiou_alpha,
            )
            eiou_penalty = 1.0 - eiou_overlap
            final_cost[active_rows, :] += float(self.cfg.lambda_eiou) * eiou_penalty[active_rows, :]

        if self.cfg.use_relation_prior:
            for i in range(n):
                if not active_rows[i]:
                    continue

                rel_row, used = relation_cost_row(
                    tracks=self.tracks,
                    track_idx=i,
                    det_boxes=det_boxes,
                    max_age=self.cfg.max_age,
                    rel_k=self.cfg.rel_k,
                    rel_radius=self.cfg.rel_radius,
                    rel_norm_px=self.cfg.rel_norm_px,
                )

                if used:
                    final_cost[i, :] += float(self.cfg.lambda_rel) * rel_row
                    self.debug["relation_rows"] += 1

        return final_cost.astype(np.float32), ambiguous

    def step(self, frame_idx: int, det_xyxy: np.ndarray, det_scores: np.ndarray):
        self.debug["frames"] += 1

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
        ambiguous = np.zeros((len(self.tracks),), dtype=bool)

        if len(self.tracks) > 0 and len(det_xyxy) > 0:
            cost, ambiguous = self._build_final_cost(pred_boxes, det_xyxy)
            row_ind, col_ind = linear_sum_assignment(cost)

            for r, c in zip(row_ind.tolist(), col_ind.tolist()):
                if cost[r, c] <= float(self.cfg.max_match_cost):
                    matches.append((r, c))
                    unmatched_dets.discard(c)

                    self.debug["matches"] += 1
                    if ambiguous[r]:
                        self.debug["ambiguous_matches"] += 1
                    else:
                        self.debug["unambiguous_matches"] += 1

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
