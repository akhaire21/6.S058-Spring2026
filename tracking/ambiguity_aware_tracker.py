
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class TrackRow:
    frame_idx: int
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0


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

    # Ambiguity triggers
    ambiguity_weak_cost: float = 0.55
    ambiguity_margin: float = 0.08
    ambiguity_radius: float = 70.0
    ambiguity_min_neighbors: int = 2

    # Ambiguity-conditioned cues
    eiou_alpha: float = 0.50
    lambda_eiou: float = 0.35
    lambda_rel: float = 0.25

    # Local interaction-aware prior
    rel_k: int = 3
    rel_radius: float = 120.0
    rel_norm_px: float = 80.0


def _area_xyxy(box: np.ndarray) -> float:
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

    union = _area_xyxy(a) + _area_xyxy(b) - inter + 1e-9
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
        [cx - 0.5 * new_w, cy - 0.5 * new_h, cx + 0.5 * new_w, cy + 0.5 * new_h],
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


class _Track:
    def __init__(self, track_id: int, init_box: np.ndarray, score: float, cfg: AAEIOUSORTConfig):
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
            [(self.box[0] + self.box[2]) * 0.5, (self.box[1] + self.box[3]) * 0.5],
            dtype=np.float32,
        )

    def predict(self):
        self.box[[0, 2]] += self.vel[0]
        self.box[[1, 3]] += self.vel[1]
        self.age += 1
        self.time_since_update += 1

    def update(self, det_box: np.ndarray, det_score: float):
        det_center = np.array(
            [(det_box[0] + det_box[2]) * 0.5, (det_box[1] + det_box[3]) * 0.5],
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


class AmbiguityAwareEIOUSORT:
    def __init__(self, cfg: AAEIOUSORTConfig, img_w: int, img_h: int):
        self.cfg = cfg
        self.img_w = int(img_w)
        self.img_h = int(img_h)
        self.img_diag = max(1.0, float((img_w ** 2 + img_h ** 2) ** 0.5))
        self.tracks: List[_Track] = []
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
        self.tracks.append(_Track(self.next_id, det_box, det_score, self.cfg))
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

    def _ambiguity_flags(
        self,
        base_cost: np.ndarray,
        pred_boxes: np.ndarray,
        det_boxes: np.ndarray,
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

            weak[i] = best > float(self.cfg.ambiguity_weak_cost)
            margin[i] = (second - best) < float(self.cfg.ambiguity_margin)

            dists = np.linalg.norm(det_centers - pred_centers[i], axis=1)
            crowded[i] = int(np.sum(dists <= float(self.cfg.ambiguity_radius))) >= int(self.cfg.ambiguity_min_neighbors)

            ambiguous[i] = weak[i] or margin[i] or crowded[i]

        return ambiguous, {"weak": weak, "margin": margin, "crowded": crowded}

    def _neighbor_mean_velocity(self, track_idx: int) -> Tuple[np.ndarray, bool]:
        if len(self.tracks) <= 1:
            return np.zeros(2, dtype=np.float32), False

        ci = self.tracks[track_idx].center()
        candidates = []

        for q, t in enumerate(self.tracks):
            if q == track_idx:
                continue
            if t.time_since_update > self.cfg.max_age:
                continue

            d = float(np.linalg.norm(t.center() - ci))
            if d <= float(self.cfg.rel_radius):
                candidates.append((d, q))

        if not candidates:
            return np.zeros(2, dtype=np.float32), False

        candidates.sort(key=lambda x: x[0])
        keep = [q for _, q in candidates[: int(self.cfg.rel_k)]]

        if len(keep) == 0:
            return np.zeros(2, dtype=np.float32), False

        mean_vel = np.mean([self.tracks[q].vel for q in keep], axis=0).astype(np.float32)
        return mean_vel, True

    def _relation_cost_row(self, track_idx: int, det_boxes: np.ndarray) -> Tuple[np.ndarray, bool]:
        mean_vel, has_neighbors = self._neighbor_mean_velocity(track_idx)
        if not has_neighbors or len(det_boxes) == 0:
            return np.zeros((len(det_boxes),), dtype=np.float32), False

        det_centers = centers_xyxy(det_boxes)
        last = self.tracks[track_idx].last_obs_center.astype(np.float32)
        implied_vel = det_centers - last[None, :]
        rel = np.linalg.norm(implied_vel - mean_vel[None, :], axis=1)
        rel = rel / max(1.0, float(self.cfg.rel_norm_px))
        return rel.astype(np.float32), True

    def _build_final_cost(
        self,
        pred_boxes: np.ndarray,
        det_boxes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        base_cost = self._build_base_cost(pred_boxes, det_boxes)
        n, m = base_cost.shape

        if n == 0 or m == 0:
            return base_cost, np.zeros((n,), dtype=bool)

        ambiguous, reasons = self._ambiguity_flags(base_cost, pred_boxes, det_boxes)

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
                rel_row, used = self._relation_cost_row(i, det_boxes)
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
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_dets = set(range(len(det_xyxy)))
        ambiguous = np.zeros((len(self.tracks),), dtype=bool)

        if len(self.tracks) > 0 and len(det_xyxy) > 0:
            cost, ambiguous = self._build_final_cost(pred_boxes, det_xyxy)
            row_ind, col_ind = linear_sum_assignment(cost)

            for r, c in zip(row_ind.tolist(), col_ind.tolist()):
                if cost[r, c] <= float(self.cfg.max_match_cost):
                    matches.append((r, c))
                    unmatched_tracks.discard(r)
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

        survivors = []
        for t in self.tracks:
            if t.time_since_update <= int(self.cfg.max_age):
                survivors.append(t)
        self.tracks = survivors

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


def _extract_det_arrays(det_obj, frame_idx: int):
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


def run_tracker_on_sequence(tracker: AmbiguityAwareEIOUSORT, sequence, detections):
    rows: List[TrackRow] = []
    seq_len = int(sequence.info.seq_length)

    for frame_idx in range(1, seq_len + 1):
        det_boxes, det_scores = _extract_det_arrays(detections, frame_idx)
        rows.extend(tracker.step(frame_idx, det_boxes, det_scores))

    return rows


def save_mot_txt(rows: List[TrackRow], out_path):
    from pathlib import Path

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for r in rows:
            w = r.x2 - r.x1
            h = r.y2 - r.y1
            f.write(
                f"{r.frame_idx},{r.track_id},{r.x1:.2f},{r.y1:.2f},{w:.2f},{h:.2f},{r.score:.4f},-1,-1,-1\n"
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
