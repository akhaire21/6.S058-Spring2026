
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
import numpy as np


def _get_attr(row: Any, name: str):
    return getattr(row, name)


def _set_attr(row: Any, name: str, value):
    setattr(row, name, value)


def _box_xyxy(row) -> np.ndarray:
    return np.array([row.x1, row.y1, row.x2, row.y2], dtype=np.float32)


def _center(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float32)


@dataclass
class TrackletSummary:
    track_id: int
    start_frame: int
    end_frame: int
    start_center: np.ndarray
    end_center: np.ndarray
    start_box: np.ndarray
    end_box: np.ndarray
    length: int
    end_vel: np.ndarray


def _group_by_track(rows: List[Any]) -> Dict[int, List[Any]]:
    out: Dict[int, List[Any]] = {}
    for r in rows:
        out.setdefault(r.track_id, []).append(r)
    for k in out:
        out[k].sort(key=lambda x: x.frame_idx)
    return out


def _summarize_track(track_id: int, rows: List[Any], tail: int = 5) -> TrackletSummary:
    boxes = np.stack([_box_xyxy(r) for r in rows], axis=0)
    centers = np.stack([_center(b) for b in boxes], axis=0)
    frames = np.array([r.frame_idx for r in rows], dtype=np.int32)

    if len(rows) >= 2:
        k = min(tail, len(rows) - 1)
        dt = max(1, int(frames[-1] - frames[-k - 1]))
        end_vel = (centers[-1] - centers[-k - 1]) / dt
    else:
        end_vel = np.zeros(2, dtype=np.float32)

    return TrackletSummary(
        track_id=track_id,
        start_frame=int(frames[0]),
        end_frame=int(frames[-1]),
        start_center=centers[0],
        end_center=centers[-1],
        start_box=boxes[0],
        end_box=boxes[-1],
        length=len(rows),
        end_vel=end_vel.astype(np.float32),
    )


def _predict_to_frame(track: TrackletSummary, target_frame: int) -> np.ndarray:
    dt = max(0, target_frame - track.end_frame)
    return track.end_center + dt * track.end_vel


def _merge_score(
    a: TrackletSummary,
    b: TrackletSummary,
    max_gap: int,
    max_center_dist: float,
    size_penalty: float = 0.25,
) -> float:
    # require a before b
    if a.end_frame >= b.start_frame:
        return math.inf

    gap = b.start_frame - a.end_frame
    if gap > max_gap:
        return math.inf

    pred_center = _predict_to_frame(a, b.start_frame)
    d = np.linalg.norm(pred_center - b.start_center)
    if d > max_center_dist:
        return math.inf

    ah = max(1.0, float(a.end_box[3] - a.end_box[1]))
    bh = max(1.0, float(b.start_box[3] - b.start_box[1]))
    size_ratio = abs(ah - bh) / max(ah, bh)

    return (d / max_center_dist) + 0.5 * (gap / max_gap) + size_penalty * size_ratio


def _apply_merge(rows: List[Any], src_id: int, dst_id: int):
    for r in rows:
        if r.track_id == src_id:
            r.track_id = dst_id


def merge_tracklets_greedy(
    rows: List[Any],
    max_gap: int = 20,
    max_center_dist: float = 80.0,
    min_track_len: int = 3,
    tail: int = 5,
    max_iters: int = 1000,
):
    rows = copy.deepcopy(rows)

    for _ in range(max_iters):
        tracks = _group_by_track(rows)
        summaries = {tid: _summarize_track(tid, trows, tail=tail) for tid, trows in tracks.items()}

        best_pair = None
        best_score = math.inf
        tids = sorted(summaries.keys())

        for i in range(len(tids)):
            for j in range(len(tids)):
                if i == j:
                    continue
                a = summaries[tids[i]]
                b = summaries[tids[j]]

                # bias toward merging short fragments, but allow any valid merge
                score = _merge_score(a, b, max_gap=max_gap, max_center_dist=max_center_dist)
                if a.length < min_track_len or b.length < min_track_len:
                    score *= 0.9

                if score < best_score:
                    best_score = score
                    best_pair = (a.track_id, b.track_id)

        if best_pair is None or not np.isfinite(best_score):
            break

        src_id, dst_id = best_pair
        s_src = summaries[src_id]
        s_dst = summaries[dst_id]

        if s_src.start_frame <= s_dst.start_frame:
            keep_id, drop_id = src_id, dst_id
        else:
            keep_id, drop_id = dst_id, src_id

        _apply_merge(rows, drop_id, keep_id)

    return rows
