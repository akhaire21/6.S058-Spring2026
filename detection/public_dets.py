"""
Public detection loader for MOT-style det/det.txt files.

This is the only detection utility needed for the final project. SoccerNet-style
tracking sequences include precomputed detections in MOT format:

    frame, track_id, x, y, w, h, confidence, ...

We convert xywh boxes to xyxy boxes and expose a small CachedDetections wrapper
used by the SORT/EIoU tracking code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FrameDetections:
    boxes_xyxy: np.ndarray
    scores: np.ndarray

    @property
    def n(self) -> int:
        return self.boxes_xyxy.shape[0]


@dataclass
class CachedDetections:
    sequence_name: str
    boxes: np.ndarray
    scores: np.ndarray
    frames: np.ndarray

    def at(self, frame_idx: int) -> FrameDetections:
        mask = self.frames == frame_idx
        return FrameDetections(
            boxes_xyxy=self.boxes[mask].astype(np.float32),
            scores=self.scores[mask].astype(np.float32),
        )


def load_public_detections(seq_dir: Path) -> CachedDetections:
    """Load public MOT-format detections from one sequence directory."""
    seq_dir = Path(seq_dir)
    det_path = seq_dir / "det" / "det.txt"

    if not det_path.exists():
        raise FileNotFoundError(f"No det/det.txt found at {seq_dir}")

    raw = np.loadtxt(det_path, delimiter=",")

    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    frames = raw[:, 0].astype(np.int32)

    xywh = raw[:, 2:6].astype(np.float32)
    boxes = np.column_stack(
        [
            xywh[:, 0],
            xywh[:, 1],
            xywh[:, 0] + xywh[:, 2],
            xywh[:, 1] + xywh[:, 3],
        ]
    ).astype(np.float32)

    if raw.shape[1] > 6:
        scores = raw[:, 6].astype(np.float32)
    else:
        scores = np.ones(len(raw), dtype=np.float32)

    return CachedDetections(
        sequence_name=seq_dir.name,
        boxes=boxes,
        scores=scores,
        frames=frames,
    )


def load_public_detections_for_split(split_dir: Path) -> dict[str, CachedDetections]:
    """Load public detections for every sequence in a split directory."""
    out = {}

    for seq_dir in sorted(Path(split_dir).iterdir()):
        if not seq_dir.is_dir():
            continue

        try:
            out[seq_dir.name] = load_public_detections(seq_dir)
        except FileNotFoundError:
            print(f"Skipping {seq_dir.name}: no det/det.txt")

    return out
