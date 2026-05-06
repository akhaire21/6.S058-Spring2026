"""
Loader for MOT-Challenge public detections (det/det.txt).

This bypasses YOLO entirely. SoccerNet, MOT17, MOT20, etc. all ship pre-computed
detections in this format:
    frame, track_id (always -1), x, y, w, h, conf, ...

We convert to the same CachedDetections object the rest of the pipeline expects,
so SORT and AGW-SORT consume them identically to YOLO outputs.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from detection.yolo_cache import CachedDetections, FrameDetections


def load_public_detections(seq_dir: Path) -> CachedDetections:
    """Load det.txt for one sequence, return CachedDetections."""
    det_path = Path(seq_dir) / "det" / "det.txt"
    if not det_path.exists():
        raise FileNotFoundError(f"No det/det.txt at {seq_dir}")
    
    raw = np.loadtxt(det_path, delimiter=",")
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    
    frames = raw[:, 0].astype(np.int32)
    # Convert xywh -> xyxy
    xywh = raw[:, 2:6].astype(np.float32)
    boxes = np.column_stack([
        xywh[:, 0],
        xywh[:, 1],
        xywh[:, 0] + xywh[:, 2],
        xywh[:, 1] + xywh[:, 3],
    ]).astype(np.float32)
    scores = raw[:, 6].astype(np.float32) if raw.shape[1] > 6 else np.ones(len(raw), dtype=np.float32)
    
    return CachedDetections(
        sequence_name=Path(seq_dir).name,
        boxes=boxes,
        scores=scores,
        frames=frames,
    )


def load_public_detections_for_split(split_dir: Path) -> dict[str, CachedDetections]:
    """Load det.txt for every sequence in a split. Returns {seq_name: CachedDetections}."""
    out = {}
    for seq_dir in sorted(Path(split_dir).iterdir()):
        if not seq_dir.is_dir():
            continue
        try:
            out[seq_dir.name] = load_public_detections(seq_dir)
        except FileNotFoundError:
            print(f"  Skipping {seq_dir.name} — no det.txt")
    return out
