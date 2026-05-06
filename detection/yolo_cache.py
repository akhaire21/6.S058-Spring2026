"""
YOLOv8 detection caching.

Run once per dataset+split. Caches per-frame person detections to:
    <cache_root>/<dataset>/<split>/<sequence>.npz

with arrays:
    boxes:  (N, 4)   float32, xyxy in image pixels
    scores: (N,)     float32
    frames: (N,)     int32, frame index (1-indexed)

This format is compact, fast to load, and reusable by every tracker variant.
The cache check-skips per sequence so a disconnect mid-cache is recoverable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data_io.datasets import Sequence


@dataclass
class FrameDetections:
    boxes_xyxy: np.ndarray  # (N, 4) float32
    scores: np.ndarray      # (N,)   float32

    @property
    def n(self) -> int:
        return self.boxes_xyxy.shape[0]


@dataclass
class CachedDetections:
    """In-memory wrapper around a cached .npz."""
    sequence_name: str
    boxes: np.ndarray   # (M, 4)
    scores: np.ndarray  # (M,)
    frames: np.ndarray  # (M,)

    def at(self, frame_idx: int) -> FrameDetections:
        mask = self.frames == frame_idx
        return FrameDetections(
            boxes_xyxy=self.boxes[mask].astype(np.float32),
            scores=self.scores[mask].astype(np.float32),
        )

    @classmethod
    def load(cls, npz_path: Path) -> "CachedDetections":
        z = np.load(npz_path)
        return cls(
            sequence_name=npz_path.stem,
            boxes=z["boxes"],
            scores=z["scores"],
            frames=z["frames"],
        )


def cache_path_for(cache_root: Path, sequence: Sequence) -> Path:
    return Path(cache_root) / sequence.dataset / sequence.split / f"{sequence.info.name}.npz"


def cache_yolov8_detections(
    sequence: Sequence,
    cache_root: str | Path,
    weights: str = "yolov8m.pt",
    conf: float = 0.25,
    iou: float = 0.7,
    person_class_id: int = 0,
    device: str = "cuda",
    imgsz: int = 1280,
    skip_if_exists: bool = True,
    verbose: bool = True,
) -> Path:
    """Detect persons frame-by-frame and cache to a .npz. Returns the cache path."""
    out = cache_path_for(cache_root, sequence)
    out.parent.mkdir(parents=True, exist_ok=True)
    if skip_if_exists and out.exists():
        if verbose:
            print(f"[cache] skip (exists): {out}")
        return out

    # Lazy import so this module is importable without ultralytics installed.
    from ultralytics import YOLO

    model = YOLO(weights)
    boxes_all, scores_all, frames_all = [], [], []
    for f_idx, img_path in sequence.iter_frames():
        results = model.predict(
            source=str(img_path),
            conf=conf,
            iou=iou,
            classes=[person_class_id],
            device=device,
            imgsz=imgsz,
            verbose=False,
        )
        if not results:
            continue
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        conf_arr = r.boxes.conf.detach().cpu().numpy().astype(np.float32)
        boxes_all.append(xyxy)
        scores_all.append(conf_arr)
        frames_all.append(np.full(len(xyxy), f_idx, dtype=np.int32))

    boxes = np.concatenate(boxes_all, axis=0) if boxes_all else np.empty((0, 4), np.float32)
    scores = np.concatenate(scores_all, axis=0) if scores_all else np.empty((0,), np.float32)
    frames = np.concatenate(frames_all, axis=0) if frames_all else np.empty((0,), np.int32)

    np.savez_compressed(out, boxes=boxes, scores=scores, frames=frames)
    if verbose:
        print(f"[cache] wrote {out}  ({len(boxes)} detections across {sequence.info.seq_length} frames)")
    return out


def cache_split(
    sequences: list[Sequence],
    cache_root: str | Path,
    **kwargs,
) -> list[Path]:
    paths = []
    for i, seq in enumerate(sequences):
        print(f"[cache] ({i+1}/{len(sequences)}) {seq.info.name}")
        paths.append(cache_yolov8_detections(seq, cache_root, **kwargs))
    return paths
