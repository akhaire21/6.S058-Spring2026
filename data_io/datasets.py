"""
Dataset loaders for MOT-Challenge-format sports tracking data.

Both SportsMOT and SoccerNet-Tracking are in MOT-Challenge layout:
    <root>/<split>/<sequence>/img1/000001.jpg ...
    <root>/<split>/<sequence>/gt/gt.txt
    <root>/<split>/<sequence>/seqinfo.ini

gt.txt rows are: frame, track_id, x, y, w, h, conf, class, visibility

We create a Sequence object so the rest of the pipeline does not care 
which dataset it came from.
"""
from __future__ import annotations

import configparser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass
class SeqInfo:
    name: str
    img_dir: Path
    frame_rate: int
    seq_length: int
    img_width: int
    img_height: int
    img_ext: str = ".jpg"


@dataclass
class Sequence:
    """One video sequence with images, optional GT, and metadata."""
    info: SeqInfo
    gt_path: Path | None
    split: str
    dataset: str  # "sportsmot" or "soccernet"

    # Lazily loaded
    _gt_cache: np.ndarray | None = field(default=None, repr=False)

    def frame_path(self, frame_idx: int) -> Path:
        # MOT-Challenge frames are 1-indexed and zero-padded to 6 digits.
        return self.info.img_dir / f"{frame_idx:06d}{self.info.img_ext}"

    def iter_frames(self) -> Iterator[tuple[int, Path]]:
        for f in range(1, self.info.seq_length + 1):
            yield f, self.frame_path(f)

    def gt(self) -> np.ndarray:
        """Returns GT as float64 array of shape (N, 9):
        frame, track_id, x, y, w, h, conf, class, visibility.
        """
        if self._gt_cache is not None:
            return self._gt_cache
        if self.gt_path is None or not self.gt_path.exists():
            self._gt_cache = np.empty((0, 9), dtype=np.float64)
            return self._gt_cache
        arr = np.loadtxt(self.gt_path, delimiter=",", dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        # Some MOT files only have 6 columns; pad with conf=1, class=1, vis=1.
        if arr.shape[1] < 9:
            pad = np.ones((arr.shape[0], 9 - arr.shape[1]), dtype=np.float64)
            arr = np.concatenate([arr, pad], axis=1)
        self._gt_cache = arr
        return arr

    def gt_at_frame(self, frame_idx: int) -> np.ndarray:
        gt = self.gt()
        return gt[gt[:, 0] == frame_idx]


def _parse_seqinfo(seqinfo_path: Path) -> SeqInfo:
    cp = configparser.ConfigParser()
    cp.read(seqinfo_path)
    s = cp["Sequence"]
    return SeqInfo(
        name=s["name"],
        img_dir=seqinfo_path.parent / s.get("imDir", "img1"),
        frame_rate=int(float(s.get("frameRate", 25))),
        seq_length=int(float(s["seqLength"])),
        img_width=int(float(s["imWidth"])),
        img_height=int(float(s["imHeight"])),
        img_ext=s.get("imExt", ".jpg"),
    )


def load_sequence(seq_dir: Path, split: str, dataset: str) -> Sequence:
    """Load one MOT-Challenge sequence directory."""
    seq_dir = Path(seq_dir)
    info = _parse_seqinfo(seq_dir / "seqinfo.ini")
    gt_path = seq_dir / "gt" / "gt.txt"
    return Sequence(
        info=info,
        gt_path=gt_path if gt_path.exists() else None,
        split=split,
        dataset=dataset,
    )


def load_split(root: str | Path, split: str, dataset: str) -> list[Sequence]:
    """Load every sequence in a split directory."""
    root = Path(root)
    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory missing: {split_dir}")
    seqs = []
    for seq_dir in sorted(split_dir.iterdir()):
        if seq_dir.is_dir() and (seq_dir / "seqinfo.ini").exists():
            seqs.append(load_sequence(seq_dir, split=split, dataset=dataset))
    return seqs


def load_sportsmot_split_subset(
    root: str | Path, split: str, sport: str | None = None
) -> list[Sequence]:
    """Load a SportsMOT split, optionally filtering by sport via splits_txt."""
    root = Path(root)
    seqs = load_split(root, split, dataset="sportsmot")
    if sport is None:
        return seqs
    sport_file = root / "splits_txt" / f"{sport}.txt"
    if not sport_file.exists():
        raise FileNotFoundError(f"No sport file: {sport_file}")
    allowed = set(sport_file.read_text().split())
    return [s for s in seqs if s.info.name in allowed]


def load_soccernet_subset(root: str | Path, split: str = "test") -> list[Sequence]:
    """Load SoccerNet-Tracking sequences. The pip-package layout puts split
    folders directly under root (e.g. soccernet-tracking/test/SNMOT-XXX/...).
    """
    return load_split(root, split, dataset="soccernet")
