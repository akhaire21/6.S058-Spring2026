"""
Shared experiment runner used by all experiment scripts.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable, Dict

from tqdm.auto import tqdm

from data_io.datasets import load_sequence
from detection.public_dets import load_public_detections
from tracking.common import save_mot_txt, summarize_rows


VariantFn = Callable[[object, object], tuple[list, dict]]


def run_variants(
    dataset_root: str | Path,
    out_root: str | Path,
    variants: Dict[str, VariantFn],
    overwrite: bool = False,
):
    dataset_root = Path(dataset_root)
    out_root = Path(out_root)

    if overwrite and out_root.exists():
        shutil.rmtree(out_root)

    out_root.mkdir(parents=True, exist_ok=True)

    seq_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])
    all_stats = []

    for seq_dir in tqdm(seq_dirs):
        seq = load_sequence(seq_dir, split="test", dataset="soccernet")
        dets = load_public_detections(seq_dir)

        seq_stat = {"sequence": seq_dir.name}

        for variant_name, variant_fn in variants.items():
            rows, extra = variant_fn(seq, dets)

            save_mot_txt(
                rows,
                out_root / variant_name / "data" / f"{seq_dir.name}.txt",
            )

            stat = summarize_rows(rows)
            stat.update(extra or {})
            seq_stat[variant_name] = stat

        all_stats.append(seq_stat)

    with open(out_root / "quick_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)

    print("Saved outputs to:", out_root)
