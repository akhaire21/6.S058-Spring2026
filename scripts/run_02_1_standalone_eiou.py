"""
Experiment 2.1: Standalone EIoU.

Runs:
1) SORT
2) EIoU-SORT without ambiguity gating
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.common_runner import run_variants
from tracking.common import run_tracker_on_sequence
from tracking.eiou_tracker import EIOUSORTConfig, StandaloneEIOUSORT


def make_runner(cfg: EIOUSORTConfig):
    def _run(seq, dets):
        tracker = StandaloneEIOUSORT(cfg, seq.info.img_width, seq.info.img_height)
        rows = run_tracker_on_sequence(tracker, seq, dets)
        return rows, {}

    return _run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    base = dict(
        alpha=0.7,
        beta=0.3,
        max_age=30,
        min_hits=3,
        max_match_cost=0.90,
        conf_threshold=0.0,
    )

    variants = {
        "sort": make_runner(
            EIOUSORTConfig(**base, use_eiou=False, eiou_alpha=0.50)
        ),
        "eiou_sort": make_runner(
            EIOUSORTConfig(**base, use_eiou=True, eiou_alpha=0.50)
        ),
    }

    run_variants(args.dataset_root, args.out_root, variants, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
