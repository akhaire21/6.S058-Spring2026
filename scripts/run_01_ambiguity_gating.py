"""
Experiment 1: Ambiguity-Aware Association.

Runs:
1) SORT baseline using the ambiguity-aware framework with all extra cues disabled
2) Ambiguity-Aware SORT with gated EIoU + InteractionPrior
3) Tuned Ambiguity-Aware SORT
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.common_runner import run_variants
from tracking.ambiguity_aware_tracker import AAEIOUSORTConfig, AmbiguityAwareEIOUSORT
from tracking.common import run_tracker_on_sequence


def make_runner(cfg: AAEIOUSORTConfig):
    def _run(seq, dets):
        tracker = AmbiguityAwareEIOUSORT(cfg, seq.info.img_width, seq.info.img_height)
        rows = run_tracker_on_sequence(tracker, seq, dets)
        return rows, {"debug": tracker.get_debug_summary()}

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
            AAEIOUSORTConfig(
                **base,
                always_eiou=False,
                use_ambiguity_gate=False,
                use_eiou_on_ambiguous=False,
                use_relation_prior=False,
            )
        ),
        "ambiguity_aware_default": make_runner(
            AAEIOUSORTConfig(
                **base,
                always_eiou=False,
                use_ambiguity_gate=True,
                use_eiou_on_ambiguous=True,
                use_relation_prior=True,
                ambiguity_weak_cost=0.55,
                ambiguity_margin=0.08,
                eiou_alpha=0.50,
                lambda_eiou=0.35,
                lambda_rel=0.25,
            )
        ),
        "ambiguity_aware_tuned": make_runner(
            AAEIOUSORTConfig(
                **base,
                always_eiou=False,
                use_ambiguity_gate=True,
                use_eiou_on_ambiguous=True,
                use_relation_prior=False,
                ambiguity_weak_cost=0.75,
                ambiguity_margin=0.08,
                eiou_alpha=0.50,
                lambda_eiou=0.35,
                lambda_rel=0.00,
            )
        ),
    }

    run_variants(args.dataset_root, args.out_root, variants, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
