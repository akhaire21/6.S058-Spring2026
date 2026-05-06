"""
Experiment 2.2: Standalone InteractionPrior.

Runs:
1) SORT
2) SORT + InteractionPrior at several uniform weights
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
        always_eiou=False,
        use_ambiguity_gate=False,
        use_eiou_on_ambiguous=False,
    )

    variants = {
        "sort": make_runner(
            AAEIOUSORTConfig(**base, use_relation_prior=False, lambda_rel=0.0)
        ),
        "sort_interaction_prior_0p10": make_runner(
            AAEIOUSORTConfig(**base, use_relation_prior=True, lambda_rel=0.10)
        ),
        "sort_interaction_prior_0p25": make_runner(
            AAEIOUSORTConfig(**base, use_relation_prior=True, lambda_rel=0.25)
        ),
        "sort_interaction_prior_0p40": make_runner(
            AAEIOUSORTConfig(**base, use_relation_prior=True, lambda_rel=0.40)
        ),
    }

    run_variants(args.dataset_root, args.out_root, variants, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
