# 6.S058 Spring 2026: Sports MOT Association Study

This repository contains our 6.S058 computer vision final project on sports multi-object tracking (MOT). We study when richer association cues help preserve player identities in SoccerNet-style sports tracking.

## Project question

We began with the hypothesis that richer association cues should only be activated when SORT-style matching is ambiguous. In sports videos, player crossings, occlusion, and dense interactions often create uncertain track-detection assignments. We tested whether an ambiguity-aware tracker could improve identity preservation by applying stronger cues only in these difficult cases.


### Experiment 1: Ambiguity-Aware Association

We tested an ambiguity-gated tracker that first computes a SORT-style association cost matrix, then activates richer cues only when a row appears ambiguous.

Cues tested:

- Expanded IoU: bounding boxes are expanded before IoU is computed.
- InteractionPrior: a local motion prior based on nearby player velocities.

Result: ambiguity-gated variants did not outperform SORT. Conservative gating helped relative to the default gated version, but the best ambiguity-aware model still underperformed the SORT baseline.

### Experiment 2: Standalone Cue Analysis

To understand whether the failure came from the gate or from the cues themselves, we tested the cues without ambiguity gating.

Result:

- Expanded IoU helped.
- InteractionPrior consistently hurt performance, both with SORT and with EIoU-SORT.

This suggests that mean-neighbor velocity is not a reliable association cue for soccer MOT, since nearby players often move in different tactical directions.

### Experiment 3: EIoU + TrackletRepair

Based on the cue analysis, we kept the useful cue and removed the harmful one. The final method applies EIoU uniformly and then uses a geometry-only offline TrackletRepair step to reconnect short identity fragments.

Result: EIoU-SORT + TrackletRepair improved over SORT on the full SoccerNet evaluation.


## Repository Layout

    data_io/        Dataset-loading utilities
    detection/      Public detection-loading utilities
    tracking/       Tracker implementations and shared tracking components
    scripts/        Reproducible experiment scripts
    results/        Final result tables and figures
    report/         Report tables and figures

## Main Code Files

    tracking/geometry.py                  Shared geometry related utilities
    tracking/common.py                    Shared tracking utilities
    tracking/ambiguity_gate.py            Ambiguity-gating logic
    tracking/interaction_prior.py         Local motion prior / InteractionPrior
    tracking/eiou_tracker.py              SORT and standalone EIoU-SORT
    tracking/ambiguity_aware_tracker.py   Ambiguity-aware tracker
    tracking/tracklet_repair.py           Geometry-only TrackletRepair

## Experiment Scripts

    scripts/run_01_ambiguity_gating.py
    scripts/run_02_1_standalone_eiou.py
    scripts/run_02_2_standalone_interaction_prior.py
    scripts/run_03_eiou_tracklet_repair.py

## Dataset

Raw SoccerNet data is not included in this repository.

Expected dataset layout:

    data/soccernet-tracking/tracking/test/
      SNMOT-116/
        det/det.txt
        gt/gt.txt
        seqinfo.ini

## Example Run

Run the final EIoU + TrackletRepair experiment:

    python scripts/run_03_eiou_tracklet_repair.py \
      --dataset-root /path/to/soccernet-tracking/tracking/test \
      --out-root results/runs/eiou_tracklet_repair \
      --overwrite

