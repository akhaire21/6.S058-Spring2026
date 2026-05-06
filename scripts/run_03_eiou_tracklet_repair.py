import argparse
import json
import shutil
import sys
from pathlib import Path

from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_io.datasets import load_sequence
from detection.public_dets import load_public_detections
from tracking.eiou_tracker import (
    EIOUSORTConfig,
    StandaloneEIOUSORT,
    run_tracker_on_sequence,
    save_mot_txt,
    summarize_rows,
)
from tracking.tracklet_repair import merge_tracklets_greedy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    out_root = Path(args.out_root)

    if args.overwrite and out_root.exists():
        shutil.rmtree(out_root)

    out_root.mkdir(parents=True, exist_ok=True)

    sort_cfg = EIOUSORTConfig(
        alpha=0.7,
        beta=0.3,
        max_age=30,
        min_hits=3,
        max_match_cost=0.90,
        use_eiou=False,
        eiou_alpha=0.50,
        conf_threshold=0.0,
    )

    eiou_cfg = EIOUSORTConfig(
        alpha=0.7,
        beta=0.3,
        max_age=30,
        min_hits=3,
        max_match_cost=0.90,
        use_eiou=True,
        eiou_alpha=0.50,
        conf_threshold=0.0,
    )

    seq_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])
    all_stats = []

    for seq_dir in tqdm(seq_dirs):
        seq = load_sequence(seq_dir, split="test", dataset="soccernet")
        dets = load_public_detections(seq_dir)

        seq_stat = {"sequence": seq_dir.name}

        # SORT baseline
        sort_tracker = StandaloneEIOUSORT(
            sort_cfg,
            seq.info.img_width,
            seq.info.img_height,
        )
        sort_rows = run_tracker_on_sequence(sort_tracker, seq, dets)
        save_mot_txt(
            sort_rows,
            out_root / "sort" / "data" / f"{seq_dir.name}.txt",
        )
        seq_stat["sort"] = summarize_rows(sort_rows)

        # EIoU-SORT
        eiou_tracker = StandaloneEIOUSORT(
            eiou_cfg,
            seq.info.img_width,
            seq.info.img_height,
        )
        eiou_rows = run_tracker_on_sequence(eiou_tracker, seq, dets)
        save_mot_txt(
            eiou_rows,
            out_root / "eiou_sort" / "data" / f"{seq_dir.name}.txt",
        )
        seq_stat["eiou_sort"] = summarize_rows(eiou_rows)

        # EIoU-SORT + TrackletRepair
        repaired_rows = merge_tracklets_greedy(
            eiou_rows,
            max_gap=20,
            max_center_dist=80.0,
            min_track_len=3,
            tail=5,
        )
        save_mot_txt(
            repaired_rows,
            out_root / "eiou_sort_tracklet_repair" / "data" / f"{seq_dir.name}.txt",
        )
        seq_stat["eiou_sort_tracklet_repair"] = summarize_rows(repaired_rows)

        all_stats.append(seq_stat)

    with open(out_root / "quick_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)

    print("Saved outputs to:", out_root)


if __name__ == "__main__":
    main()
