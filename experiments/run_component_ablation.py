"""Component ablation of the compression recipe, before any gradient update.

Each block of the recipe is applied on its own to the released weights, so the cost
of the full configuration can be attributed: pruning alone, the expert heads alone at
rank 64, the prompt pooler alone at rank 128, both (LiteSegMoTE), and both plus the
optional factorisation of the encoder feature adapter at rank 64, which the reported
configuration does not use. Scoring follows run_datafree_sweep.py exactly.

    python experiments/run_component_ablation.py --segmote_root third_party/segmote \
      --sam_checkpoint checkpoints/sam_b.pth --segmote_checkpoint checkpoints/segmote.pth \
      --test_dir datasets/isic2016_test --out_dir results/ablation
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segmote_lite import (apply_recipe, build_segmote, count_trainable,
                          evaluate_both_modes, freeze_backbone, list_pairs)

CONFIGURATIONS = [
    {"name": "prune-only", "expert_rank": None, "pooler_rank": None, "vitfeat_rank": None},
    {"name": "experts-r64", "expert_rank": 64, "pooler_rank": None, "vitfeat_rank": None},
    {"name": "pooler-r128", "expert_rank": None, "pooler_rank": 128, "vitfeat_rank": None},
    {"name": "litesegmote", "expert_rank": 64, "pooler_rank": 128, "vitfeat_rank": None},
    {"name": "litesegmote+adapter-r64", "expert_rank": 64, "pooler_rank": 128,
     "vitfeat_rank": 64},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmote_root", required=True)
    parser.add_argument("--sam_checkpoint", required=True)
    parser.add_argument("--segmote_checkpoint", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pairs = list_pairs(args.test_dir)
    results = []
    for config in CONFIGURATIONS:
        name = config["name"]
        model = build_segmote(args.segmote_root, args.sam_checkpoint,
                              args.segmote_checkpoint, args.device, args.image_size)
        freeze_backbone(model)
        record = apply_recipe(model, expert_rank=config["expert_rank"],
                              pooler_rank=config["pooler_rank"],
                              vitfeat_rank=config["vitfeat_rank"], prune=True, verbose=False)
        scores = evaluate_both_modes(model, pairs, args.device, args.image_size,
                                     csv_prefix=os.path.join(args.out_dir, f"ablation_{name}"))
        errors = [e for _, e in record["factorised"]]
        row = {**config,
               "trainable_params": count_trainable(model),
               "reduction": record["reduction"],
               "box": scores["box"],
               "prompt_free": scores["prompt_free"],
               "mean_relative_error": float(np.mean(errors)) if errors else 0.0}
        results.append(row)
        print(f"{name:26s} {row['trainable_params']/1e6:6.3f}M "
              f"({100*row['reduction']:4.1f}% smaller)  box {row['box']:.4f}  "
              f"prompt-free {row['prompt_free']:.4f}", flush=True)
        del model

    with open(os.path.join(args.out_dir, "component_ablation.json"), "w") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
