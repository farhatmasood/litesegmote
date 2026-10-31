"""Accuracy of the compressed decoder before any gradient update.

Each configuration is built from the released weights, factorised, and scored on the
test archive without optimisation. The resulting curve separates what the
factorisation itself costs from what subsequent fine-tuning recovers.
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
    {"name": "released", "expert_rank": None, "pooler_rank": None, "prune": False},
    {"name": "prune-only", "expert_rank": None, "pooler_rank": None, "prune": True},
    {"name": "r384-768", "expert_rank": 384, "pooler_rank": 768, "prune": True},
    {"name": "r192-384", "expert_rank": 192, "pooler_rank": 384, "prune": True},
    {"name": "r128-256", "expert_rank": 128, "pooler_rank": 256, "prune": True},
    {"name": "r96-192", "expert_rank": 96, "pooler_rank": 192, "prune": True},
    {"name": "r64-128", "expert_rank": 64, "pooler_rank": 128, "prune": True},
    {"name": "r32-64", "expert_rank": 32, "pooler_rank": 64, "prune": True},
    {"name": "r16-32", "expert_rank": 16, "pooler_rank": 32, "prune": True},
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
        record = apply_recipe(
            model, expert_rank=config["expert_rank"], pooler_rank=config["pooler_rank"],
            prune=config["prune"], verbose=False,
        )
        scores = evaluate_both_modes(model, pairs, args.device, args.image_size,
                                     csv_prefix=os.path.join(args.out_dir, f"datafree_{name}"))
        errors = [e for _, e in record["factorised"]]
        row = {
            "name": name,
            "expert_rank": config["expert_rank"],
            "pooler_rank": config["pooler_rank"],
            "trainable_params": count_trainable(model),
            "reduction": record["reduction"],
            "box": scores["box"],
            "prompt_free": scores["prompt_free"],
            "mean_relative_error": float(np.mean(errors)) if errors else 0.0,
        }
        results.append(row)
        print(f"{name:12s} {row['trainable_params']/1e6:6.3f}M "
              f"({100*row['reduction']:4.1f}% smaller)  box {row['box']:.4f}  "
              f"prompt-free {row['prompt_free']:.4f}")
        del model

    with open(os.path.join(args.out_dir, "datafree_sweep.json"), "w") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
