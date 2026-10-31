"""Score fine-tuned decoder checkpoints on the test set in both interaction modes.

Every checkpoint is loaded into a freshly built model (compressed with the
LiteSegMoTE recipe when ``--compressed`` is given) and scored exactly as in
run_finetune.py, so a stored checkpoint can be checked against its result files.

    python experiments/evaluate_checkpoint.py --segmote_root third_party/segmote \
      --sam_checkpoint checkpoints/sam_b.pth --segmote_checkpoint checkpoints/segmote.pth \
      --test_dir datasets/isic2016_test --compressed \
      --checkpoints checkpoints/segmote_lite_seed0.pth \
      --out_path results/finetune/check_lite_seed0.json
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segmote_lite import (apply_recipe, build_segmote, evaluate_both_modes, freeze_backbone,
                          list_pairs)

LITE = dict(expert_rank=64, pooler_rank=128, prune=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmote_root", required=True)
    parser.add_argument("--sam_checkpoint", required=True)
    parser.add_argument("--segmote_checkpoint", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--compressed", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--out_path", required=True)
    args = parser.parse_args()

    pairs = list_pairs(args.test_dir)
    rows = []
    for path in args.checkpoints:
        model = build_segmote(args.segmote_root, args.sam_checkpoint,
                              args.segmote_checkpoint, args.device, args.image_size)
        freeze_backbone(model)
        if args.compressed:
            apply_recipe(model, verbose=False, **LITE)
        blob = torch.load(path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(blob["model_state_dict"], strict=False)
        scores = evaluate_both_modes(model, pairs, args.device, args.image_size)
        meta = {k: v for k, v in blob.items()
                if k != "model_state_dict" and isinstance(v, (int, float, str))}
        rows.append({"checkpoint": os.path.basename(path), **meta,
                     "box": scores["box"], "prompt_free": scores["prompt_free"],
                     "missing_keys": len(missing), "unexpected_keys": len(unexpected)})
        print(rows[-1], flush=True)
        del model
    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    with open(args.out_path, "w") as handle:
        json.dump(rows, handle, indent=2)


if __name__ == "__main__":
    main()
