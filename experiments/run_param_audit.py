"""Per-module accounting of the trainable parameters in the SegMoTE mask decoder.

The upstream builder decides trainability by comparison against the initialisation
checkpoint: a decoder tensor already present in ``sam_b.pth`` is frozen, anything
else belongs to the added mixture-of-experts and prompt tokenisation machinery and
is optimised. Passing that checkpoint here therefore reproduces the exact set of
parameters that training updates.
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segmote_lite import build_segmote

POOLER_PREFIX = "img_prompt_pool"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmote_root", required=True)
    parser.add_argument("--sam_checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    model = build_segmote(args.segmote_root, args.sam_checkpoint,
                          segmote_checkpoint=None, device=args.device)
    for module in (model.image_encoder, model.prompt_encoder, model.text_model, model.text_out_dim):
        for p in module.parameters():
            p.requires_grad = False

    decoder = model.mask_decoder
    by_module, tensors = defaultdict(int), []
    pooler = router = 0
    for name, p in decoder.named_parameters():
        if not p.requires_grad:
            continue
        by_module[name.split(".")[0]] += p.numel()
        tensors.append((name, tuple(p.shape), p.numel()))
        if name.startswith(POOLER_PREFIX):
            pooler += p.numel()
        else:
            router += p.numel()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "param_by_module.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["module", "trainable_params"])
        for name, count in sorted(by_module.items(), key=lambda kv: -kv[1]):
            writer.writerow([name, count])
    with open(os.path.join(args.out_dir, "param_by_tensor.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tensor", "shape", "params"])
        for name, shape, count in sorted(tensors, key=lambda r: -r[2]):
            writer.writerow([name, shape, count])
    summary = {
        "total_params": total,
        "trainable_params": trainable,
        "decoder_total_params": sum(p.numel() for p in decoder.parameters()),
        "prompt_tokenisation_params": pooler,
        "mixture_of_experts_params": router,
        "by_module": dict(sorted(by_module.items(), key=lambda kv: -kv[1])),
    }
    with open(os.path.join(args.out_dir, "param_audit.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"total {total:,} | trainable {trainable:,}")
    print(f"  prompt tokenisation {pooler:,} | mixture of experts {router:,}")
    for name, count in sorted(by_module.items(), key=lambda kv: -kv[1]):
        print(f"  {name:32s} {count:>12,}")


if __name__ == "__main__":
    main()
