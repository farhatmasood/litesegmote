"""Singular value spectra of the decoder weights targeted by the compression.

The spectra quantify how much of each weight matrix carries variance, and how much
spectral energy a given rank retains. They explain why a factorisation with a large
elementwise error can leave segmentation quality untouched: the discarded directions
hold a small share of the energy that the layer actually uses.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segmote_lite import build_segmote
from segmote_lite.lowrank import spectrum

RANKS = [16, 32, 64, 96, 128, 192, 256, 384, 512, 768]


def collect(decoder):
    """Name to weight tensor for every layer the recipe factorises."""
    targets = {}
    for index, head in enumerate(decoder.expert_output_mlps):
        for layer in (0, 1):
            targets[f"expert_output_mlps.{index}.layers.{layer}"] = head.layers[layer].weight
    for index in (0, 2):
        targets[f"img_prompt_pool.mlp.{index}"] = decoder.img_prompt_pool.mlp[index].weight
    targets["compress_vit_feat.0"] = decoder.compress_vit_feat[0].weight
    return targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmote_root", required=True)
    parser.add_argument("--sam_checkpoint", required=True)
    parser.add_argument("--segmote_checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    model = build_segmote(args.segmote_root, args.sam_checkpoint,
                          args.segmote_checkpoint, args.device)
    targets = collect(model.mask_decoder)

    os.makedirs(args.out_dir, exist_ok=True)
    values, retention = {}, {}
    for name, weight in targets.items():
        singular, energy = spectrum(weight)
        values[name] = singular.cpu().numpy().tolist()
        retention[name] = {
            str(r): float(energy[min(r, len(energy)) - 1])
            for r in RANKS if r <= len(energy)
        }
        head = ", ".join(f"r={r}: {retention[name][str(r)]:.3f}"
                         for r in (64, 128, 256) if str(r) in retention[name])
        print(f"{name:44s} {head}")

    np.savez_compressed(os.path.join(args.out_dir, "singular_values.npz"),
                        **{k: np.asarray(v) for k, v in values.items()})
    with open(os.path.join(args.out_dir, "energy_retention.json"), "w") as handle:
        json.dump(retention, handle, indent=2)

    grouped = {
        "expert_heads": [k for k in targets if k.startswith("expert_output_mlps")],
        "prompt_pooler": [k for k in targets if k.startswith("img_prompt_pool")],
    }
    for group, names in grouped.items():
        rank = 64 if group == "expert_heads" else 128
        share = np.mean([retention[n][str(rank)] for n in names])
        print(f"[{group}] mean energy retained at rank {rank}: {share:.4f}")


if __name__ == "__main__":
    main()
