"""Excitation-weighted spectral energy of the projections targeted by compression.

The plain energy criterion E(r) weights every singular direction by its singular value
alone, which is equivalent to assuming the layer receives isotropic input. A decoder
layer does not: it sees only the activations the frozen encoder produces on real
images. Writing W = U S V^T, the response to an input x decomposes as

    W x = sum_i sigma_i (v_i^T x) u_i,

so the mean squared contribution of direction i is sigma_i^2 a_i with
a_i = E[(v_i^T x)^2] taken over the actual input distribution. This script measures
a_i by hooking each targeted layer during inference, which turns the claim that
discarded directions are weakly excited into a measurement.

Output is the excitation-weighted retained energy at each rank, directly comparable
with the isotropic E(r) reported by run_spectra.py.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segmote_lite import build_segmote, freeze_backbone, list_pairs, load_mask, preprocess_image
from segmote_lite.evaluate import box_from_mask

RANKS = [16, 32, 64, 96, 128, 192, 256, 384, 512, 768]


def target_layers(decoder):
    """Name to module for every layer the compression recipe factorises."""
    layers = {}
    for index, head in enumerate(decoder.expert_output_mlps):
        for position in (0, 1):
            layers[f"expert_output_mlps.{index}.layers.{position}"] = head.layers[position]
    for position in (0, 2):
        layers[f"img_prompt_pool.mlp.{position}"] = decoder.img_prompt_pool.mlp[position]
    return layers


class ExcitationProbe:
    """Accumulates sum of (v_i^T x)^2 over inputs seen by one layer."""

    def __init__(self, module):
        weight = module.weight.detach().float()
        # Right singular vectors span the input space of the layer.
        _, singular, vh = torch.linalg.svd(weight, full_matrices=False)
        self.singular = singular.cpu()
        self.vh = vh                       # (k, in_features), kept on the layer's device
        self.total = torch.zeros(vh.shape[0], dtype=torch.float64, device=vh.device)
        self.count = 0
        self.handle = module.register_forward_hook(self.__call__)

    def __call__(self, module, inputs, output):
        x = inputs[0].detach()
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])
        projected = x.float() @ self.vh.t()          # (N, k)
        self.total += (projected ** 2).sum(dim=0).double()
        self.count += projected.shape[0]

    def finish(self):
        self.handle.remove()
        excitation = (self.total / max(self.count, 1)).cpu()
        weighted = self.singular.double() ** 2 * excitation
        isotropic = self.singular.double() ** 2
        return {
            "n_inputs": self.count,
            "weighted_energy": (torch.cumsum(weighted, 0) / weighted.sum()).numpy(),
            "isotropic_energy": (torch.cumsum(isotropic, 0) / isotropic.sum()).numpy(),
        }


@torch.no_grad()
def drive(model, pairs, device, image_size, limit):
    """Run inference in both interaction modes so every targeted layer is exercised."""
    for image_path, mask_path in pairs[:limit]:
        image = preprocess_image(image_path, image_size).to(device)
        _, resized = load_mask(mask_path, image_size)
        if resized.sum() < 1:
            continue
        embedding, intermediate = model.image_forward(image)
        for prompt_free in (False, True):
            prompt = {} if prompt_free else {"bboxes": box_from_mask(resized).to(device)}
            model.forward_decoder(embedding, prompt, interm_embeddings=intermediate,
                                  test_mode=True, if_prompt=True if prompt_free else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmote_root", required=True)
    parser.add_argument("--sam_checkpoint", required=True)
    parser.add_argument("--segmote_checkpoint", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    model = build_segmote(args.segmote_root, args.sam_checkpoint,
                          args.segmote_checkpoint, args.device, args.image_size)
    freeze_backbone(model)
    model.eval()

    layers = target_layers(model.mask_decoder)
    probes = {name: ExcitationProbe(module) for name, module in layers.items()}

    pairs = list_pairs(args.test_dir)
    drive(model, pairs, args.device, args.image_size, args.limit)

    results, summary = {}, {}
    for name, probe in probes.items():
        record = probe.finish()
        results[name] = record
        summary[name] = {
            "n_inputs": record["n_inputs"],
            "weighted": {str(r): float(record["weighted_energy"][min(r, len(record["weighted_energy"])) - 1])
                         for r in RANKS if r <= len(record["weighted_energy"])},
            "isotropic": {str(r): float(record["isotropic_energy"][min(r, len(record["isotropic_energy"])) - 1])
                          for r in RANKS if r <= len(record["isotropic_energy"])},
        }

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(args.out_dir, "excitation_curves.npz"),
        **{f"{k}__weighted": v["weighted_energy"] for k, v in results.items()},
        **{f"{k}__isotropic": v["isotropic_energy"] for k, v in results.items()},
    )
    with open(os.path.join(args.out_dir, "excitation.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

    groups = {
        "expert_heads": ([k for k in layers if k.startswith("expert_output_mlps")], 64),
        "prompt_pooler": ([k for k in layers if k.startswith("img_prompt_pool")], 128),
    }
    aggregate = {}
    for group, (names, rank) in groups.items():
        weighted = float(np.mean([summary[n]["weighted"][str(rank)] for n in names]))
        isotropic = float(np.mean([summary[n]["isotropic"][str(rank)] for n in names]))
        aggregate[group] = {"rank": rank, "weighted": weighted, "isotropic": isotropic}
        print(f"{group:14s} rank {rank:4d}  isotropic E={isotropic:.4f}  "
              f"excitation-weighted E={weighted:.4f}", flush=True)
    with open(os.path.join(args.out_dir, "excitation_summary.json"), "w") as handle:
        json.dump(aggregate, handle, indent=2)


if __name__ == "__main__":
    main()
