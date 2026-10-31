"""Boundary agreement between the released and compressed decoders.

Dice rewards volumetric overlap and tolerates a systematically displaced contour, so
a compression claim resting on Dice alone is incomplete. This script re-scores the
same predictions with the 95th percentile Hausdorff distance and the normalised
surface distance, which respond to boundary displacement directly.

Four states are compared: the released decoder, the compressed decoder before any
optimisation, and both after the matched fine-tune. Per-image values are written so
the paired tests and the agreement plots draw on exactly these numbers.

Resolution. Dice is scale invariant and is scored at each image's native size, matching
the rest of the study. Distance metrics are not: ISIC 2016 test images span 576 to 4288
pixels in width, so a distance of five pixels means very different things across the
set and a mean over native-resolution distances would be dominated by the largest
images. Boundary metrics are therefore computed on the common 512 by 512 grid the
network operates on, with the reference resampled the same way as the prediction, and
the unit is pixels on that grid.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segmote_lite import (apply_recipe, build_segmote, count_trainable, freeze_backbone,
                          list_pairs, load_mask, preprocess_image)
from segmote_lite.evaluate import box_from_mask
from segmote_lite.metrics import all_metrics, dice

LITE = dict(expert_rank=64, pooler_rank=128, prune=True)


@torch.no_grad()
def score(model, pairs, device, image_size, prompt_free, tolerance):
    model.eval().to(device)
    rows = []
    for image_path, mask_path in pairs:
        image = preprocess_image(image_path, image_size).to(device)
        reference, resized = load_mask(mask_path, image_size)
        if resized.sum() < 1:
            continue
        embedding, intermediate = model.image_forward(image)
        prompt = {} if prompt_free else {"bboxes": box_from_mask(resized).to(device)}
        outputs, _, _ = model.forward_decoder(
            embedding, prompt, interm_embeddings=intermediate,
            test_mode=True, if_prompt=True if prompt_free else None)
        raw = outputs["masks"].float()

        # Overlap at native resolution, matching the rest of the study.
        native = F.interpolate(raw, size=reference.shape, mode="bilinear", align_corners=False)
        predicted_native = (torch.sigmoid(native)[0, 0].cpu() > 0.5).float()

        # Distances on the common grid, so a pixel means the same thing in every case.
        grid = F.interpolate(raw, size=(image_size, image_size), mode="bilinear",
                             align_corners=False)
        predicted_grid = (torch.sigmoid(grid)[0, 0].cpu() > 0.5).float()

        values = all_metrics(predicted_grid, resized, tolerance)
        values["dice"] = dice(predicted_native, reference)
        values["image"] = os.path.basename(image_path)
        rows.append(values)
    return rows


def summarise(rows):
    out = {}
    for key in ("dice", "hd95", "nsd"):
        values = np.array([r[key] for r in rows], dtype=float)
        finite = values[np.isfinite(values)]
        out[f"{key}_mean"] = float(finite.mean()) if finite.size else float("nan")
        out[f"{key}_median"] = float(np.median(finite)) if finite.size else float("nan")
        out[f"{key}_n"] = int(finite.size)
    return out


def write_rows(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "dice", "hd95", "nsd"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})


def build(args, compressed, weights):
    model = build_segmote(args.segmote_root, args.sam_checkpoint,
                          args.segmote_checkpoint, args.device, args.image_size)
    freeze_backbone(model)
    if compressed:
        apply_recipe(model, verbose=False, **LITE)
    if weights:
        blob = torch.load(weights, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model_state_dict"], strict=False)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmote_root", required=True)
    parser.add_argument("--sam_checkpoint", required=True)
    parser.add_argument("--segmote_checkpoint", required=True)
    parser.add_argument("--baseline_finetuned", default=None)
    parser.add_argument("--compressed_finetuned", default=None)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--tolerance", type=float, default=2.0)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    pairs = list_pairs(args.test_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    states = [("released", False, None), ("compressed_datafree", True, None)]
    if args.baseline_finetuned:
        states.append(("released_finetuned", False, args.baseline_finetuned))
    if args.compressed_finetuned:
        states.append(("compressed_finetuned", True, args.compressed_finetuned))

    summary = {"tolerance_px": args.tolerance, "n_images": len(pairs), "states": {}}
    for name, compressed, weights in states:
        model = build(args, compressed, weights)
        entry = {"trainable_params": count_trainable(model)}
        for mode, prompt_free in (("box", False), ("promptfree", True)):
            rows = score(model, pairs, args.device, args.image_size, prompt_free, args.tolerance)
            write_rows(os.path.join(args.out_dir, f"{name}_{mode}.csv"), rows)
            entry[mode] = summarise(rows)
            print(f"{name:22s} {mode:10s} dice {entry[mode]['dice_mean']:.4f}  "
                  f"HD95 {entry[mode]['hd95_mean']:6.2f} px  "
                  f"NSD {entry[mode]['nsd_mean']:.4f}", flush=True)
        summary["states"][name] = entry
        del model
        torch.cuda.empty_cache()

    with open(os.path.join(args.out_dir, "boundary_metrics.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[ok] wrote {os.path.join(args.out_dir, 'boundary_metrics.json')}")


if __name__ == "__main__":
    main()
