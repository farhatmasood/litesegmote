"""Decoder latency, throughput, memory and multiply-accumulate counts.

The encoder is identical across variants and dominates the wall clock, so decoder
cost is timed separately from the full pipeline. Both figures are reported: the
decoder measurement isolates the effect of the compression, the end-to-end
measurement states what a user actually experiences.
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segmote_lite import (apply_recipe, attach_upstream, build_segmote, count_trainable,
                          freeze_backbone, list_pairs, load_mask, preprocess_image)
from segmote_lite.evaluate import box_from_mask

VARIANTS = {
    "baseline": dict(expert_rank=None, pooler_rank=None, prune=False),
    "lite": dict(expert_rank=64, pooler_rank=128, prune=True),
}


def decoder_macs(model, embedding, intermediate, prompt) -> float:
    """Multiply-accumulate operations in one decoder call, counted by hooks."""
    total = [0]
    handles = []

    def linear_hook(module, inputs, output):
        rows = inputs[0].numel() / inputs[0].shape[-1]
        total[0] += rows * module.in_features * module.out_features

    def conv_hook(module, inputs, output):
        kernel = module.kernel_size[0] * module.kernel_size[1]
        # A strided convolution scatters each input position over the kernel, so its
        # cost follows the input grid; a normal convolution gathers into each output
        # position and follows the output grid.
        if isinstance(module, torch.nn.ConvTranspose2d):
            positions = inputs[0].numel() / module.in_channels
            total[0] += positions * module.in_channels * module.out_channels * kernel
        else:
            positions = output.numel() / module.out_channels
            total[0] += positions * module.in_channels * module.out_channels * kernel

    for module in model.mask_decoder.modules():
        if isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            handles.append(module.register_forward_hook(conv_hook))

    with torch.no_grad():
        model.forward_decoder(embedding, prompt, interm_embeddings=intermediate, test_mode=True)
    for handle in handles:
        handle.remove()
    return total[0]


@torch.no_grad()
def time_calls(function, warmup: int, repeats: int):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.mean(samples), statistics.stdev(samples) if len(samples) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmote_root", required=True)
    parser.add_argument("--sam_checkpoint", required=True)
    parser.add_argument("--segmote_checkpoint", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    # The bounding-box helper lives in the upstream checkout, which reaches sys.path
    # when a model is built. This script derives a prompt first, so attach it here.
    attach_upstream(args.segmote_root)
    image_path, mask_path = list_pairs(args.test_dir)[0]
    image = preprocess_image(image_path, args.image_size).to(args.device)
    _, resized = load_mask(mask_path, args.image_size)
    prompt = {"bboxes": box_from_mask(resized).to(args.device)}

    results = []
    for name, config in VARIANTS.items():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = build_segmote(args.segmote_root, args.sam_checkpoint,
                              args.segmote_checkpoint, args.device, args.image_size)
        freeze_backbone(model)
        apply_recipe(model, verbose=False, **config)
        model.eval()

        with torch.no_grad():
            embedding, intermediate = model.image_forward(image)

        decoder_ms, decoder_sd = time_calls(
            lambda: model.forward_decoder(embedding, prompt, interm_embeddings=intermediate,
                                          test_mode=True),
            args.warmup, args.repeats)

        def full_pipeline():
            emb, inter = model.image_forward(image)
            model.forward_decoder(emb, prompt, interm_embeddings=inter, test_mode=True)

        total_ms, total_sd = time_calls(full_pipeline, args.warmup, max(10, args.repeats // 5))
        macs = decoder_macs(model, embedding, intermediate, prompt)

        row = {
            "variant": name,
            "trainable_params": count_trainable(model),
            "decoder_ms": round(decoder_ms, 3), "decoder_ms_sd": round(decoder_sd, 3),
            "pipeline_ms": round(total_ms, 3), "pipeline_ms_sd": round(total_sd, 3),
            "decoder_gmacs": round(macs / 1e9, 3),
            "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        }
        results.append(row)
        print(f"{name:9s} decoder {decoder_ms:6.2f} ms  pipeline {total_ms:7.2f} ms  "
              f"{row['decoder_gmacs']:6.2f} GMAC  peak {row['peak_memory_gb']:.2f} GB")
        del model

    with open(os.path.join(args.out_dir, "efficiency.json"), "w") as handle:
        json.dump({"device": torch.cuda.get_device_name(0), "results": results}, handle, indent=2)


if __name__ == "__main__":
    main()
