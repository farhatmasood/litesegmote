"""The SegMoTE-Lite compression recipe.

Three reductions act on the trainable part of the mask decoder.

``prune``   removes three modules that the decoder instantiates but never calls, so
            their removal cannot alter any output.
``experts`` factorises the two square projections inside each per-expert
            hypernetwork of the token-level mixture of experts.
``pooler``  factorises the feed-forward block of the prompt tokenisation pooler.
``vitfeat`` optionally factorises the transposed convolution that expands the
            intermediate encoder features, the single widest tensor in the decoder.

Every factorisation is initialised from the released weights, so a compressed model
is usable before any further optimisation.
"""
from typing import Dict, List, Optional, Tuple

import torch.nn as nn

from .lowrank import factorize_conv_transpose2d, factorize_linear, relative_error

UNUSED_MODULES = ("expert_mlp", "text_token", "boxApoint_token")


def count_trainable(model) -> int:
    return sum(p.numel() for p in model.mask_decoder.parameters() if p.requires_grad)


def remove_unused(decoder) -> List[str]:
    """Delete modules that never appear in the decoder forward pass."""
    removed = []
    for name in UNUSED_MODULES:
        if hasattr(decoder, name):
            delattr(decoder, name)
            removed.append(name)
    return removed


def factorise_expert_heads(decoder, rank: int) -> List[Tuple[str, float]]:
    """Factorise the square layers of every per-expert output hypernetwork.

    Layer index 2 maps 768 to 96 and is already narrow, so it is left intact.
    """
    report = []
    for expert_index, head in enumerate(decoder.expert_output_mlps):
        for layer_index in (0, 1):
            original = head.layers[layer_index]
            replacement = factorize_linear(original, rank)
            report.append((
                f"expert_output_mlps.{expert_index}.layers.{layer_index}",
                relative_error(original, replacement),
            ))
            head.layers[layer_index] = replacement
    return report


def factorise_pooler(decoder, rank: int) -> List[Tuple[str, float]]:
    """Factorise the expansion and projection layers of the prompt pooler."""
    report = []
    pooler = decoder.img_prompt_pool
    for index in (0, 2):
        original = pooler.mlp[index]
        replacement = factorize_linear(original, rank)
        report.append((f"img_prompt_pool.mlp.{index}", relative_error(original, replacement)))
        pooler.mlp[index] = replacement
    return report


def factorise_vit_features(decoder, rank: int) -> List[Tuple[str, float]]:
    """Factorise the wide transposed convolution of the encoder feature adapter."""
    report = []
    original = decoder.compress_vit_feat[0]
    if not isinstance(original, nn.ConvTranspose2d):
        raise TypeError("compress_vit_feat[0] is not a transposed convolution")
    replacement = factorize_conv_transpose2d(original, rank)
    report.append(("compress_vit_feat.0", relative_error(original, replacement)))
    decoder.compress_vit_feat[0] = replacement
    return report


def apply_recipe(model, expert_rank: Optional[int] = None, pooler_rank: Optional[int] = None,
                 vitfeat_rank: Optional[int] = None, prune: bool = True,
                 verbose: bool = True) -> Dict:
    """Apply the compression recipe in place and return a record of what changed."""
    decoder = model.mask_decoder
    before = count_trainable(model)
    record = {"trainable_before": before, "removed": [], "factorised": []}

    if prune:
        record["removed"] = remove_unused(decoder)
    if expert_rank:
        record["factorised"] += factorise_expert_heads(decoder, expert_rank)
    if pooler_rank:
        record["factorised"] += factorise_pooler(decoder, pooler_rank)
    if vitfeat_rank:
        record["factorised"] += factorise_vit_features(decoder, vitfeat_rank)

    after = count_trainable(model)
    record["trainable_after"] = after
    record["reduction"] = (before - after) / before if before else 0.0

    if verbose:
        if record["removed"]:
            print(f"[prune] removed {record['removed']}")
        for name, error in record["factorised"]:
            print(f"[factorise] {name:48s} relative error {error:.3f}")
        print(f"[params] {before/1e6:.3f}M -> {after/1e6:.3f}M "
              f"({100*record['reduction']:.1f} percent reduction)")
    return record
