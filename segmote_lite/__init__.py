"""LiteSegMoTE: low-rank compression of the SegMoTE mask decoder."""
from .builder import attach_upstream, build_segmote, freeze_backbone, trainable_parameters
from .compress import apply_recipe, count_trainable, remove_unused
from .data import ISICSegmentation, list_pairs, load_mask, preprocess_image, split_pairs
from .evaluate import dice, evaluate, evaluate_both_modes
from .lowrank import (LowRankConvTranspose2d, LowRankLinear, factorize_conv_transpose2d,
                      factorize_linear, relative_error, spectrum)

__version__ = "1.0.0"

__all__ = [
    "attach_upstream", "build_segmote", "freeze_backbone", "trainable_parameters",
    "apply_recipe", "count_trainable", "remove_unused",
    "ISICSegmentation", "list_pairs", "load_mask", "preprocess_image", "split_pairs",
    "dice", "evaluate", "evaluate_both_modes",
    "LowRankLinear", "LowRankConvTranspose2d", "factorize_linear",
    "factorize_conv_transpose2d", "relative_error", "spectrum",
]
