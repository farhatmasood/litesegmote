"""Construction of the reference SegMoTE model from released weights.

The upstream SegMoTE sources are not vendored here. ``setup_upstream.py`` clones them
at a pinned commit, and this module inserts that checkout on ``sys.path`` before
importing, so the model graph is exactly the published one.
"""
import argparse
import os
import sys
from types import SimpleNamespace

import torch

_UPSTREAM_MARKERS = ("segment_anything", "model.py", "checkpoint_utils.py")


def attach_upstream(segmote_root: str) -> str:
    """Put the upstream SegMoTE checkout on ``sys.path`` and validate it."""
    root = os.path.abspath(segmote_root)
    missing = [m for m in _UPSTREAM_MARKERS if not os.path.exists(os.path.join(root, m))]
    if missing:
        raise FileNotFoundError(
            f"{root} does not look like a SegMoTE checkout (missing {missing}). "
            "Run `python setup_upstream.py` first."
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def build_segmote(segmote_root: str, sam_checkpoint: str, segmote_checkpoint: str = None,
                  device: str = "cuda", image_size: int = 512, test_mode: bool = True):
    """Build SegMoTE and optionally load the released fine-tuned weights.

    ``sam_checkpoint`` is required because the upstream builder uses it both to
    initialise the decoder and to decide which decoder tensors stay frozen: any
    parameter already present in that file is frozen, the remainder is trainable.
    """
    attach_upstream(segmote_root)
    from segment_anything import sam_model_registry
    from model import SegMoTE
    from checkpoint_utils import upgrade_legacy_state_dict

    args = SimpleNamespace(
        image_size=image_size, sam_checkpoint=sam_checkpoint,
        model_type="vit_b", device=device,
    )
    sam = sam_model_registry["vit_b"](args).to(device)
    model = SegMoTE(sam, test_mode=test_mode, category_weights=None).to(device)

    if segmote_checkpoint is not None:
        blob = torch.load(segmote_checkpoint, map_location="cpu", weights_only=False)
        state = blob.get("model_state_dict", blob)
        state = upgrade_legacy_state_dict(state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint does not match the model graph: "
                f"{len(missing)} missing, {len(unexpected)} unexpected keys"
            )
    return model


def freeze_backbone(model) -> None:
    """Freeze every component outside the mask decoder, matching the published recipe."""
    for module in (model.image_encoder, model.prompt_encoder, model.text_model, model.text_out_dim):
        for p in module.parameters():
            p.requires_grad = False


def trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.mask_decoder.parameters() if p.requires_grad)


def default_paths() -> argparse.ArgumentParser:
    """Argument group shared by every experiment script."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--segmote_root", required=True, help="upstream SegMoTE checkout")
    ap.add_argument("--sam_checkpoint", required=True)
    ap.add_argument("--segmote_checkpoint", required=True)
    ap.add_argument("--device", default="cuda")
    return ap
