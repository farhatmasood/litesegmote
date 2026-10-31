"""Dice evaluation under bounding-box and prompt-free interaction.

Two interaction modes are exercised. Under box prompting the decoder receives a box
derived from the reference mask, which is the protocol used for the published
comparison tables. Under prompt-free operation the progressive prompt tokenisation
module supplies the sparse prompt from image features instead, so the two modes
together cover both trainable subsystems of the decoder.
"""
import csv
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .data import list_pairs, load_mask, preprocess_image


def dice(prediction: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> float:
    intersection = (prediction * reference).sum()
    return float((2 * intersection) / (prediction.sum() + reference.sum() + eps))


def box_from_mask(mask_resized: torch.Tensor) -> torch.Tensor:
    """Tight bounding box of a mask, using the upstream helper for exact parity.

    Requires the upstream checkout to be on ``sys.path`` (see ``builder.attach_upstream``).
    """
    from dataloaders.data_utils import get_bboxes_from_mask
    size = mask_resized.shape[-1]
    return get_bboxes_from_mask(mask_resized.view(1, 1, size, size))


@torch.no_grad()
def evaluate(model, pairs: List[Tuple[str, str]], device: str,
             image_size: int = 512, prompt_free: bool = False,
             per_image_csv: Optional[str] = None) -> Dict:
    """Mean Dice over ``pairs``, scored at each image's original resolution."""
    model.eval().to(device)
    scores, rows = [], []
    started = time.time()

    for image_path, mask_path in pairs:
        image = preprocess_image(image_path, image_size).to(device)
        reference, resized = load_mask(mask_path, image_size)
        if resized.sum() < 1:
            continue

        embedding, intermediate = model.image_forward(image)
        if prompt_free:
            prompt, if_prompt = {}, True
        else:
            prompt = {"bboxes": box_from_mask(resized).to(device)}
            if_prompt = None

        outputs, _, _ = model.forward_decoder(
            embedding, prompt, interm_embeddings=intermediate,
            test_mode=True, if_prompt=if_prompt,
        )
        height, width = reference.shape
        logits = F.interpolate(outputs["masks"].float(), size=(height, width),
                               mode="bilinear", align_corners=False)
        predicted = (torch.sigmoid(logits)[0, 0].cpu() > 0.5).float()

        score = dice(predicted, reference)
        scores.append(score)
        rows.append((os.path.basename(image_path), round(score, 6)))

    mean_dice = float(np.mean(scores)) if scores else 0.0
    if per_image_csv:
        os.makedirs(os.path.dirname(per_image_csv), exist_ok=True)
        with open(per_image_csv, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image", "dice"])
            writer.writerows(rows)
    return {
        "mean_dice": mean_dice,
        "n": len(scores),
        "seconds": round(time.time() - started, 1),
    }


def evaluate_both_modes(model, pairs, device, image_size=512,
                        csv_prefix: Optional[str] = None) -> Dict[str, float]:
    """Mean Dice under both interaction modes, optionally writing per-image scores."""
    box = evaluate(model, pairs, device, image_size, prompt_free=False,
                   per_image_csv=f"{csv_prefix}_box.csv" if csv_prefix else None)
    free = evaluate(model, pairs, device, image_size, prompt_free=True,
                    per_image_csv=f"{csv_prefix}_promptfree.csv" if csv_prefix else None)
    return {"box": box["mean_dice"], "prompt_free": free["mean_dice"], "n": box["n"]}


__all__ = ["dice", "evaluate", "evaluate_both_modes", "box_from_mask", "list_pairs"]
