"""Fine-tune a baseline or compressed decoder and report held-out test Dice.

Protocol. The ISIC 2016 training archive is partitioned once into an optimisation
subset and a validation subset using a fixed split seed, so every run sees the same
partition. The epoch is selected on validation Dice alone. The test archive is
touched a single time, with the selected weights, which keeps model selection
independent of the reported result.

Only mask-decoder parameters receive gradients. The network stays in evaluation mode
throughout: the frozen encoder contains dropout, and the expert router injects noise
unless deterministic routing is requested, so training mode would inject variance
that has nothing to do with the parameters being optimised.
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segmote_lite import (ISICSegmentation, apply_recipe, build_segmote, count_trainable,
                          evaluate_both_modes, freeze_backbone, list_pairs, split_pairs)
from segmote_lite.evaluate import box_from_mask

VARIANTS = {
    "baseline": dict(expert_rank=None, pooler_rank=None, vitfeat_rank=None, prune=False),
    "lite": dict(expert_rank=64, pooler_rank=128, vitfeat_rank=None, prune=True),
    "lite-deep": dict(expert_rank=64, pooler_rank=128, vitfeat_rank=192, prune=True),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def training_loss(model, images, masks, criterion, device):
    """Loss for one batch, alternating between the two interaction modes.

    Reference boxes are always supplied so that the decoder token batch matches the
    image batch. The upstream prompt encoder emits a single sparse embedding when no
    prompt is given, which is only consistent at batch size one. On prompt-free
    steps the decoder ignores the boxes and builds its prompt from image features.
    """
    images, masks = images.to(device), masks.to(device)
    with torch.no_grad():
        embedding, intermediate = model.image_forward(images)

    boxes = torch.cat([box_from_mask(masks[i, 0]) for i in range(masks.size(0))], dim=0)
    prompt = {"bboxes": boxes.to(device)}
    if_prompt = True if random.random() < 0.5 else None

    outputs, balance_loss, _ = model.forward_decoder(
        embedding, prompt, interm_embeddings=intermediate,
        test_mode=True, if_prompt=if_prompt,
    )
    loss = criterion(outputs["masks"], masks, outputs["iou_pred"])
    if torch.is_tensor(balance_loss):
        loss = loss + 0.01 * balance_loss
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmote_root", required=True)
    parser.add_argument("--sam_checkpoint", required=True)
    parser.add_argument("--segmote_checkpoint", required=True)
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--save_weights", default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="truncate the run to a few batches and images to check wiring")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.variant}_seed{args.seed}"

    model = build_segmote(args.segmote_root, args.sam_checkpoint,
                          args.segmote_checkpoint, args.device, args.image_size)
    record = apply_recipe(model, **VARIANTS[args.variant])
    freeze_backbone(model)
    model.eval()

    from utils import FocalDice_MSELoss
    criterion = FocalDice_MSELoss()

    train_pairs, val_pairs = split_pairs(list_pairs(args.train_dir),
                                         args.val_fraction, args.split_seed)
    test_pairs = list_pairs(args.test_dir)
    if args.smoke:
        train_pairs, val_pairs, test_pairs = train_pairs[:32], val_pairs[:8], test_pairs[:8]
    print(f"[data] train {len(train_pairs)} | val {len(val_pairs)} | test {len(test_pairs)}")

    loader = DataLoader(ISICSegmentation(train_pairs, args.image_size),
                        batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=args.weight_decay)

    summary_path = os.path.join(args.out_dir, f"summary_{tag}.json")
    history, best_val, best_epoch = [], -1.0, 0

    for epoch in range(1, args.epochs + 1):
        running = 0.0
        for images, masks in loader:
            optimiser.zero_grad()
            loss = training_loss(model, images, masks, criterion, args.device)
            loss.backward()
            optimiser.step()
            running += loss.item()

        val = evaluate_both_modes(model, val_pairs, args.device, args.image_size)
        val_mean = 0.5 * (val["box"] + val["prompt_free"])
        history.append({"epoch": epoch, "train_loss": running / len(loader),
                        "val_box": val["box"], "val_prompt_free": val["prompt_free"],
                        "val_mean": val_mean})
        print(f"[{tag}] epoch {epoch}/{args.epochs} loss {running/len(loader):.4f} "
              f"val box {val['box']:.4f} val prompt-free {val['prompt_free']:.4f}", flush=True)

        if val_mean <= best_val:
            continue

        # A new best on validation. Score the test set now and persist the summary,
        # rather than holding weights until the end: selection still depends only on
        # validation, but a later failure cannot destroy the completed result.
        best_val, best_epoch = val_mean, epoch
        test = evaluate_both_modes(model, test_pairs, args.device, args.image_size,
                                   csv_prefix=os.path.join(args.out_dir, f"test_{tag}"))
        summary = {
            "variant": args.variant, "seed": args.seed, "split_seed": args.split_seed,
            "trainable_params": count_trainable(model),
            "trainable_params_before": record["trainable_before"],
            "reduction": record["reduction"],
            "selected_epoch": best_epoch, "history": history,
            "test_box": test["box"], "test_prompt_free": test["prompt_free"],
            "n_test": test["n"],
            "config": {k: v for k, v in vars(args).items() if k != "out_dir"},
        }
        with open(summary_path, "w") as handle:
            json.dump(summary, handle, indent=2)
        if args.save_weights:
            os.makedirs(os.path.dirname(args.save_weights), exist_ok=True)
            torch.save({"model_state_dict": model.state_dict()}, args.save_weights)
        print(f"[{tag}] epoch {epoch} is best on validation | test box {test['box']:.4f} "
              f"| test prompt-free {test['prompt_free']:.4f}", flush=True)
        torch.cuda.empty_cache()

    # The incremental writes above stop recording at the last improving epoch. Rewrite
    # once at the end with the complete curve, keeping the test scores of the selected
    # epoch, so the stored history covers every epoch actually run.
    if os.path.exists(summary_path):
        with open(summary_path) as handle:
            summary = json.load(handle)
        summary["history"] = history
        summary["epochs_run"] = len(history)
        with open(summary_path, "w") as handle:
            json.dump(summary, handle, indent=2)

    print(f"[{tag}] finished, selected epoch {best_epoch}, "
          f"trainable {count_trainable(model)/1e6:.3f}M", flush=True)


if __name__ == "__main__":
    main()
