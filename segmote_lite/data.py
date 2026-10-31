"""ISIC 2016 lesion segmentation data handling.

Preprocessing reproduces the upstream SegMoTE evaluation transform: nearest-neighbour
resize to the network resolution followed by the SAM channel normalisation. Dice is
always computed against the mask at its original resolution, so resizing never
flatters the score.
"""
import glob
import os
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
PIXEL_STD = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)


def list_pairs(data_dir: str) -> List[Tuple[str, str]]:
    """Return sorted (image, mask) paths for an extracted ISIC 2016 part 1 archive."""
    image_dirs = sorted(glob.glob(os.path.join(data_dir, "*Data*")))
    mask_dirs = sorted(glob.glob(os.path.join(data_dir, "*GroundTruth*")))
    if not image_dirs or not mask_dirs:
        raise FileNotFoundError(f"no ISIC image/mask directories under {data_dir}")
    pairs = []
    for image_path in sorted(glob.glob(os.path.join(image_dirs[0], "*.jpg"))):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = os.path.join(mask_dirs[0], f"{stem}_Segmentation.png")
        if os.path.isfile(mask_path):
            pairs.append((image_path, mask_path))
    if not pairs:
        raise FileNotFoundError(f"no matched image/mask pairs under {data_dir}")
    return pairs


def preprocess_image(image_path: str, size: int = 512) -> torch.Tensor:
    """Load an RGB image and return a normalised tensor of shape (1, 3, size, size)."""
    image = Image.open(image_path).convert("RGB").resize((size, size), Image.NEAREST)
    array = torch.from_numpy(np.array(image)).float().permute(2, 0, 1)
    return ((array - PIXEL_MEAN) / PIXEL_STD).unsqueeze(0)


def load_mask(mask_path: str, size: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the binary mask at original resolution and at the network resolution."""
    mask = Image.open(mask_path).convert("L")
    original = (torch.from_numpy(np.array(mask)) > 127).float()
    resized = mask.resize((size, size), Image.NEAREST)
    resized = (torch.from_numpy(np.array(resized)) > 127).float()
    return original, resized


def split_pairs(pairs: List[Tuple[str, str]], val_fraction: float,
                split_seed: int = 0) -> Tuple[List, List]:
    """Partition pairs into training and validation subsets.

    The split is governed by ``split_seed`` alone, held fixed across runs, so that
    every seed and every model variant sees an identical partition and only the
    optimisation trajectory differs.
    """
    order = np.random.RandomState(split_seed).permutation(len(pairs))
    n_val = int(round(len(pairs) * val_fraction))
    val_idx = set(order[:n_val].tolist())
    train = [p for i, p in enumerate(pairs) if i not in val_idx]
    val = [p for i, p in enumerate(pairs) if i in val_idx]
    return train, val


class ISICSegmentation(Dataset):
    """Images and binary masks at the network resolution."""

    def __init__(self, pairs: List[Tuple[str, str]], size: int = 512):
        self.pairs = pairs
        self.size = size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        image_path, mask_path = self.pairs[index]
        image = preprocess_image(image_path, self.size)[0]
        _, resized = load_mask(mask_path, self.size)
        return image, resized.unsqueeze(0)
