"""Overlap and boundary agreement metrics for binary segmentation.

Dice measures volumetric overlap and is insensitive to boundary displacement: a
prediction can score well while systematically mislocating the lesion border. Two
boundary metrics are provided alongside it.

``hd95``  the 95th percentile of the symmetric distance between the two contours,
          which reports the worst-case localisation error while discarding the
          extreme tail that a handful of stray pixels would otherwise dominate.
``nsd``   the normalised surface distance, the fraction of contour points on either
          shape lying within a tolerance of the other, which reports how much of the
          border would need manual correction.

Distances are in pixels. ISIC dermoscopy carries no physical spacing metadata, so a
spacing of one is the honest unit and results are comparable only within this study.
"""
from typing import Tuple

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

EMPTY = float("nan")


def _as_bool(mask) -> np.ndarray:
    array = mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)
    return array > 0.5


def _contour(mask: np.ndarray) -> np.ndarray:
    """Inner boundary: foreground pixels with at least one background neighbour."""
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
    return mask & ~eroded


def _surface_distances(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Distances from each contour of ``a`` to ``b`` and vice versa."""
    contour_a, contour_b = _contour(a), _contour(b)
    if not contour_a.any() or not contour_b.any():
        return np.array([]), np.array([])
    # Distance from every pixel to the nearest contour pixel of the other shape.
    dist_to_b = distance_transform_edt(~contour_b)
    dist_to_a = distance_transform_edt(~contour_a)
    return dist_to_b[contour_a], dist_to_a[contour_b]


def dice(prediction, reference, eps: float = 1e-8) -> float:
    p, r = _as_bool(prediction), _as_bool(reference)
    denominator = p.sum() + r.sum()
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(p, r).sum() / (denominator + eps))


def hd95(prediction, reference) -> float:
    """95th percentile symmetric Hausdorff distance in pixels.

    Returns NaN when either mask is empty, since the distance is undefined; callers
    aggregate with nan-aware statistics so such cases are excluded rather than
    silently scored as perfect or catastrophic.
    """
    p, r = _as_bool(prediction), _as_bool(reference)
    if not p.any() or not r.any():
        return EMPTY
    forward, backward = _surface_distances(p, r)
    if forward.size == 0 or backward.size == 0:
        return EMPTY
    return float(np.percentile(np.concatenate([forward, backward]), 95))


def nsd(prediction, reference, tolerance: float = 2.0) -> float:
    """Normalised surface distance at ``tolerance`` pixels, in [0, 1]."""
    p, r = _as_bool(prediction), _as_bool(reference)
    if not p.any() or not r.any():
        return EMPTY
    forward, backward = _surface_distances(p, r)
    if forward.size == 0 or backward.size == 0:
        return EMPTY
    within = (forward <= tolerance).sum() + (backward <= tolerance).sum()
    return float(within / (forward.size + backward.size))


def all_metrics(prediction, reference, tolerance: float = 2.0) -> dict:
    return {
        "dice": dice(prediction, reference),
        "hd95": hd95(prediction, reference),
        "nsd": nsd(prediction, reference, tolerance),
    }
